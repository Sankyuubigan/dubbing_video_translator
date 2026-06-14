use crate::comm::{PipelineContext, SubtitleChunk};
use anyhow::Result;
use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;

const MAX_N_CTX: u32 = 1024;

fn build_prompt(text: &str) -> String {
    format!(
        "<start_of_turn>user\nTranslate the following text into Russian. Fix STT errors. Output only the translation — no explanations, no options.\n\n{}<end_of_turn>\n<start_of_turn>model\n",
        text
    )
}

fn generate<'a>(
    model: &'a LlamaModel,
    ctx: &mut LlamaContext<'a>,
    sampler: &mut LlamaSampler,
    prompt: &str,
) -> Result<(Vec<LlamaToken>, bool)> {
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let input_token_count = tokens.len();

    let mut batch_llm = LlamaBatch::new(tokens.len(), 1);
    for (i, t) in tokens.iter().enumerate() {
        batch_llm.add(*t, i as i32, &[0], i == tokens.len() - 1)?;
    }
    ctx.decode(&mut batch_llm)?;

    let eos = model.token_eos();
    let max_new = (input_token_count / 2).max(64).min(1024);
    let mut output_toks = Vec::new();
    let mut generation_ok = true;
    let mut ctx_pos = batch_llm.n_tokens() as i32;

    for _ in 0..max_new {
        let mut cur_p = ctx.token_data_array();
        sampler.apply(&mut cur_p);

        let token = match cur_p.selected_token() {
            Some(t) => t,
            None => {
                log::warn!("Перевод: ни один токен не выбран");
                generation_ok = false;
                break;
            }
        };
        sampler.accept(token);

        if token == eos || token == LlamaToken(1) || token == LlamaToken(213) {
            break;
        }
        output_toks.push(token);

        let mut nb = LlamaBatch::new(1, 1);
        if let Err(e) = nb.add(token, ctx_pos, &[0], true) {
            log::warn!("Перевод: ошибка batch.add: {:#}", e);
            generation_ok = false;
            break;
        }
        if let Err(e) = ctx.decode(&mut nb) {
            log::warn!("Перевод: ошибка decode: {:#}", e);
            generation_ok = false;
            break;
        }
        ctx_pos += 1;
    }

    Ok((output_toks, generation_ok))
}

fn decode_tokens(model: &LlamaModel, tokens: &[LlamaToken]) -> String {
    let mut decoder = UTF_8.new_decoder();
    let mut result = String::new();
    for &token in tokens {
        if let Ok(piece) = model.token_to_piece(token, &mut decoder, false, None) {
            result.push_str(&piece);
        }
    }
    result
}

pub fn translate(ctx: PipelineContext) -> Result<PipelineContext> {
    let chunks = match ctx.subtitle_chunks.as_ref() {
        Some(c) => c,
        None => {
            log::error!("Перевод: нет чанков STT");
            anyhow::bail!("Нет чанков STT");
        }
    };
    let model_path = match ctx.config.gguf_model_path.as_ref() {
        Some(p) => p,
        None => {
            log::warn!("Перевод: не выбран GGUF-файл модели, пропускаем перевод");
            return Ok(PipelineContext {
                translated_chunks: Some(chunks.clone()),
                ..ctx
            });
        }
    };

    if !Path::new(model_path).exists() {
        log::warn!("Перевод: GGUF модель не найдена: {}, пропускаем перевод", model_path);
        return Ok(PipelineContext {
            translated_chunks: Some(chunks.clone()),
            ..ctx
        });
    }

    log::info!("Перевод: загружаем LLM из {}", model_path);

    let backend = match crate::get_llama_backend() {
        Ok(b) => b,
        Err(e) => {
            log::warn!("Перевод: {} — пропускаем перевод", e);
            return Ok(PipelineContext {
                translated_chunks: Some(chunks.clone()),
                ..ctx
            });
        }
    };
    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
    let model = match LlamaModel::load_from_file(backend, model_path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("Перевод: ошибка загрузки модели: {:#} — пропускаем перевод", e);
            return Ok(PipelineContext {
                translated_chunks: Some(chunks.clone()),
                ..ctx
            });
        }
    };

    log::info!("Перевод: {} чанков", chunks.len());

    let mut ctx_llm = model.new_context(
        backend,
        LlamaContextParams::default().with_n_ctx(NonZeroU32::new(MAX_N_CTX)),
    )?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(-1, 1.05, 0.0, 0.0),
        LlamaSampler::temp(0.0),
        LlamaSampler::dist(42),
    ]);

    let mut result: Vec<SubtitleChunk> = Vec::new();

    for chunk in chunks {
        if crate::is_cancelled() {
            anyhow::bail!("Перевод отменён пользователем");
        }

        if chunk.text.trim().is_empty() {
            result.push(chunk.clone());
            continue;
        }

        let prompt = build_prompt(&chunk.text);

        let tokens = match model.str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Перевод: ошибка токенизации: {:#}", e);
                result.push(SubtitleChunk {
                    text: format!("[{}]", chunk.text),
                    ..chunk.clone()
                });
                continue;
            }
        };

        if tokens.len() + 128 > MAX_N_CTX as usize {
            log::warn!("Перевод: промпт слишком длинный ({} токенов), fallback", tokens.len());
            result.push(SubtitleChunk {
                text: format!("[{}]", chunk.text),
                ..chunk.clone()
            });
            continue;
        }

        ctx_llm.clear_kv_cache();
        sampler.reset();

        let (toks, ok) = generate(&model, &mut ctx_llm, &mut sampler, &prompt)?;
        let output = if ok {
            decode_tokens(&model, &toks)
        } else {
            log::warn!("Перевод: генерация не удалась для '{}'", chunk.text);
            String::new()
        };

        let text = output.trim();
        let translated = if text.is_empty() {
            log::warn!("Перевод: пустой вывод для '{}', fallback", chunk.text);
            SubtitleChunk {
                text: format!("[{}]", chunk.text),
                ..chunk.clone()
            }
        } else {
            SubtitleChunk {
                text: text.to_string(),
                ..chunk.clone()
            }
        };

        result.push(translated);
    }

    log::info!("Перевод: готово {} чанков", result.len());
    Ok(PipelineContext {
        translated_chunks: Some(result),
        ..ctx
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_format() {
        let prompt = build_prompt("Hello world");
        assert!(prompt.contains("Hello world"));
        assert!(prompt.contains("Translate the following text into Russian"));
        assert!(prompt.contains("<start_of_turn>user"));
        assert!(prompt.contains("<start_of_turn>model"));
        assert!(prompt.contains("<end_of_turn>"));
    }
}
