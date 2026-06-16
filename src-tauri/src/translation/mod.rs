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

const MAX_N_CTX: u32 = 8192;
const MAX_CONTEXT_CHUNKS: usize = 8;
const SAMPLING_TEMP: f32 = 0.15;
const SAMPLING_TOP_K: i32 = 40;
const SAMPLING_TOP_P: f32 = 0.90;
const SAMPLING_REP_PENALTY: f32 = 1.0;

fn build_prompt(text: &str, prev_chunks: &[(String, String)]) -> String {
    let mut p = String::from("<|turn>system\nYou are an expert translator of video subtitles (English to Russian). The text is a raw Speech-to-Text transcript with phonetic errors. You MUST rely on the 'Previous dialogue context' to understand the topic.\n\nRULES:\n1. Fix ASR errors by sound.\n2. Translate contextually.\n3. Keep the translation natural and conversational.\n4. Output ONLY the final Russian translation without quotes, notes, or explanations.\n5. Convert imperial units (feet, inches, Fahrenheit, pounds, etc.) to metric for the Russian audience (e.g. 5'7\" → 170 cm, 32°F → 0°C). Do this naturally without explicit conversion notes.<turn|>\n<|turn>user");

    if !prev_chunks.is_empty() {
        p.push_str("\n\nPrevious dialogue context:");
        for (en, ru) in prev_chunks.iter() {
            p.push_str(&format!("\nEN: {}\nRU: {}", en, ru));
        }
    }

    p.push_str(&format!("\n\nTranslate the following text into Russian. Output only the translation.\n\n{}\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", text));

    p
}

fn clean_output(output: &str) -> String {
    // Strip everything before the first Russian/Cyrillic character
    let output = if let Some(pos) = output.find(|c: char| ('А'..='я').contains(&c) || c == '«' || c == 'ё' || c == 'Ё') {
        &output[pos..]
    } else {
        let output = output.trim();
        if let Some(pos) = output.find(|c: char| c.is_alphabetic() && !c.is_whitespace()) {
            &output[pos..]
        } else {
            ""
        }
    };
    // Strip after any angle-bracket patterns: <..., <|..., ...|>
    let output = if let Some(pos) = output.find('<') {
        &output[..pos]
    } else {
        output
    };
    // Strip trailing "thought" word (artifact from <|channel>thought)
    let output = output.trim_end().strip_suffix("thought").map(|s| s.trim_end()).unwrap_or(output.trim_end());
    // Remove known prefixes
    let output = output.trim();
    let known_prefixes = ["Russian: ", "Russian : ", "Translation: ", "RU: ", "Перевод: "];
    let output = if let Some(rest) = known_prefixes.iter().find_map(|p| output.strip_prefix(p)) {
        rest.trim()
    } else {
        output
    };
    // Strip trailing English/Latin text after the last Cyrillic character
    let output = {
        let s = output.trim_end();
        if let Some(pos) = s.rfind(|c: char| ('А'..='я').contains(&c) || c == '«' || c == 'Ё' || c == 'ё') {
            let after_cyr = &s[pos..];
            let cyr_len = after_cyr.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            let tail = &s[pos + cyr_len..];
            let allowed_bytes: usize = tail.chars().take_while(|c| {
                c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ':' | ';' | '-' | '—' | '…' | ')' | ']' | '}' | '»' | '"')
            }).map(|c| c.len_utf8()).sum();
            if allowed_bytes < tail.len() {
                let keep = pos + cyr_len + allowed_bytes;
                s[..keep].trim_end().to_string()
            } else {
                s.to_string()
            }
        } else {
            s.to_string()
        }
    };
    // If multiple lines remain, model likely generated multiple translation candidates;
    // keep only the last (most refined) line
    let output = output.trim();
    let lines: Vec<&str> = output.lines().filter(|l| !l.trim().is_empty()).collect();
    let output = if lines.len() > 1 {
        lines.last().unwrap().trim()
    } else {
        output
    };
    output.trim().to_string()
}

fn generate<'a>(
    model: &'a LlamaModel,
    ctx: &mut LlamaContext<'a>,
    sampler: &mut LlamaSampler,
    prompt: &str,
    eos: LlamaToken,
    max_new: usize,
) -> Result<Vec<LlamaToken>> {
    let tokens = model.str_to_token(prompt, AddBos::Always)?;

    let mut batch_llm = LlamaBatch::new(tokens.len(), 1);
    for (i, t) in tokens.iter().enumerate() {
        batch_llm.add(*t, i as i32, &[0], i == tokens.len() - 1)?;
    }
    ctx.decode(&mut batch_llm)?;

    let mut output_toks = Vec::new();
    let mut ctx_pos = batch_llm.n_tokens() as i32;

    for _ in 0..max_new {
        let mut cur_p = ctx.token_data_array();
        sampler.apply(&mut cur_p);

        let token = match cur_p.selected_token() {
            Some(t) => t,
            None => break,
        };

        if token == eos {
            break;
        }

        sampler.accept(token);
        output_toks.push(token);

        // Early stopping: if model tries to open a new channel or turn, stop
        let current_text = decode_tokens(model, &output_toks);
        if current_text.contains("<|channel>") || current_text.contains("<turn|>") || current_text.contains("\n\n") {
            break;
        }

        let mut nb = LlamaBatch::new(1, 1);
        if let Err(e) = nb.add(token, ctx_pos, &[0], true) {
            log::warn!("Перевод: ошибка batch.add: {:#}", e);
            break;
        }
        if let Err(e) = ctx.decode(&mut nb) {
            log::warn!("Перевод: ошибка decode: {:#}", e);
            break;
        }
        ctx_pos += 1;
    }

    Ok(output_toks)
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
        LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(MAX_N_CTX))
            .with_n_batch(MAX_N_CTX),
    )?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(512, SAMPLING_REP_PENALTY, 0.0, 0.0),
        LlamaSampler::top_k(SAMPLING_TOP_K),
        LlamaSampler::top_p(SAMPLING_TOP_P, 1),
        LlamaSampler::temp(SAMPLING_TEMP),
        LlamaSampler::dist(42),
    ]);

    let eos = model.token_eos();

    let mut result: Vec<SubtitleChunk> = Vec::new();
    let mut prev_chunks: Vec<(String, String)> = Vec::new();

    for chunk in chunks {
        if crate::is_cancelled() {
            anyhow::bail!("Перевод отменён пользователем");
        }

        if chunk.text.trim().is_empty() {
            result.push(chunk.clone());
            continue;
        }

        let prompt = build_prompt(&chunk.text, &prev_chunks);

        let tokens = match model.str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Перевод: ошибка токенизации: {:#}", e);
                result.push(SubtitleChunk {
                    text: format!("[{}]", chunk.text),
                    ..chunk.clone()
                });
                prev_chunks.push((chunk.text.clone(), String::new()));
                if prev_chunks.len() > MAX_CONTEXT_CHUNKS {
                    prev_chunks.remove(0);
                }
                continue;
            }
        };

        let max_new = 150.min(MAX_N_CTX as usize - tokens.len() - 50);
        if max_new < 16 {
            log::warn!("Перевод: промпт слишком длинный ({} токенов), fallback", tokens.len());
            result.push(SubtitleChunk {
                text: format!("[{}]", chunk.text),
                ..chunk.clone()
            });
            prev_chunks.push((chunk.text.clone(), String::new()));
            if prev_chunks.len() > MAX_CONTEXT_CHUNKS {
                prev_chunks.remove(0);
            }
            continue;
        }

        ctx_llm.clear_kv_cache();
        sampler.reset();

        let toks = generate(&model, &mut ctx_llm, &mut sampler, &prompt, eos, max_new)?;
        let raw = decode_tokens(&model, &toks);
        let chunk_preview: String = chunk.text.chars().take(20).collect();
        let raw_start: String = raw.chars().take(80).collect();
        let raw_end: String = raw.chars().rev().take(40).collect::<Vec<_>>().into_iter().rev().collect();
        log::info!("Перевод: чанк '{}' -> {} токенов, raw_start='{}', raw_end='{}'", chunk_preview, toks.len(), raw_start, raw_end);
        let output = clean_output(&raw);

        let translated = if output.is_empty() {
            log::warn!("Перевод: пустой вывод для '{}' ({} токенов)", chunk.text, toks.len());
            SubtitleChunk {
                text: format!("[{}]", chunk.text),
                ..chunk.clone()
            }
        } else {
            SubtitleChunk {
                text: output,
                ..chunk.clone()
            }
        };

        let ru_trimmed = if translated.text.chars().count() > 200 {
            translated.text.chars().take(200).collect::<String>()
        } else {
            translated.text.clone()
        };
        prev_chunks.push((chunk.text.clone(), ru_trimmed));
        if prev_chunks.len() > MAX_CONTEXT_CHUNKS {
            prev_chunks.remove(0);
        }
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
        let prompt = build_prompt("Hello world", &[]);
        assert!(prompt.contains("Hello world"));
        assert!(prompt.contains("<|turn>system"));
        assert!(prompt.contains("<|turn>user"));
        assert!(prompt.contains("<turn|>"));
        assert!(prompt.contains("<|turn>model"));
        assert!(prompt.contains("expert translator"));
        assert!(!prompt.contains("Fix any ASR errors"));
    }

    #[test]
    fn test_build_prompt_with_context() {
        let ctx = vec![("First EN text".to_string(), "Первая".to_string())];
        let prompt = build_prompt("Second", &ctx);
        assert!(prompt.contains("Second"));
        assert!(prompt.contains("Previous dialogue context"));
        assert!(prompt.contains("Первая"));
        assert!(prompt.contains("First EN text"));
        assert!(prompt.contains("EN:"));
        assert!(prompt.contains("RU:"));
    }

    #[test]
    fn test_build_prompt_with_multiple_contexts() {
        let ctx = vec![
            ("First EN".to_string(), "Первая".to_string()),
            ("Second EN".to_string(), "Вторая".to_string()),
            ("Third EN".to_string(), "Третья".to_string()),
        ];
        let prompt = build_prompt("Fourth", &ctx);
        assert!(prompt.contains("Fourth"));
        assert!(prompt.contains("First EN"));
        assert!(prompt.contains("Second EN"));
        assert!(prompt.contains("Third EN"));
        assert!(prompt.contains("Первая"));
        assert!(prompt.contains("Вторая"));
        assert!(prompt.contains("Третья"));
    }

    #[test]
    fn test_clean_output_plain_text() {
        assert_eq!(clean_output("Привет мир"), "Привет мир");
    }

    #[test]
    fn test_clean_output_strips_garbage_prefix() {
        let out = clean_output("<|channel>thought<channel|>Привет мир");
        assert_eq!(out, "Привет мир", "garbage prefix stripped");
    }

    #[test]
    fn test_clean_output_strips_turn_end() {
        let out = clean_output("Привет мир<turn|>\n<|turn>model");
        assert_eq!(out, "Привет мир", "turn suffix stripped");
    }

    #[test]
    fn test_clean_output_empty() {
        assert_eq!(clean_output(""), "");
    }

    #[test]
    fn test_clean_output_prefixes() {
        assert_eq!(clean_output("Russian: Привет"), "Привет");
        assert_eq!(clean_output("Translation: Как дела?"), "Как дела?");
        assert_eq!(clean_output("RU: Привет"), "Привет");
        assert_eq!(clean_output("Перевод: Привет"), "Привет");
    }

    #[test]
    fn test_clean_output_trailing_english() {
        assert_eq!(clean_output("Привет мир (let's go with)"), "Привет мир");
        assert_eq!(clean_output("Привет мир. (Thought) yeah"), "Привет мир.");
        assert_eq!(clean_output("Привет мир High confidence"), "Привет мир");
        assert_eq!(clean_output("Привет мир. (Let's go with"), "Привет мир.");
        assert_eq!(clean_output("девочка./крошка. (Let's go with"), "девочка./крошка.");
    }

    #[test]
    fn test_clean_output_keeps_valid_russian() {
        assert_eq!(clean_output("Привет мир."), "Привет мир.");
        assert_eq!(clean_output("Как дела?"), "Как дела?");
        assert_eq!(clean_output("Отлично!"), "Отлично!");
    }
}
