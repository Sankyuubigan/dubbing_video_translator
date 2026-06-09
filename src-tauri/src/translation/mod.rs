use crate::comm::{PipelineContext, SubtitleChunk};
use anyhow::Result;
use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;

/// Парсит вывод LLM вида:
/// [0] translated text
/// [1] more text
/// и возвращает HashMap<index, text>.
fn parse_tagged_translation(output: &str) -> std::collections::HashMap<usize, String> {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"^\[(\d+)\]\s*(.*)").unwrap());
    let mut result = std::collections::HashMap::new();
    for line in output.lines() {
        if let Some(caps) = re.captures(line.trim()) {
            let idx: usize = caps[1].parse().unwrap_or(usize::MAX);
            let text = caps[2].trim().to_string();
            if idx != usize::MAX && !text.is_empty() {
                result.insert(idx, text);
            }
        }
    }
    result
}

pub fn translate(ctx: PipelineContext) -> Result<PipelineContext> {
    let chunks = match ctx.subtitle_chunks.as_ref() {
        Some(c) => c,
        None => {
            log::error!("Перевод: Нет чанков STT");
            anyhow::bail!("Нет чанков STT");
        }
    };
    let model_path = match ctx.config.gguf_model_path.as_ref() {
        Some(p) => p,
        None => {
            log::warn!("Перевод: Не выбран GGUF-файл модели, пропускаем перевод");
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
    let model = match LlamaModel::load_from_file(backend, model_path, &LlamaModelParams::default())
    {
        Ok(m) => m,
        Err(e) => {
            log::warn!("Перевод: Ошибка загрузки модели: {:#} — пропускаем перевод", e);
            return Ok(PipelineContext {
                translated_chunks: Some(chunks.clone()),
                ..ctx
            });
        }
    };

    let n_vocab = model.n_vocab() as i32;
    let eos = model.token_eos().0;
    log::info!("Перевод: {} чанков, n_vocab={}, eos={}", chunks.len(), n_vocab, eos);
    let mut result = Vec::new();

    for batch in chunks.chunks(5) {
        let batch_text = format_batch(batch);

        let mut ctx_llm = match model.new_context(
            backend,
            LlamaContextParams::default().with_n_ctx(NonZeroU32::new(16384)),
        ) {
            Ok(c) => c,
            Err(e) => {
                log::error!("Перевод: Ошибка создания контекста: {:#}", e);
                anyhow::bail!("Ошибка создания контекста: {:#}", e);
            }
        };

        let prompt = format!(
            "Translate to Russian. Keep the tags [0], [1], etc. Do NOT output JSON. Output ONLY:\n\
             [0] translation\n[1] translation\n\n\
             Input:\n{}\n\n\
             Output:",
            batch_text
        );

        let tokens = match model.str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Перевод: Ошибка токенизации: {:#}", e);
                anyhow::bail!("Ошибка токенизации: {:#}", e);
            }
        };

        let input_token_count = tokens.len();

        if tokens.len() > 14000 {
            log::warn!("Перевод: промпт слишком длинный ({} токенов), уменьшаем batch", tokens.len());
            for chunk in batch {
                if chunk.text.trim().is_empty() {
                    continue;
                }
                result.push(SubtitleChunk {
                    text: format!("[{}]", chunk.text),
                    ..chunk.clone()
                });
            }
            continue;
        }

        let mut batch_llm = LlamaBatch::new(tokens.len(), 1);
        for (i, t) in tokens.iter().enumerate() {
            if let Err(e) = batch_llm.add(*t, i as i32, &[0], i == tokens.len() - 1) {
                log::error!("Перевод: Ошибка добавления токена: {:#}", e);
                anyhow::bail!("Ошибка добавления токена: {:#}", e);
            }
        }

        if let Err(e) = ctx_llm.decode(&mut batch_llm) {
            log::error!("Перевод: Ошибка декодирования: {:#}", e);
            anyhow::bail!("Ошибка декодирования: {:#}", e);
        }

        let mut output_toks = Vec::new();
        let mut pos = batch_llm.n_tokens() as i32;
        let mut generation_ok = true;

        let max_new = (input_token_count * 2).max(128).min(4096);
        for _ in 0..max_new {
            let logits_last = ctx_llm.get_logits();

            let next_idx = logits_last
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i32)
                .unwrap_or(0);

            if next_idx < 0 || next_idx >= n_vocab {
                log::warn!("Перевод: токен {} вне диапазона vocab, прерываем", next_idx);
                break;
            }
            if next_idx == eos {
                break;
            }
            output_toks.push(LlamaToken(next_idx));

            let mut nb = LlamaBatch::new(1, 1);
            if let Err(e) = nb.add(LlamaToken(next_idx), pos, &[0], true) {
                log::warn!("Перевод: Ошибка добавления токена: {:#}, прерываем генерацию", e);
                generation_ok = false;
                break;
            }
            if let Err(e) = ctx_llm.decode(&mut nb) {
                log::warn!("Перевод: Ошибка декодирования: {:#}, прерываем генерацию", e);
                generation_ok = false;
                break;
            }
            pos += 1;
        }

        let output = if generation_ok {
            decode_tokens(&model, &output_toks)
        } else {
            String::new()
        };

        let parsed = parse_tagged_translation(&output);
        let has_repeats = detect_repeated_translations(&parsed, 3);
        if parsed.is_empty() || has_repeats {
            if has_repeats {
                log::warn!("Перевод: LLM зациклилась — повторяющиеся переводы ({} записей, {} уникальных)",
                    parsed.len(),
                    parsed.values().collect::<std::collections::HashSet<_>>().len());
            } else {
                log::warn!("Перевод: LLM не вернула тегированный вывод ({} bytes). Вывод LLM:\n{}",
                    output.len(), &output);
            }
            log::warn!("Перевод: используем оригинал");
            for chunk in batch {
                if chunk.text.trim().is_empty() {
                    continue;
                }
                result.push(SubtitleChunk {
                    text: format!("[{}]", chunk.text),
                    ..chunk.clone()
                });
            }
            continue;
        }

        for (idx, chunk) in batch.iter().enumerate() {
            if chunk.text.trim().is_empty() {
                continue;
            }
            let text = match parsed.get(&idx) {
                Some(t) => t.clone(),
                None => {
                    log::warn!("Перевод: LLM пропустила индекс {}, оставляем оригинал", idx);
                    chunk.text.clone()
                }
            };
            result.push(SubtitleChunk {
                text,
                ..chunk.clone()
            });
        }
    }

    log::info!("Перевод: готово {} чанков", result.len());
    Ok(PipelineContext {
        translated_chunks: Some(result),
        ..ctx
    })
}

fn decode_tokens(model: &LlamaModel, tokens: &[LlamaToken]) -> String {
    let n_vocab = model.n_vocab() as i32;
    let eos = model.token_eos().0;
    let bos = model.token_bos().0;
    let mut out = Vec::new();

    for &token in tokens {
        let id = token.0;
        if id < 0 || id >= n_vocab {
            log::warn!("Перевод: токен {} вне диапазона vocab (0..{}), пропускаем", id, n_vocab);
            continue;
        }
        if id == eos || id == bos {
            continue;
        }

        let mut decoder = UTF_8.new_decoder();
        if let Ok(piece) = model.token_to_piece(token, &mut decoder, false, None) {
            if piece.chars().any(|c| c as u32 > 0xFF) {
                out.extend_from_slice(piece.as_bytes());
            } else {
                out.extend(piece.chars().map(|c| c as u8));
            }
        }
    }

    String::from_utf8_lossy(&out).into_owned()
}

/// Детектит зацикливание LLM: если больше `max_repeat` переводов совпадают — галлюцинация.
fn detect_repeated_translations(parsed: &std::collections::HashMap<usize, String>, max_repeat: usize) -> bool {
    if parsed.len() < max_repeat {
        return false;
    }
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for text in parsed.values() {
        let entry = text.trim();
        if entry.is_empty() {
            continue;
        }
        *counts.entry(entry).or_insert(0) += 1;
        if counts[entry] >= max_repeat {
            return true;
        }
    }
    false
}

fn format_batch(chunks: &[SubtitleChunk]) -> String {
    let mut out = String::new();
    for (i, c) in chunks.iter().enumerate() {
        if c.text.trim().is_empty() {
            continue;
        }
        out.push_str(&format!("[{}] {}\n", i, c.text));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tagged_translation_simple() {
        let input = "[0] Привет мир\n[1] Это тест\n";
        let result = parse_tagged_translation(input);
        assert_eq!(result.len(), 2);
        assert_eq!(result.get(&0).unwrap(), "Привет мир");
        assert_eq!(result.get(&1).unwrap(), "Это тест");
    }

    #[test]
    fn test_parse_tagged_translation_extra_text() {
        let input = "Here is the translation:\n[0] Привет\n[1] Как дела?\nDone!";
        let result = parse_tagged_translation(input);
        assert_eq!(result.len(), 2);
        assert_eq!(result.get(&0).unwrap(), "Привет");
    }

    #[test]
    fn test_parse_tagged_translation_empty() {
        assert!(parse_tagged_translation("").is_empty());
        assert!(parse_tagged_translation("no tags here").is_empty());
    }

    #[test]
    fn test_parse_tagged_translation_skipped_index() {
        let input = "[0] Привет\n[2] Пропущен 1\n";
        let result = parse_tagged_translation(input);
        assert_eq!(result.len(), 2);
        assert!(result.get(&1).is_none());
    }

    #[test]
    fn test_format_batch() {
        let chunks = vec![
            SubtitleChunk {
                start_sec: 0.0,
                end_sec: 1.0,
                text: "Hello".to_string(),
                speaker_id: None,
                word_timestamps: None,
            },
            SubtitleChunk {
                start_sec: 1.0,
                end_sec: 2.0,
                text: "World".to_string(),
                speaker_id: None,
                word_timestamps: None,
            },
        ];
        let output = format_batch(&chunks);
        assert_eq!(output, "[0] Hello\n[1] World\n");
    }

    #[test]
    fn test_format_batch_skips_empty() {
        let chunks = vec![
            SubtitleChunk {
                start_sec: 0.0,
                end_sec: 1.0,
                text: "".to_string(),
                speaker_id: None,
                word_timestamps: None,
            },
            SubtitleChunk {
                start_sec: 1.0,
                end_sec: 2.0,
                text: "Hello".to_string(),
                speaker_id: None,
                word_timestamps: None,
            },
        ];
        let output = format_batch(&chunks);
        assert_eq!(output, "[0] Hello\n");
    }
}
