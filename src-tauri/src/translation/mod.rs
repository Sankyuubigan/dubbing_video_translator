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

/// Tries to parse JSON array from LLM output with cleanup for common errors.
fn parse_llm_json(output: &str) -> Option<Vec<SubtitleChunk>> {
    let js = output.find('[')?;
    let je = output[js..].rfind(']')?;
    let raw = output[js..=js + je].to_string();
    let raw = raw.replace("\"\"", "\"");
    if let Ok(parsed) = serde_json::from_str::<Vec<SubtitleChunk>>(&raw) {
        return Some(parsed);
    }
    let cleaned = fix_json_quotes(&raw);
    if let Ok(parsed) = serde_json::from_str::<Vec<SubtitleChunk>>(&cleaned) {
        return Some(parsed);
    }
    let mut results = Vec::new();
    let mut pos = 0;
    let bytes = raw.as_bytes();
    while pos < bytes.len() {
        let obj_start = match bytes[pos..].iter().position(|&b| b == b'{') {
            Some(i) => pos + i,
            None => break,
        };
        let mut depth = 0i32;
        let mut obj_end: Option<usize> = None;
        for i in obj_start..bytes.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => { depth -= 1; if depth == 0 { obj_end = Some(i + 1); break; } }
                _ => {}
            }
        }
        let obj_end = match obj_end { Some(e) => e, None => break };
        let obj_str = &raw[obj_start..obj_end];
        let obj_clean = fix_json_quotes(obj_str);
        if let Ok(chunk) = serde_json::from_str::<SubtitleChunk>(&obj_clean) {
            results.push(chunk);
        }
        pos = obj_end;
    }
    if !results.is_empty() { return Some(results); }
    None
}

/// Escapes unescaped double quotes inside JSON string values (heuristic).
fn fix_json_quotes(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 32);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            out.push('"');
            i += 1;
            if i >= 2 && bytes[i - 2] == b'\\' { continue; }
            loop {
                if i >= bytes.len() { break; }
                if bytes[i] == b'\\' {
                    out.push('\\'); i += 1;
                    if i < bytes.len() { out.push(bytes[i] as char); i += 1; }
                    continue;
                }
                if bytes[i] == b'"' { out.push('"'); i += 1; break; }
                out.push(bytes[i] as char); i += 1;
            }
        } else {
            out.push(bytes[i] as char); i += 1;
        }
    }
    out
}

pub fn translate(ctx: PipelineContext) -> Result<PipelineContext> {
    let chunks = match ctx.subtitle_chunks.as_ref() {
        Some(c) => c,
        None => {
            log::error!("Перевод: Нет чанков STT");
            anyhow::bail!("Нет чанков STT");
        }
    };
    let speakers = ctx.speaker_segments.as_ref();
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

    for batch in chunks.chunks(10) {
        let batch_json = format_batch(batch, speakers);

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
            "Translate the following English text to Russian. Output ONLY a valid JSON array, no extra text.\n\
             IMPORTANT: If the translated text contains double quotes, escape them with backslash.\n\
             Format: [{{\"start_sec\":0,\"end_sec\":1,\"text\":\"translated text\",\"speaker_id\":\"S1\"}}]\n\n\
             Input: {}\n\n\
             Output:",
            batch_json
        );

        let tokens = match model.str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Перевод: Ошибка токенизации: {:#}", e);
                anyhow::bail!("Ошибка токенизации: {:#}", e);
            }
        };

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

        for _ in 0..4096 {
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

        if let Some(parsed) = parse_llm_json(&output) {
            result.extend(parsed);
            continue;
        }

        log::warn!("Перевод: LLM JSON не парсится ({} bytes). Вывод LLM:\n{}",
            output.len(), &output);
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
                // Already valid UTF-8 (Cyrillic, etc.) — take bytes as-is
                out.extend_from_slice(piece.as_bytes());
            } else {
                // Byte-level token: Latin-1 chars → bytes → valid UTF-8
                out.extend(piece.chars().map(|c| c as u8));
            }
        }
    }

    String::from_utf8_lossy(&out).into_owned()
}

fn format_batch(
    chunks: &[SubtitleChunk],
    speakers: Option<&Vec<crate::comm::SpeakerSegment>>,
) -> String {
    let mut out = String::from("[\n");
    for (i, c) in chunks.iter().enumerate() {
        let s = speakers
            .and_then(|sp| {
                sp.iter()
                    .find(|s| (s.start_sec - c.start_sec).abs() < 0.5)
                    .map(|s| s.speaker_id.as_str())
            })
            .unwrap_or("S?");
        let escaped = c.text
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\t', "\\t")
            .replace('\r', "\\r");
        out.push_str(&format!(
            "  {{\"start_sec\":{},\"end_sec\":{},\"text\":\"{}\",\"speaker_id\":\"{}\"}}",
            c.start_sec,
            c.end_sec,
            escaped,
            s
        ));
        if i < chunks.len() - 1 {
            out.push(',');
        }
        out.push('\n');
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_json_quotes_no_change() {
        let input = r#"{"start_sec":0,"end_sec":8,"text":"hello world","speaker_id":"S1"}"#;
        let result = fix_json_quotes(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_fix_json_quotes_unescaped_inside() {
        let input = r#"{"start_sec":0,"end_sec":8,"text":"say "hello" world","speaker_id":"S1"}"#;
        let expected = r#"{"start_sec":0,"end_sec":8,"text":"say \"hello\" world","speaker_id":"S1"}"#;
        let result = fix_json_quotes(input);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_fix_json_quotes_already_escaped() {
        let input = r#"{"start_sec":0,"end_sec":8,"text":"say \"hello\" world","speaker_id":"S1"}"#;
        let result = fix_json_quotes(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_fix_json_quotes_with_backslash() {
        let input = r#"{"start_sec":0,"end_sec":8,"text":"path\\to\\file","speaker_id":"S1"}"#;
        let result = fix_json_quotes(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_parse_llm_json_valid() {
        let input = r#"[{"start_sec":0.0,"end_sec":8.0,"text":"Привет мир","speaker_id":"S?"}]"#;
        let result = parse_llm_json(input);
        assert!(result.is_some());
        let chunks = result.unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Привет мир");
        assert!((chunks[0].start_sec - 0.0).abs() < 0.01);
        assert!((chunks[0].end_sec - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_llm_json_with_extra_text() {
        let input = r#"Here is the translation: [{"start_sec":0.0,"end_sec":8.0,"text":"Привет","speaker_id":"S?"}]"#;
        let result = parse_llm_json(input);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_parse_llm_json_unescaped_quotes() {
        let input = r#"Here: [{"start_sec":0.0,"end_sec":8.0,"text":"say "hello" world","speaker_id":"S?"}]"#;
        let result = parse_llm_json(input);
        assert!(result.is_some(), "Should recover from unescaped quotes");
        assert_eq!(result.unwrap()[0].text, "say \"hello\" world");
    }

    #[test]
    fn test_parse_llm_json_empty() {
        assert!(parse_llm_json("").is_none());
        assert!(parse_llm_json("no brackets here").is_none());
    }
}
