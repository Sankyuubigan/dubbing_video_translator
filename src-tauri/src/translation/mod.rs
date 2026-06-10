use crate::comm::{PipelineContext, SubtitleChunk};
use anyhow::Result;
use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use serde::Deserialize;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::Path;

fn has_cyrillic(s: &str) -> bool {
    s.chars().any(|c| matches!(c, '\u{0400}'..='\u{04FF}' | '\u{0500}'..='\u{052F}' | '\u{2DE0}'..='\u{2DFF}' | '\u{A640}'..='\u{A69F}'))
}

fn extract_ru_texts(output: &str) -> Vec<String> {
    let mut results = Vec::new();
    let mut byte_pos = 0;
    let bs = output.as_bytes();

    while byte_pos < bs.len() {
        if !output.is_char_boundary(byte_pos) {
            byte_pos += 1;
            continue;
        }

        let remaining = &output[byte_pos..];
        let key_start = remaining.find('"');
        let key_start = match key_start { Some(p) => byte_pos + p, None => break };
        let after_key = key_start + 1;
        if after_key + 4 > bs.len() { break; }
        if !output.is_char_boundary(after_key) { byte_pos = after_key; continue; }

        let (key_end, is_ru) = {
            let k = &output[after_key..].find('"');
            match k {
                Some(0) => { byte_pos = after_key + 1; continue; }
                Some(p) => {
                    let key_text = &output[after_key..after_key + p];
                    let lower = key_text.to_lowercase();
                    let ru = lower.starts_with("ru") || lower.starts_with("ру");
                    if ru || lower.starts_with("en") || lower.starts_with("src") || lower.starts_with("source") {
                        (after_key + p, ru)
                    } else {
                        byte_pos = after_key + 1;
                        continue;
                    }
                }
                None => break,
            }
        };
        if !output.is_char_boundary(key_end) { byte_pos = key_end + 1; continue; }

        let after_key_close = output[key_end..].find(|c| c == ':' || c == '：');
        let after_colon = match after_key_close {
            Some(p) => key_end + p + 1,
            None => { byte_pos = key_end + 1; continue; }
        };
        if !output.is_char_boundary(after_colon) { byte_pos = after_colon; continue; }

        let value_quote = output[after_colon..].find('"');
        let value_start = match value_quote {
            Some(p) => after_colon + p + 1,
            None => { byte_pos = after_colon; continue; }
        };

        let mut ve = value_start;
        while ve < bs.len() {
            if bs[ve] == b'\\' && ve + 1 < bs.len() && bs[ve + 1] == b'"' {
                ve += 2;
                continue;
            }
            if bs[ve] == b'"' {
                break;
            }
            ve += 1;
        }
        if ve > value_start && ve < bs.len() {
            let raw = &output[value_start..ve];
            let text = raw.replace("\\\"", "\"").replace("\\n", "\n");
            if !text.trim().is_empty() && (is_ru || has_cyrillic(&text)) {
                results.push(text);
            }
        }
        byte_pos = ve + 1;
    }
    results
}

const BATCH_SIZE: usize = 5;
const MAX_CONTEXT: usize = 2;

#[derive(Debug, Deserialize)]
struct TranslationItem {
    id: usize,
    #[serde(alias = "ru")]
    text: Option<String>,
}

fn parse_json_translation(output: &str) -> HashMap<usize, String> {
    let output = output.trim();
    let start = output.find('[');
    let end = output.rfind(']');
    let json_str = match (start, end) {
        (Some(s), Some(e)) if e > s => &output[s..=e],
        _ => {
            log::debug!("Перевод: JSON массив не найден в выводе");
            return HashMap::new();
        }
    };

    match serde_json::from_str::<Vec<TranslationItem>>(json_str) {
        Ok(items) => items
            .into_iter()
            .filter_map(|item| {
                let text = item.text.filter(|t| !t.is_empty())?;
                Some((item.id, text))
            })
            .collect(),
        Err(e) => {
            log::debug!("Перевод: ошибка парсинга JSON: {}", e);
            HashMap::new()
        }
    }
}

fn build_grammar(count: usize) -> String {
    let mut g = String::new();
    g.push_str("root ::= \"[\" space ");
    for i in 0..count {
        if i > 0 { g.push_str("\",\" space "); }
        g.push_str(&format!("item{}", i));
    }
    g.push_str("\"]\" space\n");
    for i in 0..count {
        g.push_str(&format!(
            "item{} ::= \"{{\" space \"\\\"id\\\"\" space \":\" space \"{}\" \",\" space \"\\\"ru\\\"\" space \":\" space string \"}}\" space\n",
            i, i
        ));
    }
    g.push_str(r#"string ::= "\"" char* "\"""#);
    g.push_str("\n");
    g.push_str(r#"char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})"#);
    g.push_str("\n");
    g.push_str(r#"space ::= " "?"#);
    g.push_str("\n");
    g
}

fn build_prompt(batch: &[SubtitleChunk], context: &[SubtitleChunk]) -> String {
    let mut p = String::from("<|hy_begin_of_sentence|>You are a professional subtitle translator from English to Russian.<|hy_sep|>\n<|hy_User|>");

    if !context.is_empty() {
        p.push_str("Context (already translated):\n[");
        for (i, c) in context.iter().enumerate() {
            if c.text.trim().is_empty() {
                continue;
            }
            let ru_escaped = serde_json::to_string(&c.text).unwrap_or_default();
            if i > 0 {
                p.push(',');
            }
            p.push_str(&format!("{{\"id\":{},\"ru\":{}}}", i, ru_escaped));
        }
        p.push_str("]\n\n");
    }

    p.push_str("Translate these lines to Russian.\n\nInput:\n[");
    let mut first = true;
    for (i, c) in batch.iter().enumerate() {
        if c.text.trim().is_empty() {
            continue;
        }
        let en_escaped = serde_json::to_string(&c.text).unwrap_or_default();
        if !first {
            p.push(',');
        }
        first = false;
        p.push_str(&format!("{{\"id\":{},\"en\":{}}}", i, en_escaped));
    }
    p.push_str("]\n\n<|hy_Assistant|>");

    p
}

fn generate(
    model: &LlamaModel,
    backend: &LlamaBackend,
    prompt: &str,
    grammar_str: &str,
) -> Result<(Vec<LlamaToken>, bool)> {
    let mut ctx = model.new_context(
        backend,
        LlamaContextParams::default().with_n_ctx(NonZeroU32::new(16384)),
    )?;

    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let input_token_count = tokens.len();

    let mut batch_llm = LlamaBatch::new(tokens.len(), 1);
    for (i, t) in tokens.iter().enumerate() {
        batch_llm.add(*t, i as i32, &[0], i == tokens.len() - 1)?;
    }
    ctx.decode(&mut batch_llm)?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::grammar(model, grammar_str, "root")?,
        LlamaSampler::greedy(),
    ]);

    let eos = model.token_eos().0;
    let max_new = (input_token_count * 2).max(128).min(4096);
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

        if token == LlamaToken(eos) {
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
    let n_vocab = model.n_vocab() as i32;
    let eos = model.token_eos().0;
    let bos = model.token_bos().0;
    let mut out = Vec::new();

    for &token in tokens {
        let id = token.0;
        if id < 0 || id >= n_vocab {
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

fn try_parse_batch(
    parsed: &HashMap<usize, String>,
    batch: &[SubtitleChunk],
) -> Option<HashMap<usize, String>> {
    let expected: Vec<usize> = batch
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.text.trim().is_empty())
        .map(|(i, _)| i)
        .collect();

    if expected.is_empty() {
        return Some(HashMap::new());
    }

    let all_present = expected.iter().all(|i| parsed.contains_key(i));
    if !all_present {
        return None;
    }

    let has_repeats = detect_repeated_translations(parsed, 3);
    if has_repeats {
        return None;
    }

    Some(parsed.clone())
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
    let model = match LlamaModel::load_from_file(backend, model_path, &LlamaModelParams::default()) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("Перевод: ошибка загрузки модели: {:#} — пропускаем перевод", e);
            return Ok(PipelineContext {
                translated_chunks: Some(chunks.clone()),
                ..ctx
            });
        }
    };

    let n_vocab = model.n_vocab() as i32;
    let eos = model.token_eos().0;
    log::info!("Перевод: {} чанков, n_vocab={}, eos={}", chunks.len(), n_vocab, eos);

    let mut result: Vec<SubtitleChunk> = Vec::new();
    let mut context: Vec<SubtitleChunk> = Vec::new();

    for batch in chunks.chunks(BATCH_SIZE) {
        if crate::is_cancelled() {
            anyhow::bail!("Перевод отменён пользователем");
        }

        let prompt = build_prompt(batch, &context);

        let tokens = match model.str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Перевод: ошибка токенизации: {:#}", e);
                anyhow::bail!("Ошибка токенизации: {:#}", e);
            }
        };

        if tokens.len() > 14000 {
            log::warn!("Перевод: промпт слишком длинный ({} токенов), уменьшаем batch", tokens.len());
            for chunk in batch {
                if chunk.text.trim().is_empty() {
                    continue;
                }
                let translated = SubtitleChunk {
                    text: format!("[{}]", chunk.text),
                    ..chunk.clone()
                };
                result.push(translated.clone());
                context.push(translated);
            }
            if context.len() > MAX_CONTEXT {
                context.drain(0..context.len() - MAX_CONTEXT);
            }
            continue;
        }

        let expected_count = batch.iter().filter(|c| !c.text.trim().is_empty()).count();
        let grammar_str = build_grammar(expected_count);
        log::info!("Перевод: grammar (expected_count={}):\n{}", expected_count, grammar_str);

        let (toks, ok) = generate(&model, backend, &prompt, &grammar_str)?;
        let output = if ok { decode_tokens(&model, &toks) } else { String::new() };

        // === Grammar guarantees valid JSON — direct parse ===
        let parsed = parse_json_translation(&output);
        let valid_map = if let Some(valid) = try_parse_batch(&parsed, batch) {
            valid
        } else {
            let ru_texts = extract_ru_texts(&output);
            if ru_texts.is_empty() {
                log::warn!("Перевод: парсинг не удался, fallback на оригинал");
                let mut m = HashMap::new();
                for (idx, chunk) in batch.iter().enumerate() {
                    if chunk.text.trim().is_empty() { continue; }
                    m.insert(idx, format!("[{}]", chunk.text));
                }
                m
            } else {
                let ru_texts: Vec<_> = ru_texts.into_iter().take(expected_count).collect();
                log::info!("Перевод: lenient-парсинг ({}/{})", ru_texts.len(), expected_count);
                let mut m = HashMap::new();
                let mut ri = 0;
                for (idx, chunk) in batch.iter().enumerate() {
                    if chunk.text.trim().is_empty() { continue; }
                    let text = if ri < ru_texts.len() { ru_texts[ri].clone() } else { chunk.text.clone() };
                    m.insert(idx, text);
                    ri += 1;
                }
                m
            }
        };

        for (idx, chunk) in batch.iter().enumerate() {
            if chunk.text.trim().is_empty() {
                continue;
            }
            let text = valid_map.get(&idx).cloned().unwrap_or_else(|| chunk.text.clone());
            let translated = SubtitleChunk { text, ..chunk.clone() };
            result.push(translated.clone());
            context.push(translated);
        }
        if context.len() > MAX_CONTEXT {
            context.drain(0..context.len() - MAX_CONTEXT);
        }
    }

    log::info!("Перевод: готово {} чанков", result.len());
    Ok(PipelineContext {
        translated_chunks: Some(result),
        ..ctx
    })
}

fn detect_repeated_translations(parsed: &HashMap<usize, String>, max_repeat: usize) -> bool {
    if parsed.len() < max_repeat {
        return false;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_translation_simple() {
        let input = r#"[{"id":0,"ru":"Привет мир"},{"id":1,"ru":"Это тест"}]"#;
        let result = parse_json_translation(input);
        assert_eq!(result.len(), 2);
        assert_eq!(result.get(&0).unwrap(), "Привет мир");
        assert_eq!(result.get(&1).unwrap(), "Это тест");
    }

    #[test]
    fn test_parse_json_translation_extra_text() {
        let input = "Here is the translation:\n[{\"id\":0,\"ru\":\"Привет\"},{\"id\":1,\"ru\":\"Как дела?\"}]\nDone!";
        let result = parse_json_translation(input);
        assert_eq!(result.len(), 2);
        assert_eq!(result.get(&0).unwrap(), "Привет");
    }

    #[test]
    fn test_parse_json_translation_empty() {
        assert!(parse_json_translation("").is_empty());
        assert!(parse_json_translation("no json here").is_empty());
    }

    #[test]
    fn test_detect_repeated_translations() {
        let mut map = HashMap::new();
        map.insert(0, "hello".to_string());
        map.insert(1, "hello".to_string());
        map.insert(2, "hello".to_string());
        assert!(detect_repeated_translations(&map, 3));
    }

    #[test]
    fn test_detect_repeated_translations_not_enough() {
        let mut map = HashMap::new();
        map.insert(0, "hello".to_string());
        map.insert(1, "world".to_string());
        assert!(!detect_repeated_translations(&map, 3));
    }

    #[test]
    fn test_build_prompt() {
        let batch = vec![
            SubtitleChunk { start_sec: 0.0, end_sec: 1.0, text: "Hello".to_string(), speaker_id: None, word_timestamps: None },
            SubtitleChunk { start_sec: 1.0, end_sec: 2.0, text: "World".to_string(), speaker_id: None, word_timestamps: None },
        ];
        let prompt = build_prompt(&batch, &[]);
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains("World"));
        assert!(prompt.contains("<|hy_Assistant|>"));
    }

    #[test]
    fn test_build_prompt_with_context() {
        let context = vec![
            SubtitleChunk { start_sec: 0.0, end_sec: 1.0, text: "Привет".to_string(), speaker_id: None, word_timestamps: None },
        ];
        let batch = vec![
            SubtitleChunk { start_sec: 1.0, end_sec: 2.0, text: "World".to_string(), speaker_id: None, word_timestamps: None },
        ];
        let prompt = build_prompt(&batch, &context);
        assert!(prompt.contains("Context"));
        assert!(prompt.contains("Привет"));
        assert!(prompt.contains("World"));
    }

    #[test]
    fn test_build_grammar_3() {
        let g = build_grammar(3);
        assert!(g.contains("item0"));
        assert!(g.contains("item1"));
        assert!(g.contains("item2"));
        assert!(!g.contains("item3"));
    }

    #[test]
    fn test_try_parse_batch_all_ok() {
        let mut parsed = HashMap::new();
        parsed.insert(0, "A".to_string());
        parsed.insert(1, "B".to_string());
        let batch = vec![
            SubtitleChunk { start_sec: 0.0, end_sec: 1.0, text: "a".to_string(), speaker_id: None, word_timestamps: None },
            SubtitleChunk { start_sec: 1.0, end_sec: 2.0, text: "b".to_string(), speaker_id: None, word_timestamps: None },
        ];
        assert!(try_parse_batch(&parsed, &batch).is_some());
    }

    #[test]
    fn test_try_parse_batch_missing_index() {
        let mut parsed = HashMap::new();
        parsed.insert(0, "A".to_string());
        let batch = vec![
            SubtitleChunk { start_sec: 0.0, end_sec: 1.0, text: "a".to_string(), speaker_id: None, word_timestamps: None },
            SubtitleChunk { start_sec: 1.0, end_sec: 2.0, text: "b".to_string(), speaker_id: None, word_timestamps: None },
        ];
        assert!(try_parse_batch(&parsed, &batch).is_none());
    }

    #[test]
    fn test_try_parse_batch_with_empty() {
        let mut parsed = HashMap::new();
        parsed.insert(0, "A".to_string());
        let batch = vec![
            SubtitleChunk { start_sec: 0.0, end_sec: 1.0, text: "a".to_string(), speaker_id: None, word_timestamps: None },
            SubtitleChunk { start_sec: 1.0, end_sec: 2.0, text: "".to_string(), speaker_id: None, word_timestamps: None },
        ];
        assert!(try_parse_batch(&parsed, &batch).is_some());
    }
}
