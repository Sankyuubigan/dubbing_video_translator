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

/// Пытается распарсить JSON-массив из вывода LLM, с очисткой от распространённых ошибок.
fn parse_llm_json(output: &str) -> Option<Vec<SubtitleChunk>> {
    let js = output.find('[')?;
    let je = output[js..].rfind(']')?;
    let raw = &output[js..=js + je];

    if let Ok(parsed) = serde_json::from_str::<Vec<SubtitleChunk>>(raw) {
        return Some(parsed);
    }

    let cleaned = fix_json_quotes(raw);
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
        let mut depth = 0;
        let mut obj_end = None;
        for i in obj_start..bytes.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        obj_end = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let obj_end = match obj_end {
            Some(e) => e,
            None => break,
        };
        let obj_str = &raw[obj_start..obj_end];
        let obj_clean = fix_json_quotes(obj_str);
        if let Ok(chunk) = serde_json::from_str::<SubtitleChunk>(&obj_clean) {
            results.push(chunk);
        }
        pos = obj_end;
    }

    if !results.is_empty() {
        return Some(results);
    }

    None
}

/// Экранирует неэкранированные кавычки внутри строковых значений JSON.
fn fix_json_quotes(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 32);
    let bytes = s.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'"' {
            out.push('"');
            i += 1;

            if i >= 2 && bytes[i - 2] == b'\\' {
                continue;
            }

            loop {
                if i >= bytes.len() {
                    break;
                }
                if bytes[i] == b'\\' {
                    out.push('\\');
                    i += 1;
                    if i < bytes.len() {
                        out.push(bytes[i] as char);
                        i += 1;
                    }
                    continue;
                }
                if bytes[i] == b'"' {
                    out.push('"');
                    i += 1;
                    break;
                }
                out.push(bytes[i] as char);
                i += 1;
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}
