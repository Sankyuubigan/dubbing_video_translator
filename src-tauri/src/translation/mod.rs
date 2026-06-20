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
const SAMPLING_TEMP: f32 = 1.0;
const SAMPLING_TOP_K: i32 = 64;
const SAMPLING_TOP_P: f32 = 0.95;
const SAMPLING_REP_PENALTY: f32 = 1.05;

fn build_prompt(
    current_text: &str,
    prev_chunks: &[(String, String)],
    next_chunks: &[String],
    speaker_gender: &str,
) -> String {
    let mut p = String::from(
        "<|turn>system\n\
        You are an expert audiovisual translator adapting English video subtitles into Russian for voiceover.\n\n\
        STRICT RULES:\n\
        1. CONTEXT & ASR FIXES: The input contains Speech-to-Text errors (e.g., 'Three to C' might mean '32C' bra size). Fix them based on the surrounding context.\n\
        2. ISOCHRONY (TIMING): The Russian text MUST be exactly as short as the English text \
        to fit the audio duration. Omit filler words (oh, well, yeah) and use concise synonyms.\n\
        3. NUMBERS TO WORDS: You MUST spell out ALL numbers and symbols using Russian words \
        (e.g. 'сто семьдесят сантиметров' instead of '170 см', 'тридцать два' instead of '32'). TTS engine cannot read digits.\n\
        4. METRIC: Convert imperial units to metric before spelling them out.\n\
        5. GENDER: Current speaker is [SPEAKER_GENDER]. Ensure Russian verbs/adjectives match this gender perfectly.\n\
        6. NO CENSORSHIP: Preserve all slang and profanity accurately.\n\
        7. OUTPUT FORMAT: Output the translation ONLY ONCE. Do NOT repeat or revise the answer. \
        Do NOT add prefixes like \"Translation:\" or quotation marks.\n\n\
        First, use <|channel>thought to analyze the context, fix errors, and plan a concise translation. Then, output ONLY the final spoken Russian text in the normal channel.<turn|>\n\
        <|turn>user\n",
    );
    p = p.replace("[SPEAKER_GENDER]", speaker_gender);

    if !prev_chunks.is_empty() {
        p.push_str("\nPrevious context:\n");
        for (en, ru) in prev_chunks.iter() {
            p.push_str(&format!("EN: {}\nRU: {}\n", en, ru));
        }
    }

    if !next_chunks.is_empty() {
        p.push_str("\nFuture context (DO NOT translate yet):\n");
        for en in next_chunks.iter() {
            p.push_str(&format!("EN: {}\n", en));
        }
    }

    p.push_str(&format!(
        "\nTranslate this current text:\nEN: {}\n<turn|>\n<|turn>model\n<|channel>thought\n",
        current_text,
    ));

    p
}

/// Merges adjacent chunks that belong to the same speaker and form an incomplete sentence.
/// If a chunk doesn't end with sentence-ending punctuation (. ? !) or the next chunk
/// starts with a lowercase letter (continuing the thought), it's merged with gap up to 2.0s.
fn merge_short_chunks(chunks: &[SubtitleChunk]) -> Vec<SubtitleChunk> {
    if chunks.is_empty() {
        return chunks.to_vec();
    }

    let mut merged: Vec<SubtitleChunk> = Vec::new();

    for chunk in chunks {
        if chunk.text.trim().is_empty() {
            merged.push(chunk.clone());
            continue;
        }

        if let Some(last) = merged.last_mut() {
            let ends_with_punct = last
                .text
                .trim_end()
                .ends_with(|c: char| matches!(c, '.' | '?' | '!'));
            let same_speaker =
                last.speaker_id.is_some() && last.speaker_id == chunk.speaker_id;
            let gap = chunk.start_sec - last.end_sec;
            let next_starts_lower = chunk
                .text
                .trim_start()
                .starts_with(|c: char| c.is_ascii_lowercase());

            if same_speaker
                && gap >= 0.0
                && gap < 2.0
                && (!ends_with_punct || next_starts_lower)
            {
                log::debug!(
                    "Перевод: merging chunks '{:.60}...' + '{:.60}...' (gap={:.2}s, speaker={:?})",
                    last.text,
                    chunk.text,
                    gap,
                    last.speaker_id,
                );
                last.text.push(' ');
                last.text.push_str(&chunk.text);
                last.end_sec = chunk.end_sec;
                continue;
            }
        }

        merged.push(chunk.clone());
    }

    merged
}

fn clean_output(output: &str) -> String {
    // Extract the final answer from the CoT format: <|channel>thought...<channel|>FINAL_ANSWER
    // Handle edge cases: stray <|channel> after <channel|>, or answer before <channel|>
    // If multiple <channel|> blocks exist and the last one has insufficient Cyrillic content
    // (e.g., model ran out of tokens during CoT), fall back to the previous block.
    // Each block is bounded by the next <channel|> marker (or end of string) to avoid
    // contamination from later blocks' CoT content.
    let output = {
        let positions: Vec<usize> = output.match_indices("<channel|>")
            .map(|(pos, _)| pos)
            .collect();

        let mut selected: Option<&str> = None;
        let mut last_valid: Option<&str> = None;

        for i in (0..positions.len()).rev() {
            let start = positions[i] + "<channel|>".len();
            let end = if i + 1 < positions.len() {
                positions[i + 1]
            } else {
                output.len()
            };
            let block_text = &output[start..end];

            let after = block_text.trim_start();
            let cleaned = if after.starts_with("<|") {
                if let Some(end_tag) = after.find('>') {
                    after[end_tag + 1..].trim_start()
                } else {
                    after
                }
            } else {
                after
            };

            if cleaned.is_empty() || cleaned.trim().eq_ignore_ascii_case("thought") {
                log::info!("CLEANOUTPUT_DEBUG: block[{}] skipped (empty/thought)", i);
                continue;
            }

            let cyrillic_count = cleaned.chars()
                .filter(|c| ('А'..='я').contains(c) || *c == 'Ё' || *c == 'ё')
                .count();
            let total_alpha = cleaned.chars().filter(|c| c.is_alphabetic()).count();
            // Accept block only if Cyrillic makes up >= 25% of alphabetic chars.
            // Using ratio avoids selecting long English CoT blocks that happen to
            // start with a few Cyrillic words (e.g., "Пять футов семь. (Wait, I must...").
            let has_cyrillic = total_alpha > 0 && cyrillic_count as f64 / total_alpha as f64 >= 0.25;

            log::info!(
                "CLEANOUTPUT_DEBUG: block[{}] cyr={} alpha={} ratio={:.2} has_cyrillic={} cleaned='{}'",
                i, cyrillic_count, total_alpha,
                if total_alpha > 0 { cyrillic_count as f64 / total_alpha as f64 } else { 0.0 },
                has_cyrillic, cleaned.chars().take(60).collect::<String>()
            );

            if has_cyrillic {
                selected = Some(cleaned);
                break;
            }
            if last_valid.is_none() {
                last_valid = Some(cleaned);
            }
        }

        match selected.or(last_valid) {
            Some(text) => text,
            None => {
                if let Some(pos) = output.find("<|channel>") {
                    &output[pos + "<|channel>".len()..]
                } else {
                    output
                }
            }
        }
    };

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
    // keep only the last (most refined) line, but prefer lines without known prefixes
    let output = output.trim();
    let lines: Vec<&str> = output.lines().filter(|l| !l.trim().is_empty()).collect();
    let output = if lines.len() > 1 {
        let known_prefixes = ["Russian: ", "Russian : ", "Translation: ", "RU: ", "Перевод: "];
        // Iterate in reverse, prefer last line WITHOUT a known prefix
        lines.iter()
            .rev()
            .find(|l| !known_prefixes.iter().any(|p| l.trim().starts_with(p)))
            .map(|l| l.trim())
            .unwrap_or_else(|| lines.last().unwrap().trim())
    } else {
        output
    };
    // Strip known prefixes from final result (handles single-line + edge cases)
    let output = {
        let known_prefixes = ["Russian: ", "Russian : ", "Translation: ", "RU: ", "Перевод: "];
        if let Some(rest) = known_prefixes.iter().find_map(|p| output.strip_prefix(p)) {
            rest.trim()
        } else {
            output
        }
    };
    // Strip bold/italic markers (**) from markdown formatting that model sometimes adds
    let output = output.trim_start_matches(|c: char| c == '*' || c == '_');
    let output = output.trim_end_matches(|c: char| c == '*' || c == '_');
    output.trim().to_string()
}

/// Returns true if the text consists only of interjections/filler words (≤4 words).
/// These chunks waste TTS time and can be safely merged with neighbors.
fn is_interjection_only(text: &str) -> bool {
    const INTERJECTIONS: &[&str] = &[
        "yeah", "yes", "no", "oh", "ah", "uh", "um", "hmm", "wow", "hey",
        "hi", "hello", "ok", "okay", "well", "so", "like", "you know",
        "uh-huh", "mm-hmm", "nah", "nope", "yep", "aye", "alas",
        "gosh", "gee", "ooh", "aah", "whoa", "dude", "man",
        "right", "sure", "great", "good", "fine", "cool", "nice",
        "huh", "eh", "ow", "ouch", "phew", "whew", "duh",
        "hooray", "bravo", "encore",
    ];

    let text = text.trim().trim_matches(|c: char| matches!(c, '.' | '!' | '?' | ',' | ';' | ':' | '-' | '—' | '"' | '\''));
    if text.is_empty() {
        return false;
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() || words.len() > 4 {
        return false;
    }

    words.iter().all(|w| {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        INTERJECTIONS.iter().any(|&i| i.eq_ignore_ascii_case(w))
    })
}

/// Merges interjection-only chunks (e.g. "Yeah.", "Oh.") with the previous chunk
/// to avoid wasting LLM context and TTS time on non-semantic filler.
fn merge_interjections(chunks: &[SubtitleChunk]) -> Vec<SubtitleChunk> {
    if chunks.is_empty() {
        return chunks.to_vec();
    }

    let mut merged: Vec<SubtitleChunk> = Vec::new();

    for chunk in chunks {
        if is_interjection_only(&chunk.text) {
            // Merge backward into the previous chunk if close in time
            if let Some(last) = merged.last_mut() {
                let gap = chunk.start_sec - last.end_sec;
                if gap < 2.0 || last.speaker_id == chunk.speaker_id {
                    last.text.push(' ');
                    last.text.push_str(chunk.text.trim());
                    last.end_sec = chunk.end_sec;
                    continue;
                }
            }
        }
        merged.push(chunk.clone());
    }

    merged
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

        // Early stopping: if model tries to open a new turn or channel, stop
        let current_text = decode_tokens(model, &output_toks);
        if current_text.contains("<turn|>") || current_text.contains("<|channel>") {
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
    let chunks_src = match ctx.subtitle_chunks.as_ref() {
        Some(c) => c,
        None => {
            log::error!("Перевод: нет чанков STT");
            anyhow::bail!("Нет чанков STT");
        }
    };

    // Pre-merge: combine incomplete chunks from the same speaker before translation
    let chunks = merge_short_chunks(chunks_src);
    log::info!(
        "Перевод: {} чанков после merge_short_chunks (было {})",
        chunks.len(),
        chunks_src.len()
    );
    let chunks = merge_interjections(&chunks);
    log::info!(
        "Перевод: {} чанков после merge_interjections",
        chunks.len()
    );

    let model_path = match ctx.config.gguf_model_path.as_ref() {
        Some(p) => p,
        None => {
            log::warn!("Перевод: не выбран GGUF-файл модели, пропускаем перевод");
            return Ok(PipelineContext {
                translated_chunks: Some(chunks),
                ..ctx
            });
        }
    };

    if !Path::new(model_path).exists() {
        log::warn!("Перевод: GGUF модель не найдена: {}, пропускаем перевод", model_path);
        return Ok(PipelineContext {
            translated_chunks: Some(chunks),
            ..ctx
        });
    }

    log::info!("Перевод: загружаем LLM из {}", model_path);

    let backend = match crate::get_llama_backend() {
        Ok(b) => b,
        Err(e) => {
            log::warn!("Перевод: {} — пропускаем перевод", e);
            return Ok(PipelineContext {
                translated_chunks: Some(chunks),
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
                translated_chunks: Some(chunks),
                ..ctx
            });
        }
    };

    log::info!("Перевод: {} чанков после мержа", chunks.len());

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

    for (i, chunk) in chunks.iter().enumerate() {
        if crate::is_cancelled() {
            anyhow::bail!("Перевод отменён пользователем");
        }

        if chunk.text.trim().is_empty() {
            result.push(chunk.clone());
            continue;
        }

        // 1. Lookahead: collect up to 3 future chunks for context
        let mut next_chunks: Vec<String> = Vec::new();
        for j in 1..=3 {
            if let Some(next_chunk) = chunks.get(i + j) {
                if !next_chunk.text.trim().is_empty() {
                    next_chunks.push(next_chunk.text.clone());
                }
            }
        }

        // 2. Determine speaker gender from diarization data
        let speaker_gender = chunk
            .speaker_id
            .as_ref()
            .and_then(|spk_id| {
                ctx.speaker_segments.as_ref().and_then(|speakers| {
                    speakers
                        .iter()
                        .find(|s| s.speaker_id == *spk_id)
                        .and_then(|s| s.gender.as_ref())
                        .map(|g| {
                            if g == "female" { "Female" } else { "Male" }
                        })
                })
            })
            .unwrap_or("Person");

        // 3. Build prompt with full context
        let prompt = build_prompt(&chunk.text, &prev_chunks, &next_chunks, speaker_gender);

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

        // Dynamic max_new: more tokens for longer prompts
        let max_new = ((tokens.len() / 2)
            .max(64))
            .min(512)
            .min(MAX_N_CTX as usize - tokens.len() - 50);
        if max_new < 16 {
            log::warn!(
                "Перевод: промпт слишком длинный ({} токенов), fallback",
                tokens.len()
            );
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
        let raw_end: String = raw
            .chars()
            .rev()
            .take(40)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        log::info!(
            "Перевод: чанк '{}' -> {} токенов, raw_start='{}', raw_end='{}'",
            chunk_preview,
            toks.len(),
            raw_start,
            raw_end,
        );
        let output = clean_output(&raw);

        if chunk.text.trim() == "Five seven." {
            log::info!("CLEANOUTPUT_DEBUG: 'Five seven.' raw_len={}, output='{}'", raw.len(), output);
        }

        let translated = if output.is_empty() {
            log::warn!(
                "Перевод: пустой вывод для '{}' ({} токенов)",
                chunk.text,
                toks.len()
            );
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
        let prompt = build_prompt("Hello world", &[], &[], "Person");
        assert!(prompt.contains("Hello world"));
        assert!(prompt.contains("<|turn>system"));
        assert!(prompt.contains("<|turn>user"));
        assert!(prompt.contains("<turn|>"));
        assert!(prompt.contains("<|turn>model"));
        assert!(prompt.contains("expert audiovisual translator"));
        assert!(prompt.contains("ASR FIXES"));
        assert!(prompt.contains("ISOCHRONY"));
        assert!(prompt.contains("NUMBERS TO WORDS"));
        assert!(prompt.contains("NO CENSORSHIP"));
        assert!(prompt.contains("<|channel>thought"));
        assert!(!prompt.contains("[SPEAKER_GENDER]"));
    }

    #[test]
    fn test_build_prompt_with_context() {
        let ctx = vec![("First EN text".to_string(), "Первая".to_string())];
        let prompt = build_prompt("Second", &ctx, &[], "Person");
        assert!(prompt.contains("Second"));
        assert!(prompt.contains("Previous context:"));
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
        let prompt = build_prompt("Fourth", &ctx, &[], "Person");
        assert!(prompt.contains("Fourth"));
        assert!(prompt.contains("First EN"));
        assert!(prompt.contains("Second EN"));
        assert!(prompt.contains("Third EN"));
        assert!(prompt.contains("Первая"));
        assert!(prompt.contains("Вторая"));
        assert!(prompt.contains("Третья"));
    }

    #[test]
    fn test_build_prompt_with_lookahead() {
        let next = vec!["Next text".to_string(), "Future text".to_string()];
        let prompt = build_prompt("Current", &[], &next, "Male");
        assert!(prompt.contains("Current"));
        assert!(prompt.contains("Future context"));
        assert!(prompt.contains("Next text"));
        assert!(prompt.contains("Future text"));
        assert!(prompt.contains("DO NOT translate"));
    }

    #[test]
    fn test_build_prompt_with_gender() {
        let prompt = build_prompt("Hello", &[], &[], "Female");
        assert!(prompt.contains("Female"));
        assert!(prompt.contains("match this gender perfectly"));

        let prompt = build_prompt("Hello", &[], &[], "Male");
        assert!(prompt.contains("Male"));
        assert!(prompt.contains("match this gender perfectly"));
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

    #[test]
    fn test_merge_short_chunks_no_merge() {
        let chunks = vec![
            SubtitleChunk {
                start_sec: 0.0,
                end_sec: 1.0,
                text: "Hello world.".to_string(),
                speaker_id: Some("Speaker_1".to_string()),
                word_timestamps: None,
            },
            SubtitleChunk {
                start_sec: 1.5,
                end_sec: 2.5,
                text: "How are you?".to_string(),
                speaker_id: Some("Speaker_1".to_string()),
                word_timestamps: None,
            },
        ];
        let result = merge_short_chunks(&chunks);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_merge_short_chunks_merge_incomplete() {
        let chunks = vec![
            SubtitleChunk {
                start_sec: 0.0,
                end_sec: 1.0,
                text: "Hello I am".to_string(),
                speaker_id: Some("Speaker_1".to_string()),
                word_timestamps: None,
            },
            SubtitleChunk {
                start_sec: 1.2,
                end_sec: 2.0,
                text: "going home".to_string(),
                speaker_id: Some("Speaker_1".to_string()),
                word_timestamps: None,
            },
        ];
        let result = merge_short_chunks(&chunks);
        assert_eq!(result.len(), 1);
        assert!(result[0].text.contains("Hello I am going home"));
        assert!((result[0].end_sec - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_short_chunks_different_speakers() {
        let chunks = vec![
            SubtitleChunk {
                start_sec: 0.0,
                end_sec: 1.0,
                text: "Hello I am".to_string(),
                speaker_id: Some("Speaker_1".to_string()),
                word_timestamps: None,
            },
            SubtitleChunk {
                start_sec: 1.2,
                end_sec: 2.0,
                text: "going home".to_string(),
                speaker_id: Some("Speaker_2".to_string()),
                word_timestamps: None,
            },
        ];
        let result = merge_short_chunks(&chunks);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_merge_short_chunks_large_gap() {
        let chunks = vec![
            SubtitleChunk {
                start_sec: 0.0,
                end_sec: 1.0,
                text: "Hello I am".to_string(),
                speaker_id: Some("Speaker_1".to_string()),
                word_timestamps: None,
            },
            SubtitleChunk {
                start_sec: 4.0,
                end_sec: 5.0,
                text: "going home".to_string(),
                speaker_id: Some("Speaker_1".to_string()),
                word_timestamps: None,
            },
        ];
        let result = merge_short_chunks(&chunks);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_merge_short_chunks_lowercase_continues() {
        let chunks = vec![
            SubtitleChunk {
                start_sec: 0.0,
                end_sec: 1.0,
                text: "Hello world.".to_string(),
                speaker_id: Some("Speaker_1".to_string()),
                word_timestamps: None,
            },
            SubtitleChunk {
                start_sec: 1.5,
                end_sec: 2.5,
                text: "going home now.".to_string(),
                speaker_id: Some("Speaker_1".to_string()),
                word_timestamps: None,
            },
        ];
        let result = merge_short_chunks(&chunks);
        assert_eq!(result.len(), 1);
        assert!(result[0].text.contains("Hello world. going home now."));
    }

    #[test]
    fn test_is_interjection_only_true() {
        assert!(is_interjection_only("Yeah"));
        assert!(is_interjection_only("Oh."));
        assert!(is_interjection_only("Wow!"));
        assert!(is_interjection_only("Oh yeah"));
        assert!(is_interjection_only("Uh-huh"));
    }

    #[test]
    fn test_is_interjection_only_false() {
        assert!(!is_interjection_only("I am going home"));
        assert!(!is_interjection_only("That is great!"));
        assert!(!is_interjection_only("No way, really?"));
        assert!(!is_interjection_only(""));
    }

    #[test]
    fn test_clean_output_strips_stray_channel_tag() {
        let out = clean_output("<|channel>thought\nReasoning...\n<channel|><|channel>thought\nПривет мир");
        assert_eq!(out, "Привет мир", "stray <|channel> after <channel|> stripped");
    }

    #[test]
    fn test_clean_output_strips_stray_channel_fallback() {
        let out = clean_output("Reasoning text\n<channel|><|channel>");
        assert_eq!(out, "Reasoning text", "fallback to content before <channel|> when no answer after it");
    }

    #[test]
    fn test_clean_output_multiline_prefers_without_prefix() {
        let out = clean_output(
            "Свет и камера уже стоят, всё готово к работе.\n\
             Translation: Свет и камера уже стоят, всё готово к работе."
        );
        assert_eq!(out, "Свет и камера уже стоят, всё готово к работе.",
            "multiline with prefixed last line: should pick clean first line");
    }

    #[test]
    fn test_clean_output_multiline_both_prefixed_fallback() {
        // When ALL lines have known prefixes, fall back to last line
        let out = clean_output(
            "Russian: Первая строка\n\
             Translation: Вторая строка"
        );
        assert_eq!(out, "Вторая строка",
            "all prefixed: should pick last line as fallback");
    }

    #[test]
    fn test_clean_output_single_line_known_prefix() {
        // Single-line output with prefix should still be stripped
        let out = clean_output("Translation: Привет мир");
        assert_eq!(out, "Привет мир");
    }

    #[test]
    fn test_clean_output_multiline_translation_quotes() {
        // Simulates the actual bug scenario from the logs
        let out = clean_output(
            "Свет и камера уже стоят, всё готово к работе. Хорошо?\"\n\
             Translation: \"Свет и камера уже стоят, всё готово к работе. Хорошо?\"\n\n\
             Is there any number? No."
        );
        assert_eq!(out, "Свет и камера уже стоят, всё готово к работе. Хорошо?\"",
            "multiline with Translation: prefix + trailing CoT: should pick clean first line");
    }
}
