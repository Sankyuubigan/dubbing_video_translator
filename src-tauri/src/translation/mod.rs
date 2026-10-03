use crate::comm::{PipelineContext, SubtitleChunk};
use crate::llm::{message, LlmSession};
use anyhow::Result;
use tauri_plugin_llama_engine::engine::LlmMessage;

const MAX_CONTEXT_CHUNKS: usize = 8;

/// Системная часть промпта. Раньше она собиралась в строку Gemma-4 вручную
/// (`<|turn>system ... <turn|>`); теперь промпт рендерит сам движок по
/// `tokenizer.chat_template` модели, а мы отдаём обычные роли.
const SYSTEM_RULES: &str = "\
You are an expert audiovisual translator adapting English video subtitles into Russian for voiceover.

STRICT RULES:
1. CONTEXT & ASR FIXES: The input contains Speech-to-Text errors (e.g., 'Three to C' might mean '32C' bra size). Fix them based on the surrounding context.
2. ISOCHRONY (TIMING): The Russian text MUST be exactly as short as the English text to fit the audio duration. Omit filler words (oh, well, yeah) and use concise synonyms.
3. NUMBERS TO WORDS: You MUST spell out ALL numbers and symbols using Russian words (e.g. 'сто семьдесят сантиметров' instead of '170 см', 'тридцать два' instead of '32'). TTS engine cannot read digits.
4. METRIC: Convert imperial units to metric before spelling them out.
5. GENDER: Current speaker is [SPEAKER_GENDER]. Ensure Russian verbs/adjectives match this gender perfectly.
6. NO CENSORSHIP: Preserve all slang and profanity accurately.
7. OUTPUT FORMAT: Output ONLY the final spoken Russian text. Do NOT think out loud, do NOT analyze, do NOT offer options or variants. Do NOT repeat or revise the answer. Do NOT add prefixes like \"Translation:\" or quotation marks.";

/// Собирает сообщения для одного чанка: правила + предыдущие пары EN/RU +
/// lookahead по английскому оригиналу (source-side контекст важнее целевого).
fn build_messages(
    current_text: &str,
    prev_chunks: &[(String, String)],
    next_chunks: &[String],
    speaker_gender: &str,
) -> Vec<LlmMessage> {
    let system = SYSTEM_RULES.replace("[SPEAKER_GENDER]", speaker_gender);

    let mut user = String::new();
    if !prev_chunks.is_empty() {
        user.push_str("Previous context:\n");
        for (en, ru) in prev_chunks.iter() {
            user.push_str(&format!("EN: {}\nRU: {}\n", en, ru));
        }
    }
    if !next_chunks.is_empty() {
        user.push_str("\nFuture context (DO NOT translate yet):\n");
        for en in next_chunks.iter() {
            user.push_str(&format!("EN: {}\n", en));
        }
    }
    user.push_str(&format!(
        "\nTranslate this current text:\nEN: {}\n",
        current_text
    ));

    vec![message("system", system), message("user", user)]
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

/// Очистка вывода модели (общая для основного перевода и строгих ретраев
/// верификации): извлекает ответ из CoT-цикла `<|channel>thought...<channel|>ANSWER`.
pub(crate) fn clean_output(output: &str) -> String {
    // Извлекаем ответ из CoT-формата: <|channel>thought...<channel|>ANSWER
    // Если закрывающих <channel|> несколько (модель ушла в «режим редактора»:
    // первый ответ корректен, дальше идёт мусор), берём ПЕРВЫЙ блок с
    // достаточным содержанием кириллицы. Второй и последующие — хвост-правки.
    let output = {
        let positions: Vec<usize> = output.match_indices("<channel|>")
            .map(|(pos, _)| pos)
            .collect();

        let mut selected: Option<&str> = None;
        let mut last_touched: Option<&str> = None;

        for i in 0..positions.len() {
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
            last_touched = Some(cleaned);
        }

        match selected.or(last_touched) {
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
    // keep the FIRST line (most direct answer), but prefer lines without known prefixes
    let output = output.trim();
    let lines: Vec<&str> = output.lines().filter(|l| !l.trim().is_empty()).collect();
    let output = if lines.len() > 1 {
        let known_prefixes = ["Russian: ", "Russian : ", "Translation: ", "RU: ", "Перевод: "];
        // Iterate forward, prefer first line WITHOUT a known prefix
        lines.iter()
            .find(|l| !known_prefixes.iter().any(|p| l.trim().starts_with(p)))
            .map(|l| l.trim())
            .unwrap_or_else(|| lines.first().unwrap().trim())
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
                if (gap >= 0.0 && gap < 2.0) || last.speaker_id == chunk.speaker_id {
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

/// Переводит чанки и возвращает вместе с контекстом ЖИВУЮ сессию движка.
///
/// Сессия возвращается наружу не «на всякий случай», а потому что следующий
/// этап (`verification`) — тоже LLM-этап: раньше он поднимал ВТОРОЙ
/// `llama-server.exe` (5 с загрузки 6.5 ГиБ в VRAM) ради десятка запросов по
/// бракованным чанкам. Теперь один процесс живёт на оба этапа, а `drop`
/// сессии происходит строго до старта TTS — VRAM освобождается по desktop §6.5.
pub fn translate_with_session(
    ctx: PipelineContext,
) -> Result<(PipelineContext, Option<LlmSession>)> {
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

    // Сессия движка на весь этап: один запуск llama-server, по чанку — запрос.
    let app = crate::app_handle()
        .ok_or_else(|| anyhow::anyhow!("AppHandle не инициализирован — движок LLM недоступен"))?;
    let session = LlmSession::open(app).map_err(anyhow::Error::msg)?;
    log::info!("Перевод: {} чанков после мержа", chunks.len());

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

        // 3. Build messages with full context
        let messages = build_messages(&chunk.text, &prev_chunks, &next_chunks, speaker_gender);

        let generation = match session.generate(&messages, None) {
            Ok(g) => g,
            Err(e) => {
                if crate::is_cancelled() {
                    anyhow::bail!("Перевод отменён пользователем");
                }
                log::error!("Перевод: ошибка генерации для '{}': {}", chunk.text, e);
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

        let output = clean_output(&generation.text);
        let chunk_preview: String = chunk.text.chars().take(20).collect();
        log::info!(
            "Перевод: '{}' -> {} токенов, '{}'",
            chunk_preview,
            generation.metrics.generated_tokens,
            output,
        );

        let translated = if output.is_empty() {
            log::warn!(
                "Перевод: пустой вывод для '{}' ({} токенов, stop_reason={})",
                chunk.text,
                generation.metrics.generated_tokens,
                generation.stop_reason
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
    Ok((
        PipelineContext {
            translated_chunks: Some(result),
            ..ctx
        },
        Some(session),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Текст промпта собирается из system+user: движок сам рендерит его по
    /// `tokenizer.chat_template` модели, поэтому тест проверяет роли и
    /// содержимое, а не теги Gemma-4.
    fn flatten(messages: &[LlmMessage]) -> String {
        messages
            .iter()
            .map(|m| format!("[{}] {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn test_build_messages_format() {
        let messages = build_messages("Hello world", &[], &[], "Person");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        let prompt = flatten(&messages);
        assert!(prompt.contains("Hello world"));
        assert!(prompt.contains("expert audiovisual translator"));
        assert!(prompt.contains("ASR FIXES"));
        assert!(prompt.contains("ISOCHRONY"));
        assert!(prompt.contains("NUMBERS TO WORDS"));
        assert!(prompt.contains("NO CENSORSHIP"));
        assert!(!prompt.contains("[SPEAKER_GENDER]"));
    }

    #[test]
    fn test_build_messages_with_context() {
        let ctx = vec![("First EN text".to_string(), "Первая".to_string())];
        let messages = build_messages("Second", &ctx, &[], "Person");
        let prompt = flatten(&messages);
        assert!(prompt.contains("Second"));
        assert!(prompt.contains("Previous context:"));
        assert!(prompt.contains("Первая"));
        assert!(prompt.contains("First EN text"));
        assert!(prompt.contains("EN:"));
        assert!(prompt.contains("RU:"));
    }

    #[test]
    fn test_build_messages_with_multiple_contexts() {
        let ctx = vec![
            ("First EN".to_string(), "Первая".to_string()),
            ("Second EN".to_string(), "Вторая".to_string()),
            ("Third EN".to_string(), "Третья".to_string()),
        ];
        let messages = build_messages("Fourth", &ctx, &[], "Person");
        let prompt = flatten(&messages);
        assert!(prompt.contains("Fourth"));
        assert!(prompt.contains("First EN"));
        assert!(prompt.contains("Second EN"));
        assert!(prompt.contains("Third EN"));
        assert!(prompt.contains("Первая"));
        assert!(prompt.contains("Вторая"));
        assert!(prompt.contains("Третья"));
    }

    #[test]
    fn test_build_messages_with_lookahead() {
        let next = vec!["Next text".to_string(), "Future text".to_string()];
        let messages = build_messages("Current", &[], &next, "Male");
        let prompt = flatten(&messages);
        assert!(prompt.contains("Current"));
        assert!(prompt.contains("Future context"));
        assert!(prompt.contains("Next text"));
        assert!(prompt.contains("Future text"));
        assert!(prompt.contains("DO NOT translate"));
    }

    #[test]
    fn test_build_messages_with_gender() {
        let prompt = flatten(&build_messages("Hello", &[], &[], "Female"));
        assert!(prompt.contains("Female"));
        assert!(prompt.contains("match this gender perfectly"));

        let prompt = flatten(&build_messages("Hello", &[], &[], "Male"));
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
        assert_eq!(out, "", "no Cyrillic in any block → empty (CoT only, no answer)");
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
        // When ALL lines have known prefixes, fall back to first line
        let out = clean_output(
            "Russian: Первая строка\n\
             Translation: Вторая строка"
        );
        assert_eq!(out, "Первая строка",
            "all prefixed: should pick first line as fallback");
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
