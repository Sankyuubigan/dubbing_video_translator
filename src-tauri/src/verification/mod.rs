use crate::comm::{PipelineContext, SubtitleChunk, SKIP_MARKER};
use anyhow::Result;

/// Эмпирическая норма: ≤ 20 символов на секунду речи (Netflix ~42 символа
/// на строку, строка ≈ 2 сек). Точность: 10% запас (22 симв/сек) чтобы
/// не отбрасывать короткие строки.
const CHARS_PER_SEC_MAX: f64 = 22.0;
/// Минимальная длина RU текста для проверки бюджета (чтобы не триггерить
/// короткие фразы типа «Да.», «Нет.»).
const MIN_LEN_FOR_BUDGET: usize = 10;

/// Метки-мусор которые модельfois генерирует вместо перевода
const GARBAGE_MARKERS: &[&str] = &[
    "Вот перевод для озвучки",
    "Вот перевод для закадрового",
    "Вот текущий вариант",
    "Итоговый вариант",
    "ИТОГОВЫЙ ВАРИАНТ",
    "Финальный вариант",
    "Готовый вариант",
    "Перевод для озвучки",
    "Перевод для субтитров",
    "Мужской род:",
    "Женский род:",
    "Мужской род :",
    "Женский род :",
    "Russian:",
    "Russian :",
    "Translation:",
    "Translation :",
    "RU:",
    "RU :",
    "Перевод:",
    "Перевод :",
    "Вот этот вариант",
    "В этом варианте",
    "Вариант для озвучки",
];

pub fn verify(mut ctx: PipelineContext) -> Result<PipelineContext> {
    let mut chunks = match std::mem::take(&mut ctx.translated_chunks) {
        Some(c) => c,
        None => {
            log::warn!("VERIFY: нет переведённых чанков");
            return Ok(ctx);
        }
    };

    let src = match ctx.subtitle_chunks.as_ref() {
        Some(c) => c,
        None => {
            log::warn!("VERIFY: нет исходных чанков STT для сравнения");
            return Ok(PipelineContext {
                translated_chunks: Some(chunks),
                ..ctx
            });
        }
    };

    // EN-текст для каждого RU-чанка ищем по пересечению времени, а не по
    // индексу: после merge_short_chunks/merge_interjections индексы не совпадают.
    let matched_en: Vec<String> = chunks
        .iter()
        .map(|c| en_matching(c, src))
        .collect();

    let mut model_path: Option<String> = None;
    let mut ok_count = 0;
    let mut retry_count = 0;
    let mut fail_count = 0;
    // Последние 3 обработанных пары (EN, RU) — для prev-контекста строгого ретрая
    let mut prev_pairs: Vec<(String, String)> = Vec::new();

    for (i, chunk) in chunks.iter_mut().enumerate() {
        let en = matched_en.get(i).map(|s| s.as_str()).unwrap_or("");
        let duration = chunk.end_sec - chunk.start_sec;
        let issues = assess(&chunk.text, en, duration);

        if issues.is_empty() {
            ok_count += 1;
            prev_pairs.push((en.to_string(), chunk.text.clone()));
            if prev_pairs.len() > 3 {
                prev_pairs.remove(0);
            }
            continue;
        }

        log::warn!(
            "VERIFY: [{}] t={:.1}s FAILED: {:?} | text='{}' | en='{}'",
            i, chunk.start_sec, issues,
            chunk.text.chars().take(60).collect::<String>(),
            en.chars().take(60).collect::<String>(),
        );

        let mp = match model_path.clone().or_else(|| ctx.config.gguf_model_path.clone()) {
            Some(p) => {
                model_path = Some(p.clone());
                p
            }
            None => {
                chunk.text = SKIP_MARKER.to_string();
                fail_count += 1;
                log::warn!("VERIFY: [{}] FAIL (нет модели) → {}", i, SKIP_MARKER);
                prev_pairs.push((en.to_string(), chunk.text.clone()));
                if prev_pairs.len() > 3 {
                    prev_pairs.remove(0);
                }
                continue;
            }
        };

        let gender = extract_gender(&ctx, chunk);
        // lookahead: до 3 будущих EN-чанков
        let next_ens: Vec<String> = matched_en
            .get(i + 1..)
            .unwrap_or(&[])
            .iter()
            .take(3)
            .cloned()
            .collect();

        let mut fixed: Option<String> = None;

        for attempt in 0..2 {
            let temp = if attempt == 0 { 0.2 } else { 0.1 };
            let seed = 42 + i as u32 * 10 + attempt as u32;
            let prompt = build_strict_prompt(en, &gender, &prev_pairs, &next_ens);
            let raw = crate::llm::generate_once(&mp, &prompt, temp, seed);
            let cleaned = match raw {
                Ok(r) => crate::translation::clean_output(&r),
                Err(e) => {
                    log::warn!("VERIFY: [{}] attempt {} generate error: {:#}", i, attempt, e);
                    String::new()
                }
            };
            log::info!(
                "VERIFY: [{}] attempt {} → cleaned='{}'",
                i, attempt, cleaned.chars().take(80).collect::<String>(),
            );
            if !cleaned.is_empty() && assess(&cleaned, en, duration).is_empty() {
                fixed = Some(cleaned);
                break;
            }
        }

        match fixed {
            Some(ok) => {
                chunk.text = ok;
                retry_count += 1;
                log::info!(
                    "VERIFY: [{}] RETRY OK → '{}'",
                    i, chunk.text.chars().take(60).collect::<String>(),
                );
            }
            None => {
                chunk.text = SKIP_MARKER.to_string();
                fail_count += 1;
                log::warn!("VERIFY: [{}] FAIL (после 2 ретраев) → {}", i, SKIP_MARKER);
            }
        }
        prev_pairs.push((en.to_string(), chunk.text.clone()));
        if prev_pairs.len() > 3 {
            prev_pairs.remove(0);
        }
    }

    log::info!(
        "VERIFY: итог OK={} RETRY={} FAIL={} (из {})",
        ok_count, retry_count, fail_count, chunks.len()
    );
    Ok(PipelineContext {
        translated_chunks: Some(chunks),
        ..ctx
    })
}

/// Находит EN-текст, пересекающийся по времени с чанком (мерджинг смешивает
/// время и число EN-чанков). Если пересечений нет — ближайший по start_sec.
fn en_matching(chunk: &SubtitleChunk, src: &[SubtitleChunk]) -> String {
    let mut overlapping: Vec<String> = Vec::new();
    let mut best: Option<&SubtitleChunk> = None;
    let mut best_dist = f64::MAX;

    for s in src {
        let overlap = chunk.end_sec.min(s.end_sec) - chunk.start_sec.max(s.start_sec);
        if overlap > 0.05 {
            overlapping.push(s.text.clone());
        }
        let dist = (s.start_sec - chunk.start_sec).abs();
        if dist < best_dist {
            best_dist = dist;
            best = Some(s);
        }
    }

    if !overlapping.is_empty() {
        overlapping.join(" ")
    } else {
        best.map(|s| s.text.clone()).unwrap_or_default()
    }
}

/// EN-фраза считается завершённой, если оканчивается терминальным знаком
/// (`.` `!` `?` `…`) после trim-кавычек. STT-чанки часто обрываются на полуслове
/// (`'You'`, `'can't'`) — их честный перевод и должен заканчиваться «...».
fn en_is_complete(en: &str) -> bool {
    let t = en.trim().trim_end_matches(|c: char| matches!(c, '"' | '»' | ')' | ']' | ' '));
    let last = t.chars().last();
    matches!(last, Some('.') | Some('!') | Some('?') | Some('…'))
}

/// Возвращает причину мусора (непустой вектор), либо пустой если всё ок.
fn assess(text: &str, en: &str, duration: f64) -> Vec<&'static str> {
    let mut issues: Vec<&'static str> = Vec::new();
    let trimmed = text.trim();

    // 1. Пустой / fallback
    if trimmed.is_empty() || trimmed.starts_with('[') {
        issues.push("empty/fallback");
        return issues;
    }

    // 2. Уже маркер пропуска
    if trimmed == SKIP_MARKER {
        issues.push("already_skip");
        return issues;
    }

    // 3. Спецсимволы / markdown-мусор
    if trimmed.contains('<') {
        issues.push("angle_bracket");
    }
    if trimmed.contains("**") || trimmed.contains("__") {
        issues.push("markdown_bold");
    }

    // 4. Метки CoT / ревизии
    for &marker in GARBAGE_MARKERS {
        if trimmed.contains(marker) {
            issues.push("garbage_marker");
            break;
        }
    }

    // 5. Незакрытая кавычка «
    {
        let open = trimmed.matches('«').count();
        let close = trimmed.matches('»').count();
        if open != close {
            issues.push("unclosed_quote");
        }
    }

    // 6. Обрезанность: последний символ — connector
    {
        let en_complete = en_is_complete(en);
        let last_chars: &str = trimmed.rsplit(|c: char| c.is_whitespace() || c == ',' || c == '.')
            .next().unwrap_or("");
        let connectors = ["что", "и", "а", "или", "но", "на", "в", "из", "с", "к", "о"];
        if en_complete && connectors.iter().any(|&c| last_chars.eq_ignore_ascii_case(c)) {
            issues.push("truncated_connector");
        }
    }

    // 6b. Обрыв на многоточии («Кстати, если вы...») — модель отрезала мысль.
    // Флагим ТОЛЬКО если EN завершён: если EN сам оборван STT, честный перевод
    // оборванных слов тоже заканчивается «...» — это не мусор.
    if trimmed.ends_with("...") || trimmed.ends_with("…") || trimmed.ends_with("..") {
        if en_is_complete(en) {
            issues.push("truncated_ellipsis");
        }
    }

    // 6c. Незакрытая кавычка-тире/двоеточие в конце (префикс «RU: ...» хвост)
    if let Some(last) = trimmed.chars().last() {
        if matches!(last, ':' | ';' | ',') && !trimmed.ends_with("...") && en_is_complete(en) {
            // Vive la phrase: модель могла начать перечисление и оборваться
            let words_after_cyr = trimmed.split_whitespace().count();
            if words_after_cyr >= 2 {
                issues.push("truncated_trailing_punct");
            }
        }
    }

    // 7. Cyrillic ratio < 25%
    {
        let cyr = trimmed.chars()
            .filter(|c| ('А'..='я').contains(c) || *c == 'ё' || *c == 'Ё')
            .count();
        let alpha = trimmed.chars().filter(|c| c.is_alphabetic()).count();
        if alpha > 3 && (cyr as f64 / alpha as f64) < 0.25 {
            issues.push("low_cyrillic");
        }
    }

    // 8. Временной бюджет (> 22 симв/сек × длительность)
    {
        let len = trimmed.chars().count();
        if len > MIN_LEN_FOR_BUDGET && duration > 0.0 {
            let limit = (duration * CHARS_PER_SEC_MAX) as usize + 10;
            if len > limit {
                issues.push("too_long_for_timing");
            }
        }
    }

    // 9. RU << EN (обрезка) или RU >> EN (раздутие)
    {
        let en_len = en.chars().count();
        let ru_len = trimmed.chars().count();
        if en_len >= 6 && ru_len > 0 {
            let ratio = ru_len as f64 / en_len as f64;
            if ratio < 0.30 {
                issues.push("ru_too_short_vs_en");
            } else if ratio > 3.5 {
                issues.push("ru_too_long_vs_en");
            }
        }
    }

    issues
}

fn extract_gender(ctx: &PipelineContext, chunk: &SubtitleChunk) -> String {
    chunk
        .speaker_id
        .as_ref()
        .and_then(|spk_id| {
            ctx.speaker_segments.as_ref().and_then(|segments| {
                segments
                    .iter()
                    .find(|s| s.speaker_id == *spk_id)
                    .and_then(|s| s.gender.as_ref())
                    .map(|g| g.to_string())
            })
        })
        .unwrap_or_else(|| "Person".to_string())
}

/// Строгий промпт (без CoT): модель отвечает сразу переводом, без
/// «давайте подумаю», без ревизий и вариантов. Temp 0.2 = детерминированно.
fn build_strict_prompt(
    en_text: &str,
    gender: &str,
    prev: &[(String, String)],
    next: &[String],
) -> String {
    let speaker_gender = if gender == "female" { "Female" } else if gender == "male" { "Male" } else { "Person" };

    let mut p = String::from(
        "<|turn>system\n\
        You are an expert audiovisual translator adapting English subtitles into Russian for voiceover.\n\n\
        STRICT RULES:\n\
        1. Output ONLY the final Russian translation. No explanations, no variants, no quotation marks, no markdown.\n\
        2. Fix Speech-to-Text errors contextually.\n\
        3. Keep Russian as short as the English so it fits the audio timing.\n\
        4. GENDER: Current speaker is ").to_string();
    p.push_str(speaker_gender);
    p.push_str(". Match verbs/adjectives.\n\
        5. NO CENSORSHIP: preserve slang and profanity.\n<turn|>\n\
        <|turn>user\n");

    if !prev.is_empty() {
        p.push_str("\nPrevious context:\n");
        for (en, ru) in prev.iter().rev() {
            p.push_str(&format!("EN: {}\nRU: {}\n", en, ru));
        }
    }
    if !next.is_empty() {
        p.push_str("\nFuture context (DO NOT translate yet):\n");
        for en in next.iter() {
            p.push_str(&format!("EN: {}\n", en));
        }
    }

    p.push_str(&format!("\nTranslate ONLY this line:\nEN: {}\n<turn|>\n<|turn>model\n", en_text));
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assess_clean() {
        let issues = assess("Привет мир", "Hello world", 2.0);
        assert!(issues.is_empty(), "clean text → no issues");
    }

    #[test]
    fn test_assess_empty() {
        let issues = assess("", "", 2.0);
        assert!(issues.contains(&"empty/fallback"));
    }

    #[test]
    fn test_assess_skip_marker() {
        let issues = assess(crate::comm::SKIP_MARKER, "Hello", 2.0);
        assert!(issues.contains(&"already_skip"));
    }

    #[test]
    fn test_assess_garbage_marker() {
        let issues = assess("ИТОГОВЫЙ ВАРИАН", "Hello", 2.0);
        assert!(issues.contains(&"garbage_marker"));
    }

    #[test]
    fn test_assess_unclosed_quote() {
        let issues = assess("«Но если ваше", "But if you", 2.0);
        assert!(issues.contains(&"unclosed_quote"), "unclosed « detected");
    }

    #[test]
    fn test_assess_unclosed_quote_closed() {
        let issues = assess("«Привет» мир", "Hello world", 2.0);
        assert!(!issues.contains(&"unclosed_quote"), "balanced «» OK");
    }

    #[test]
    fn test_assess_truncated_connector() {
        let issues = assess("Просто даю знать. Просто предупреждаю о том, что", "I'm letting you know.", 2.0);
        assert!(issues.contains(&"truncated_connector"), "ends with «что» detected");
    }

    #[test]
    fn test_assess_low_cyrillic() {
        let issues = assess("Wait, I see the user wants me to translate", "Please translate", 3.0);
        assert!(issues.contains(&"low_cyrillic"));
    }

    #[test]
    fn test_assess_russian_prefix() {
        let issues = assess("RU: Кстати, если вы...", "By the way", 3.0);
        assert!(issues.contains(&"garbage_marker"), "RU: prefix detected as garbage");
    }

    #[test]
    fn test_assess_too_long_for_timing() {
        let issues = assess(
            "Это очень длинный текст который не влезет в отведённое время даже приблизительно",
            "Hello",
            1.0,
        );
        assert!(issues.contains(&"too_long_for_timing"), "22 chars/sec limit");
    }

    #[test]
    fn test_assess_ru_short_vs_en() {
        let issues = assess("Да.", "I am going to the grocery store right now to buy some bread", 3.0);
        assert!(issues.contains(&"ru_too_short_vs_en"));
    }

    #[test]
    fn test_assess_ok_long_russian() {
        let issues = assess(
            "Может, мы узнаем это после честно говоря очень ужасного девять с половиной.",
            "Maybe we'll find out after honestly a really awful nine and a half",
            6.0,
        );
        assert!(issues.is_empty(), "длинный, но содержательный перевод — ок");
    }

    #[test]
    fn test_assess_truncated_ellipsis() {
        // EN завершён точкой, а RU оборван — это мусор
        let issues = assess("Кстати, если вы...", "By the way, if you pack it.", 1.5);
        assert!(issues.contains(&"truncated_ellipsis"));
    }

    #[test]
    fn test_assess_ellipsis_ok_when_en_unfinished() {
        let issues = assess("Кстати, если вы...", "But people by the way, if you make if you", 1.5);
        assert!(!issues.contains(&"truncated_ellipsis"), "EN оборван — RU с «...» это норма");
    }

    #[test]
    fn test_assess_truncated_trailing_punct() {
        let issues = assess("И ещё одна вещь:", "And one more thing", 1.5);
        assert!(issues.contains(&"truncated_trailing_punct"));
    }

    #[test]
    fn test_clean_strict_output_uses_shared_cleaner() {
        let out = crate::translation::clean_output("<|channel>thought\n<channel|>Привет мир world");
        assert_eq!(out, "Привет мир");
    }

    #[test]
    fn test_clean_strict_output_markdown() {
        let out = crate::translation::clean_output("**Привет мир**");
        assert_eq!(out, "Привет мир");
    }
}