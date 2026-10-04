use crate::comm::{PipelineContext, SubtitleChunk, SKIP_MARKER};
use crate::llm::{message, LlmSession};
use anyhow::Result;
use tauri_plugin_llama_engine::engine::LlmMessage;

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

/// Проверяет изохронию/мусор в переводе и чинит брак ретраями.
///
/// Принимает сессию движка, оставшуюся от `translate`: оба этапа — LLM-этапы,
/// и поднимать второй `llama-server.exe` ради десятка запросов означало лишние
/// ~5 с загрузки модели в VRAM. Если сессии нет (этап вызван отдельно) — она
/// поднимается лениво, как раньше. Сессия уничтожается на выходе, то есть
/// строго ДО старта TTS (desktop §6.5).
pub fn verify_with_session(
    ctx_and_session: (PipelineContext, Option<LlmSession>),
) -> Result<PipelineContext> {
    let (mut ctx, mut session) = ctx_and_session;

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

    let mut ok_count = 0;
    let mut retry_count = 0;
    let mut fail_count = 0;
    // Последние 3 обработанные пары (EN, RU) — для prev-контекста строгого ретрая
    let mut prev_pairs: Vec<(String, String)> = Vec::new();
    // Формат промпта ретрая: вычисляется один раз на весь проход, см. ниже.
    let mut prompt_style: Option<crate::translation::PromptStyle> = None;

    for (i, chunk) in chunks.iter_mut().enumerate() {
        if crate::is_cancelled() {
            log::warn!("VERIFY: отменено пользователем на чанке {}", i);
            return Ok(PipelineContext {
                translated_chunks: Some(chunks),
                ..ctx
            });
        }
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

        // Движок поднимаем только когда реально понадобился ретрай: обычно
        // верификация проходит без единого обращения к LLM (desktop §6.5).
        // Чаще всего сессия уже есть — её оставил `translate`.
        if session.is_none() {
            let app = crate::app_handle().ok_or_else(|| {
                anyhow::anyhow!("AppHandle не инициализирован — ретрай перевода невозможен")
            })?;
            log::info!(
                "VERIFY: есть бракованные чанки — поднимаем движок LLM для ретраев"
            );
            session = Some(LlmSession::open(app).map_err(anyhow::Error::msg)?);
        }
        let session = session.as_ref().expect("сессия LLM открыта выше");
        // Формат определяется ОДИН за весь проход верификации, а не на каждый
        // ретрай: иначе в лог уходит «формат промпта …» по разу на чанк, а
        // главное — решение формата перестаёт быть одним решением и может
        // разойтись между попытками. Кэш заполняется при первом обращении к
        // движку: раньше открыть его нечем (сессия поднимается лениво).
        let prompt_style = *prompt_style.get_or_insert_with(|| {
            crate::translation::PromptStyle::resolve(
                ctx.config.prompt_style.as_deref(),
                session.model_path(),
            )
        });

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
            // Дефекты пересобираются на КАЖДОЙ попытке из фактического текста
            // последней генерации: после первой попытки исходный список уже не
            // описывает то, что модель вернула (и могла исправить одно и
            // сломать другое).
            let defects: Vec<String> = assess(&chunk.text, en, duration)
                .iter()
                .map(|c| issue_hint(c))
                .collect();
            let messages = build_strict_messages(
                en,
                &gender,
                &prev_pairs,
                &next_ens,
                &defects,
                prompt_style,
            );
            let cleaned = match session.generate(&messages, Some(temp)) {
                Ok(generation) => crate::translation::clean_output(&generation.text),
                Err(e) => {
                    log::warn!("VERIFY: [{}] attempt {} generate error: {}", i, attempt, e);
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

    // Движок (если поднимался) освобождает VRAM при выходе из функции —
    // до этапа TTS (desktop §6.5).
    drop(session);

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

    // 3a. Замаскированное слово: `бл*ть`, `п*здец`, `f**k`. Проверка `**`
    // выше это не ловит — она про markdown, а не про цензуру.
    //
    // Наблюдалось на Index-Translate-9B: чанк с «F you broke cheaters» пришёл
    // как «Бл*ть, обманщики…». Причина не в промпте — то же самое получалось в
    // chat- и в instTrans-формате, тогда как 2B и Gemma в этом же тесте не дали
    // ни одной звёздочки. То есть модель узнаёт слово и цензурит его сама.
    //
    // Проверять надо именно букву с `*` внутри слова, а не любое вхождение `*`:
    // одиночная звёздочка в «нас*ледие» или курсив — нормальный текст.
    if trimmed.split_whitespace().any(|w| w.chars().any(|c| c == '*')) {
        issues.push("censored_symbol");
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

    // 10. Цифры в тексте. Правило «пиши числа словами» есть и в системном
    // промпте, и в требованиях instTrans, но ДО этого места оно не
    // проверялось вообще: ни в одном из 13 предыдущих пунктов. Модель
    // нарушала его («Сезон 10», «9.5») — и нарушение уходило в SRT и в
    // озвучку, где CosyVoice3 цифры не читает.
    if trimmed.chars().any(|c| c.is_ascii_digit()) {
        issues.push("digits_present");
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

/// Переводит технический код проверки в требование к переводу.
///
/// Коды `assess` — это внутренний язык верификатора («too_long_for_timing»);
/// в промпт их отдавать бессмысленно, модель не знает этих слов. Раньше
/// повторная генерация вообще не получала списка проблем и просто
/// переспрашивала то же самое при меньшей температуре — поэтому дефекты
/// вроде «9.5» цифрами переживали ретрай, а счётчик RETRY оставался нулевым.
fn issue_hint(code: &str) -> String {
    match code {
        "digits_present" => "The previous attempt left digits in the text. Write EVERY number \
             as Russian words, including decimals and ratings (9.5 -> девять и пять десятых)."
            .to_string(),
        "censored_symbol" => "The previous attempt masked a word with an asterisk, like бл*ть. \
             Write the whole word in full, in plain letters. Never use *, x, dots or other \
             symbols to stand in for letters — the speech engine cannot pronounce them."
            .to_string(),
        "too_long_for_timing" => {
            "The previous attempt was too long for its time window. Shorten it: drop fillers \
             and pick shorter words. Output fewer words than the source."
                .to_string()
        }
        "ru_too_short_vs_en" => {
            "The previous attempt was truncated. Translate the whole source text, do not cut \
             off the ending."
                .to_string()
        }
        "ru_too_long_vs_en" => {
            "The previous attempt was bloated compared to the source. Be concise and literal."
                .to_string()
        }
        "unclosed_quote" => "The previous attempt left an unclosed quotation mark.".to_string(),
        "truncated_connector" => {
            "The previous attempt ended on a dangling conjunction. Complete the phrase or \
             shorten it to a whole sentence."
                .to_string()
        }
        "truncated_ellipsis" => {
            "The previous attempt ended with an ellipsis mid-sentence. Finish the thought or \
             drop it."
                .to_string()
        }
        "truncated_trailing_punct" => "The previous attempt ends abruptly.".to_string(),
        "low_cyrillic" => {
            "The previous attempt is mostly not Russian. Translate the whole text into Russian."
                .to_string()
        }
        "garbage_marker" => {
            "The previous attempt contained meta-commentary. Output only the translation.".to_string()
        }
        "empty/fallback" => "The previous attempt was empty. Output the translation.".to_string(),
        "angle_bracket" => {
            "The previous attempt contained angle brackets. Output plain text.".to_string()
        }
        "markdown_bold" => "The previous attempt contained markdown. Output plain text.".to_string(),
        other => format!("The previous attempt had this problem: {other}."),
    }
}

/// Строгий промпт (без CoT): модель отвечает сразу переводом, без
/// «давайте подумаю», без ревизий и вариантов. Низкая температура =
/// детерминированно.
///
/// Формат повторяет формат основного перевода (`PromptStyle`): ретрай,
/// собранный в чужом для этой модели виде, чинит один дефект и ломает всё
/// остальное — проверено, что Index-Translate вне instTrans начинает
/// дублировать и переводить соседние чанки.
fn build_strict_messages(
    en_text: &str,
    gender: &str,
    prev: &[(String, String)],
    next: &[String],
    defects: &[String],
    prompt_style: crate::translation::PromptStyle,
) -> Vec<LlmMessage> {
    let speaker_gender = if gender == "female" { "Female" } else if gender == "male" { "Male" } else { "Person" };

    // Формат передаётся готовым, а не разрешается заново: перевод и верификация
    // обязаны говорить с моделью на одном языке. Повторное разрешение здесь
    // означало бы второе чтение env/настроек — и возможность разойтись.
    match prompt_style {
        crate::translation::PromptStyle::InstTrans => {
            crate::translation::build_messages_insttrans(en_text, speaker_gender, defects)
        }
        crate::translation::PromptStyle::Chat => {
            build_strict_messages_chat(en_text, speaker_gender, prev, next, defects)
        }
    }
}

fn build_strict_messages_chat(
    en_text: &str,
    speaker_gender: &str,
    prev: &[(String, String)],
    next: &[String],
    defects: &[String],
) -> Vec<LlmMessage> {
    let mut system = String::from(
        "You are an expert audiovisual translator adapting English subtitles into Russian for voiceover.\n\n\
         STRICT RULES:\n\
         1. Output ONLY the final Russian translation. No explanations, no variants, no quotation marks, no markdown.\n\
         2. Fix Speech-to-Text errors contextually.\n\
         3. Keep Russian as short as the English so it fits the audio timing.\n\
         4. NUMBERS TO WORDS: spell out ALL numbers as Russian words, no digits at all — the speech engine cannot pronounce digits.\n\
         5. GENDER: Current speaker is ",
    );
    system.push_str(speaker_gender);
    system.push_str(
        ". Match verbs/adjectives.\n\
         6. NO CENSORSHIP: preserve slang and profanity.",
    );
    if !defects.is_empty() {
        system.push_str("\n\nFIX THESE SPECIFIC PROBLEMS — your previous attempt was rejected for them:");
        for d in defects {
            system.push_str(&format!("\n- {d}"));
        }
    }

    let mut user = String::new();
    if !prev.is_empty() {
        user.push_str("Previous context:\n");
        for (en, ru) in prev.iter().rev() {
            user.push_str(&format!("EN: {}\nRU: {}\n", en, ru));
        }
    }
    if !next.is_empty() {
        user.push_str("\nFuture context (DO NOT translate yet):\n");
        for en in next.iter() {
            user.push_str(&format!("EN: {}\n", en));
        }
    }
    user.push_str(&format!("\nTranslate ONLY this line:\nEN: {}\n", en_text));

    vec![message("system", system), message("user", user)]
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
    fn test_assess_digits_flagged() {
        // Реальные нарушения из прогона Index-Translate: «Сезон 10» и «9.5».
        // CosyVoice3 цифры не читает, поэтому такой текст обязан уйти в ретрай.
        assert!(assess("Сезон 10 наконец-то здесь", "Season 10 is here", 3.0)
            .contains(&"digits_present"));
        assert!(assess(
            "Я считаю, что 9.5 был худшим сезоном",
            "I think 9.5 was the worst season",
            4.0
        )
        .contains(&"digits_present"));
    }

    #[test]
    fn test_assess_spelled_numbers_ok() {
        let issues = assess(
            "Девять целых пять десятых был худшим сезоном",
            "9.5 was the worst season",
            4.0,
        );
        assert!(!issues.contains(&"digits_present"));
    }

    // ---- Цензура слов (Index-Translate-9B) ----

    #[test]
    fn test_assess_masked_word_flagged() {
        // Реальный вывод 9B: чанк с матом пришёл с `Бл*ть`. Проверка `**`
        // (markdown) это не ловит, нужна отдельная.
        let issues = assess(
            "Не могут позволить себе ПК. Бл*ть, обманщики",
            "F you broke cheaters",
            3.0,
        );
        assert!(
            issues.contains(&"censored_symbol"),
            "замаскированное слово должно уходить в ретрай, а не в озвучку"
        );
    }

    #[test]
    fn test_assess_plain_profanity_not_flagged() {
        // Мат без цензуры — законный результат, ретрай тут не нужен.
        let issues = assess("Чёрт, это не весело", "This is not fun", 2.0);
        assert!(!issues.contains(&"censored_symbol"));
    }

    #[test]
    fn test_assess_markdown_bold_not_flagged_as_censorship() {
        // `**Привет**` — markdown, цензуры нет. Иначе формат ломался бы вхолостую.
        let issues = assess("**Привет** мир", "Hello world", 2.0);
        assert!(issues.contains(&"markdown_bold"));
        assert!(!issues.contains(&"censored_symbol"));
    }

    #[test]
    fn test_issue_hint_for_masked_word_mentions_no_symbols() {
        let hint = issue_hint("censored_symbol");
        assert!(
            hint.contains("whole word in full"),
            "подсказка ретраю должна требовать слово целиком: {hint}"
        );
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