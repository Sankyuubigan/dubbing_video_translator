# Task Plan: Improve Gemma 4 Translation Quality

## Goal
Настроить Gemma 4 E4B с правильным промпт-форматом (`<|turn>`/`<turn|>`), параметрами сэмплирования (temp=1.0, top_p=0.95, top_k=64), контекстом соседних чанков и STT-correction инструкцией. Добиться правильного перевода "соскальзывает" (контекст топика) и "груди" (STT fix "sea"→"sight").

## Phases

### Phase 1: ✅ Исследование
- [x] Прочитать translation_report.md
- [x] Прочитать AGENTS.md и текущий mod.rs
- [x] Изучить официальный промпт-формат Gemma 4 (Google AI docs)
- [x] Найти правильные параметры сэмплирования
- [x] Записать findings.md

### Phase 2: Изменить mod.rs
- [ ] `build_prompt()` → формат `<|turn>`/`<turn|>`
- [ ] `MAX_N_CTX` → 4096 (Gemma 4 поддерживает больше)
- [ ] Стоп-токены: `token_eos()` + `<turn|>`
- [ ] Сэмплинг: temp=1.0, top_p=0.95, top_k=64 (через `chain_simple`)
- [ ] System prompt с инструкциями: STT fix, род, контекст
- [ ] Добавить контекст предыдущего чанка
- [ ] Обновить тест `test_build_prompt_format()`

### Phase 3: Сборка и тестирование
- [ ] Собрать release (`cargo run --bin test-pipeline --release ...`)
- [ ] Запустить на `test/for_test.mp4`
- [ ] Проверить `%TEMP%\dubvidtra_subtitles_ru.srt`

### Phase 4: Оценка
- [ ] Сравнить с `test/ru_gemma4.srt`
- [ ] Проверить chunk 10: "соскальзывает"
- [ ] Проверить chunk 18: "груди" вместо "море"
- [ ] При необходимости скорректировать prompt и перетестировать

## Новый prompt format (план)

```rust
fn build_prompt(current_chunk: &str, prev_context: Option<&str>) -> String {
    let system = "You are a professional translator from English to Russian. \
        Fix speech recognition errors using context. \
        Keep the speaker's gender. \
        Output only the translation, no explanations.";

    let ctx = match prev_context {
        Some(c) => format!("Previous context:\n{}\n\nTranslate the following text into Russian. Fix any STT errors (the text may contain misheard words):\n{}", c, current_chunk),
        None => format!("Translate the following text into Russian. Fix any STT errors (the text may contain misheard words):\n{}", current_chunk),
    };

    format!(
        "<|turn>system\n{}<turn|>\n<|turn>user\n{}<turn|>\n<|turn>model\n",
        system, ctx
    )
}
```

## Ключевые изменения sampling

```rust
let mut sampler = LlamaSampler::chain_simple([
    LlamaSampler::top_k(64),
    LlamaSampler::top_p(0.95, 1), // min_keep
    LlamaSampler::temp(1.0),
    LlamaSampler::dist(42),
]);
```

## Стоп-токены

Нужно определить `<turn|>` токен. В Gemma 4 это специальный токен.
В llama.cpp: `model.token_eos()` для EOS, плюс добавить стоп на `<turn|>`.

В контексте llama-cpp-2, нужно передавать stop tokens:
```rust
// Стоп-токены Gemma 4:
// - token_eos() — стандартный EOS
// - token_custom("<turn|>") — конец turn'а
```

## Важные детали

1. `AddBos::Always` — Gemini 4 сама добавляет BOS, это правильно
2. `n_ctx` = 4096 — минимум для работы с контекстом
3. Контекст: 1-2 предыдущих чанка достаточно для связности
4. Speaker ID: можно добавить в промпт "[Speaker_1]: {text}"
