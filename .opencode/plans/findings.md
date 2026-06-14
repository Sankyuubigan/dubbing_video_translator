# Findings: Gemma 4 Translation Improvement

## 1. Текущая проблема

Gemma 4 E4B в `mod.rs` работает через **чужой промпт-формат** (Gemma 2):

```
<start_of_turn>user
Translate the following text into Russian. Fix STT errors. Output only the translation — no explanations, no options.

{text}<end_of_turn>
<start_of_turn>model
```

Это формат Gemma 2 (`<start_of_turn>`/`<end_of_turn>`), а Gemma 4 использует `<|turn>`/`<turn|>`. Из-за этого модель не понимает структуру диалога и качество страдает.

## 2. Официальный промпт-формат Gemma 4

Из документации Google AI for Developers:
- Использует `<|turn>`/`<turn|>` токены
- Роли: `system`, `user`, `model`
- BOS токен добавляется автоматически

### Пример диалога:
```
<|turn>system
You are a helpful assistant.<turn|>
<|turn>user
Hello.<turn|>
<|turn>model
Hi there!<turn|>
```

### Для перевода:
```
<|turn>user
Translate the following text into Russian. Output only the translation without any additional explanation.

{text}<turn|>
<|turn>model
```

Или с system prompt:
```
<|turn>system
You are a professional translator. Fix STT errors. Keep speaker gender.<turn|>
<|turn>user
Previous context: {prev_text}

Translate: {text}<turn|>
<|turn>model
```

## 3. Официальные параметры сэмплирования

Источник: https://ai.google.dev/gemma/docs/core/model_card_4

| Параметр | Официально | Сейчас |
|----------|-----------|--------|
| temperature | 1.0 | 0 (greedy) |
| top_p | 0.95 | — |
| top_k | 64 | — |
| repetition_penalty | не указан | 1.05 (от Tower) |

Текущие параметры скопированы от Tower-Plus-9B и НЕ подходят для Gemma 4.
temp=0 (greedy) убивает вариативность и естественность перевода.

## 4. Thinking Mode

У Gemma 4 есть режим размышлений через `<|think|>` токен.
- Если `<|think|>` в system prompt → модель генерирует reasoning + ответ
- Если `<|think|>` НЕТ → модель сразу даёт ответ
- Для перевода `<|think|>` НЕ НУЖЕН

## 5. Стоп-токены для Gemma 4

Текущие стоп-токены в коде: `token_eos()`, `LlamaToken(1)`, `LlamaToken(213)` — это от Tower.
Для Gemma 4 нужно:
- `token_eos()` (общий)
- `<turn|>` токен (конец turn'а Gemma 4)

## 6. Решения конкретных проблем

### Проблема 1: "соскальзывает" (chunk 10: "I kind of slipped sometimes")
Корень: модель не знает контекст — предыдущий чанк про "tube top".
Решение: добавить предыдущий чанк(и) в промпт как контекст.

### Проблема 2: "sea" → "груди" (chunk 18: STT ошибка "sea" вместо "sight")
Корень: модель не пытается исправить STT ошибки.
Решение: в system prompt добавить явную инструкцию фиксить STT ошибки по контексту + в user prompt добавить напоминание.

### Проблема 3: мужской род для Speaker_2
Корень: модель не знает что Speaker_2 женщина.
Решение: передавать speaker_id в промпте, либо добавить system инструкцию про род.
