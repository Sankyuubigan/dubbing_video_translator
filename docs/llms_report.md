# Translation Model Comparison

## 1. Модели (в хронологическом порядке)

| Параметр | Hy-MT2-7B | Gemma 4 E4B | Tower-Plus-9B | Gemma 4 12B |
|---|---|---|---|---|---|
| Архитектура | Hunyuan-dense | Gemma 4 (MoE) | Gemma 2 | Gemma 4 (dense) |
| Размер | 7.5B params | 4B active / 8B total | 9.24B params | 12B params |
| Разработчик | Tencent | Google | (community) | (community) |
| Файл GGUF | Hy-MT2-7B.i1-IQ4_NL.gguf | gemma-4-E4B-it-heretic-QAT-UD-Q4_K_XL.gguf | Tower-Plus-9B-abliterated-hf-data.i1-IQ4_NL.gguf | gemma-4-12B-it-heretic-QAT-UD-Q4_K_XL.gguf |
| Quant | IQ4_NL (4.5 bpw) | Q4_K_XL | IQ4_NL | Q4_K_XL |
| Размер на диске | ~4.07 GiB | ~3.0 GiB | ~5.06 GiB | ~7.2 GiB |
| Vocab | 128167, BPE | 256k, SPM | 256k, SPM | 256k, SPM |
| Позиция в pipeline | 1-я (первая попытка) | 2-я (после Hy-MT2) | 3-я (предыдущая) | 4-я (текущая) |

## 2. Важное замечание

Gemma 4 (все версии) использует **родной `<|turn>...<turn|>` формат**.
Остальные модели (Hy-MT2, Tower-Plus-9B) — старый Gemma 2 формат.

## 3. Текущие параметры Gemma 4 12B (в коде)

Константы в `src-tauri/src/translation/mod.rs`:

| Параметр | Константа | Значение |
|----------|-----------|----------|
| Контекст | `MAX_N_CTX` | 8192 |
| Предыдущих чанков | `MAX_CONTEXT_CHUNKS` | 3 |
| Температура | `SAMPLING_TEMP` | 1.0 |
| Top-K | `SAMPLING_TOP_K` | 64 |
| Top-P | `SAMPLING_TOP_P` | 0.95 |
| Repetition penalty | `SAMPLING_REP_PENALTY` | 1.05 |
| Seed | — | 42 (`dist(42)`) |
| BOS | `AddBos` | `Always` (токен 2) |

Формат промпта (plain instruction, без turn-токенов):
```
Translate the following English subtitle to Russian.

Previous translations for context:
[id:0] EN: "prev text"
RU: "prev translation"
[id:1] EN: "older text"
RU: "older translation"

Fix any ASR errors. Output only the Russian translation.

{current_text}
```

Стоп-токен: EOS (3). `<turn|>` и `<|channel>`/`<channel|>` обрезаются в `clean_output`.

## 4. Промпт-форматы (официальные)

### Hy-MT2 (Tencent)

Официальный [Hy-MT2](https://github.com/Tencent-Hunyuan/Hy-MT2) промпт — без спецтокенов:
```
Translate the following text into Russian. Note that you should only output the translated result without any additional explanation:{source_text}
```
BOS (127958) через `AddBos::Always`. Стоп-токены: 3 (EOS), 127960, 127962, 127958.

### Gemma 4 (Google)

Официальный промпт использует `<|turn>`/`<turn|>`:
```
<|turn>user
Translate the following text into Russian. Output only the translation without any additional explanation.

{text}<turn|>
<|turn>model
```
Сэмплинг: temp=1.0, top_p=0.95, top_k=64. Стоп-токен: `<turn|>`.

**Gemma 4 12B** (`gemma-4-12B-it-heretic-QAT-UD-Q4_K_XL.gguf`) — та же архитектура и формат промпта, но dense 12B params (не MoE). Файл: `D:\nn\models\llm\uncen\gemma-4-12B\`.

### Tower-Plus-9B

Формат Gemma 2 (тот же что в коде):
```
<start_of_turn>user
Translate the following text into Russian. Fix STT errors. Output only the translation — no explanations, no options.

{text}<end_of_turn>
<start_of_turn>model
```

## 5. Результаты перевода (26 chunks) — Original Comparison (old Tower)

```
EN ======================== | Tower (old JSON+0.7) ========== | ============== Hy-MT2-7B ============== | =============================== Gemma 4 ===============================
...
```
(Original comparison unchanged — see git history)

## 6. Финальный результат: Tower-Plus-9B v4 (temp=0.15, plain text, EN+RU context)

**Конфигурация:**
- `SAMPLING_TEMP=0.15`, `top_k=20`, `top_p=0.6`
- `MAX_N_CTX=8192` (полный контекст)
- Plain text output (без JSON/grammar)
- Source-side context (EN+RU в предыдущих переводах)
- Gemma 2 prompt format

| EN | Tower v4 (финальный) | Hy-MT2 (лучший) |
|---|---|---|
| 'Cause you are oh, nineteen | Потому что тебе всего девятнадцать. ✅ | Потому что тебе девятнадцать. ✅ |
| Unless you went to | Если вы не посещали один из клубов... ⚠️ (формальное Вы) | Если только ты не ходила... ✅ (неформальное ты) |
| you get naked on stage | Ты раздеваешься на сцене. ✅ | Иначе приходится раздеваться. ✅ |
| But this body looks amazing | Но это тело выглядит потрясающе. **Оно лучше вживую**, чем на FaceTime. ✅ | Но тело у тебя потрясающее. Вживую... ✅ |
| How tall are you? | Эй, как ты ростом? ✅ | Сколько тебя рост? ❌ |
| Five seven | Пять семь. ✅ | 175 см. ✅ |
| Okay. Yeah you look | **я была**, О да... ✅ (правильный род) | ты выглядишь просто идеально. ✅ |
| Yeah, it's a little tube top | Да, это маленькое **топик**, которое просто держит всё вместе. ✅ | Это просто **топ-трубка**, он всё держит. ✅ |
| I kinda slipped sometimes | Иногда я немного спотыкалась. ❌ (род ✅, смысл ❌) | Иногда он слегка соскальзывает. ✅ |
| Well let it slip | Ну давай, пусть это **проскользнет**. ✅ | Ну, пусть соскальзывает. ✅ |
| Jesus Christ, man | Иисус Христос, **чувак**, это же фантастическое море. ❌ (STT) | Чёрт, это просто невероятные груди! ✅ (STT fix) |
| A little bit of | **Немного.** ✅ | (N/A) |
| What how do you like | Как тебе нравится, когда играют с **твоей грудью**? ✅ | Как тебе, когда играют с твоей грудью? ✅ |
| I think I like the squeeze | Мне кажется, мне нравится **сжимать**. ✅ | Мне нравится, когда сжимают её. ✅ |
| That's good | **Это хорошо.** ✅ (нет "орошо"!) | Это хорошо. ✅ |
| Any nipple play you like? | Любопытствуешь ли ты по поводу игры с сосками? ✅ | Как тебе, когда играют с твоей грудью? ✅ |
| **Кириллица** | **Нет галлюцинаций** 🎉 | "орошо", "потрошающе" ❌ |
| **Скорость** | **8.3s translate / 42.7s total** ⚡ | ~7-10s (не замерялось) |

## 7. Эволюция Tower-Plus-9B

| Версия | Изменения | Результат |
|--------|-----------|-----------|
| v1 (JSON+0.7) | JSON grammar, temp=0.7, старая модель | Кириллица ок, но "человек", повторы, "трусики" |
| v2 (no grammar+0.1) | temp=0.1, plain text, старый промпт | **Повторы контекста** ❌ (критический баг) |
| v3 (English:X\nRussian:) | Новый промпт English: X\nRussian: | **Без повторов** ✅, быстрее |
| v4 (EN+RU context) | Source-side контекст, temp=0.15, 8192 ctx | **"топик"** ✅, **"я была"** ✅, **"вживую"** ✅ |

## 8. Вердикт

### Hy-MT2-7B — непригоден
**Hy-MT2-7B — говнище.** BPE токенизатор ломает кириллицу: "орошо", "потрошающе", "потрясающе"→"потрошающе". Тексты выглядят как битые кракозябры. Теоретически перевод осмысленный, но на практике результат unusable из-за токенизаторного бага. Tower-Plus-9B со SPM токенизатором эту проблему не имеет.

### Качество перевода (Tower-Plus-9B v4):
- **Главный плюс**: Нет галлюцинаций кириллицы (SPM токенизатор) 🎉
- **Значительное улучшение**: "топик" вместо "трусики", "я была" вместо "я был", "вживую" вместо "в реальности"
- **Source-side контекст (EN)** даёт лучшие результаты, чем только target-side (RU)
- **temp=0.15** — оптимально для перевода (research consensus)

### Остаточные проблемы (лимит 9B модели):
- "sea" → "море" (STT ошибка не исправлена)
- "спотыкалась" вместо "соскальзывал" (slip контекст топика)
- Иногда формальное "Вы" вместо "ты"

### Скорость:
- Tower-Plus-9B v4: **8.3s translate**, 42.7s total pipeline ⚡
- В 4 раза быстрее, чем с JSON grammar (35.3s)

### Файлы:
- `resource_tower_plus_9b/` — полные research findings по Tower-Plus-9B
- `src-tauri/src/translation/mod.rs` — код пайплайна

## 9. Переход на Gemma 4 12B (2026-06-16)

**Причина замены:** Tower-Plus-9B (9.24B, Gemma 2) упирается в лимит 9B — STT ошибки не исправляются ("sea"→"море"), контекст топика не понимается ("спотыкалась"). Gemma 4 12B — архитектура Gemma 4, 12B dense params.

### Изменения в коде

| Параметр | Tower-Plus-9B | Gemma 4 12B |
|----------|---------------|-------------|
| `AddBos` | `Always` (токен 2) | `Always` (токен 2) |
| EOS | токен 107 (`<end_of_turn>`) | токен 3 (модельный) |
| `SAMPLING_TEMP` | 0.15 | 1.0 |
| `SAMPLING_TOP_K` | 20 | 64 |
| `SAMPLING_TOP_P` | 0.6 | 0.95 |
| Prompt | Gemma 2 (`<start_of_turn>`) | Plain instruction (turn-токены не работают) |
| Модель | `D:\nn\models\translation\Tower-Plus-9B-...` | `D:\nn\models\llm\uncen\gemma-4-12B\gemma-4-12B-it-heretic-QAT-UD-Q4_K_XL.gguf` |

### Новая конфигурация

- **Файл:** `D:\nn\models\llm\uncen\gemma-4-12B\gemma-4-12B-it-heretic-QAT-UD-Q4_K_XL.gguf`
- **Архитектура:** Gemma 4 (dense), 12B params
- **Quant:** Q4_K_XL, ~7.2 GiB
- **Vocab:** 256k, SPM
- **Контекст:** 8192 токенов
- **BOS:** Не добавляется (промпт начинается с `<|turn>`)
- **EOS:** Токен 3 (проверка в цикле генерации) + `<turn|>` (пост-обработка)
- **Сэмплинг:** temp=1.0, top_k=64, top_p=0.95, rep_penalty=1.05, seed=42
- **Prompt format:** Gemma 4 native (`<|turn>system...<turn|><|turn>user...<turn|><|turn>model`)
- **System prompt:** "You are a professional translator. Translate English subtitles to Russian. Use previous dialogue segments to resolve pronouns and maintain topic consistency. Output only the translation, no explanations."
- **Контекст:** до 3 предыдущих чанков (EN + RU), обратный порядок (сначала свежие)
- **max_new:** `(prompt_tokens / 2).max(64).min(512)`
- **KV cache:** очищается на каждый чанк

### Ожидания
- **12B параметров** должны лучше исправлять STT ошибки
- Большая модель должна лучше понимать контекст (tube top slip и т.п.)
- temp=1.0 — стандарт для Gemma 4 (официальные рекомендации Google)
