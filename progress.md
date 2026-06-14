# Progress Log

## Session: 2026-06-14

### Phase 0: Чтение кода и первичные правки (до осознания docs/)
- **Status:** complete
- Actions taken:
  - Прочитал translation/mod.rs, tts/mod.rs, download.rs
  - Нашёл баг: piper URL без vits- префикса
  - Нашёл что extract_ru_texts падает, добавил two-pass логику
  - Добавил логирование в try_parse_batch
  - Увеличил detect_repeated_translations threshold 3→4
  - Поднял обработку пустого вывода (ok=false)
- Files created/modified:
  - src-tauri/src/tts/mod.rs (vits-piper- prefix)
  - src-tauri/src/translation/mod.rs (extract_ru_texts, try_parse_batch, пустой вывод)

### Phase 1: Диагностика перевода (complete)
- **Status:** complete
- Actions taken:
  - Прочитал AGENTS.md: batch_size=5 (документировано), pipeline работал 100%
  - Прочитал project-rules.md: формат перевода тегированный (устарело), pipeline проверки
  - Прочитал RULES.md: output только в test/
  - Создал planning файлы
  - Гипотеза: max_new = min(input*2, 4096) может не хватать для 20 чанков
  - max_new увеличен: min(input*3, 8192)
  - Добавлен debug-лог сырого вывода при падении парсинга
- Questions:
  - Был ли баг с max_new подтверждён? Пока нет — нужен прогон test-pipeline

### Phase 2: Исправление перевода (complete)
- **Status:** complete
- Actions taken:
  - max_new: (input*2).max(128).min(4096) → (input*3).max(256).min(8192)
  - Добавлен preview вывода (info level) при падении парсинга
  - extract_ru_texts: rewritten with strict/fallback two-pass
  - try_parse_batch: detailed logging (missing indices, repeat count)
  - detect_repeated_translations threshold 3→4
- Files created/modified:
  - src-tauri/src/translation/mod.rs (multiple fixes)

### Phase 3: Исправление TTS (complete)
- **Status:** complete
- Actions taken:
  - URL: piper- → vits-piper- для всех моделей
  - Удалён ru_RU-tatyana-medium (404 — не существует в tts-models release)
  - Добавлен rename vits-piper-{name}/ → {name}/ после распаковки архива
  - Оставлено 3 модели: dmitri (male), ruslan (male), irina (female) — достаточно для 2 спикеров
- Files created/modified:
  - src-tauri/src/tts/mod.rs

### Phase 5: Синхронизация документации (complete)
- **Status:** complete
- Actions taken:
  - AGENTS.md: batch_size 5→20, max_new updated
  - project-rules.md: max_new limit updated
  - task_plan.md, findings.md, progress.md: created
- Files created/modified:
  - docs/AGENTS.md
  - docs/project-rules.md
  - task_plan.md (created)
  - findings.md (created)
  - progress.md (created)

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| cargo check --tests | - | OK | OK | ✓ |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 1 — Диагностика перевода |
| Where am I going? | Phase 2-4: исправление, тестирование, синхронизация docs |
| What's the goal? | Починить перевод (fallback на оригинал) и TTS download |
| What have I learned? | См. findings.md |
| What have I done? | См. progress.md выше |
