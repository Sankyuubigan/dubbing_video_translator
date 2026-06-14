# Task Plan: Исправление багов DubVidTra2

## Goal
Починить пайплайн: исправить регрессию перевода (GBNF grammar выдаёт пустой вывод для batch из 20 чанков) и TTS скачивание (404 piper model + network reset).

## Current Phase
Phase 2

## Phases

### Phase 1: Диагностика перевода
- [x] Собрать информацию: что изменилось между рабочей версией (batch=5) и текущей (batch=20)
- [x] Выявить причину: max_new не хватает для 20 чанков (гипотеза)
- [x] Увеличить max_new: (input*3).max(256).min(8192)
- [x] Добавить debug-лог вывода при падении парсинга
- **Status:** complete

### Phase 2: Исправление перевода
- [x] max_new увеличен
- [x] extract_ru_texts two-pass (strict/fallback)
- [x] try_parse_batch logging improved
- [x] detect_repeated_translations threshold 3→4
- [ ] Добавить обработку случая когда ru_texts меньше expected_count (заполнять оригиналом)
- **Status:** in_progress

### Phase 3: Исправление TTS
- [x] URL piper- → vits-piper-
- [ ] Проверить структуру архива vits-piper: распаковывается в ru_RU-dmitri-medium/ или vits-piper-*/?
- [ ] Исправить путь поиска модели после распаковки если нужно
- **Status:** pending

### Phase 4: Тестирование
- [ ] Собрать release и прогнать test-pipeline
- [ ] Проверить субтитры (все переведены, нет [original] маркеров)
- [ ] Проверить TTS (dubbing успешен)
- **Status:** pending

### Phase 5: Синхронизация документации
- [ ] Обновить AGENTS.md (batch_size=20, текущее состояние)
- **Status:** pending

## Key Questions
1. Почему grammar fails для 20 чанков? (max_new? грамматика? модель?)
2. Какова структура vits-piper архива? (с vits- префиксом или без?)
3. Есть ли проблема с network connectivity до GitHub?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| BATCH_SIZE=20 | Было увеличено сознательно (подтверждено пользователем) |
| extract_ru_texts two-pass | strict/fallback разделение — берём ru/cyrillic, если нет — остальное |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| piper model 404 | 1 | URL: piper- → vits-piper- |
| espeak-ng-data connection reset | 1 | Network issue, не исправляется кодом |
| translation parsing fallback | 1 | Добавлено логирование причины |

## Notes
- Всегда читать docs/AGENTS.md, docs/RULES.md, docs/project-rules.md, README.md перед работой
- Тестировать через `cargo run --bin test-pipeline --release`
