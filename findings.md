# Findings & Decisions

## Requirements
- Исправить баги в логе: translation parsing fallback, TTS download fail
- BATCH_SIZE=20 (сознательно увеличено)

## Research Findings

### Перевод: GBNF grammar с batch=20
- AGENTS.md: pipeline работал с batch=5, 100% перевод 26/26 чанков
- Сейчас batch=20 → grammar на 20/6 items, parsing падает
- Возможные причины:
  1. max_new = (input_token_count * 2).max(128).min(4096) — не хватает токенов для 20 предметов
  2. GBNF grammar с 20 правилами слишком большая — баг в llama.cpp grammar engine
  3. Модель генерирует EOS раньше времени при большой грамматике
  4. `generate()` возвращает ok=false → output пустой

### TTS: Piper model URL
- Старый URL: `piper-{name}.tar.bz2` → 404
- Новый URL: `vits-piper-{name}.tar.bz2` → существует
- Вопрос: структура архива внутри — с vits-префиксом в директории или без?
- espeak-ng-data: Connection reset — сетевая проблема (GitHub блокируется?)

### Download механизм
- Три fallback: PowerShell → curl → bitsadmin
- PowerShell: Invoke-WebRequest с TLS 1.2
- curl: with --ssl-reqd
- bitsadmin: Windows built-in

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| BATCH_SIZE=20 | Пользователь подтвердил, intentional |
| detect_repeated_translations threshold: 4 | 3 слишком агрессивно для коротких фраз |
| extract_ru_texts two-pass | strict (ru/cyrillic) → fallback (всё остальное) |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| sccache build error | Build env issue, cargo check проходит |
| piper model 404 | Исправлен URL на vits-piper- |
| espeak-ng-data connection reset | Сетевая проблема GitHub |

## Resources
- project-rules.md: архитектурные правила, pipeline порядок
- AGENTS.md: история решений по переводу, batch_size, GBNF
- README.md: команды тестирования
- RULES.md: test output только в test/
