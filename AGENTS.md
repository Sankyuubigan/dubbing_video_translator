# Documentation

## ОБЯЗАТЕЛЬНО: Чтение глобальной документации (core §3.1)

Перед любой работой над этим проектом ОБЯЗАТЕЛЬНО прочитать:

1. `D:\Projects\docusaurus-starter\docs\Sega Mega Note\Моя картотека\software\настройки\global_ai_docs\core\rules.md` — базовые правила (универсальные, для всех проектов)
2. `D:\Projects\docusaurus-starter\docs\Sega Mega Note\Моя картотека\software\настройки\global_ai_docs\desktop_rust_tauri\rules.md` — правила для Rust + Tauri

### Ключевая выжимка, обязательная к соблюдению
- **Git**: любые git-операции (commit/push/add/checkout/...) — ТОЛЬКО с явного письменного разрешения пользователя. Разрешены только `git diff/status/log`.
- **Временные файлы и логи**: ТОЛЬКО в проектных папках `temp/`, `test/` или `target/`. Системный `%TEMP%` и корень репо — запрещены. Лог последней сессии — `test/last_logs.txt` (пишет `tauri-plugin-logs`).
- **Поиск корня бага**: запрещено чинить симптомы/костылями, только root cause. Догадка ≠ причина: проверять кодом/логами/тестами.
- **Поиск в интернете**: только keyless-сервисы (DuckDuckGo/Wikipedia/...). Использование API-ключей (Tavily/Exa/Brave/Bing/SerpAPI) — ЗАПРЕЩЕНО (core §2.6).
- **Крупные задачи**: обязателен план-файл в `tasks/` с именем `ДД.ММ.ГГ <название>.md` и чекбоксами (core §1.8).
- **Сборка/тесты**: НЕ вызывать `cargo`/`npx tauri` напрямую — только через `.bat`-обёртки из папки проекта (desktop §2).

## Stack
- **Backend:** Rust + Tauri 2.0
- **LLM (перевод):** плагин `tauri-plugin-llama-engine` — инференс отдельным процессом `llama-server.exe` (desktop §6.7). Нативного `llama-cpp-2` в проекте НЕТ.
- **Речь (STT/TTS):** плагин `tauri-plugin-speech` (CrispASR)
- **Логи:** плагин `tauri-plugin-logs` (core §2.5.2). Свой логгер в хосте запрещён.
- **Frontend:** React/TypeScript
- **Media:** FFmpeg (external, audio extraction + subtitle muxing)
- **Модели:** Gemma-4-12B GGUF (перевод), CrispASR parakeet (STT), CosyVoice3 (TTS)

## Architecture
Modules are isolated — they don't import each other directly. All communication goes through `comm.rs` (PipelineContext hub).

```
Pipeline order (DON'T CHANGE):
1. audio_extractor — extract WAV from video via FFmpeg
2. diarization — CrispASR one-pass: parakeet ASR + VAD + speaker diarization (returns speaker segments AND transcript)
3. stt — takes the ready transcript from diarization (does NOT re-run the engine)
4. translation — LLM translation (Gemma-4-12B GGUF через плагин `tauri-plugin-llama-engine`)
5. verification — проверка изохронии/мусора, ретраи через тот же движок
6. tts — CrispASR CosyVoice3
7. output — burn subtitles into video via FFmpeg
```

Каждый LLM-этап сам поднимает и роняет движок (`llama-server.exe`): при выходе
из этапа процесс убивается и VRAM освобождается до следующего (desktop §6.5).
`pipeline.rs` о движке ничего не знает.

**No Silero VAD module and no sherpa-onnx fallback diarization** — CrispASR is the single source of truth for ASR+diarization. If CrispASR gives no segments, the pipeline fails with a clear error (no silent fallback).

## Translation Module (`src-tauri/src/translation/mod.rs`)

### Model
Выбирается пользователем в **Настройках → «Локальные модели перевода»**
(`<llama-models-panel>` плагина). Выбор хранится в конфиге плагина
(`%APPDATA%\com.deedub.desktop\app_config.json`, ключ `last_model`) — это
единственный источник правды (core §2.1). Движок ставится и обновляется
плашкой «Движок перевода (LLM)» (`<llama-engine-panel>`), папку можно сменить
там же («Изменить путь»).
- **Типичная модель:** `D:\nn\models\llm\uncen\gemma-4-12B\gemma-4-12B-it-qat-q4_0-unquantized-heretic-ja-v2.i1-IQ4_NL.gguf`
- **Base:** `google/gemma-4` (Gemma4ForCausalLM)
- **Arch:** Gemma 4, 12B dense params (not MoE)
- **Quant:** IQ4_NL, ~6.5 GiB
- **Vocab:** 256000, SPM (SentencePiece — no BPE Cyrillic bug)
- **Context window:** 8192 токенов (`llm.rs: CONTEXT_SIZE`)
- В GGUF есть `tokenizer.chat_template` — движок рендерит промпт сам

### Prompt format
Раньше промпт собирался вручную в теги Gemma-4 (`<|turn>…<|channel>thought`).
**Теперь этого кода нет**: хост отдаёт обычные роли `system` + `user`, а промпт
рендерит сам движок по `tokenizer.chat_template` модели (в GGUF это
«Google Gemma 4 Canonical Chat Template»).

```
system:
You are an expert audiovisual translator adapting English video subtitles into Russian for voiceover.

STRICT RULES:
1. CONTEXT & ASR FIXES: ...
2. ISOCHRONY (TIMING): ...
3. NUMBERS TO WORDS: ... (TTS engine cannot read digits)
4. METRIC: ...
5. GENDER: Current speaker is [SPEAKER_GENDER]. ...
6. NO CENSORSHIP: ...
7. OUTPUT FORMAT: Output ONLY the final spoken Russian text. Do NOT think out loud, do NOT analyze, do NOT offer options or variants. ...

user:
Previous context:
EN: prev text
RU: prev translation

Future context (DO NOT translate yet):
EN: next text

Translate this current text:
EN: source text
```

- Контекст: до 8 предыдущих пар (EN, RU) + до 3 lookahead (только EN)
- Source-side контекст (EN) важнее целевого — модель чинит ASR-ошибки по нему
- **Думатель выключен**: `REASONING_BUDGET = 0` + `disable_reasoning = true`, движок
  получает `chat_template_kwargs {"enable_thinking": false}` и поднимается без
  `--reasoning-*` флагов. Модель отдаёт перевод сразу в `GenerationResult.text`
- `clean_output` прогоняет ответ через пост-обработку: CoT-блоки `<channel|>`,
  обрезка до первой кириллицы, префиксы («Russian:», «Перевод:»), хвост латиницы

### Параметры сэмплинга (`src-tauri/src/llm.rs`)
Нативных сэмплеров больше нет — параметры уходят в `llama-server` как параметры запроса (`ModelParams` плагина). База берётся из самого GGUF (`tokenizer.ggml.*`), сверху накладываются значения проекта:
1. `temperature = 0.6`
2. `top_k = 64`
3. `top_p = 0.95`
4. `min_p = 0.0`
5. `repetition_penalty = 1.05`
6. `presence_penalty = 0.0`, `dry_* = 0`, `xtc_* = 0` (раньше не использовались)

Фиксированного seed больше нет: `llama-server` получает `seed: -1` (случайный).

### Генерация
- Контекст: 8192 токенов (`llm.rs: CONTEXT_SIZE`), передаётся движку при старте
- Бюджет: `n_predict = токены_ответа`, где ответ = `(prompt/2).clamp(64, 512)`
  - Резерв «на размышления» не нужен: думатель выключен (см. Prompt format)
- Замер (ролик 192 с / 38 чанков, RTX 4070 Ti SUPER, полный оффлоад):
  - `translate: 47.7s` — **~1.25 с на чанк**, все запросы по EOS, 0 токенов думателя
  - `verify: 6.8s` (OK=36 RETRY=2 FAIL=0), `tts: 118.1s`, **TOTAL 186.3s**
  - 14 чанков на `tts_60s.mp4`: `translate: 15.1s` (было 348.7s при бюджете 1500)
- **Почему думатель выключен:** при бюджете 1500 модель писала 600-1500 токенов
  рассуждений на каждый чанк (на «Bam. Right now.» — 611 токенов думателя) →
  25-30 с на реплику, 238 чанков ≈ 100 минут. Выигрыша в качестве не было:
  ответ одинаковый, а ошибки распознавания чинит source-side контекст, а не
  длинный внутренний монолог. Старый нативный код тоже «думал», но стоп по
  закрытию `<channel|>` ограничивал весь вывод 64-512 токенами
- Отмена пайплайна передаётся в движок напрямую (`Arc<AtomicBool>` в
  `generate_chat`), так что отмена работает и на середине генерации

## TTS Module (`src-tauri/src/tts/mod.rs`)

### Модель по умолчанию: `cosyvoice3-tts-rl` (RL)
- **Файл LLM:** `D:\nn\models\tts\cosyvoice3-tts-rl\cosyvoice3-llm-rl-q4_k.gguf` (~366 МБ)
- **Прочее (общие GGUF в той же папке):** flow-q8_0 + campplus-f16 + s3tok-f16 + hift-f16 + voices.gguf
- Base-версия `cosyvoice3-tts` (т.е. `cosyvoice3-llm-q4_k.gguf`) УДАЛЕНА с диска и из `speech_models.json` — она глючила (обрезание финальных слогов в cross-lingual режиме). RL-постобучение стабильнее: на тесте 38 чанков обрезаний нет.
- Выбор модели: env `DEEDUB_TTS_PRESET` (приоритет, для headless A/B) → `tts_settings.json` → `preset` → дефолт `cosyvoice3-tts-rl`.
- **Пресеты — ЕДИНЫЙ источник: `tauri-plugin-speech/speech_models.json`.** Плагин подключается через path-зависимость (`Cargo.toml:42`), его `include_str!` компилируется из этой папки при каждой сборке. `src-tauri/speech_models.json` УДАЛЁН (и из `tauri.conf.json` resources тоже) — внешний файл рядом с exe больше не создаётся. Любая правка пресетов делается только в плагине (`D:\Projects\my-tauri-plugins\tauri-plugin-speech\speech_models.json`).

### Retry-костыль и declick (только для base `cosyvoice3-tts`)
- Историческая причина: base-модель в cross-lingual (EN-голос → RU-текст) выбрасывает LM reference-tokens и «обрезает» финальные слоги; RAS-семплер seed-зависим и даёт щелчки.
- Retry-механизм: `RetryTracker` + `TTS_RETRY_SEEDS` + `TTS_FULL_FLOOR=0.62` — повторы с разными seed, пока ДВЕ попытки не сойдутся на полноте ≥ 0.62×ожидаемой длительности (`expected_spoken_secs` по числу гласных, темп 4.4 гласных/сек).
- Деклик: `declick_spikes` — сглаживание 1-сэмпловых выбросов в финальных сэмплах (O(n), без перегенераций). Клики — артефакт base (фикстуры ч.3/ч.27 сняты с её прогонов 17.09 01:58, до установки RL).
- **Оба включаются строго по бэкенду:** `attempt_count = if tts_backend == "cosyvoice3-tts" { TTS_MAX_ATTEMPTS } else { 1 }`, `if tts_backend == "cosyvoice3-tts" { declick_spikes(...) }`.
- Сейчас пресета `cosyvoice3-tts` в JSON нет → оба костыля фактически выключены. Оставлены в коде на будущее: если base вернётся в плагинский `speech_models.json`, retry и declick восстановятся автоматически. RL синтезирует 1 попыткой (seed=None) и без кликов → костыли для неё — zero overhead.

## Build & Release (Tauri Build Toolkit)

Сборка/тесты выполняются ТОЛЬКО через `.bat`-обёртки (desktop §2 — прямой вызов `cargo`/`npx tauri` запрещён).
`build.bat`, `test.bat`, `generate_installer.bat`, `release.bat` — шаблоны общего тулкита
`..\my-tauri-plugins\tauri-build-toolkit` (единый источник правды для всех проектов). Настройки
проекта — `.build-config.json` в корне (repo, appExe, productName, signing, ...).

**ВАЖНО:** команды `build`/`prep`/`installer`/`release` автоматически бампят версию по схеме YY.M.P
в `src-tauri/tauri.conf.json` + `Cargo.toml` при каждом запуске. Dev-сборка (`build`) собирает без
бандла (dev-override), `installer`/`release` включают `bundle.active`.

Read-only диагностика: `node ..\my-tauri-plugins\tauri-build-toolkit\cli.cjs doctor --project "%CD%"`.

### Commands

```batch
REM Dev-сборка: prep (бамп версии + npm install + иконки) + tauri build без бандла + запуск deedub.exe
build.bat

REM Unit-тесты: cargo test [фильтр] (компиляция+прогон; харнесс на этой машине
REM не стартует — 0xc0000139, поэтому есть test_release.bat с release-профилем)
test.bat
test_release.bat

REM Установщик NSIS: prep + tauri build --bundles nsis + верификация установщика и .sig
generate_installer.bat

REM Полный релиз на GitHub: build + sign + gh release + latest.json + commit/push
release.bat

REM Полный pipeline-тест (TTS-дубляж, сохраняет raw/stretched/final WAV в temp/)
run_tts_pipeline.bat

REM Полный pipeline-тест без дубляжа
run_pipeline.bat

REM Только компиляция тест-бинаря
build_test_pipeline.bat

REM Деклик red-тест (бинарь, работает): клики против порога <0.20 FS (костыль base, функция)
build_test_declick.bat

REM RetryTracker-тесты по всем сценариям обрезания (бинарь, работает)
build_test_retry.bat
```

### Batch files
| File | Purpose |
|------|---------|
| `build.bat` | Toolkit: dev build (`prep` + `npx tauri build` без бандла + запуск `deedub.exe`) |
| `test.bat` | Toolkit: `cargo test [фильтр]` (компиляция + прогон unit-тестов) |
| `test_release.bat` | `cargo test --release` — обход падения debug-харнесса на этой машине |
| `generate_installer.bat` | Toolkit: сборка NSIS-установщика + верификация `.sig` |
| `release.bat` | Toolkit: полный релиз (build + sign + `gh release` + `latest.json` + commit/push) |
| `.build-config.json` | Конфиг тулкита (repo, appExe, productName, signing, ...) |
| `run_pipeline.bat` | Full pipeline test (no dubbing) |
| `run_tts_pipeline.bat` | Full pipeline with TTS dubbing, keeps raw/stretched/final WAVs (`DEEDUB_KEEP_TTS_WAV=1`) |
| `build_test_pipeline.bat` | Compile `test-pipeline` binary |
| `build_test_declick.bat` | Build+run `test-tts-declick`: red test for base-костыля click spikes (ch03/ch27 must drop below 0.20 FS, controls ch01/05/14 untouched) |
| `build_test_retry.bat` | Build+run `test-retry-tracker`: all truncation scenarios (ch7/ch27/ch34/ch35/ch38/ch36/monotonic) via real RetryTracker |

Пайплайн-батники (`run_pipeline`, `run_tts_pipeline`, `build_test_pipeline`, `build_test_declick`,
`build_test_retry`, `run_gemma4`) используют тот же MSVC-прелюд, что и шаблоны тулкита
(vswhere → vcvarsall x64 + сброс `RUSTC_WRAPPER`/`CC`/`CXX`/`CARGO_PROFILE_*`), но запускают
проектные `cargo run`/`cargo build` — у тулкита нет команды для произвольного бинаря.
`run_gemma4.bat` — тонкий алиас на `run_pipeline.bat`.

### Output files (проектные папки, core §1.2)
- Video with subtitles: `test/<name>_subbed.mp4` (рядом с исходником)
- Dubbed WAV: `temp/deedub_dubbed.wav`
- Per-chunk raw/stretched/final WAV: `temp/deedub_tts_{raw,stretched,final}_<N>.wav`
- English SRT: `temp/deedub_subtitles_en.srt`
- Russian SRT: `temp/deedub_subtitles_ru.srt`
- Session log: `test/last_logs.txt` (зеркало от `tauri-plugin-logs`), рядом с exe — `deedub.log`

## Config
File: `~/.deedub/config.toml`
Fields: `ffmpeg_path`, `output_format`, `enable_dubbing`, `mix_volume`
Модель перевода здесь НЕ хранится — она в конфиге плагина движка
(`%APPDATA%\com.deedub.desktop\app_config.json`, ключ `last_model`).

## Known Issues
- Occasional STT errors not corrected (e.g., "sea" → "море" instead of "sight" → "зрелище") — 12B model limitation
- "spотыкалась" instead of "соскальзывал" for "slipped" context — model doesn't infer the tube top slip meaning
- CoT occasionally produces "thought: ..." prefix instead of proper `<|channel>thought` format — handled by `clean_output`

## Relevant Source Files
| File | Purpose |
|------|---------|
| `src-tauri/src/translation/mod.rs` | Translation pipeline |
| `src-tauri/src/llm.rs` | Фасад LLM: `LlmSession` поверх `tauri-plugin-llama-engine` (движок, модель, параметры, бюджет) |
| `src-tauri/src/lib.rs` | App init, плагины (logs/downloader/llama-engine), PIPELINE_CANCEL |
| `src-tauri/src/pipeline.rs` | Pipeline orchestrator |
| `src-tauri/src/comm.rs` | PipelineContext, SubtitleChunk |
| `src-tauri/src/config/mod.rs` | Config load/save |
| `src-tauri/src/audio_extractor/mod.rs` | WAV extraction |
| `src-tauri/src/diarization/mod.rs` | CrispASR ASR + speaker diarization (single source) |
| `src-tauri/src/stt/mod.rs` | Consumes diarization transcript |
| `src-tauri/src/output/mod.rs` | Subtitle muxing |
| `src-tauri/src/tts/mod.rs` | TTS дubляж: синтез + retry и declick только для base (`RetryTracker`, `TTS_FULL_FLOOR`, `declick_spikes`) |
| `src-tauri/src/bin/test_retry_tracker.rs` | Автономный тест сценариев обрезания (работает на этой машине) |
| `src-tauri/src/paths.rs` | Единый источник путей (core §1.2/§2.5.1): `temp/`, `test/` |
| `src-tauri/Cargo.toml` | Rust dependencies |

## Research
- `research_tower_plus_9b/` — full research findings on Tower-Plus-9B prompt format, translation prompting, and alternative approaches
- Key finding: source-side context (previous EN text) is more important than target-side context for translation quality
- Key finding: temperature 0.1-0.3 is optimal for factual translation tasks
- Key finding: GBNF grammar not needed for standard instruction-tuned models like Tower-Plus-9B