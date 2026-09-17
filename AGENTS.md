# Documentation

## ОБЯЗАТЕЛЬНО: Чтение глобальной документации (core §3.1)

Перед любой работой над этим проектом ОБЯЗАТЕЛЬНО прочитать:

1. `D:\Projects\docusaurus-starter\docs\Sega Mega Note\Моя картотека\software\настройки\global_ai_docs\core\rules.md` — базовые правила (универсальные, для всех проектов)
2. `D:\Projects\docusaurus-starter\docs\Sega Mega Note\Моя картотека\software\настройки\global_ai_docs\desktop_rust_tauri\rules.md` — правила для Rust + Tauri

### Ключевая выжимка, обязательная к соблюдению
- **Git**: любые git-операции (commit/push/add/checkout/...) — ТОЛЬКО с явного письменного разрешения пользователя. Разрешены только `git diff/status/log`.
- **Временные файлы и логи**: ТОЛЬКО в проектных папках `temp/`, `test/` или `target/`. Системный `%TEMP%` и корень репо — запрещены. Лог последней сессии — `test/last_logs` (с меткой `[ГГГГ-ММ-ДД ЧЧ:ММ:СС]Z` в каждой строке).
- **Поиск корня бага**: запрещено чинить симптомы/костылями, только root cause. Догадка ≠ причина: проверять кодом/логами/тестами.
- **Поиск в интернете**: только keyless-сервисы (DuckDuckGo/Wikipedia/...). Использование API-ключей (Tavily/Exa/Brave/Bing/SerpAPI) — ЗАПРЕЩЕНО (core §2.6).
- **Крупные задачи**: обязателен план-файл в `tasks/` с именем `ДД.ММ.ГГ <название>.md` и чекбоксами (core §1.8).
- **Сборка/тесты**: НЕ вызывать `cargo`/`npx tauri` напрямую — только через `.bat`-обёртки из папки проекта (desktop §2).

## Stack
- **Backend:** Rust + Tauri 2.0, `llama-cpp-2` v0.1.146 (CUDA)
- **Frontend:** React/TypeScript
- **Media:** FFmpeg (external, audio extraction + subtitle muxing)
- **Models:** CrispASR (parakeet ASR + VAD + diarization), Gemma-4-12B (translation)

## Architecture
Modules are isolated — they don't import each other directly. All communication goes through `comm.rs` (PipelineContext hub).

```
Pipeline order (DON'T CHANGE):
1. audio_extractor — extract WAV from video via FFmpeg
2. diarization — CrispASR one-pass: parakeet ASR + VAD + speaker diarization (returns speaker segments AND transcript)
3. stt — takes the ready transcript from diarization (does NOT re-run the engine)
4. translation — LLM translation (Gemma-4-12B GGUF via llama-cpp-2)
5. output — burn subtitles into video via FFmpeg
```

**No Silero VAD module and no sherpa-onnx fallback diarization** — CrispASR is the single source of truth for ASR+diarization. If CrispASR gives no segments, the pipeline fails with a clear error (no silent fallback).

## Translation Module (`src-tauri/src/translation/mod.rs`)

### Model
- **File:** `D:\nn\models\llm\uncen\gemma-4-12B\gemma-4-12B-it-heretic-QAT-UD-Q4_K_XL.gguf`
- **Base:** `google/gemma-4` (Gemma4ForCausalLM)
- **Arch:** Gemma 4, 12B dense params (not MoE)
- **Quant:** Q4_K_XL, ~7.2 GiB
- **Vocab:** 256000, SPM (SentencePiece — no BPE Cyrillic bug)
- **BOS token:** 2 — **prepended** (`AddBos::Always`, BOS needed before `<|turn>` tokens)
- **EOS token:** 3 (model default)
- **Context window:** 8192 tokens

### Prompt format
**Gemma 4 turn format** with **Chain of Thought** (uses `<|turn>`/`<turn|>` control tokens + `<|channel>thought` for internal reasoning):

```
<|turn>system
You are a top-tier audiovisual translator and dubbing adapter. Translate English to Russian.

STRICT RULES:
1. ASR FIX: The source text has Speech-to-Text errors. Fix them contextually.
2. DUBBING LENGTH (ISOCHRONY): Russian text MUST be exactly as short as the English text so it fits the audio timing. Drop filler words, use short synonyms. CONCISE IS KING.
3. METRIC: Convert imperial units to metric (e.g., 'Five seven' -> '170 см', '32C' -> '32C').
4. GENDER: Current speaker is [SPEAKER_GENDER]. Match Russian verbs/adjectives to this gender.
5. TONE: Preserve slang, cursing, or formality.

PROCESS:
First, use <|channel>thought to analyze context, fix ASR errors, and compress the text length.
Then, output ONLY the final Russian translation in the normal channel.<turn|>
<|turn>user

Previous context:
EN: prev text
RU: prev translation

Future context (DO NOT translate yet):
EN: next text

Translate this current text:
EN: source text<turn|>
<|turn>model
<|channel>thought
```

- BOS token is prepended automatically via `AddBos::Always`
- Context: up to 3 previous (EN, RU) pairs in order (most recent first), plus up to 3 lookahead (EN only)
- Source-side context (EN) is critical — research shows it contributes more than target-side
- Model first generates Chain of Thought (analysis, ASR fix, draft), then `<channel|>`, then final Russian answer
- Generation stops at EOS token 3 or `<turn|>`
- `clean_output` extracts text after `<channel|>` (if present), then strips `<|...>` garbage tokens, known prefixes, and trailing artifacts

### Sampling chain (in order)
1. `LlamaSampler::penalties(512, 1.05, 0.0, 0.0)` — repetition penalty
2. `LlamaSampler::top_k(64)` — top-K filtering
3. `LlamaSampler::top_p(0.95, 1)` — nucleus sampling
4. `LlamaSampler::temp(1.0)` — temperature scaling (Gemma 4 standard)
5. `LlamaSampler::dist(42)` — random selection with seed

### Generation
- `max_new = (prompt_tokens / 2).max(64).min(512)`
- KV cache cleared per chunk: `ctx.clear_kv_cache()` + `sampler.reset()`
- Context window: 8192 tokens
- EOS check (token 3) in generation loop as safety stop
- `clean_output` pipeline: extract after `<channel|>` (CoT) → strip before first Cyrillic → strip from `<` → strip "thought" suffix → strip known prefixes → strip trailing non-Cyrillic after last Cyrillic char → keep only last line if multiline (multiple candidates)

### Key differences from Tower-Plus-9B
- **`AddBos::Always`** — BOS needed before `<|turn>` tokens (same as Tower-Plus-9B)
- **temp=1.0** instead of 0.15 — official Gemma 4 recommendation
- **top_k=64, top_p=0.95** — wider sampling for better translation diversity
- **Chain of Thought** — `<|channel>thought` for internal reasoning before final answer
- **12B params** — better STT error correction and context understanding

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

REM Unit-тесты: cargo test [фильтр] (компиляция+прогон; харнесс на этой машине часто падает на CUDA DLL)
test.bat

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
- Session log: `test/last_logs`

## Config
File: `~/.deedub/config.toml`
Fields: `gguf_model_path`, `ffmpeg_path`, `output_format`, `enable_dubbing`, `mix_volume`

## Known Issues
- Occasional STT errors not corrected (e.g., "sea" → "море" instead of "sight" → "зрелище") — 12B model limitation
- "spотыкалась" instead of "соскальзывал" for "slipped" context — model doesn't infer the tube top slip meaning
- CoT occasionally produces "thought: ..." prefix instead of proper `<|channel>thought` format — handled by `clean_output`

## Relevant Source Files
| File | Purpose |
|------|---------|
| `src-tauri/src/translation/mod.rs` | Translation pipeline |
| `src-tauri/src/lib.rs` | App init, PIPELINE_CANCEL, LlamaBackend |
| `src-tauri/src/pipeline.rs` | Pipeline orchestrator |
| `src-tauri/src/comm.rs` | PipelineContext, SubtitleChunk |
| `src-tauri/src/config/mod.rs` | Config load/save |
| `src-tauri/src/audio_extractor/mod.rs` | WAV extraction |
| `src-tauri/src/diarization/mod.rs` | CrispASR ASR + speaker diarization (single source) |
| `src-tauri/src/stt/mod.rs` | Consumes diarization transcript |
| `src-tauri/src/output/mod.rs` | Subtitle muxing |
| `src-tauri/src/tts/mod.rs` | TTS дubляж: синтез + retry и declick только для base (`RetryTracker`, `TTS_FULL_FLOOR`, `declick_spikes`) |
| `src-tauri/src/bin/test_retry_tracker.rs` | Автономный тест сценариев обрезания (работает на этой машине) |
| `src-tauri/src/paths.rs` | Единый источник путей (core §1.2/§2.5.1): `temp/`, `test/`, `test/last_logs` |
| `src-tauri/Cargo.toml` | Rust dependencies |

## Research
- `research_tower_plus_9b/` — full research findings on Tower-Plus-9B prompt format, translation prompting, and alternative approaches
- Key finding: source-side context (previous EN text) is more important than target-side context for translation quality
- Key finding: temperature 0.1-0.3 is optimal for factual translation tasks
- Key finding: GBNF grammar not needed for standard instruction-tuned models like Tower-Plus-9B