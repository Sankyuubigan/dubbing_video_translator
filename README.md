# DeeDub

**Deep Engineered Dubbing** — an open-source AI-powered video dubbing pipeline utilizing pre-trained deep learning models.

**Deep Engineered Dubbing** — ИИ-пайплайн для дубляжа видео с открытым исходным кодом, использующий предобученные глубокие модели.

Файлы с расширением `.mp4`, аргументами движка и пресетами — локально, без облака. Всё распознавание, перевод и синтез речи выполняются на вашем компьютере.

---

## Русский

### О проекте

Локальный перевод видео с субтитрами и дубляжом. Английская речь распознаётся (ASR + диаризация спикеров), переводится на русский большой языковой моделью, затем может быть озвучена русским синтезом речи с сохранением темпа оригинала.

Пайплайн построен из готовых нейросетей и инженерно собрано в единую цепочку:

```
audio_extractor → diarization → stt → translation → output
```

| Стадия | Что делает |
|--------|------------|
| `audio_extractor` | Извлекает WAV из видео через FFmpeg |
| `diarization` | CrispASR одним проходом: ASR (parakeet) + VAD + диаризация спикеров (возвращает сегменты спикеров и транскрипт) |
| `stt` | Потребляет готовый транскрипт от диаризации (не перезапускает движок) |
| `translation` | Перевод LLM (Gemma-4-12B GGUF через llama-cpp-2) с исправлением STT-ошибок и учётом изохронии |
| `output` | Сжигает субтитры в видео или сводит дубляж через FFmpeg |

### Возможности

- Распознавание английской речи + **диаризация спикеров** (CrispASR: parakeet ASR + VAD)
- Перевод на русский с **исправлением ошибок распознавания** через Chain of Thought Gemma-4-12B
- Субтитры: английские и русские (SRT)
- **Дубляж**: синтез русской речи движком CosyVoice3-TTS-RL, темп подгоняется под оригинал (изохрония)
- Адаптация перевода под пол говорящего, сохранение тона/сленга
- Всё локально: ваши видео не покидают компьютер

### Стек

| Слой | Технология |
|------|------------|
| Backend | Rust + Tauri 2.0, `llama-cpp-2` v0.1.146 (CUDA) |
| Frontend | React + TypeScript (Vite) |
| Медиа | FFmpeg (внешний; извлечение аудио + mux субтитров) |
| Модели | CrispASR (parakeet ASR + VAD + диаризация), Gemma-4-12B GGUF (перевод), CosyVoice3-TTS-RL (дубляж) |

### Требования

- Windows + Visual Studio Build Tools (MSVC x64)
- Rust (stable, x86_64-pc-windows-msvc) и Node.js
- FFmpeg, доступный из командной строки (или указанный в конфиге)
- CUDA-совместимая видеокарта для llama-cpp (опционально ускоряет перевод)
- Модели в `D:\nn\models\...` (пути настраиваются env-переменными и конфигом)

### Сборка и запуск

Сборка/тесты выполняются только через `.bat`-обёртки проекта:

```batch
REM Dev-сборка: prep + tauri build (без бандла) + запуск app
build.bat

REM Unit-тесты
test.bat
```

Пайплайн-тесты:

```batch
REM Полный pipeline-тест с TTS-дубляжом (сохраняет raw/final WAV в temp/)
run_tts_pipeline.bat

REM Полный pipeline-тест без дубляжа
run_pipeline.bat
```

Напрямую (без GUI):

```batch
cd src-tauri
cargo run --bin test-pipeline --release -- --video "path\to\video.mp4" --dub
```

### Конфигурация

Файл: `~/.deedub/config.toml` (создаётся автоматически при первом запуске).

| Поле | Описание |
|------|----------|
| `gguf_model_path` | Путь к GGUF-модели перевода |
| `ffmpeg_path` | Путь к FFmpeg (если не в PATH) |
| `output_format` | Формат результата (`mp4`) |
| `enable_dubbing` | Включает TTS-дубляж |
| `mix_volume` | Громкость дубляжа при микшировании |

Env-переменные (диагностика/разработка):

| Переменная | Назначение |
|------------|------------|
| `DEEDUB_STT_MODEL` | Явный путь к STT-модели |
| `DEEDUB_TTS_PRESET` | TTS-пресет (headless A/B) |
| `DEEDUB_TEST_VIDEO` | Авто-запуск пайплайна на старте приложения |
| `DEEDUB_KEEP_TTS_WAV` | Не удалять raw/final WAV чанков |

### Выходные файлы

- Субтитры в видео: `test/<name>_subbed.mp4`
- Дубляж: `temp/deedub_dubbed.wav`
- Чанки: `temp/deedub_tts_{raw,final}_<N>.wav`
- Английские/русские субтитры: `temp/deedub_subtitles_{en,ru}.srt`
- Лог сессии: `test/last_logs`

### Известные ограничения

- Не все ошибки STT исправляются моделью (например, «sea» → «море» вместо «sight» → «зрелище») — ограничение Gemma-4-12B
- Контекстная интерпретация редких значений может ошибаться (например, «slipped» в значении «соскальзывал с бортика»)

---

## English

### About

DeeDub (**Deep Engineered Dubbing**) is a fully local, open-source video dubbing pipeline that chains pre-trained deep learning models end-to-end: speech recognition, speaker diarization, LLM translation and TTS synthesis run entirely on your machine.

### Pipeline

```
audio_extractor → diarization → stt → translation → output
```

1. **audio_extractor** — extracts WAV from the video via FFmpeg
2. **diarization** — CrispASR one-pass: parakeet ASR + VAD + speaker diarization (returns speaker segments and transcript)
3. **stt** — consumes the ready transcript from diarization (does not re-run the engine)
4. **translation** — LLM translation (Gemma-4-12B GGUF via llama-cpp-2) with STT error correction and isochrony-aware phrasing
5. **output** — burns subtitles into the video or mixes the dub via FFmpeg

### Features

- English ASR + **speaker diarization** (CrispASR)
- English→Russian translation with **STT error correction** via Gemma-4-12B Chain of Thought
- English and Russian subtitles (SRT)
- **Dubbing** with CosyVoice3-TTS-RL, speech rate matched to the original (isochrony)
- Gender-matched verbs/adjectives, tone and register preservation
- 100% offline — your footage never leaves your computer

### Stack

- **Backend:** Rust + Tauri 2.0, `llama-cpp-2` 0.1.146 (CUDA)
- **Frontend:** React + TypeScript (Vite)
- **Media:** FFmpeg (external, audio extraction + subtitle muxing)
- **Models:** CrispASR (parakeet ASR + VAD + diarization), Gemma-4-12B GGUF (translation), CosyVoice3-TTS-RL (dubbing)

### Building & Testing

Use the project `.bat` wrappers:

```batch
build.bat            REM dev build + launch
test.bat             REM unit tests
run_tts_pipeline.bat REM full dubbing pipeline test
run_pipeline.bat     REM full pipeline test (no dubbing)
```

Headless CLI:

```batch
cd src-tauri
cargo run --bin test-pipeline --release -- --video "path\to\video.mp4" --dub
```

### Configuration

File: `~/.deedub/config.toml` (auto-created on first run): `gguf_model_path`, `ffmpeg_path`, `output_format`, `enable_dubbing`, `mix_volume`.

Dev env vars: `DEEDUB_STT_MODEL`, `DEEDUB_TTS_PRESET`, `DEEDUB_TEST_VIDEO`, `DEEDUB_KEEP_TTS_WAV`.

### Output files

- Subbed video: `test/<name>_subbed.mp4`
- Dub: `temp/deedub_dubbed.wav`
- Per-chunk WAVs: `temp/deedub_tts_{raw,final}_<N>.wav`
- SRTs: `temp/deedub_subtitles_{en,ru}.srt`
- Session log: `test/last_logs`

### Known limitations

- Some STT errors are not corrected by the model (e.g. "sea" → "море" instead of "sight" → "зрелище") — a Gemma-4-12B limitation
- Rare contextual meanings can be missed (e.g. "slipped" as sliding off a tube top)