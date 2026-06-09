# Architecture DubVidTra2

## ობщий принцип: Микромодульная архитектура

Каждый модуль = изолированная фича бизнес-логики. Модули **не импортят друг друга напрямую**. Взаимодействие — только через единый **Communications Hub** (comm.rs).

`
┌─────────────────────────────────────────────────────┐
│                   GUI (Tauri)                       │
│  drag-drop video → настройки (GGUF путь) → старт    │
└─────────────────┬───────────────────────────────────┘
                  │ вызов команды
┌─────────────────▼───────────────────────────────────┐
│              main.rs / orchestrator                  │
│  Pipeline::run(PipelineContext)                      │
└─────────────────┬───────────────────────────────────┘
                  │ передаёт PipelineContext
                  ▼
┌─────────────────────────────────────────────────────┐
│                  comm.rs (Hub)                       │
│  PipelineContext { input_path, audio, chunks, subs } │
│  enum ModuleMessage { Ready, Progress, Done }        │
└──────┬──────┬──────┬──────┬──────┬──────┬───────────┘
       │      │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼      ▼
     audio    vad    stt   diar  transl output
     ext.                        ize    ate
`

## Модули

### 1. udio_extractor
- **Вход:** путь к видео (input_path)
- **Выход:** WAV 16kHz mono файл
- **Зависимость:** FFmpeg (внешний бинарник)
- **Функция:** extract_audio(path: &str) -> Result<String> — возвращает путь к WAV

### 2. ad (Voice Activity Detection)
- **Вход:** WAV файл
- **Выход:** Vec<TimeSegment> — список фрагментов {start_sec, end_sec} с речью
- **Модель:** Silero VAD v5 (.onnx) через крейт ort
- **Функция:** detect_voice(wav_path: &str) -> Result<Vec<TimeSegment>>

### 3. stt (Speech-to-Text)
- **Вход:** WAV + Vec<TimeSegment>
- **Выход:** Vec<SubtitleChunk> — {start, end, text}
- **Модель:** distil-whisper / whisper.cpp через whisper-rs
- **Функция:** 	ranscribe(wav_path: &str, segments: &[TimeSegment]) -> Result<Vec<SubtitleChunk>>

### 4. diarization (Speaker Identification)
- **Вход:** WAV + Vec<TimeSegment>
- **Выход:** Vec<SpeakerSegment> — {start, end, speaker_id}
- **Модель:** Speaker Embedding ONNX + K-Means (крейт linfa-clustering)
- **Функция:** diarize(wav_path: &str, segments: &[TimeSegment]) -> Result<Vec<SpeakerSegment>>

### 5. 	ranslation
- **Вход:** Vec<SubtitleChunk> + Vec<SpeakerSegment>
- **Выход:** Vec<SubtitleChunk> с переведённым текстом
- **Модель:** Qwen-2.5-7B GGUF (выбирается пользователем в GUI) через llama-cpp-2
- **Функция:** 	ranslate(chunks: Vec<SubtitleChunk>) -> Result<Vec<SubtitleChunk>>
- **Особенность:** модель грузится из пути, указанного в config.toml / GUI

### 6. output
- **Вход:** Vec<SubtitleChunk> + оригинальное видео
- **Выход:** видео с вшитыми субтитрами (софтсаб)
- **Формат:** определяется входным видео (MP4 по умолчанию). Если вход .mp4 → выход .mp4, если .mkv → .mkv и т.д.
- **Функция:** mux(video_path: &str, subs: &[SubtitleChunk], format: &str) -> Result<String>

### 7. config
- **Назначение:** управление настройками (путь к GGUF, модель VAD, выходной формат)
- **Функция:** load() -> Config, save(cfg: Config)
- **Формат:** config.toml в ~/.dubvidtra2/config.toml

## Communications Hub (comm.rs)

`ust
pub struct PipelineContext {
    pub input_path: String,
    pub output_format: String,     // "mp4" | "mkv" | etc., по умолчанию — как входной
    pub gguf_model_path: String,   // путь к GGUF-файлу LLM
    pub wav_path: Option<String>,
    pub voice_segments: Option<Vec<TimeSegment>>,
    pub subtitle_chunks: Option<Vec<SubtitleChunk>>,
    pub speaker_segments: Option<Vec<SpeakerSegment>>,
    pub output_path: Option<String>,
}

pub enum ModuleMessage {
    Progress { stage: String, percent: f32 },
    Error(String),
    Done,
}
`

## Pipeline (оркестратор)

В main.rs или pipeline.rs:

`ust
pub fn run(ctx: PipelineContext) -> Result<PipelineContext> {
    let mut ctx = ctx;
    ctx = audio_extractor::extract(ctx)?;
    ctx = vad::detect(ctx)?;
    ctx = stt::transcribe(ctx)?;
    ctx = diarization::diarize(ctx)?;
    ctx = translation::translate(ctx)?;
    ctx = output::mux(ctx)?;
    Ok(ctx)
}
`

## Порядок загрузки моделей (VRAM safety)

1. Загрузить VAD (.onnx) → прогнать весь WAV → выгрузить (drop(model))
2. Загрузить whisper → прогнать чанки → выгрузить
3. Загрузить speaker embedding (.onnx) → вычислить эмбеддинги → выгрузить
4. Загрузить LLM (GGUF) → перевести порциями по 20-30 строк → выгрузить

Это гарантирует, что в VRAM одновременно находится **только одна модель**.
