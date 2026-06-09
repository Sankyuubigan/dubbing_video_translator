# Architecture DubVidTra2

## Общий принцип: Микромодульная архитектура

Каждый модуль = изолированная фича бизнес-логики. Модули **не импортят друг друга напрямую**. Взаимодействие — только через единый **Communications Hub** (comm.rs).

```
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
└──────┬──────┬──────┬──────┬──────┬──────┬───────────┘
       │      │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼      ▼
     audio   vad   diariz   stt   transl  output
     ext.              ️    ate    ate
```

## Модули

### 1. audio_extractor
- **Вход:** путь к видео (input_path)
- **Выход:** WAV 16kHz mono файл
- **Зависимость:** FFmpeg (внешний бинарник)
- **Функция:** extract(ctx: PipelineContext) -> Result<PipelineContext>

### 2. vad (Voice Activity Detection)
- **Вход:** WAV файл
- **Выход:** Vec<TimeSegment> — список фрагментов {start_sec, end_sec} с речью
- **Модель:** Silero VAD v5 (.onnx) через sherpa-onnx (VoiceActivityDetector)
- **Функция:** detect(ctx: PipelineContext) -> Result<PipelineContext>
- **Загрузка модели:** автоматическое скачивание silero_vad.onnx в models/vad/ при первом запуске
- **Параметры:** threshold=0.5, window_size=1536 (для 16kHz), padding=0.15с по краям

### 3. diarization (Speaker Identification)
- **Вход:** WAV
- **Выход:** Vec<SpeakerSegment> — {start, end, speaker_id}
- **Модель:** PyAnnote segmentation + NeMo TitaNet embedding (ONNX) через sherpa-onnx
- **Функция:** diarize(ctx: PipelineContext) -> Result<PipelineContext>
- **Параметры:**
  - `diarization_threshold`: 0.95 (по умолчанию). Порог кластеризации speaker embedding'ов.
    - Выше → меньше кластеров (агрессивнее объединение)
    - Ниже → больше кластеров (больше спикеров)
  - `min_duration_on: 0.5` — мин. длительность речевого сегмента
  - `min_duration_off: 0.3` — мин. пауза между репликами для смены спикера
- **Пост-обработка (postprocess_segments):**
  1. Удаление сегментов короче 0.5с
  2. Объединение соседних сегментов одного спикера (разрыв < 1.0с)
  3. Минорные спикеры (<5% эфирного времени) переприсваиваются ближайшему мажорному
  4. **Гарантия:** если до слияния было 2+ спикеров, после слияния останется минимум 2
  5. Перенумерация спикеров (Speaker_1, Speaker_2, ...)

### 4. stt (Speech-to-Text) — спикер-аварное распознавание
- **Вход:** WAV + Vec<TimeSegment> (VAD) + Vec<SpeakerSegment> (диаризация)
- **Выход:** Vec<SubtitleChunk> — {start, end, text, speaker_id, word_timestamps}
- **Модель:** Qwen3-ASR / Parakeet-TDT через sherpa-onnx (ONNX)
- **Функция:** transcribe(ctx: PipelineContext) -> Result<PipelineContext>
- **Ограничение:** Qwen3-ASR не поддерживает word-level тайминги (word_timestamps = None)
- **Макс. длина чанка:** 6 секунд (константа MAX_CHUNK_LEN)
- **Макс. длина текста:** 65 символов (константа MAX_TEXT_CHARS). Длинные чанки автоматически разбиваются `split_long_chunks()`

#### Как работает спикер-аварное распознавание (split_vad_by_speakers)

```
VAD-сегменты:        │███████████████████████████████████████│
                          25.1s                        36.2s

Speaker-сегменты:    │███████████████████│███████│████████████│
                     │  Speaker_1        │Spk_2  │ Speaker_1  │
                     25.1s         32.4s 33.8    33.9    36.1s

Результат (субтитры): │███████████████████│███████│████████████│
                     │ [Speaker_1] фраза │[Spk_2]│ [Speaker_1]│
```

Каждый VAD-сегмент разрезается по границам спикеров из диаризации. После нарезки:
- Соседние подсегменты **одного спикера** сливаются, если разрыв ≤ 0.5с и суммарная длина ≤ 6с
- **Пустые** подсегменты (нет текста от STT) расширяют предыдущий чанк
- **Перекрывающиеся** сегменты спикеров обрезаются (более поздний сдвигается к концу более раннего)

#### Разбивка длинных субтитров (split_long_chunks)

После распознавания текста, перед переводом, все чанки длиннее **65 символов** разбиваются на несколько:

```
До:  [0.0s-17.0s] [Speaker_1] 'Cause you are oh, nineteen. So yeah. Unfortunately they're all twenty one and over. Unless you went to one of the eighteen and over strip clubs...

После:
[0.0s-4.25s]  [Speaker_1] 'Cause you are oh, nineteen. So yeah.
[4.25s-8.5s]  [Speaker_1] Unfortunately they're all twenty one and over.
[8.5s-12.75s] [Speaker_1] Unless you went to one of the eighteen and over strip clubs...
```

Алгоритм:
1. Текст разбивается на предложения (по `. `, `? `, `! ` с учётом сокращений Mr., Dr., etc.)
2. Предложения группируются, каждая группа ≤ MAX_TEXT_CHARS
3. Если предложение длиннее лимита — режется по словам
4. Время между частями распределяется **пропорционально длине текста**

### 5. translation
- **Вход:** Vec<SubtitleChunk>
- **Выход:** Vec<SubtitleChunk> с переведённым текстом
- **Модель:** Qwen-2.5-7B GGUF (выбирается пользователем в GUI) через llama-cpp-2
- **Функция:** translate(ctx: PipelineContext) -> Result<PipelineContext>
- **Формат общения с LLM:** тегированная построчная генерация (не JSON!)
  ```
  Input:
  [0] English text line 1
  [1] English text line 2
  Output:
  [0] Привет мир
  [1] Это тест
  ```
- **Парсер:** регулярное выражение `\[(\d+)\]\s*(.*)`
- **Ограничение генерации:** max_new_tokens = input_tokens * 2 (макс. 4096)
- **Fallback:** если LLM пропустила индекс — оставляется оригинальный текст

### 6. output
- **Вход:** Vec<SubtitleChunk> + оригинальное видео
- **Выход:** видео с вшитыми субтитрами (софтсаб) + отдельные SRT файлы
- **Формат:** определяется входным видео (MP4 по умолчанию)
- **Функция:** mux(video_path: &str, subs: &[SubtitleChunk], format: &str) -> Result<String>
- **Формат SRT:** `[Speaker_N] текст` — для каждого чанка (если speaker_id = None, пишется `[Speaker]`)

### 7. config
- **Назначение:** управление настройками (путь к GGUF, модель VAD, выходной формат)
- **Функция:** load() -> Config, save(cfg: Config)
- **Формат:** config.toml в ~/.dubvidtra2/config.toml

## Communications Hub (comm.rs)

```rust
pub struct PipelineContext {
    pub input_path: String,
    pub output_format: String,
    pub gguf_model_path: String,
    pub wav_path: Option<String>,
    pub voice_segments: Option<Vec<TimeSegment>>,
    pub subtitle_chunks: Option<Vec<SubtitleChunk>>,
    pub speaker_segments: Option<Vec<SpeakerSegment>>,
    pub output_path: Option<String>,
}

pub struct SubtitleChunk {
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
    pub speaker_id: Option<String>,
    pub word_timestamps: Option<Vec<WordTimestamp>>,
}
```

## Pipeline (оркестратор)

```rust
pub fn run(ctx: PipelineContext) -> Result<PipelineContext> {
    ctx = audio_extractor::extract(ctx)?;     // 1. Извлечение WAV
    ctx = vad::detect(ctx)?;                  // 2. VAD — речевые сегменты
    ctx = diarization::diarize(ctx)?;         // 3. Диаризация — кто говорит
    ctx = stt::transcribe(ctx)?;              // 4. STT — что говорит (с учётом спикеров)
    ctx = translation::translate(ctx)?;       // 5. Перевод
    ctx = output::mux(ctx)?;                  // 6. Вшивание субтитров
    Ok(ctx)
}
```

**Порядок ВАЖЕН:** Диаризация ДО STT, чтобы STT мог нарезать аудио по границам спикеров.

## Порядок загрузки моделей (VRAM safety)

1. Загрузить VAD (Silero .onnx через sherpa-onnx) → прогнать весь WAV → выгрузить
2. Загрузить speaker embedding (.onnx) → вычислить эмбеддинги → выгрузить
3. Загрузить Qwen3-ASR (.onnx через sherpa-onnx) → прогнать чанки → выгрузить
4. Загрузить LLM (GGUF через llama-cpp-2) → перевести порциями → выгрузить

Это гарантирует, что в VRAM одновременно находится **только одна модель**.
