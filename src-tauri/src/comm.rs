use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSegment {
    pub start_sec: f64,
    pub end_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start_sec: f64,
    pub end_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleChunk {
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
    pub speaker_id: Option<String>,
    pub word_timestamps: Option<Vec<WordTimestamp>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub input_path: String,
    pub output_format: String,
    pub gguf_model_path: Option<String>,
    pub ffmpeg_path: Option<String>,
    pub vad_threshold_db: Option<String>,
    pub sherpa_onnx_dir: Option<String>,
    pub stt_model: Option<String>,
    pub diarization_threshold: Option<f64>,
    pub diarization_num_speakers: Option<i32>,
    pub enable_dubbing: bool,
    pub mix_volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineContext {
    pub config: PipelineConfig,
    pub resolved_input_path: Option<String>,
    pub wav_path: Option<String>,
    pub voice_segments: Option<Vec<TimeSegment>>,
    pub subtitle_chunks: Option<Vec<SubtitleChunk>>,
    pub speaker_segments: Option<Vec<SpeakerSegment>>,
    pub translated_chunks: Option<Vec<SubtitleChunk>>,
    pub dubbed_audio_path: Option<String>,
    pub output_path: Option<String>,
}

impl PipelineContext {
    pub fn new(config: PipelineConfig) -> Self {
        let output_format = if config.output_format.is_empty() {
            let ext = std::path::Path::new(&config.input_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("mp4")
                .to_string();
            ext
        } else {
            config.output_format.clone()
        };

        Self {
            config: PipelineConfig {
                output_format,
                ..config
            },
            resolved_input_path: None,
            wav_path: None,
            voice_segments: None,
            subtitle_chunks: None,
            speaker_segments: None,
            translated_chunks: None,
            dubbed_audio_path: None,
            output_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    pub speaker_id: String,
    pub gender: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProgressUpdate {
    pub stage: String,
    pub percent: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// Находит спикера для чанка субтитров по максимальному перекрытию во времени.
/// Если перекрытия нет — берёт ближайшего по start_sec.
pub fn find_speaker_for_chunk<'a>(chunk: &SubtitleChunk, speakers: &'a [SpeakerSegment]) -> Option<&'a str> {
    if speakers.is_empty() {
        return None;
    }
    // Сначала ищем спикера с максимальным overlap
    let best = speakers
        .iter()
        .filter(|s| s.start_sec < chunk.end_sec && s.end_sec > chunk.start_sec)
        .max_by(|a, b| {
            let a_overlap = a.end_sec.min(chunk.end_sec) - a.start_sec.max(chunk.start_sec);
            let b_overlap = b.end_sec.min(chunk.end_sec) - b.start_sec.max(chunk.start_sec);
            a_overlap.partial_cmp(&b_overlap).unwrap_or(std::cmp::Ordering::Equal)
        });
    if best.is_some() {
        return best.map(|s| s.speaker_id.as_str());
    }
    // Нет перекрытия — берём ближайший по start_sec
    speakers
        .iter()
        .min_by(|a, b| {
            let da = (a.start_sec - chunk.start_sec).abs();
            let db = (b.start_sec - chunk.start_sec).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| s.speaker_id.as_str())
}
