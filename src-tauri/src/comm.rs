use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSegment {
    pub start_sec: f64,
    pub end_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleChunk {
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
    pub speaker_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub input_path: String,
    pub output_format: String,
    pub gguf_model_path: Option<String>,
    pub ffmpeg_path: Option<String>,
    pub vad_threshold_db: Option<String>,
    pub sherpa_onnx_dir: Option<String>,
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
            output_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    pub speaker_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProgressUpdate {
    pub stage: String,
    pub percent: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_path: Option<String>,
}
