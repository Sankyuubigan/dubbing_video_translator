use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub gguf_model_path: Option<String>,
    pub output_format: String,
    pub ffmpeg_path: Option<String>,
    pub vad_threshold_db: Option<String>,
    pub vad_model_dir: Option<String>,
    pub sherpa_onnx_dir: Option<String>,
    pub stt_model: Option<String>,
    pub diarization_models_dir: Option<String>,
    pub diarization_threshold: Option<f64>,
    pub diarization_num_speakers: Option<i32>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            gguf_model_path: None,
            output_format: "mp4".to_string(),
            ffmpeg_path: None,
            vad_threshold_db: None,
            vad_model_dir: None,
            sherpa_onnx_dir: None,
            stt_model: None,
            diarization_models_dir: None,
            diarization_threshold: None,
            diarization_num_speakers: None,
        }
    }
}

pub fn config_path() -> PathBuf {
    let mut path = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    path.push(".dubvidtra2");
    path
}

pub fn config_file() -> PathBuf {
    let mut path = config_path();
    std::fs::create_dir_all(&path).ok();
    path.push("config.toml");
    path
}

pub fn load() -> AppConfig {
    let path = config_file();
    if !path.exists() {
        let cfg = AppConfig::default();
        save(&cfg).ok();
        return cfg;
    }
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    toml::from_str(&content).unwrap_or_default()
}

pub fn save(cfg: &AppConfig) -> Result<()> {
    let path = config_file();
    let content = toml::to_string_pretty(cfg)?;
    std::fs::create_dir_all(config_path())?;
    std::fs::write(&path, content)?;
    Ok(())
}
