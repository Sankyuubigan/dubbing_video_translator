use crate::comm::PipelineContext;
use anyhow::{Context, Result};
use std::os::windows::process::CommandExt;
use std::process::Command;

const CREATE_NO_WINDOW: u32 = 0x08000000;

fn resolve_input_path(path: &str) -> String {
    let p = std::path::Path::new(path);
    if p.exists() {
        return path.to_string();
    }
    // fallback: ищем в подпапке test/
    if let Some(parent) = p.parent() {
        let stem = p.file_name().unwrap_or_default();
        let candidate = parent.join("test").join(stem);
        if candidate.exists() {
            log::info!("audio_extractor: файл не найден по пути {}, используем {}", path, candidate.display());
            return candidate.to_string_lossy().to_string();
        }
    }
    path.to_string()
}

pub fn extract(ctx: PipelineContext) -> Result<PipelineContext> {
    let input = resolve_input_path(&ctx.config.input_path);
    let wav_path = std::env::temp_dir().join("dubvidtra_audio.wav");
    let wav_str = wav_path.to_string_lossy().to_string();

    log::info!("Извлекаем аудио из: {}", input);

    let ffmpeg = crate::ffmpeg::resolve(&ctx.config.ffmpeg_path);
    let output = Command::new(&ffmpeg)
        .creation_flags(CREATE_NO_WINDOW)
        .arg("-i")
        .arg(&input)
        .arg("-vn")
        .arg("-acodec")
        .arg("pcm_s16le")
        .arg("-ar")
        .arg("16000")
        .arg("-ac")
        .arg("1")
        .arg("-y")
        .arg(&wav_str)
        .output()
        .map_err(|e| {
            log::error!("audio_extractor: FFmpeg не найден: {:#}", e);
            e
        })
        .context("FFmpeg не найден. Установите FFmpeg и добавьте в PATH.")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::error!("audio_extractor: FFmpeg error: {}", stderr);
        anyhow::bail!("FFmpeg error: {}", stderr);
    }

    log::info!("Аудио извлечено: {}", wav_str);

    Ok(PipelineContext {
        resolved_input_path: Some(input),
        wav_path: Some(wav_str),
        ..ctx
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::{PipelineConfig, PipelineContext};

    fn project_root() -> std::path::PathBuf {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.parent().unwrap().to_path_buf()
    }

    #[test]
    fn test_extract_audio() {
        let video_path = project_root().join("test").join("for_test.mp4");
        assert!(video_path.exists(), "Тестовый файл не найден: {:?}", video_path);
        let cfg = PipelineConfig {
            input_path: video_path.to_string_lossy().to_string(),
            output_format: "mp4".to_string(),
            gguf_model_path: None,
            ffmpeg_path: None,
            vad_threshold_db: None,
            sherpa_onnx_dir: None,
        };
        let ctx = PipelineContext::new(cfg);
        let result = extract(ctx);
        assert!(result.is_ok(), "Ошибка извлечения аудио: {:?}", result.err());
        let ctx = result.unwrap();
        assert!(ctx.wav_path.is_some());
        assert!(std::path::Path::new(ctx.wav_path.as_ref().unwrap()).exists());
    }
}
