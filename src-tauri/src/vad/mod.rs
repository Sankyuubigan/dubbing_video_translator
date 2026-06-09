use crate::comm::{PipelineContext, TimeSegment};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

const MODELS_SUBDIR: &str = "models\\vad";
const MODEL_FILENAME: &str = "silero_vad.onnx";
const MODEL_URL: &str = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx";

const MIN_SEGMENT_SEC: f64 = 0.3;
const PADDING_SEC: f64 = 0.15;
const MERGE_GAP_SEC: f64 = 0.5;

fn project_root() -> PathBuf {
    if cfg!(debug_assertions) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        return root;
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().unwrap();
        loop {
            if dir.join("test").join("for_test.mp4").exists() || dir.join("src").join("App.tsx").exists() {
                return dir.to_path_buf();
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }
    }
    std::env::current_dir().unwrap_or_default()
}

fn models_dir() -> PathBuf {
    if let Some(cfg_dir) = crate::config::load().vad_model_dir {
        let p = PathBuf::from(cfg_dir);
        if p.is_absolute() {
            return p;
        }
        return project_root().join(&p);
    }
    project_root().join(MODELS_SUBDIR)
}

fn ensure_model() -> Result<PathBuf> {
    let dir = models_dir();
    let model_path = dir.join(MODEL_FILENAME);

    if model_path.exists() {
        log::info!("VAD: модель уже загружена: {}", model_path.display());
        return Ok(model_path);
    }

    log::info!("VAD: скачиваем Silero VAD модель в {}", dir.display());
    std::fs::create_dir_all(&dir).context("Ошибка создания директории для VAD модели")?;

    let result = try_download_powershell(MODEL_URL, &model_path);
    if result.is_ok() {
        return Ok(model_path);
    }
    let err1 = result.unwrap_err();
    log::warn!("VAD: PowerShell download failed: {}", err1);

    let result = try_download_curl(MODEL_URL, &model_path);
    if result.is_ok() {
        return Ok(model_path);
    }
    let err2 = result.unwrap_err();
    log::warn!("VAD: curl download failed: {}", err2);

    anyhow::bail!(
        "Не удалось скачать Silero VAD модель.\n\
         URL: {}\n\
         Путь: {}\n\
         PowerShell: {}\n\
         curl: {}\n\n\
         Скачайте модель вручную и положите в {}",
        MODEL_URL,
        model_path.display(),
        err1,
        err2,
        model_path.display()
    );
}

fn try_download_powershell(url: &str, path: &Path) -> Result<()> {
    let script = format!(
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; \
         $p = Invoke-WebRequest -Uri \"{url}\" -OutFile \"{path}\" -UseBasicParsing -PassThru; \
         if ($p.StatusCode -ne 200) {{ throw \"HTTP $($p.StatusCode)\" }}",
        url = url,
        path = path.to_string_lossy()
    );

    let output = Command::new("powershell")
        .arg("-NoProfile")
        .arg("-Command")
        .arg(&script)
        .output()
        .context("Ошибка запуска PowerShell")?;

    if output.status.success() {
        if path.exists() && std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) > 1000 {
            return Ok(());
        }
        anyhow::bail!("Файл скачан, но слишком мал или отсутствует");
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!("PowerShell: {}", stderr.trim())
}

fn try_download_curl(url: &str, path: &Path) -> Result<()> {
    let which = Command::new("where").arg("curl").output();
    match which {
        Ok(out) if out.status.success() => {}
        _ => anyhow::bail!("curl не найден"),
    }

    let output = Command::new("curl")
        .args(&["-L", "-o", &path.to_string_lossy(), "-f", "--ssl-reqd", url])
        .output()
        .context("Ошибка запуска curl")?;

    if output.status.success() && path.exists() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!("curl: {}", stderr.trim())
}

fn merge_segments(segments: Vec<TimeSegment>, min_duration: f64) -> Vec<TimeSegment> {
    let mut merged: Vec<TimeSegment> = Vec::new();
    for seg in segments {
        if seg.end_sec - seg.start_sec < min_duration {
            continue;
        }
        if let Some(last) = merged.last_mut() {
            if seg.start_sec <= last.end_sec + MERGE_GAP_SEC {
                last.end_sec = last.end_sec.max(seg.end_sec);
                continue;
            }
        }
        merged.push(seg);
    }
    merged
}

pub fn detect(ctx: PipelineContext) -> Result<PipelineContext> {
    let wav_path = match ctx.wav_path.as_ref() {
        Some(p) => p,
        None => anyhow::bail!("Нет WAV файла"),
    };

    let model_path = ensure_model()?;
    log::info!("VAD: загружаем Silero VAD из {}", model_path.display());

    let reader = hound::WavReader::open(wav_path)
        .context("Ошибка открытия WAV для VAD")?;
    let spec = reader.spec();
    let total_duration = reader.duration() as f64 / spec.sample_rate as f64;
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / 32768.0)
        .collect();

    log::info!("VAD: аудио {:.1}s, {} сэмплов, {}Hz",
        total_duration, samples.len(), spec.sample_rate);

    let vad_config = sherpa_onnx::VadModelConfig {
        silero_vad: sherpa_onnx::SileroVadModelConfig {
            model: Some(model_path.to_string_lossy().to_string()),
            threshold: 0.3,
            min_silence_duration: 0.3,
            min_speech_duration: 0.1,
            max_speech_duration: 8.0,
            window_size: 512,
        },
        sample_rate: spec.sample_rate as i32,
        ..Default::default()
    };

    let vad = sherpa_onnx::VoiceActivityDetector::create(&vad_config, 3.0)
        .context("Ошибка создания VoiceActivityDetector — проверьте модель")?;

    // Подаём аудио порциями, чтобы VAD мог выдавать промежуточные сегменты
    let chunk_size = 512;
    for chunk in samples.chunks(chunk_size) {
        vad.accept_waveform(chunk);
    }
    vad.flush();

    let sr = spec.sample_rate as f64;
    let mut raw_segments: Vec<TimeSegment> = Vec::new();
    while let Some(seg) = vad.front() {
        let start_sample = seg.start() as f64;
        let n_samples = seg.n() as f64;
        raw_segments.push(TimeSegment {
            start_sec: ((start_sample / sr) - PADDING_SEC).max(0.0),
            end_sec: (((start_sample + n_samples) / sr) + PADDING_SEC).min(total_duration),
        });
        vad.pop();
    }

    if raw_segments.is_empty() {
        log::warn!("VAD: нет речевых сегментов, весь файл как один сегмент");
        return Ok(PipelineContext {
            voice_segments: Some(vec![TimeSegment {
                start_sec: 0.0,
                end_sec: total_duration,
            }]),
            ..ctx
        });
    }

    let final_segments = merge_segments(raw_segments, MIN_SEGMENT_SEC);

    log::info!("VAD: найдено {} речевых сегментов (Silero VAD)", final_segments.len());
    for (i, s) in final_segments.iter().enumerate() {
        log::info!("VAD: сегмент {}: {:.1}s–{:.1}s ({:.1}s)",
            i, s.start_sec, s.end_sec, s.end_sec - s.start_sec);
    }

    Ok(PipelineContext {
        voice_segments: Some(final_segments),
        ..ctx
    })
}
