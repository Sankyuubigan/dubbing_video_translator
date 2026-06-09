use crate::comm::{PipelineContext, TimeSegment};
use anyhow::{Context, Result};
use std::os::windows::process::CommandExt;
use std::process::Command;

const CREATE_NO_WINDOW: u32 = 0x08000000;

const DEFAULT_THRESHOLD: &str = "-40";
const MIN_SEGMENT_SEC: f64 = 0.5;
const MAX_SEGMENT_SEC: f64 = 8.0;

pub fn detect(ctx: PipelineContext) -> Result<PipelineContext> {
    let wav_path = match ctx.wav_path.as_ref() {
        Some(p) => p,
        None => {
            log::error!("VAD: Нет WAV файла");
            anyhow::bail!("Нет WAV файла");
        }
    };

    let threshold = ctx
        .config
        .vad_threshold_db
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_THRESHOLD);

    log::info!("VAD: ffmpeg silencedetect threshold={}dB", threshold);

    let ffmpeg = crate::ffmpeg::resolve(&ctx.config.ffmpeg_path);
    let output = Command::new(&ffmpeg)
        .creation_flags(CREATE_NO_WINDOW)
        .arg("-i")
        .arg(wav_path)
        .arg("-af")
        .arg(format!("silencedetect=noise={}dB:d=0.5", threshold))
        .arg("-f")
        .arg("null")
        .arg("NUL")
        .output()
        .context("FFmpeg не найден")?;

    let stderr = String::from_utf8_lossy(&output.stderr);

    let total_duration = get_wav_duration(wav_path)?;
    let mut segments = parse_speech_segments(&stderr, total_duration);

    segments.retain(|s| s.end_sec - s.start_sec >= MIN_SEGMENT_SEC);

    let mut final_segments = Vec::new();
    for seg in segments {
        let dur = seg.end_sec - seg.start_sec;
        if dur > MAX_SEGMENT_SEC {
            let mut t = seg.start_sec;
            while t < seg.end_sec {
                let end = (t + MAX_SEGMENT_SEC).min(seg.end_sec);
                final_segments.push(TimeSegment {
                    start_sec: t,
                    end_sec: end,
                });
                t = end;
            }
        } else {
            final_segments.push(seg);
        }
    }

    if final_segments.is_empty() {
        log::warn!("VAD: нет речевых сегментов, весь файл как один сегмент");
        final_segments.push(TimeSegment {
            start_sec: 0.0,
            end_sec: total_duration,
        });
    }

    log::info!(
        "VAD: найдено {} речевых сегментов",
        final_segments.len()
    );
    Ok(PipelineContext {
        voice_segments: Some(final_segments),
        ..ctx
    })
}

fn parse_speech_segments(stderr: &str, total_duration: f64) -> Vec<TimeSegment> {
    let mut segments = Vec::new();
    let mut speech_start: f64 = 0.0;
    let mut in_silence = false;

    for line in stderr.lines() {
        let line = line.trim();

        if let Some(pos) = line.find("silence_start: ") {
            let val = line[pos + "silence_start: ".len()..].trim();
            if let Ok(t) = val.parse::<f64>() {
                if t > speech_start {
                    segments.push(TimeSegment {
                        start_sec: speech_start,
                        end_sec: t,
                    });
                }
                in_silence = true;
            }
        }

        if let Some(pos) = line.find("silence_end: ") {
            let rest = &line[pos + "silence_end: ".len()..];
            if let Some(pipe_pos) = rest.find('|') {
                let val = rest[..pipe_pos].trim();
                if let Ok(t) = val.parse::<f64>() {
                    speech_start = t;
                }
            }
            in_silence = false;
        }
    }

    if !in_silence && total_duration > speech_start {
        segments.push(TimeSegment {
            start_sec: speech_start,
            end_sec: total_duration,
        });
    }

    for s in segments.iter_mut() {
        let overlap = 0.1;
        s.start_sec = (s.start_sec - overlap).max(0.0);
        s.end_sec = (s.end_sec + overlap).min(total_duration);
    }

    segments
}

fn get_wav_duration(path: &str) -> Result<f64> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| {
            log::error!("VAD: Ошибка открытия WAV: {:#}", e);
            e
        })
        .context("Ошибка открытия WAV")?;
    let spec = reader.spec();
    Ok(reader.duration() as f64 / spec.sample_rate as f64)
}
