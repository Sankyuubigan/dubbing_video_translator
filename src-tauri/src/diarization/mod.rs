use crate::comm::{PipelineContext, SpeakerSegment, SubtitleChunk};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, Stdio};

fn load_wav_spec(path: &str) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path).context("Ошибка открытия WAV")?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / 32768.0)
        .collect();
    Ok((samples, spec.sample_rate))
}

/// Определение основного тона (F0) через автокорреляцию.
/// Возвращает медианную частоту в Гц или None, если речь не обнаружена.
fn estimate_median_pitch(samples: &[f32], sample_rate: u32) -> Option<f64> {
    let min_freq = 50.0;
    let max_freq = 500.0;
    let min_lag = (sample_rate as f64 / max_freq) as usize;
    let max_lag = (sample_rate as f64 / min_freq) as usize;
    let frame_size = 1024;
    let hop_size = 512;

    let mut pitches: Vec<f64> = Vec::new();
    let mut pos = 0;

    while pos + frame_size <= samples.len() {
        let frame = &samples[pos..pos + frame_size];

        let energy: f32 = frame.iter().map(|&x| x * x).sum();
        if energy < 1e-6 {
            pos += hop_size;
            continue;
        }

        let mut best_lag = 0;
        let mut best_corr = 0.0f32;

        for lag in min_lag..=max_lag.min(frame_size / 2) {
            let mut corr = 0.0f32;
            for i in 0..(frame_size - lag) {
                corr += frame[i] * frame[i + lag];
            }
            let lag_energy: f32 = frame[lag..].iter().map(|&x| x * x).sum();
            let denom = (energy * lag_energy).sqrt();
            if denom > 1e-8 {
                corr /= denom;
            }
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        if best_corr > 0.3 && best_lag > 0 {
            pitches.push(sample_rate as f64 / best_lag as f64);
        }

        pos += hop_size;
    }

    if pitches.is_empty() {
        return None;
    }
    pitches.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(pitches[pitches.len() / 2])
}

/// Определяет пол для каждого уникального спикера на основе высоты тона.
/// Порог: < 160 Гц → male, >= 160 Гц → female.
fn detect_speaker_genders(
    wav_path: &str,
    speaker_segments: &[SpeakerSegment],
) -> HashMap<String, String> {
    let (samples, sample_rate) = match load_wav_spec(wav_path) {
        Ok(v) => v,
        Err(e) => {
            log::error!("GenderDetection: не удалось загрузить WAV: {}", e);
            return HashMap::new();
        }
    };

    let mut speaker_genders: HashMap<String, Vec<f64>> = HashMap::new();

    for seg in speaker_segments {
        let start_sample = (seg.start_sec * sample_rate as f64) as usize;
        let end_sample = (seg.end_sec * sample_rate as f64) as usize;
        if end_sample > samples.len() || start_sample >= end_sample {
            continue;
        }
        let seg_samples = &samples[start_sample..end_sample];
        if let Some(pitch) = estimate_median_pitch(seg_samples, sample_rate) {
            speaker_genders
                .entry(seg.speaker_id.clone())
                .or_default()
                .push(pitch);
        }
    }

    let mut result = HashMap::new();
    for (speaker_id, pitches) in &speaker_genders {
        if pitches.is_empty() {
            continue;
        }
        let mut sorted = pitches.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let gender = if median < 160.0 { "male" } else { "female" };
        log::info!(
            "GenderDetection: {} median pitch={:.0} Hz → {}",
            speaker_id,
            median,
            gender
        );
        result.insert(speaker_id.clone(), gender.to_string());
    }
    result
}

/// ---- Путь диаризации: движок CrispASR (parakeet ASR + диаризация) ----

#[derive(Deserialize, Default)]
#[serde(default)]
struct EngineSeg {
    offsets: Option<Offsets>,
    timestamps: Option<Timestamps>,
    speaker: String,
    text: String,
}

#[derive(Deserialize)]
struct Offsets {
    from: i64,
    to: i64,
}

#[derive(Deserialize)]
struct Timestamps {
    from: String,
    to: String,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct EngineTranscription {
    transcription: Vec<EngineSeg>,
}

/// Парсит "00:00:02,880" (допускаются '.' и "mm:ss") в миллисекунды.
fn ts_to_ms(s: &str) -> Option<i64> {
    let parts: Vec<&str> = s.split(':').collect();
    let (h, m, rest) = match parts.as_slice() {
        [h, m, rest] => (h.parse::<i64>().ok()?, m.parse::<i64>().ok()?, rest),
        [m, rest] => (0i64, m.parse::<i64>().ok()?, rest),
        _ => return None,
    };
    let mut sec_parts = rest.split(|c: char| c == ',' || c == '.');
    let sec: i64 = sec_parts.next()?.parse().ok()?;
    let ms: i64 = sec_parts
        .next()
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);
    Some(h * 3_600_000 + m * 60_000 + sec * 1000 + ms)
}

/// Извлекает номер из "(speaker 0)".
fn parse_speaker_num(s: &str) -> Option<usize> {
    let num: String = s
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num.parse().ok()
}

fn log_tail(path: &Path, prefix: &str) {
    if let Ok(content) = std::fs::read_to_string(path) {
        let lines: Vec<&str> = content.lines().collect();
        let start = lines.len().saturating_sub(20);
        log::warn!("{prefix}");
        for l in &lines[start..] {
            log::warn!("  {l}");
        }
    }
}

/// Разбирает diarized_json движка в (speaker_segments, subtitle_chunks).
fn parse_engine_diarization(
    json_path: &Path,
) -> Option<(Vec<SpeakerSegment>, Vec<SubtitleChunk>)> {
    let content = std::fs::read_to_string(json_path).ok()?;
    let parsed: EngineTranscription = serde_json::from_str(&content).ok()?;
    if parsed.transcription.is_empty() {
        return None;
    }

    // Стабильные Speaker_N в порядке первого появления спикера.
    let mut order: HashMap<usize, usize> = HashMap::new();
    let mut seq = 1usize;
    for seg in &parsed.transcription {
        let key = parse_speaker_num(&seg.speaker).unwrap_or(usize::MAX);
        if !order.contains_key(&key) {
            order.insert(key, seq);
            seq += 1;
        }
    }

    let mut speaker_segments: Vec<SpeakerSegment> = Vec::new();
    let mut subtitle_chunks: Vec<SubtitleChunk> = Vec::new();
    for seg in &parsed.transcription {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        let (from_ms, to_ms) = match &seg.offsets {
            Some(o) => (o.from, o.to),
            None => match &seg.timestamps {
                Some(t) => match (ts_to_ms(&t.from), ts_to_ms(&t.to)) {
                    (Some(a), Some(b)) => (a, b),
                    _ => continue,
                },
                None => continue,
            },
        };
        let start = from_ms as f64 / 1000.0;
        let end = to_ms as f64 / 1000.0;
        if end <= start || end - start < 0.15 {
            continue;
        }
        let key = parse_speaker_num(&seg.speaker).unwrap_or(usize::MAX);
        let num = *order.get(&key).unwrap_or(&1);
        let speaker_id = format!("Speaker_{num}");
        speaker_segments.push(SpeakerSegment {
            start_sec: start,
            end_sec: end,
            speaker_id: speaker_id.clone(),
            gender: None,
        });
        subtitle_chunks.push(SubtitleChunk {
            start_sec: start,
            end_sec: end,
            text: text.to_string(),
            speaker_id: Some(speaker_id),
            word_timestamps: None,
        });
    }

    if speaker_segments.is_empty() {
        return None;
    }
    speaker_segments.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());
    Some((speaker_segments, subtitle_chunks))
}

/// Один прогон CrispASR: parakeet ASR + VAD + диаризация с авто-оценкой
/// числа спикеров (`--diarize-speakers`, сессионная кластеризация TitaNet).
/// Возвращает None, если движок недоступен/не дал сегментов.
fn try_engine_diarization(wav_path: &str) -> Option<(Vec<SpeakerSegment>, Vec<SubtitleChunk>)> {
    if crate::is_cancelled() {
        log::info!("Diarization: отменено пользователем");
        return None;
    }

    let engine_exe = match crate::pick_engine_exe() {
        Ok(p) => p,
        Err(e) => {
            log::warn!("Diarization: CrispASR недоступен: {e}");
            return None;
        }
    };
    let model = match crate::resolve_stt_model() {
        Ok(p) => p,
        Err(e) => {
            log::warn!("Diarization: STT-модель недоступна: {e}");
            return None;
        }
    };

    let prefix = crate::paths::temp_file(&format!("deedub_diar_{}", std::process::id()));
    let json_path = prefix.with_extension("json");
    let stderr_log = prefix.with_extension("log");

    log::info!(
        "Diarization: [CrispASR] запускаем parakeet + диаризацию (--diarize-speakers, авто-число спикеров)..."
    );

    let stderr_file = match std::fs::File::create(&stderr_log) {
        Ok(f) => f,
        Err(e) => {
            log::warn!("Diarization: не удалось открыть лог движка: {e}");
            return None;
        }
    };

    let start = std::time::Instant::now();
    let mut cmd = Command::new(&engine_exe);
    cmd.arg("--backend")
        .arg("parakeet")
        .arg("-m")
        .arg(&model)
        .arg("-f")
        .arg(wav_path)
        .arg("--diarize-speakers")
        .arg("--diarize-embedder")
        .arg("auto")
        .arg("--auto-download")
        .arg("-ojf")
        .arg("-of")
        .arg(&prefix)
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }

    let status = match cmd.status() {
        Ok(s) => s,
        Err(e) => {
            log::warn!("Diarization: не удалось запустить движок: {e}");
            return None;
        }
    };
    log::info!(
        "Diarization: [CrispASR] завершился за {:.1}s (exit={:?})",
        start.elapsed().as_secs_f64(),
        status.code()
    );

    if !status.success() {
        log_tail(&stderr_log, "Diarization: движок завершился с ошибкой, stderr:");
        return None;
    }

    let parsed = parse_engine_diarization(&json_path);
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&stderr_log);
    if parsed.is_none() {
        log::warn!("Diarization: движок завершился успешно, но JSON без сегментов");
    }
    parsed
}

pub fn diarize(ctx: PipelineContext) -> Result<PipelineContext> {
    let wav_path = match ctx.wav_path.as_deref() {
        Some(p) => p.to_string(),
        None => anyhow::bail!("Нет WAV файла"),
    };

    let (mut speaker_segments, subtitle_chunks) = try_engine_diarization(&wav_path).ok_or_else(|| {
        anyhow::anyhow!(
            "CrispASR не дал сегментов диаризации. Проверьте, что движок установлен \
             (TTS-настройки) и STT-модель доступна."
        )
    })?;

    let genders = detect_speaker_genders(&wav_path, &speaker_segments);
    for seg in &mut speaker_segments {
        seg.gender = genders.get(&seg.speaker_id).cloned();
    }
    let unique: std::collections::HashSet<&str> = speaker_segments
        .iter()
        .map(|s| s.speaker_id.as_str())
        .collect();
    log::info!(
        "Diarization: [CrispASR] итого {} спикеров, {} речевых сегментов (транскрипция прилагается)",
        unique.len(),
        speaker_segments.len()
    );

    Ok(PipelineContext {
        speaker_segments: Some(speaker_segments),
        subtitle_chunks: Some(subtitle_chunks),
        ..ctx
    })
}
