use crate::comm::{PipelineContext, SpeakerSegment, TimeSegment};
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};

pub fn diarize(ctx: PipelineContext) -> Result<PipelineContext> {
    let wav_path = match ctx.wav_path.as_deref() {
        Some(p) => p.to_string(),
        None => {
            log::error!("diarization: Нет WAV файла");
            anyhow::bail!("Нет WAV файла");
        }
    };
    let segments: Vec<TimeSegment> = match ctx.voice_segments.as_ref() {
        Some(s) => s.clone(),
        None => {
            log::warn!("diarization: Нет сегментов VAD, создаём один сегмент на всё аудио");
            let duration = get_wav_duration(&wav_path)?;
            vec![TimeSegment { start_sec: 0.0, end_sec: duration }]
        }
    };
    run_diarize(ctx, &segments, &wav_path)
}

fn get_wav_duration(path: &str) -> Result<f64> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| { log::error!("diarization: Ошибка открытия WAV: {:#}", e); e })
        .context("Ошибка открытия WAV")?;
    let spec = reader.spec();
    Ok(reader.duration() as f64 / spec.sample_rate as f64)
}

fn run_diarize(ctx: PipelineContext, segments: &[TimeSegment], wav_path: &str) -> Result<PipelineContext> {

    log::info!("Диаризация: анализируем {} сегментов", segments.len());

    if segments.len() < 2 {
        let speaker_segments: Vec<SpeakerSegment> = segments.iter().map(|s| SpeakerSegment {
            start_sec: s.start_sec, end_sec: s.end_sec, speaker_id: "Speaker_1".to_string(),
        }).collect();
        return Ok(PipelineContext { speaker_segments: Some(speaker_segments), ..ctx });
    }

    let audio = match load_audio(wav_path) {
        Ok(a) => a,
        Err(e) => {
            log::error!("diarization: Ошибка загрузки аудио: {:#}", e);
            return Err(e);
        }
    };
    let features = match extract_mfcc_features(&audio, segments) {
        Ok(f) => f,
        Err(e) => {
            log::error!("diarization: Ошибка извлечения MFCC: {:#}", e);
            return Err(e);
        }
    };

    // Простая агломеративная кластеризация по косинусному расстоянию
    let n = features.len();
    let threshold: f64 = 1.2;
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for i in 0..n {
        let mut best_cluster = None;
        let mut best_dist = threshold;

        for (ci, cluster) in clusters.iter().enumerate() {
            let rep = cluster[0]; // representative
            let dist = cosine_distance(&features[i], &features[rep]);
            if dist < best_dist {
                best_dist = dist;
                best_cluster = Some(ci);
            }
        }

        match best_cluster {
            Some(ci) => clusters[ci].push(i),
            None => clusters.push(vec![i]),
        }
    }

    // Назначаем спикеров
    let mut cluster_to_speaker: HashMap<usize, String> = HashMap::new();
    let mut counter = 0;
    let mut speaker_segments = Vec::new();

    for (i, seg) in segments.iter().enumerate() {
        let ci = clusters.iter().position(|c| c.contains(&i)).unwrap_or(0);
        let speaker_id = cluster_to_speaker.entry(ci).or_insert_with(|| {
            counter += 1;
            format!("Speaker_{}", counter)
        }).clone();

        speaker_segments.push(SpeakerSegment {
            start_sec: seg.start_sec,
            end_sec: seg.end_sec,
            speaker_id,
        });
    }

    let unique: HashSet<_> = speaker_segments.iter().map(|s| &s.speaker_id).collect();
    log::info!("Диаризация: {} спикеров, {} сегментов", unique.len(), speaker_segments.len());
    Ok(PipelineContext { speaker_segments: Some(speaker_segments), ..ctx })
}

fn load_audio(path: &str) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path).context("Ошибка открытия WAV")?;
    Ok(reader.into_samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / 32768.0).collect())
}

fn extract_mfcc_features(audio: &[f32], segments: &[TimeSegment]) -> Result<Vec<Vec<f64>>> {
    let n_mels = 13;
    let n_fft = 256;
    let hop_length = 128;

    let mut features = Vec::new();

    for seg in segments {
        let start = (seg.start_sec * 16000.0) as usize;
        let end = (seg.end_sec * 16000.0).min(audio.len() as f64) as usize;
        if start >= end || end - start < n_fft {
            features.push(vec![0.0; n_mels]);
            continue;
        }

        let seg_audio = &audio[start..end];
        let n_frames = (seg_audio.len() - n_fft) / hop_length;
        if n_frames == 0 {
            features.push(vec![0.0; n_mels]);
            continue;
        }

        let mut mels = vec![0.0_f64; n_mels];
        let max_frames = n_frames.min(100);

        for frame in 0..max_frames {
            let fs = frame * hop_length;
            let half_fft = n_fft / 2;
            let spectrum = vec![0.0_f64; half_fft + 1];

            for k in 0..spectrum.len() {
                let (mut re, mut im) = (0.0_f64, 0.0_f64);
                for n in 0..n_fft {
                    if fs + n < seg_audio.len() {
                        let w = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / n_fft as f64).cos();
                        let angle = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / n_fft as f64;
                        re += seg_audio[fs + n] as f64 * w * angle.cos();
                        im += seg_audio[fs + n] as f64 * w * angle.sin();
                    }
                }
                let mag = (re * re + im * im).sqrt();
                let mk = (k as f64 / spectrum.len() as f64 * n_mels as f64) as usize;
                if mk < n_mels { mels[mk] += mag; }
            }
        }

        for m in 0..n_mels {
            mels[m] = (mels[m] / max_frames as f64 + 1e-10).ln();
        }

        features.push(mels);
    }

    Ok(features)
}

fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum();
    let nb: f64 = b.iter().map(|x| x * x).sum();
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 2.0 } else { 1.0 - dot / denom }
}
