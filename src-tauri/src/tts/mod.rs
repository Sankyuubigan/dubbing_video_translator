use crate::comm::{PipelineContext, SpeakerSegment};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use tauri::Manager;

const CREATE_NO_WINDOW: u32 = 0x08000000;

// --- Референсы голосового клона ---

/// Выбирает сегмент спикера для клонирования голоса.
/// Приоритет: длительность в диапазоне [3, 10] с ближайшей к 7 с;
/// иначе — самый длинный сегмент (но не короче 1 с).
fn pick_ref_segment(speaker: &str, segments: &[SpeakerSegment]) -> Option<(f64, f64)> {
    let mine: Vec<&SpeakerSegment> = segments
        .iter()
        .filter(|s| s.speaker_id == speaker)
        .collect();
    if mine.is_empty() {
        return None;
    }

    let mut best: Option<(f64, f64, f64)> = None;
    for s in &mine {
        let d = s.end_sec - s.start_sec;
        if (3.0..=10.0).contains(&d) {
            let diff = (d - 7.0).abs();
            if best.map_or(true, |b| diff < b.2) {
                best = Some((s.start_sec, s.end_sec, diff));
            }
        }
    }
    if let Some(b) = best {
        return Some((b.0, b.1));
    }

    let longest = mine
        .iter()
        .max_by(|a, b| {
            let da = a.end_sec - a.start_sec;
            let db = b.end_sec - b.start_sec;
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })?;
    let d = longest.end_sec - longest.start_sec;
    if d >= 1.0 {
        Some((longest.start_sec, longest.end_sec))
    } else {
        None
    }
}

fn safe_speaker(speaker: &str) -> String {
    let sanitized: String = speaker
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "speaker".to_string()
    } else {
        sanitized
    }
}

// --- Время и длительность WAV ---

fn wav_sr_and_duration(path: &Path) -> Result<(u32, f64)> {
    let r = hound::WavReader::open(path).with_context(|| {
        format!("TTS: открытие {}", path.display())
    })?;
    let sr = r.spec().sample_rate;
    let dur = r.duration() as f64 / sr as f64;
    Ok((sr, dur))
}

/// Вырезает сегмент [start_sec, end_sec) из preloaded-сэмплов и пишет mono WAV.
fn write_segment_wav(out: &str, sample_rate: u32, samples: &[i16]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out, spec)
        .with_context(|| format!("TTS: создание {}", out))?;
    for &s in samples {
        writer.write_sample(s).ok();
    }
    writer.finalize().ok();
    Ok(())
}

// --- FFmpeg помощники ---

/// Применяет time-stretch через FFmpeg rubberband-фильтр.
/// rubberband сохраняет pitch (голос не становится бурундуком),
/// поддерживает ratio [0.25, 4.0].
fn time_stretch_wav(
    input_wav: &str,
    output_wav: &str,
    target_duration_sec: f64,
    current_duration_sec: f64,
    ffmpeg_path: &Option<String>,
) -> Result<()> {
    if current_duration_sec <= 0.0 || target_duration_sec <= 0.0 {
        std::fs::copy(input_wav, output_wav).ok();
        return Ok(());
    }
    let ratio = current_duration_sec / target_duration_sec;
    let clamped = ratio.clamp(0.25, 4.0);
    if (clamped - 1.0).abs() < 0.02 {
        std::fs::copy(input_wav, output_wav)
            .context("Ошибка копирования WAV")?;
        return Ok(());
    }
    log::info!(
        "TTS: rubberband ratio={:.2} (gen={:.1}s, target={:.1}s)",
        clamped,
        current_duration_sec,
        target_duration_sec
    );
    let ffmpeg = crate::ffmpeg::resolve(ffmpeg_path);
    let output = Command::new(&ffmpeg)
        .creation_flags(CREATE_NO_WINDOW)
        .arg("-y")
        .arg("-i")
        .arg(input_wav)
        .arg("-filter:a")
        .arg(format!("rubberband=tempo={:.4}", clamped))
        .arg(output_wav)
        .output()
        .context("Ошибка запуска FFmpeg для rubberband")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("FFmpeg rubberband error: {}", stderr);
    }
    Ok(())
}

fn resample_wav(
    input_wav: &str,
    output_wav: &str,
    out_sample_rate: u32,
    ffmpeg_path: &Option<String>,
) -> Result<()> {
    let ffmpeg = crate::ffmpeg::resolve(ffmpeg_path);
    let output = Command::new(&ffmpeg)
        .creation_flags(CREATE_NO_WINDOW)
        .arg("-y")
        .arg("-i")
        .arg(input_wav)
        .arg("-ar")
        .arg(out_sample_rate.to_string())
        .arg("-ac")
        .arg("1")
        .arg(output_wav)
        .output()
        .context("Ошибка запуска FFmpeg для ресемплинга")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("FFmpeg resample error: {}", stderr);
    }
    Ok(())
}

// --- Основной пайплайн озвучки ---

pub fn dub(ctx: PipelineContext) -> Result<PipelineContext> {
    if !ctx.config.enable_dubbing {
        log::info!("TTS: озвучка отключена, пропускаем");
        return Ok(ctx);
    }

    let translated = match ctx.translated_chunks.as_ref() {
        Some(c) => c,
        None => {
            log::warn!("TTS: нет переведённых чанков");
            return Ok(ctx);
        }
    };
    let speaker_segments = match ctx.speaker_segments.as_ref() {
        Some(s) => s,
        None => {
            log::warn!("TTS: нет сегментов спикеров");
            return Ok(ctx);
        }
    };
    let wav_path = match ctx.wav_path.as_ref() {
        Some(p) => p,
        None => {
            log::warn!("TTS: нет WAV файла для определения длительности");
            return Ok(ctx);
        }
    };

    // Инициализация движка CrispASR (cosyvoice3-tts, GPU).
    let app = crate::app_handle()
        .ok_or_else(|| anyhow::anyhow!("TTS: AppHandle не инициализирован (запуск вне tauri?)"))?;
    let state = app.state::<tauri_plugin_speech::PluginState>();
    let tss = tauri_plugin_speech::tts_settings::load(app);
    let models_dir = if tss.models_dir.trim().is_empty() {
        tauri_plugin_speech::download::default_models_dir()
    } else {
        PathBuf::from(tss.models_dir.clone())
    };
    let engine_exe = crate::pick_engine_exe().map_err(|e| anyhow::anyhow!(e))?;
    log::info!("TTS: движок {} / модели {}", engine_exe, models_dir.display());

    // Движок стартует ЛЕНИВО, на каждый спикер отдельно: cosyvoice3 синтезирует
    // WAV-клон ТОЛЬКО из голоса, поданного при старте через `--voice <ref>`
    // (имена, зарегистрированные через /v1/voices, бэкенд не резолвит —
    // «voice not found (have 8)»). При смене спикера ensure() перезапускает движок
    // с новым референсом.

    // Preload исходный WAV (16 кГц mono) для вырезания референсов.
    let reader = hound::WavReader::open(wav_path)
        .with_context(|| format!("TTS: открытие {}", wav_path))?;
    let src_sr = reader.spec().sample_rate;
    let src_samples: Vec<i16> = reader
        .into_samples::<i16>()
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("TTS: чтение сэмплов: {e}"))?;
    let total_duration_sec = src_samples.len() as f64 / src_sr as f64;

    // Холст для финальной озвучки (16 кГц).
    let out_sr: usize = 16000;
    let canvas_len = (total_duration_sec * out_sr as f64).ceil() as usize;
    let mut canvas: Vec<f32> = vec![0.0; canvas_len];

    let tmp_dir = std::env::temp_dir();
    let ffmpeg_path = ctx.config.ffmpeg_path.clone();

    // Карта спикера → путь к стартовому референсу (24кГц).
    let mut voice_map: HashMap<String, PathBuf> = HashMap::new();
    let mut current_startup_voice = String::new();

    for (idx, chunk) in translated.iter().enumerate() {
        let text = chunk.text.trim();
        if text.is_empty() || text == crate::comm::SKIP_MARKER {
            continue;
        }
        let speaker_id = chunk
            .speaker_id
            .as_deref()
            .unwrap_or("Speaker_1");

        let ref24 = match voice_map.get(speaker_id) {
            Some(v) => v.clone(),
            None => {
                // Создаём клон-референс по голосу спикера (16кГц → 24кГц для --voice).
                let seg = pick_ref_segment(speaker_id, speaker_segments)
                    .ok_or_else(|| anyhow::anyhow!("TTS: нет референсного сегмента для {}", speaker_id))?;
                let start = (seg.0 * src_sr as f64) as usize;
                let end = ((seg.1 * src_sr as f64) as usize).min(src_samples.len());
                if start >= end {
                    anyhow::bail!("TTS: пустой референс для {}", speaker_id);
                }
                let safe = safe_speaker(speaker_id);
                let ref16 = tmp_dir.join(format!("dubvidtra_tts_ref_{}.wav", safe));
                write_segment_wav(
                    &ref16.to_string_lossy(),
                    src_sr,
                    &src_samples[start..end],
                )?;
                let ref24 = tmp_dir.join(format!("dubvidtra_tts_ref_{}_24k.wav", safe));
                resample_wav(
                    &ref16.to_string_lossy(),
                    &ref24.to_string_lossy(),
                    24000,
                    &ffmpeg_path,
                )?;

                log::info!(
                    "TTS: спикер {} → стартовый голос {} (референс {:.1}–{:.1}с)",
                    speaker_id,
                    ref24.display(),
                    seg.0,
                    seg.1
                );
                voice_map.insert(speaker_id.to_string(), ref24.clone());
                ref24
            }
        };

        // Перезапуск движка с референсом спикера, если он сменился.
        let startup_str = ref24.to_string_lossy().to_string();
        if startup_str != current_startup_voice {
            crate::block_on(state.tts.ensure(
                app,
                &engine_exe,
                "cosyvoice3-tts",
                &models_dir.to_string_lossy().to_string(),
                "cosyvoice3-tts",
                &startup_str,
            ))
            .map_err(|e| anyhow::anyhow!("TTS: запуск движка: {e}"))?;
            current_startup_voice = startup_str;
        }

        log::info!(
            "TTS: [{}/{}] ({:.1}s–{:.1}s) [{}] {}",
            idx + 1,
            translated.len(),
            chunk.start_sec,
            chunk.end_sec,
            speaker_id,
            text,
        );

        // Cross-lingual clone: EN-референс (startup voice) → RU-синтез.
        let (wav_bytes, _timing) = crate::block_on(state.tts.speak(
            text,
            "",
            "",
            "",
            1.0,
            true,
            "ru",
            "en",
        ))
        .map_err(|e| {
            anyhow::anyhow!("TTS: синтез [{}] '{}': {}", idx + 1, text, e)
        })?;

        // Сырой ответ движка — WAV 24 кГц mono.
        let raw_wav = tmp_dir.join(format!("dubvidtra_tts_raw_{}.wav", idx));
        std::fs::write(&raw_wav, &wav_bytes)
            .with_context(|| format!("TTS: запись raw WAV {}", raw_wav.display()))?;
        let (gen_sr, gen_duration) = wav_sr_and_duration(&raw_wav)?;
        let target_duration = chunk.end_sec - chunk.start_sec;

        // Time-stretch при необходимости.
        let stretched_wav = tmp_dir.join(format!("dubvidtra_tts_stretched_{}.wav", idx));
        if gen_duration > target_duration {
            time_stretch_wav(
                &raw_wav.to_string_lossy(),
                &stretched_wav.to_string_lossy(),
                target_duration,
                gen_duration,
                &ffmpeg_path,
            )?;
        } else {
            std::fs::copy(&raw_wav, &stretched_wav).ok();
        }

        // Ресемплинг до 16 кГц.
        let resampled_wav = tmp_dir.join(format!("dubvidtra_tts_final_{}.wav", idx));
        resample_wav(
            &stretched_wav.to_string_lossy(),
            &resampled_wav.to_string_lossy(),
            out_sr as u32,
            &ffmpeg_path,
        )?;

        // Читаем обработанный WAV.
        let final_samples: Vec<f32> = {
            let r = hound::WavReader::open(&resampled_wav)
                .context("TTS: чтение финального WAV")?;
            r.into_samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect()
        };

        // Вставляем в холст по смещению.
        let offset = (chunk.start_sec * out_sr as f64) as usize;
        for (i, &s) in final_samples.iter().enumerate() {
            let pos = offset + i;
            if pos < canvas.len() {
                canvas[pos] = s;
            }
        }

        log::info!(
            "TTS: [{}] готово (gen={:.1}s @{}k, target={:.1}s)",
            idx + 1,
            gen_duration,
            gen_sr / 1000,
            target_duration
        );

        // Чистим временные файлы чанка.
        std::fs::remove_file(&raw_wav).ok();
        std::fs::remove_file(&stretched_wav).ok();
        std::fs::remove_file(&resampled_wav).ok();
    }

    // Чистим референсные WAV.
    for key in voice_map.keys() {
        let safe = safe_speaker(key);
        std::fs::remove_file(tmp_dir.join(format!("dubvidtra_tts_ref_{}.wav", safe))).ok();
        std::fs::remove_file(tmp_dir.join(format!("dubvidtra_tts_ref_{}_24k.wav", safe))).ok();
    }

    // Останавливаем движок.
    crate::block_on(state.tts.stop());

    // Сохраняем финальную дорожку озвучки.
    let dubbed_path = tmp_dir
        .join("dubvidtra_dubbed.wav")
        .to_string_lossy()
        .to_string();
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: out_sr as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&dubbed_path, spec)
            .context("Ошибка создания финальной WAV дорожки")?;
        for &s in &canvas {
            let clamped = s.clamp(-1.0, 1.0);
            writer
                .write_sample((clamped * 32767.0) as i16)
                .ok();
        }
        writer.finalize().ok();
    }

    log::info!("TTS: дорожка озвучки сохранена: {}", dubbed_path);

    Ok(PipelineContext {
        dubbed_audio_path: Some(dubbed_path),
        ..ctx
    })
}