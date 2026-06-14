use crate::comm::{PipelineContext, SpeakerSegment};
use anyhow::{Context, Result};
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
    OfflineTtsVitsModelConfig,
};
use std::collections::HashMap;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

const CREATE_NO_WINDOW: u32 = 0x08000000;
const TTS_MODELS_SUBDIR: &str = "models\\tts";
static MODELS: &[(&str, &str)] = &[
    ("ru_RU-dmitri-medium", "male"),
    ("ru_RU-ruslan-medium", "male"),
    ("ru_RU-irina-medium", "female"),
];

fn project_root() -> PathBuf {
    if cfg!(debug_assertions) {
        return Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().unwrap();
        loop {
            if dir.join("test").join("for_test.mp4").exists()
                || dir.join("src").join("App.tsx").exists()
            {
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
    project_root().join(TTS_MODELS_SUBDIR)
}

fn ensure_models_downloaded(models_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(models_dir)
        .context("Ошибка создания директории для TTS моделей")?;

    let espeak_dir = models_dir.join("espeak-ng-data");
    if !espeak_dir.exists() {
        log::info!("TTS: скачиваем espeak-ng-data (общий для всех Piper моделей)...");
        let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2";
        let tar_path = models_dir.join("espeak-ng-data.tar.bz2");
        download_file(url, &tar_path)?;
        extract_tar_bz2(&tar_path, models_dir)?;
        std::fs::remove_file(&tar_path).ok();
    }

    for &(name, _gender) in MODELS {
        let model_dir = models_dir.join(name);
        let onnx = find_onnx_in_dir(&model_dir);
        let tokens = model_dir.join("tokens.txt");
        if onnx.is_some() && tokens.exists() {
            log::info!("TTS: модель {} уже загружена", name);
            continue;
        }
        log::info!("TTS: скачиваем модель {}...", name);
        let url = format!(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-{}.tar.bz2",
            name
        );
        let tar_path = models_dir.join(format!("piper-{}.tar.bz2", name));
        download_file(&url, &tar_path)?;
        extract_tar_bz2(&tar_path, models_dir)?;
        std::fs::remove_file(&tar_path).ok();

        // Архив vits-piper-*.tar.bz2 распаковывается в vits-piper-{name}/,
        // а код ожидает {name}/ — переименовываем при необходимости
        let vits_dir = models_dir.join(format!("vits-piper-{}", name));
        if vits_dir.is_dir() && !model_dir.exists() {
            log::info!("TTS: переименовываем {} → {}", vits_dir.display(), model_dir.display());
            std::fs::rename(&vits_dir, &model_dir)
                .context(format!("Ошибка переименования {} → {}", vits_dir.display(), model_dir.display()))?;
        }
    }

    log::info!("TTS: все модели готовы");
    Ok(())
}

fn find_onnx_in_dir(dir: &Path) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    for entry in std::fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
            return Some(path);
        }
    }
    None
}

fn download_file(url: &str, path: &Path) -> Result<()> {
    crate::download::download_file(url, path)
}

fn extract_tar_bz2(archive: &Path, dest: &Path) -> Result<()> {
    crate::download::extract_tar_bz2(archive, dest)
}

fn assign_voices(
    speaker_segments: &[SpeakerSegment],
) -> HashMap<String, String> {
    let unique_speakers: Vec<&str> = {
        let mut seen: Vec<&str> = speaker_segments
            .iter()
            .map(|s| s.speaker_id.as_str())
            .collect();
        seen.sort();
        seen.dedup();
        seen
    };

    let mut genders: HashMap<&str, &str> = HashMap::new();
    for seg in speaker_segments {
        if let Some(ref g) = seg.gender {
            genders.entry(seg.speaker_id.as_str()).or_insert(g);
        }
    }

    let male_voices: Vec<&str> = MODELS
        .iter()
        .filter(|(_, g)| *g == "male")
        .map(|(n, _)| *n)
        .collect();
    let female_voices: Vec<&str> = MODELS
        .iter()
        .filter(|(_, g)| *g == "female")
        .map(|(n, _)| *n)
        .collect();

    let mut voice_map: HashMap<String, String> = HashMap::new();
    let mut male_idx = 0usize;
    let mut female_idx = 0usize;

    for speaker in &unique_speakers {
        let gender = genders.get(speaker).copied().unwrap_or("male");
        let voice = if gender == "female" {
            let v = female_voices[female_idx % female_voices.len()];
            female_idx += 1;
            v
        } else {
            let v = male_voices[male_idx % male_voices.len()];
            male_idx += 1;
            v
        };
        log::info!("TTS: {} ({}) → {}", speaker, gender, voice);
        voice_map.insert(speaker.to_string(), voice.to_string());
    }
    voice_map
}

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

    let models_dir = models_dir();
    ensure_models_downloaded(&models_dir)?;

    let voice_map = assign_voices(speaker_segments);
    let espeak_dir = models_dir.join("espeak-ng-data");

    // Инициализируем TTS движки для нужных голосов
    let mut tts_cache: HashMap<String, OfflineTts> = HashMap::new();
    for voice_name in voice_map.values() {
        if tts_cache.contains_key(voice_name) {
            continue;
        }
        let model_dir = models_dir.join(voice_name);
        let onnx = find_onnx_in_dir(&model_dir)
            .unwrap_or_else(|| panic!("TTS: не найден .onnx для {}", voice_name));
        let tokens = model_dir.join("tokens.txt");
        let tokens_str = tokens.to_string_lossy().to_string();
        let onnx_str = onnx.to_string_lossy().to_string();

        let config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                vits: OfflineTtsVitsModelConfig {
                    model: Some(onnx_str),
                    tokens: Some(tokens_str),
                    data_dir: Some(espeak_dir.to_string_lossy().to_string()),
                    ..Default::default()
                },
                num_threads: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        match OfflineTts::create(&config) {
            Some(tts) => {
                log::info!("TTS: загружен голос {}", voice_name);
                tts_cache.insert(voice_name.to_string(), tts);
            }
            None => {
                anyhow::bail!("TTS: ошибка загрузки {}: OfflineTts::create вернул None", voice_name);
            }
        }
    }

    // Определяем общую длительность видео из WAV
    let reader = hound::WavReader::open(wav_path)
        .context("Ошибка открытия WAV для определения длительности")?;
    let total_samples = reader.duration() as usize;
    let canvas_sr = reader.spec().sample_rate as usize;
    drop(reader);
    let total_duration_sec = total_samples as f64 / canvas_sr as f64;

    // Холст для финального аудиоозвучки (16000 Гц)
    let out_sr: usize = 16000;
    let canvas_len = (total_duration_sec * out_sr as f64).ceil() as usize;
    let mut canvas: Vec<f32> = vec![0.0; canvas_len];

    let tmp_dir = std::env::temp_dir();
    let ffmpeg_path = ctx.config.ffmpeg_path.clone();

    for (idx, chunk) in translated.iter().enumerate() {
        let text = chunk.text.trim();
        if text.is_empty() {
            continue;
        }
        let speaker_id = chunk
            .speaker_id
            .as_deref()
            .unwrap_or("Speaker_1");
        let voice_name = match voice_map.get(speaker_id) {
            Some(v) => v,
            None => {
                log::warn!("TTS: нет голоса для {}, пропускаем", speaker_id);
                continue;
            }
        };
        let tts = match tts_cache.get(voice_name) {
            Some(t) => t,
            None => continue,
        };

        log::info!(
            "TTS: [{}/{}] ({:.1}s–{:.1}s) [{}] {} -> {}",
            idx + 1,
            translated.len(),
            chunk.start_sec,
            chunk.end_sec,
            speaker_id,
            voice_name,
            text
        );

        let callback: Option<fn(&[f32], f32) -> bool> = None;

        // Первая генерация со скоростью 1.0
        let gen_config = GenerationConfig {
            sid: 0,
            speed: 1.0,
            ..Default::default()
        };
        let audio = match tts.generate_with_config(text, &gen_config, callback) {
            Some(a) => a,
            None => {
                log::warn!("TTS: генерация вернула None для '{}'", text);
                continue;
            }
        };
        let mut audio_data: Vec<f32> = audio.samples().to_vec();
        let mut gen_sr = audio.sample_rate() as usize;
        let mut gen_duration = audio_data.len() as f64 / gen_sr as f64;

        // Если сгенерированное аудио слишком длинное — перегенерируем
        // с повышенной скоростью VITS, чтобы уменьшить time-stretch ratio
        let target_duration = chunk.end_sec - chunk.start_sec;
        if gen_duration > target_duration {
            let needed_speed = gen_duration / target_duration;
            let max_speed = 3.0;
            let adjusted_speed = needed_speed.min(max_speed);
            if adjusted_speed > 1.05 {
                let orig_duration = gen_duration;
                log::info!(
                    "TTS: перегенерация [{}/{}] со speed={:.2} (needed={:.2})",
                    idx + 1, translated.len(), adjusted_speed, needed_speed
                );
                let fast_config = GenerationConfig {
                    sid: 0,
                    speed: adjusted_speed as f32,
                    ..Default::default()
                };
                if let Some(fast_audio) = tts.generate_with_config(text, &fast_config, callback) {
                    audio_data = fast_audio.samples().to_vec();
                    gen_sr = fast_audio.sample_rate() as usize;
                    gen_duration = audio_data.len() as f64 / gen_sr as f64;
                    log::info!(
                        "TTS: после перегенерации длительность={:.1}s (было {:.1}s)",
                        gen_duration, orig_duration
                    );
                }
            }
        }

        // Сохраняем сырой аудио-кусок во временный WAV
        let raw_wav = tmp_dir.join(format!("dubvidtra_tts_raw_{}.wav", idx));
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: gen_sr as u32,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer =
                hound::WavWriter::create(&raw_wav, spec).context("Ошибка создания raw WAV")?;
            for &s in &audio_data {
                let clamped = s.clamp(-1.0, 1.0);
                writer
                    .write_sample((clamped * 32767.0) as i16)
                    .ok();
            }
            writer.finalize().ok();
        }

        // Time-stretch при необходимости
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

        // Ресемплинг до 16 кГц
        let resampled_wav = tmp_dir.join(format!("dubvidtra_tts_final_{}.wav", idx));
        resample_wav(
            &stretched_wav.to_string_lossy(),
            &resampled_wav.to_string_lossy(),
            out_sr as u32,
            &ffmpeg_path,
        )?;

        // Читаем обработанный WAV
        let final_samples: Vec<f32> = {
            let r = hound::WavReader::open(&resampled_wav)
                .context("Ошибка чтения финального WAV")?;
            r.into_samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect()
        };

        // Вставляем в холст по смещению
        let offset = (chunk.start_sec * out_sr as f64) as usize;
        for (i, &s) in final_samples.iter().enumerate() {
            let pos = offset + i;
            if pos < canvas.len() {
                canvas[pos] = s;
            }
        }

        // Чистим временные файлы
        std::fs::remove_file(&raw_wav).ok();
        std::fs::remove_file(&stretched_wav).ok();
        std::fs::remove_file(&resampled_wav).ok();
    }

    // Сохраняем финальную дорожку озвучки
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
