use crate::comm::{PipelineContext, SubtitleChunk};
use anyhow::Result;
use std::path::Path;
use std::time::Instant;

fn find_model_file(dir: &str, name: &str) -> Option<std::path::PathBuf> {
    for variant in &[format!("{}.int8.onnx", name), format!("{}.onnx", name)] {
        let p = std::path::Path::new(dir).join(variant);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn resolve_model_dir(cfg_dir: &Option<String>) -> Option<String> {
    if let Some(d) = cfg_dir {
        let p = Path::new(d);
        if p.join("encoder.int8.onnx").exists() || p.join("encoder.onnx").exists() {
            return Some(d.clone());
        }
    }

    let candidates = vec![
        r"D:\nn\models\stt\qwen3-asr-1.7b-sherpa-onnx".to_string(),
        r"D:\nn\models\stt\qwen3-asr-0.6b-sherpa-onnx".to_string(),
    ];

    for c in &candidates {
        let p = Path::new(c);
        if p.join("encoder.int8.onnx").exists() || p.join("encoder.onnx").exists() {
            log::info!("STT: найдена модель по пути {}", c);
            return Some(c.clone());
        }
    }

    cfg_dir.clone()
}

pub fn transcribe(ctx: PipelineContext) -> Result<PipelineContext> {
    let wav_path = match ctx.wav_path.as_ref() {
        Some(p) => p,
        None => anyhow::bail!("Нет WAV файла"),
    };
    let model_dir = resolve_model_dir(&ctx.config.sherpa_onnx_dir);
    let model_dir = model_dir.as_deref().ok_or_else(|| anyhow::anyhow!(
        "Не указана директория sherpa-onnx модели (sherpa_onnx_dir)"
    ))?;

    let conv_frontend = Path::new(model_dir).join("conv_frontend.onnx");
    let encoder = find_model_file(model_dir, "encoder")
        .ok_or_else(|| anyhow::anyhow!("STT: encoder.onnx не найден в {}", model_dir))?;
    let decoder = find_model_file(model_dir, "decoder")
        .ok_or_else(|| anyhow::anyhow!("STT: decoder.onnx не найден в {}", model_dir))?;
    let tokenizer = Path::new(model_dir).join("tokenizer");

    if !tokenizer.is_dir() {
        anyhow::bail!("STT: директория tokenizer не найдена: {}", tokenizer.display());
    }

    let segments = match ctx.voice_segments.as_ref() {
        Some(s) => s,
        None => anyhow::bail!("Нет VAD-сегментов"),
    };

    let wav_path_check = std::path::Path::new(wav_path);
    if !wav_path_check.exists() {
        anyhow::bail!("STT: WAV файл не найден: {}", wav_path);
    }
    let wav_meta = std::fs::metadata(wav_path)
        .map(|m| m.len())
        .unwrap_or(0);
    log::info!("STT: WAV файл {} ({} bytes)", wav_path, wav_meta);

    log::info!("STT: загружаем Qwen3-ASR via sherpa-onnx из {}", model_dir);

    let t_load = Instant::now();

    let asr_cfg = sherpa_onnx::OfflineQwen3ASRModelConfig {
        conv_frontend: Some(conv_frontend.to_string_lossy().to_string()),
        encoder: Some(encoder.to_string_lossy().to_string()),
        decoder: Some(decoder.to_string_lossy().to_string()),
        tokenizer: Some(tokenizer.to_string_lossy().to_string()),
        max_total_len: 1024,
        max_new_tokens: 512,
        temperature: 1e-6,
        top_p: 0.8,
        seed: 42,
        hotwords: None,
    };

    let mut cfg = sherpa_onnx::OfflineRecognizerConfig::default();
    cfg.model_config.qwen3_asr = asr_cfg;
    cfg.model_config.num_threads = 4;
    cfg.model_config.debug = false;
    cfg.model_config.provider = Some("cuda".into());
    cfg.model_config.num_threads = 2;

    let recognizer = sherpa_onnx::OfflineRecognizer::create(&cfg)
        .unwrap_or_else(|| {
            log::info!("STT: CUDA недоступен, пробуем CPU");
            cfg.model_config.provider = Some("cpu".into());
            cfg.model_config.num_threads = 4;
            sherpa_onnx::OfflineRecognizer::create(&cfg)
                .expect("STT: ошибка создания OfflineRecognizer (CPU)")
        });

    log::info!("STT: модель загружена за {:.1}s (provider={})",
        t_load.elapsed().as_secs_f64(),
        cfg.model_config.provider.as_deref().unwrap_or("?"));

    let wave = sherpa_onnx::Wave::read(wav_path)
        .ok_or_else(|| anyhow::anyhow!("STT: ошибка чтения WAV: {}", wav_path))?;
    let sample_rate = wave.sample_rate();
    let all_samples = wave.samples();
    log::info!("STT: аудио sample_rate={}, len={} samples ({:.1}s)",
        sample_rate, all_samples.len(),
        all_samples.len() as f64 / sample_rate as f64);

    let t_stt = Instant::now();

    // Создаём все потоки и передаём аудио
    let mut streams: Vec<sherpa_onnx::OfflineStream> = Vec::new();
    let mut segment_info: Vec<(usize, f64, f64)> = Vec::new(); // (idx, start, end)

    for (idx, seg) in segments.iter().enumerate() {
        log::info!("STT: сегмент {}: {:.1}с–{:.1}с", idx, seg.start_sec, seg.end_sec);

        let start_sample = (seg.start_sec * sample_rate as f64) as usize;
        let end_sample = (seg.end_sec * sample_rate as f64).min(all_samples.len() as f64) as usize;

        if start_sample >= end_sample || end_sample > all_samples.len() {
            log::warn!("STT: сегмент {} пустой, пропускаем", idx);
            continue;
        }

        let seg_samples = &all_samples[start_sample..end_sample];
        if seg_samples.is_empty() {
            log::warn!("STT: сегмент {} пустой, пропускаем", idx);
            continue;
        }

        let stream = recognizer.create_stream();
        stream.accept_waveform(sample_rate, seg_samples);
        streams.push(stream);
        segment_info.push((idx, seg.start_sec, seg.end_sec));
    }

    // Пакетное распознавание всех сегментов (GPU параллелизация)
    if !streams.is_empty() {
        let stream_refs: Vec<&sherpa_onnx::OfflineStream> = streams.iter().collect();
        recognizer.decode_multiple_streams(&stream_refs);
    }

    // Собираем результаты
    let mut subtitle_chunks = Vec::new();
    for (offset, (orig_idx, start_sec, end_sec)) in segment_info.iter().enumerate() {
        let text = match streams[offset].get_result() {
            Some(r) => {
                log::info!("STT: сырой результат сегмента {}: {:?}", orig_idx, r.text);
                r.text
            }
            None => {
                log::error!("STT: пустой результат сегмента {}", orig_idx);
                continue;
            }
        };

        let clean = parse_qwen3asr(&text);
        if clean.is_empty() {
            log::warn!("STT: сегмент {} пустой (распознавание не дало текста)", orig_idx);
            continue;
        }

        log::info!("STT: сегмент {} распознан: {}", orig_idx, clean);
        subtitle_chunks.push(SubtitleChunk {
            start_sec: *start_sec,
            end_sec: *end_sec,
            text: clean,
            speaker_id: None,
        });
    }

    log::info!("STT: распознано {} из {} сегментов за {:.1}s",
        subtitle_chunks.len(), segments.len(), t_stt.elapsed().as_secs_f64());

    if subtitle_chunks.is_empty() {
        log::error!("STT: ни один сегмент не распознан, прерываем pipeline");
        anyhow::bail!("STT: модель не распознала речь ни в одном сегменте. Проверьте аудио или модель.");
    }

    Ok(PipelineContext {
        subtitle_chunks: Some(subtitle_chunks),
        ..ctx
    })
}

fn parse_qwen3asr(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    if let Some(start) = text.find("<asr_text>") {
        let after = &text[start + "<asr_text>".len()..];
        if let Some(end) = after.find("</asr_text>") {
            let content = after[..end].trim().to_string();
            if content.is_empty() {
                log::warn!("STT: Qwen3-ASR не обнаружил речи в аудио");
            }
            return content;
        }
        let trimmed = after.trim().to_string();
        if trimmed.is_empty() {
            log::warn!("STT: Qwen3-ASR не обнаружил речи в аудио");
        }
        return trimmed;
    }
    text.trim().to_string()
}
