use crate::comm::{PipelineContext, SubtitleChunk, TimeSegment};
use anyhow::Result;
use std::path::Path;
use std::time::Instant;

fn find_model_file(dir: &str, name: &str) -> Option<std::path::PathBuf> {
    for variant in &[
        format!("{}.int8.onnx", name),
        format!("{}.fp16.onnx", name),
        format!("{}.onnx", name),
    ] {
        let p = std::path::Path::new(dir).join(variant);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn resolve_model_dir(cfg_dir: &Option<String>, stt_model: &str) -> Option<String> {
    if let Some(d) = cfg_dir {
        let p = Path::new(d);
        if stt_model == "parakeet-tdt" {
            if p.join("joiner.int8.onnx").exists() || p.join("joiner.onnx").exists() || p.join("joiner.fp16.onnx").exists() {
                return Some(d.clone());
            }
        } else if p.join("conv_frontend.onnx").exists() {
            return Some(d.clone());
        }
    }

    let candidates = if stt_model == "parakeet-tdt" {
        vec![
            r"D:\nn\models\stt\parakeet-tdt-0.6b-v3-sherpa-onnx-fp16".to_string(),
            r"D:\nn\models\stt\parakeet-tdt-0.6b-v2-sherpa-onnx-int8".to_string(),
        ]
    } else {
        vec![
            r"D:\nn\models\stt\qwen3-asr-1.7b-sherpa-onnx".to_string(),
            r"D:\nn\models\stt\qwen3-asr-0.6b-sherpa-onnx".to_string(),
        ]
    };

    for c in &candidates {
        let p = Path::new(c);
        if stt_model == "parakeet-tdt" {
            if p.join("joiner.int8.onnx").exists() || p.join("joiner.onnx").exists() {
                log::info!("STT: найдена модель по пути {}", c);
                return Some(c.clone());
            }
        } else if p.join("conv_frontend.onnx").exists() {
            log::info!("STT: найдена модель по пути {}", c);
            return Some(c.clone());
        }
    }

    cfg_dir.clone()
}

fn create_qwen3_recognizer(model_dir: &str) -> Result<sherpa_onnx::OfflineRecognizer> {
    let conv_frontend = Path::new(model_dir).join("conv_frontend.onnx");
    if !conv_frontend.exists() {
        anyhow::bail!("STT Qwen3-ASR: conv_frontend.onnx не найден в {}", model_dir);
    }
    let encoder = find_model_file(model_dir, "encoder")
        .ok_or_else(|| anyhow::anyhow!("STT Qwen3-ASR: encoder.onnx не найден в {}", model_dir))?;
    let decoder = find_model_file(model_dir, "decoder")
        .ok_or_else(|| anyhow::anyhow!("STT Qwen3-ASR: decoder.onnx не найден в {}", model_dir))?;
    let tokenizer = Path::new(model_dir).join("tokenizer");
    if !tokenizer.is_dir() {
        anyhow::bail!("STT Qwen3-ASR: директория tokenizer не найдена: {}", tokenizer.display());
    }

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

    Ok(recognizer)
}

fn create_parakeet_recognizer(model_dir: &str) -> Result<sherpa_onnx::OfflineRecognizer> {
    let encoder = find_model_file(model_dir, "encoder")
        .ok_or_else(|| anyhow::anyhow!("STT Parakeet: encoder.onnx не найден в {}", model_dir))?;
    let decoder = find_model_file(model_dir, "decoder")
        .ok_or_else(|| anyhow::anyhow!("STT Parakeet: decoder.onnx не найден в {}", model_dir))?;
    let joiner = find_model_file(model_dir, "joiner")
        .ok_or_else(|| anyhow::anyhow!("STT Parakeet: joiner.onnx не найден в {}", model_dir))?;
    let tokens = Path::new(model_dir).join("tokens.txt");
    if !tokens.exists() {
        anyhow::bail!("STT Parakeet: tokens.txt не найден в {}", model_dir);
    }

    let transducer_cfg = sherpa_onnx::OfflineTransducerModelConfig {
        encoder: Some(encoder.to_string_lossy().to_string()),
        decoder: Some(decoder.to_string_lossy().to_string()),
        joiner: Some(joiner.to_string_lossy().to_string()),
    };

    let mut cfg = sherpa_onnx::OfflineRecognizerConfig::default();
    cfg.model_config.transducer = transducer_cfg;
    cfg.model_config.tokens = Some(tokens.to_string_lossy().to_string());
    cfg.model_config.model_type = Some("nemo_transducer".into());
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

    Ok(recognizer)
}

pub fn transcribe(ctx: PipelineContext) -> Result<PipelineContext> {
    let wav_path = match ctx.wav_path.as_ref() {
        Some(p) => p,
        None => anyhow::bail!("Нет WAV файла"),
    };

    let stt_model = ctx.config.stt_model.as_deref().unwrap_or("qwen3-asr");
    let model_dir = resolve_model_dir(&ctx.config.sherpa_onnx_dir, stt_model);
    let model_dir = model_dir.as_deref().ok_or_else(|| anyhow::anyhow!(
        "Не указана директория sherpa-onnx модели"
    ))?;

    // ПРОФЕССИОНАЛЬНЫЙ ПАЙПЛАЙН: Используем сегменты диаризации как источник истины.
    // Это гарантирует, что в каждом куске аудио только один спикер!
    let base_segments = if let Some(speakers) = ctx.speaker_segments.as_ref() {
        speakers.iter().map(|s| TimeSegment {
            start_sec: s.start_sec,
            end_sec: s.end_sec,
        }).collect()
    } else if let Some(vad_segs) = ctx.voice_segments.as_ref() {
        vad_segs.clone()
    } else {
        anyhow::bail!("Нет ни сегментов диаризации, ни VAD");
    };

    // ИСПРАВЛЕНИЕ ПРОПУСКОВ РЕЧИ:
    // Qwen3-ASR "захлебывается" и выдает пустоту на кусках длиннее 15 секунд.
    // Жестко дробим любые сегменты длиннее 15 секунд.
    let mut segments = Vec::new();
    for seg in base_segments {
        let mut curr_start = seg.start_sec;
        while curr_start < seg.end_sec {
            let curr_end = (curr_start + 15.0).min(seg.end_sec);
            segments.push(TimeSegment {
                start_sec: curr_start,
                end_sec: curr_end,
            });
            curr_start = curr_end;
        }
    }

    let wave = sherpa_onnx::Wave::read(wav_path)
        .ok_or_else(|| anyhow::anyhow!("STT: ошибка чтения WAV: {}", wav_path))?;
    let sample_rate = wave.sample_rate();
    let all_samples = wave.samples();

    let recognizer = if stt_model == "parakeet-tdt" {
        create_parakeet_recognizer(model_dir)?
    } else {
        create_qwen3_recognizer(model_dir)?
    };

    let t_stt = Instant::now();
    let mut subtitle_chunks: Vec<SubtitleChunk> = Vec::new();
    const BATCH_SIZE: usize = 16;

    let n_batches = (segments.len() + BATCH_SIZE - 1) / BATCH_SIZE;
    let log_step = (n_batches / 10).max(1);

    for (batch_idx, batch_segments) in segments.chunks(BATCH_SIZE).enumerate() {
        if batch_idx == 0 || batch_idx == n_batches - 1 || batch_idx % log_step == 0 {
            let pct = (batch_idx + 1) * 100 / n_batches;
            log::info!("STT: обработка {}/{} пачек ({}%)", batch_idx + 1, n_batches, pct);
        }

        let mut streams = Vec::new();
        let mut valid_segments = Vec::new();

        for seg in batch_segments {
            let start_sample = (seg.start_sec * sample_rate as f64) as usize;
            let end_sample = (seg.end_sec * sample_rate as f64).min(all_samples.len() as f64) as usize;

            if start_sample >= end_sample { continue; }
            let seg_samples = &all_samples[start_sample..end_sample];
            if seg_samples.is_empty() { continue; }

            let stream = recognizer.create_stream();
            if stt_model == "qwen3-asr" { stream.set_option("language", "English"); }
            stream.accept_waveform(sample_rate, seg_samples);

            streams.push(stream);
            valid_segments.push(seg);
        }

        if streams.is_empty() { continue; }

        let stream_refs: Vec<&sherpa_onnx::OfflineStream> = streams.iter().collect();
        recognizer.decode_multiple_streams(&stream_refs);

        for (offset, seg) in valid_segments.into_iter().enumerate() {
            if let Some(r) = streams[offset].get_result() {
                let clean = if stt_model == "qwen3-asr" { parse_qwen3asr(&r.text) } else { parse_text(&r.text) };

                if !clean.is_empty() {
                    subtitle_chunks.push(SubtitleChunk {
                        start_sec: seg.start_sec,
                        end_sec: seg.end_sec,
                        text: clean,
                        speaker_id: None,
                        word_timestamps: None,
                    });
                }
            }
        }
    }

    let mut final_chunks = split_long_chunks(subtitle_chunks);

    if let Some(speakers) = ctx.speaker_segments.as_ref() {
        for chunk in final_chunks.iter_mut() {
            if let Some(speaker) = crate::comm::find_speaker_for_chunk(chunk, speakers) {
                chunk.speaker_id = Some(speaker.to_string());
            }
        }
    }

    log::info!("STT: распознано {} чанков за {:.1}s", final_chunks.len(), t_stt.elapsed().as_secs_f64());

    if final_chunks.is_empty() {
        anyhow::bail!("STT: модель не распознала речь ни в одном сегменте.");
    }

    Ok(PipelineContext {
        subtitle_chunks: Some(final_chunks),
        ..ctx
    })
}

fn split_long_chunks(chunks: Vec<SubtitleChunk>) -> Vec<SubtitleChunk> {
    let mut result = Vec::new();
    for chunk in chunks {
        let text = chunk.text.trim();
        if text.is_empty() { continue; }

        let parts = split_text_for_display(text, 80);
        if parts.is_empty() { continue; }

        let total_chars: usize = parts.iter().map(|p| p.chars().count()).sum();
        let total_duration = chunk.end_sec - chunk.start_sec;
        let mut current_start = chunk.start_sec;

        for part in parts {
            let part_duration = if total_chars > 0 {
                (part.chars().count() as f64 / total_chars as f64) * total_duration
            } else {
                total_duration
            };
            let current_end = (current_start + part_duration).min(chunk.end_sec);

            result.push(SubtitleChunk {
                start_sec: current_start,
                end_sec: current_end,
                text: part,
                speaker_id: None,
                word_timestamps: None,
            });
            current_start = current_end;
        }
    }
    result
}

/// Разбивает текст на части для отображения, каждая не длиннее max_chars.
/// Сначала пытается разделить по границам предложений (. ? !),
/// затем группирует предложения, если одно предложение слишком длинное — режет по словам.
fn split_text_for_display(text: &str, max_chars: usize) -> Vec<String> {
    let sentences = split_sentences(text);

    let mut result: Vec<String> = Vec::new();
    let mut current = String::new();

    for sentence in &sentences {
        if sentence.len() > max_chars {
            if !current.is_empty() {
                result.push(current.clone());
                current.clear();
            }
            // Длинное предложение — режем по словам
            let mut line = String::new();
            for word in sentence.split_whitespace() {
                if !line.is_empty() && line.len() + word.len() + 1 > max_chars {
                    result.push(line.clone());
                    line.clear();
                }
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(word);
            }
            if !line.is_empty() {
                current = line;
            }
        } else if current.is_empty() {
            current = sentence.clone();
        } else if current.len() + 1 + sentence.len() <= max_chars {
            current.push(' ');
            current.push_str(sentence);
        } else {
            result.push(current.clone());
            current = sentence.clone();
        }
    }

    if !current.is_empty() {
        result.push(current);
    }

    result
}

/// Разбивает текст на предложения по . ? ! (с пробелом после или конец строки).
/// Пытается не резать на распространённых сокращениях (Mr., Dr., Ms., Mrs., etc.).
fn split_sentences(text: &str) -> Vec<String> {
    let abbrevs = ["mr.", "dr.", "ms.", "mrs.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "dept.", "inc.", "ltd.", "co.", "corp.", "gen.", "sgt.", "capt.", "col.", "maj.", "gov.", "rep.", "sen.", "ave.", "blvd.", "pl.", "sq.", "mt.", "ft.", "approx.", "decr.", "incr.", "temp.", "est."];

    let mut result = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        current.push(chars[i]);

        if matches!(chars[i], '.' | '?' | '!') {
            let end_of_sentence = if i + 1 >= len {
                true // конец строки
            } else if chars[i + 1] == ' ' {
                // Не разбивать на сокращениях: проверить слово перед точкой
                if chars[i] == '.' {
                    let word_start = current[..current.len() - 1]
                        .rfind(|c: char| !c.is_alphabetic())
                        .map(|p| p + 1)
                        .unwrap_or(0);
                    let word: String = current[word_start..current.len() - 1]
                        .chars()
                        .filter(|c| c.is_alphabetic())
                        .collect();
                    !abbrevs.contains(&word.to_lowercase().as_str())
                } else {
                    true // ? и ! всегда конец предложения
                }
            } else if chars[i + 1].is_uppercase() {
                // Заглавная буква без пробела (например "Hello!World")
                true
            } else {
                false
            };

            if end_of_sentence {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    result.push(trimmed);
                }
                current.clear();
                if i + 1 < len && chars[i + 1] == ' ' {
                    i += 1; // пропускаем пробел
                }
            }
        }

        i += 1;
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        result.push(trimmed);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_basic() {
        let text = "Hello world. How are you? I am fine! Yes.";
        let sentences = split_sentences(text);
        assert_eq!(sentences, vec![
            "Hello world.",
            "How are you?",
            "I am fine!",
            "Yes.",
        ]);
    }

    #[test]
    fn test_split_sentences_no_punct() {
        let text = "Hello world how are you";
        let sentences = split_sentences(text);
        assert_eq!(sentences, vec!["Hello world how are you"]);
    }

    #[test]
    fn test_split_sentences_abbrev() {
        let text = "Dr. Smith is here. He is a doctor.";
        let sentences = split_sentences(text);
        assert_eq!(sentences, vec!["Dr. Smith is here.", "He is a doctor."]);
    }

    #[test]
    fn test_split_sentences_empty() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   ").is_empty());
    }

    #[test]
    fn test_split_text_for_display_short() {
        let text = "Hello world.";
        let parts = split_text_for_display(text, 65);
        assert_eq!(parts, vec!["Hello world."]);
    }

    #[test]
    fn test_split_text_for_display_long() {
        let text = "Hello world. This is a test of the subtitle splitting functionality. It should work correctly.";
        let parts = split_text_for_display(text, 50);
        assert!(parts.len() >= 2);
        assert!(parts.iter().all(|p| p.len() <= 50));
    }

    #[test]
    fn test_split_text_for_display_single_long_sentence() {
        let text = "Jesus Christ that is a fantastic C right there thank you those are absolutely amazing";
        let parts = split_text_for_display(text, 65);
        assert!(parts.len() >= 2);
        assert!(parts.iter().all(|p| p.len() <= 65));
    }

    #[test]
    fn test_split_text_for_display_empty() {
        assert!(split_text_for_display("", 65).is_empty());
        assert!(split_text_for_display("   ", 65).is_empty());
    }

    #[test]
    fn test_split_long_chunks_short() {
        let chunks = vec![SubtitleChunk {
            start_sec: 0.0,
            end_sec: 2.0,
            text: "Hello world.".to_string(),
            speaker_id: Some("Speaker_1".to_string()),
            word_timestamps: None,
        }];
        let result = split_long_chunks(chunks);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "Hello world.");
        assert!(result[0].speaker_id.is_none());
    }

    #[test]
    fn test_split_long_chunks_long() {
        let chunks = vec![SubtitleChunk {
            start_sec: 0.0,
            end_sec: 10.0,
            text: "Hello world. How are you? I am fine. This is a test of the long subtitle splitting feature."
                .to_string(),
            speaker_id: Some("Speaker_1".to_string()),
            word_timestamps: None,
        }];
        let result = split_long_chunks(chunks);
        assert!(result.len() >= 2);
        assert!(result[0].start_sec < result[0].end_sec);
        assert!(result[1].start_sec >= result[0].end_sec);
        for chunk in &result {
            assert!(chunk.speaker_id.is_none());
        }
    }

    #[test]
    fn test_split_long_chunks_empty_text() {
        let chunks = vec![SubtitleChunk {
            start_sec: 0.0,
            end_sec: 2.0,
            text: String::new(),
            speaker_id: None,
            word_timestamps: None,
        }];
        let result = split_long_chunks(chunks);
        assert!(result.is_empty());
    }
}

fn parse_qwen3asr(text: &str) -> String {
    if text.is_empty() { return String::new(); }
    if let Some(start) = text.find("<asr_text>") {
        let after = &text[start + "<asr_text>".len()..];
        if let Some(end) = after.find("</asr_text>") {
            return after[..end].trim().to_string();
        }
        return after.trim().to_string();
    }

    let mut trimmed = text.trim();
    let lower = trimmed.to_lowercase();

    // Спасаем текст: если модель забыла теги, но написала "language English [текст]"
    if lower.starts_with("language ") {
        if let Some(space_idx) = trimmed[9..].find(' ') {
            trimmed = trimmed[9 + space_idx..].trim();
            trimmed = trimmed.trim_start_matches(|c: char| !c.is_alphabetic());
        } else {
            return String::new(); // Это просто тег языка без самого текста
        }
    }

    trimmed.to_string()
}

/// Парсит обычный текстовый вывод (для Parakeet TDT и других моделей).
fn parse_text(text: &str) -> String {
    text.trim().to_string()
}
