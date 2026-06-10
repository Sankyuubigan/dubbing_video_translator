use crate::comm::{PipelineContext, SpeakerSegment, SubtitleChunk, TimeSegment};
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

/// Разбивает VAD-сегменты по границам спикеров из диаризации.
/// Возвращает (индекс_исходного_VAD, подсегмент, speaker_id).
/// Если диаризация не дала результатов — возвращает оригинальные VAD-сегменты.
fn split_vad_by_speakers<'a>(
    vad_segments: &'a [TimeSegment],
    speaker_segments: Option<&'a Vec<SpeakerSegment>>,
) -> Vec<(usize, TimeSegment, Option<String>)> {
    let speakers = match speaker_segments {
        Some(s) if !s.is_empty() => s,
        _ => {
            return vad_segments.iter().enumerate()
                .map(|(i, s)| (i, s.clone(), None))
                .collect();
        }
    };

    let mut result = Vec::new();
    const MIN_SUB_SEGMENT: f64 = 0.5;

    for (vad_idx, vad) in vad_segments.iter().enumerate() {
        let mut overlapping: Vec<SpeakerSegment> = speakers.iter()
            .filter(|sp| sp.start_sec < vad.end_sec && sp.end_sec > vad.start_sec)
            .cloned()
            .collect();
        overlapping.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());

        // Разрешаем перекрытия: делим пересечение пополам между двумя спикерами
        // Вместо того чтобы обрезать поздний сегмент (что выкидывает его целиком),
        // отдаём каждому спикеру половину overlapping-региона.
        for i in 1..overlapping.len() {
            if overlapping[i].start_sec < overlapping[i - 1].end_sec {
                let overlap_start = overlapping[i].start_sec;
                let overlap_end = overlapping[i - 1].end_sec.min(overlapping[i].end_sec);
                let midpoint = (overlap_start + overlap_end) / 2.0;
                log::debug!(
                    "STT: разрешаем перекрытие сегментов спикеров: {:.1}s–{:.1}s [{}] и {:.1}s–{:.1}s [{}], делим по {:.1}s",
                    overlapping[i - 1].start_sec, overlapping[i - 1].end_sec, overlapping[i - 1].speaker_id,
                    overlapping[i].start_sec, overlapping[i].end_sec, overlapping[i].speaker_id,
                    midpoint
                );
                overlapping[i - 1].end_sec = midpoint;
                overlapping[i].start_sec = midpoint;
            }
        }
        overlapping.retain(|s| s.end_sec - s.start_sec >= 0.01);

        if overlapping.is_empty() {
            result.push((vad_idx, vad.clone(), None));
            continue;
        }

        let mut current = vad.start_sec;

        for sp in &overlapping {
            if sp.start_sec > current + 0.01 {
                let gap_end = sp.start_sec.min(vad.end_sec);
                if gap_end - current >= MIN_SUB_SEGMENT {
                    result.push((vad_idx, TimeSegment {
                        start_sec: current,
                        end_sec: gap_end,
                    }, None));
                }
                current = gap_end;
            }

            let sp_start = current.max(sp.start_sec);
            let sp_end = sp.end_sec.min(vad.end_sec);
            if sp_end > sp_start + 0.01 {
                let dur = sp_end - sp_start;
                if dur >= MIN_SUB_SEGMENT {
                    result.push((vad_idx, TimeSegment {
                        start_sec: sp_start,
                        end_sec: sp_end,
                    }, Some(sp.speaker_id.clone())));
                }
                current = sp_end;
            }
        }

        if vad.end_sec > current + 0.01 {
            let remaining = vad.end_sec - current;
            if remaining >= MIN_SUB_SEGMENT {
                result.push((vad_idx, TimeSegment {
                    start_sec: current,
                    end_sec: vad.end_sec,
                }, None));
            }
        }
    }

    const MAX_CHUNK_LEN: f64 = 6.0;
    let mut merged: Vec<(usize, TimeSegment, Option<String>)> = Vec::new();
    for item in result {
        if let Some(last) = merged.last_mut() {
            let gap = item.1.start_sec - last.1.end_sec;
            let same_speaker = last.2.is_some() && last.2 == item.2;
            let merged_len = item.1.end_sec - last.1.start_sec;
            if same_speaker && gap <= 0.5 && merged_len <= MAX_CHUNK_LEN {
                last.1.end_sec = item.1.end_sec;
                continue;
            }
        }
        merged.push(item);
    }

    // Заполняем gap-сегменты (None speaker) ближайшим спикером
    for item in merged.iter_mut() {
        if item.2.is_some() {
            continue;
        }
        let nearest = speakers.iter()
            .min_by(|a, b| {
                let da = (a.start_sec - item.1.start_sec).abs()
                    .min((a.end_sec - item.1.start_sec).abs());
                let db = (b.start_sec - item.1.start_sec).abs()
                    .min((b.end_sec - item.1.start_sec).abs());
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(ns) = nearest {
            log::debug!(
                "STT: назначаем спикера {} для gap-сегмента {:.1}s–{:.1}s",
                ns.speaker_id, item.1.start_sec, item.1.end_sec
            );
            item.2 = Some(ns.speaker_id.clone());
        }
    }

    merged
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
        "Не указана директория sherpa-onnx модели (sherpa_onnx_dir)"
    ))?;

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

    log::info!("STT: загружаем {} via sherpa-onnx из {}", stt_model, model_dir);

    let t_load = Instant::now();

    let recognizer = if stt_model == "parakeet-tdt" {
        create_parakeet_recognizer(model_dir)?
    } else {
        create_qwen3_recognizer(model_dir)?
    };

    log::info!("STT: модель загружена за {:.1}s", t_load.elapsed().as_secs_f64());

    let wave = sherpa_onnx::Wave::read(wav_path)
        .ok_or_else(|| anyhow::anyhow!("STT: ошибка чтения WAV: {}", wav_path))?;
    let sample_rate = wave.sample_rate();
    let all_samples = wave.samples();
    log::info!("STT: аудио sample_rate={}, len={} samples ({:.1}s)",
        sample_rate, all_samples.len(),
        all_samples.len() as f64 / sample_rate as f64);

    let t_stt = Instant::now();

    let speaker_segments = ctx.speaker_segments.as_ref();
    let sub_segments = split_vad_by_speakers(segments, speaker_segments);
    log::info!("STT: {} VAD-сегментов разбито на {} подсегментов по границам спикеров",
        segments.len(), sub_segments.len());

    let mut streams: Vec<sherpa_onnx::OfflineStream> = Vec::new();
    let mut segment_info: Vec<(usize, f64, f64, Option<String>)> = Vec::new();

    for (idx, seg, speaker_id) in &sub_segments {
        log::debug!("STT: сегмент {}: {:.1}с–{:.1}с{}",
            idx, seg.start_sec, seg.end_sec,
            speaker_id.as_ref().map_or(String::new(), |s| format!(" [{}]", s)));

        let start_sample = (seg.start_sec * sample_rate as f64) as usize;
        let end_sample = (seg.end_sec * sample_rate as f64).min(all_samples.len() as f64) as usize;

        if start_sample >= end_sample || end_sample > all_samples.len() {
            log::warn!("STT: подсегмент {} пустой, пропускаем", idx);
            continue;
        }

        let seg_samples = &all_samples[start_sample..end_sample];
        if seg_samples.is_empty() {
            log::warn!("STT: подсегмент {} пустой, пропускаем", idx);
            continue;
        }

        let stream = recognizer.create_stream();

        if stt_model == "qwen3-asr" {
            stream.set_option("language", "English");
        }

        stream.accept_waveform(sample_rate, seg_samples);
        streams.push(stream);
        segment_info.push((*idx, seg.start_sec, seg.end_sec, speaker_id.clone()));
    }

    // Декодируем пачками, а не всё сразу — иначе GPU OOM на длинных видео
    const BATCH_SIZE: usize = 8;
    if !streams.is_empty() {
        let total = streams.len();
        let n_batches = (total + BATCH_SIZE - 1) / BATCH_SIZE;
        let log_step = (n_batches / 10).max(1);
        for (batch_idx, batch) in streams.chunks(BATCH_SIZE).enumerate() {
            if batch_idx == 0 || batch_idx == n_batches - 1 || batch_idx % log_step == 0 {
                let pct = (batch_idx + 1) * 100 / n_batches;
                log::info!("STT: декодировано {}/{} пачек ({}%)",
                    batch_idx + 1, n_batches, pct);
            }
            let stream_refs: Vec<&sherpa_onnx::OfflineStream> = batch.iter().collect();
            recognizer.decode_multiple_streams(&stream_refs);
        }
    }

    let mut subtitle_chunks: Vec<SubtitleChunk> = Vec::new();
    for (offset, (orig_idx, start_sec, end_sec, speaker_id)) in segment_info.iter().enumerate() {
        let text = match streams[offset].get_result() {
            Some(r) => {
                log::debug!("STT: сырой результат сегмента {}: {:?}", orig_idx, r.text);
                r.text
            }
            None => {
                log::error!("STT: пустой результат сегмента {}", orig_idx);
                continue;
            }
        };

        let clean = if stt_model == "qwen3-asr" {
            parse_qwen3asr(&text)
        } else {
            parse_text(&text)
        };

        if clean.is_empty() {
            log::warn!("STT: сегмент {} пустой (распознавание не дало текста)", orig_idx);
            merge_empty_segment(&mut subtitle_chunks, *start_sec, *end_sec, *orig_idx);
            continue;
        }

        log::debug!("STT: сегмент {} распознан: {}", orig_idx, clean);
        subtitle_chunks.push(SubtitleChunk {
            start_sec: *start_sec,
            end_sec: *end_sec,
            text: clean,
            speaker_id: speaker_id.clone(),
            word_timestamps: None,
        });
    }

    fill_placeholder_gaps(&mut subtitle_chunks);
    subtitle_chunks = split_long_chunks(subtitle_chunks);

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

fn merge_empty_segment(chunks: &mut Vec<SubtitleChunk>, start_sec: f64, end_sec: f64, idx: usize) {
    const MERGE_MAX_GAP: f64 = 2.0;

    if let Some(last) = chunks.last_mut() {
        let gap = start_sec - last.end_sec;
        if gap <= MERGE_MAX_GAP {
            log::debug!(
                "STT: мержим пустой сегмент {} ({:.1}s–{:.1}s) с предыдущим чанком (расширяем до {:.1}s)",
                idx, start_sec, end_sec, end_sec
            );
            last.end_sec = end_sec;
            return;
        }
    }

    chunks.push(SubtitleChunk {
        start_sec,
        end_sec,
        text: String::new(),
        speaker_id: None,
        word_timestamps: None,
    });
    log::debug!(
        "STT: пустой сегмент {} ({:.1}s–{:.1}s) добавлен как плейсхолдер для заполнения",
        idx, start_sec, end_sec
    );
}

fn fill_placeholder_gaps(chunks: &mut Vec<SubtitleChunk>) {
    let mut i = 0;
    while i < chunks.len() {
        if !chunks[i].text.is_empty() || i == 0 {
            i += 1;
            continue;
        }
        if i > 0 {
            chunks[i - 1].end_sec = chunks[i].end_sec;
            log::debug!("STT: заполняем плейсхолдер {:.1}s–{:.1}s — расширяем предыдущий чанк",
                chunks[i].start_sec, chunks[i].end_sec);
        }
        chunks.remove(i);
    }
}

const MAX_TEXT_CHARS: usize = 120;

/// Разбивает длинные субтитры на несколько более коротких по границам предложений.
/// Время распределяется пропорционально длине текста.
fn split_long_chunks(chunks: Vec<SubtitleChunk>) -> Vec<SubtitleChunk> {
    let mut result = Vec::new();
    for chunk in chunks {
        if chunk.text.len() <= MAX_TEXT_CHARS || chunk.text.is_empty() {
            result.push(chunk);
            continue;
        }

        let parts = split_text_for_display(&chunk.text, MAX_TEXT_CHARS);
        let total_duration = chunk.end_sec - chunk.start_sec;
        let total_chars = chunk.text.len() as f64;
        let mut current_start = chunk.start_sec;

        for part in parts {
            let part_chars = part.len() as f64;
            let part_duration = (part_chars / total_chars) * total_duration;
            let current_end = (current_start + part_duration).min(chunk.end_sec);

            log::debug!(
                "STT: разбиваем длинный чанк ({:.1}s–{:.1}s, {} символов) на: {:.1}s–{:.1}s [{}]",
                chunk.start_sec, chunk.end_sec, chunk.text.len(),
                current_start, current_end, part,
            );

            result.push(SubtitleChunk {
                start_sec: current_start,
                end_sec: current_end,
                text: part,
                speaker_id: chunk.speaker_id.clone(),
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
        // Check times are distributed
        assert!(result[0].start_sec < result[0].end_sec);
        assert!(result[1].start_sec >= result[0].end_sec);
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
        assert_eq!(result.len(), 1);
    }
}

/// Парсит вывод Qwen3-ASR.
/// Модель выдаёт формат: `language {lang}<asr_text>{text}</asr_text>`
/// Иногда без тегов, просто `language Dutch.` — фильтруем такое.
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

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Qwen3-ASR иногда выдаёт только языковой тег без <asr_text>
    // например "language Dutch." или " language English"
    if trimmed.to_lowercase().contains("language") {
        log::warn!("STT: Qwen3-ASR не обнаружил речи в аудио (языковой тег: {:?})", trimmed);
        return String::new();
    }

    trimmed.to_string()
}

/// Парсит обычный текстовый вывод (для Parakeet TDT и других моделей).
fn parse_text(text: &str) -> String {
    text.trim().to_string()
}
