use crate::comm::{PipelineContext, SubtitleChunk, TimeSegment};
use anyhow::{Context, Result};
use std::time::Instant;
use tauri::Manager;

/// Вырезает сегмент [start_sec, end_sec) из preloaded-сэмплов и пишет mono WAV.
fn write_segment_wav(out: &str, sample_rate: u32, samples: &[i16]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out, spec)
        .with_context(|| format!("STT: создание {}", out))?;
    for &s in samples {
        writer.write_sample(s).ok();
    }
    writer.finalize().ok();
    Ok(())
}

pub fn transcribe(ctx: PipelineContext) -> Result<PipelineContext> {
    // Быстрый путь: диаризация (CrispASR) уже вернула готовую транскрипцию
    // с таймингами и спикерами — движок повторно не запускаем.
    if let Some(chunks) = ctx.subtitle_chunks.as_ref() {
        if chunks.iter().any(|c| c.speaker_id.is_some()) {
            log::info!(
                "STT: берём готовую транскрипцию от диаризации ({} чанков), CRISPAQR не запускаем",
                chunks.len()
            );
            let final_chunks = split_long_chunks(chunks.clone());
            if final_chunks.is_empty() {
                anyhow::bail!("STT: транскрипция от диаризации пуста");
            }
            let total_s: f64 = final_chunks.iter().map(|c| c.end_sec - c.start_sec).sum();
            log::info!(
                "STT: распознано {} чанков ({} с речи) из готовой транскрипции",
                final_chunks.len(),
                total_s
            );
            return Ok(PipelineContext {
                subtitle_chunks: Some(final_chunks),
                ..ctx
            });
        }
    }

    let wav_path = match ctx.wav_path.as_ref() {
        Some(p) => p,
        None => anyhow::bail!("Нет WAV файла"),
    };

    // ПРОФЕССИОНАЛЬНЫЙ ПАЙПЛАЙН: сегменты диаризации — источник истины.
    // Диаризация (CrispASR) — единственный источник сегментов; без неё пайплайн упал раньше.
    let base_segments: Vec<TimeSegment> = match ctx.speaker_segments.as_ref() {
        Some(speakers) => speakers.iter().map(|s| TimeSegment {
            start_sec: s.start_sec,
            end_sec: s.end_sec,
        }).collect(),
        None => anyhow::bail!("Нет сегментов диаризации"),
    };

    // Дробим сегменты: слишком длинные куски (компрессия контекста) дают
    // пропуски при распознавании. Держим ≤ 15 с.
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

    // Preload WAV один раз (16 кГц mono, PCM s16le — как выдаёт audio_extractor).
    let reader = hound::WavReader::open(wav_path)
        .with_context(|| format!("STT: открытие {}", wav_path))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        anyhow::bail!("STT: ожидался mono WAV, получено {} каналов", spec.channels);
    }
    if spec.sample_rate != 16000 {
        anyhow::bail!("STT: ожидался WAV 16 кГц, получено {}", spec.sample_rate);
    }
    let sample_rate = spec.sample_rate;
    let all_samples: Vec<i16> = reader
        .into_samples::<i16>()
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("STT: чтение сэмплов: {e}"))?;
    log::info!("STT: WAV {} — {} сэмплов ({} с)", wav_path, all_samples.len(), all_samples.len() as f64 / sample_rate as f64);
    let min_segment_samples = (sample_rate as usize) / 4; // < 250 мс — шум, пропускаем

    // Инициализация движка CrispASR (parakeet, GPU).
    let app = crate::app_handle()
        .ok_or_else(|| anyhow::anyhow!("STT: AppHandle не инициализирован (запуск вне tauri?)"))?;
    let state = app.state::<tauri_plugin_speech::PluginState>();
    let engine_exe = crate::pick_engine_exe().map_err(|e| anyhow::anyhow!(e))?;
    let model_path = crate::resolve_stt_model().map_err(anyhow::Error::msg)?;
    let settings = tauri_plugin_speech::SttSettings {
        backend: "parakeet".into(),
        model: model_path,
        engine_exe,
        vad: false, // голосовую активность уже определила диаризация (CrispASR)
        ws_port: 0,
        ..Default::default()
    };
    let port = crate::block_on(state.stt.ensure(app, &settings))
        .map_err(|e| anyhow::anyhow!("STT: запуск движка: {e}"))?;
    log::info!("STT: CrispASR parakeet на порту {port}");

    let t_stt = Instant::now();
    let tmp_seg = crate::paths::temp_file("deedub_stt_segment.wav");
    let tmp_str = tmp_seg.to_string_lossy().to_string();

    // OCR-аналог: no speaker per chunk до назначения ниже.
    let subtitle_chunks = (|| -> Result<Vec<SubtitleChunk>> {
        let mut chunks: Vec<SubtitleChunk> = Vec::new();
        for (i, seg) in segments.iter().enumerate() {
            if crate::is_cancelled() {
                log::info!("STT: отменено, останов после текущей пачки");
                break;
            }
            let start = (seg.start_sec * sample_rate as f64) as usize;
            let end = ((seg.end_sec * sample_rate as f64) as usize).min(all_samples.len());
            if start >= end || end - start < min_segment_samples {
                continue;
            }

            write_segment_wav(&tmp_str, sample_rate, &all_samples[start..end])?;

            let text = match crate::block_on(state.stt.transcribe(&tmp_str, "en")) {
                Ok(t) => t,
                Err(e) if e.contains("пустой текст") => {
                    log::warn!(
                        "STT: сегмент {:.1}–{:.1}с — пустая транскрипция, пропускаем",
                        seg.start_sec,
                        seg.end_sec
                    );
                    String::new()
                }
                Err(e) => return Err(anyhow::anyhow!("STT: распознавание сегмента: {e}")),
            };
            if !text.is_empty() {
                chunks.push(SubtitleChunk {
                    start_sec: seg.start_sec,
                    end_sec: seg.end_sec,
                    text,
                    speaker_id: None,
                    word_timestamps: None,
                });
            }

            if (i + 1) % 10 == 0 || i + 1 == segments.len() {
                log::info!("STT: {}/{} сегментов, распознано {}", i + 1, segments.len(), chunks.len());
            }
        }
        Ok(chunks)
    })();

    let subtitle_chunks = subtitle_chunks?;
    crate::block_on(state.stt.stop(app));
    log::info!("STT: движок остановлен");

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
                speaker_id: chunk.speaker_id.clone(),
                word_timestamps: chunk.word_timestamps.clone(),
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
        assert_eq!(result[0].speaker_id.as_deref(), Some("Speaker_1"));
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
            assert_eq!(chunk.speaker_id.as_deref(), Some("Speaker_1"));
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
