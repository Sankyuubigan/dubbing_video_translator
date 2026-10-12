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
/// Определяет пол для каждого уникального спикера на основе высоты тона.
///
/// Тон меряется общим модулем `comm::voice`, а не кодом здесь: пороги среза,
/// гейта и приёмки подобраны абляцией на фикстурах (`test/fixtures/gender/`)
/// и обязаны быть одними и теми же у определения пола и у выбора
/// клон-референса в TTS. Копия этого кода в двух местах разъехалась бы при
/// первой же правке порога — ровно так и вышло: старый локальный эстиматор
/// без ФВЧ и без гейта читал голос Мелоди (208 Гц по гармоникам) как 130 Гц
/// и помечал всех спикеров мужскими.
///
/// Медиана считается по **всем** кадрам всех реплик спикера. Раньше брались
/// пять самых длинных реплик, а длина не имеет отношения к тону: на тестовом
/// ролике это давало 140 Гц вместо 229, то есть «мужской» пол вместо
/// «женского» на одной и той же записи и одном и том же детекторе.
fn detect_speaker_genders(
    wav_path: &str,
    speaker_segments: &[SpeakerSegment],
) -> HashMap<String, String> {
    use crate::comm::voice;
    let (samples, sample_rate) = match load_wav_spec(wav_path) {
        Ok(v) => v,
        Err(e) => {
            log::error!("GenderDetection: не удалось загрузить WAV: {}", e);
            return HashMap::new();
        }
    };
    // Спикер → окна [start, end) в сэмплах.
    let mut by_speaker: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for seg in speaker_segments {
        let start = (seg.start_sec * sample_rate as f64) as usize;
        let end = (seg.end_sec * sample_rate as f64) as usize;
        if end > samples.len() || start >= end {
            continue;
        }
        by_speaker
            .entry(seg.speaker_id.clone())
            .or_default()
            .push((start, end));
    }
    let mut result = HashMap::new();
    for (speaker_id, windows) in by_speaker {
        let mut pitches: Vec<f64> = Vec::new();
        let mut used = 0usize;
        // Считаем до нужного ЧИСЛА ОЦЕНОК, а не до числа сегментов.
        // Сегмент, где тон не определился, просто не занимает места в
        // выборке, и мы идём дальше по списку.
        for (start, end) in windows.iter() {
            if crate::is_cancelled() {
                log::info!("GenderDetection: прервано пользователем");
                return HashMap::new();
            }
            let track = voice::voice_track(&samples[*start..*end], sample_rate);
            if track.pitches.is_empty() {
                continue;
            }
            pitches.extend_from_slice(&track.pitches);
            used += 1;
        }
        if pitches.is_empty() {
            log::warn!(
                "GenderDetection: {} — тон не измерен ни в одном сегменте, пол неизвестен \
                 (в промпте будет [Person], окончания не задаются)",
                speaker_id
            );
            continue;
        }
        let total_frames = pitches.len();
        let median = voice::median_pitch(&mut pitches).unwrap_or(0.0);
        match voice::gender_of(Some(median)) {
            Some(g) => {
                log::info!(
                    "GenderDetection: {} median pitch={:.0} Hz → {} ({} репл. из {}, {} кадров)",
                    speaker_id,
                    median,
                    g,
                    used,
                    windows.len(),
                    total_frames
                );
                result.insert(speaker_id, g.to_string());
            }
            None => {
                log::warn!(
                    "GenderDetection: {} — пол не определён по тону {:.0} Hz",
                    speaker_id,
                    median
                );
            }
        }
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
    words: Option<Vec<EngineWord>>,
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

/// Слово с таймингом — движок отдаёт их в каждом сегменте, см. `words` в JSON.
/// Раньше поле `words` не разбиралось вовсе, и `SubtitleChunk::word_timestamps`
/// оставалось `None` по всему пайплайну: данные приходили из движка и
/// выбрасывались.
#[derive(Deserialize)]
struct EngineWord {
    text: String,
    offsets: Option<TimestampsMs>,
}

#[derive(Deserialize)]
struct TimestampsMs {
    from: i64,
    to: i64,
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

/// Разбивает длинный сегмент по пословным таймингам на реплики по концам предложений и паузам.
fn split_segment_by_words(
    speaker_id: &str,
    text: &str,
    start: f64,
    end: f64,
    words: &[crate::comm::WordTimestamp],
) -> Vec<SubtitleChunk> {
    if words.is_empty() || end - start <= 8.0 {
        return vec![SubtitleChunk {
            start_sec: start,
            end_sec: end,
            text: text.to_string(),
            speaker_id: Some(speaker_id.to_string()),
            word_timestamps: if words.is_empty() { None } else { Some(words.to_vec()) },
        }];
    }

    let mut result = Vec::new();
    let mut current_words = Vec::new();
    let mut chunk_start = words[0].start_sec;

    for (i, w) in words.iter().enumerate() {
        current_words.push(w.clone());
        let is_end_punct = w.word.ends_with('.') || w.word.ends_with('!') || w.word.ends_with('?') || w.word.ends_with('…');
        let is_last = i + 1 == words.len();
        let next_gap = if !is_last { words[i + 1].start_sec - w.end_sec } else { 0.0 };

        let chunk_dur = w.end_sec - chunk_start;
        if is_end_punct || next_gap > 0.5 || chunk_dur > 7.0 || is_last {
            if !current_words.is_empty() {
                let chunk_end = current_words.last().unwrap().end_sec;
                let chunk_text = current_words.iter().map(|kw| kw.word.clone()).collect::<Vec<_>>().join(" ");
                if !chunk_text.trim().is_empty() && chunk_end > chunk_start {
                    result.push(SubtitleChunk {
                        start_sec: chunk_start,
                        end_sec: chunk_end.max(chunk_start + 0.2),
                        text: chunk_text,
                        speaker_id: Some(speaker_id.to_string()),
                        word_timestamps: Some(current_words.clone()),
                    });
                }
                current_words.clear();
                if !is_last {
                    chunk_start = words[i + 1].start_sec;
                }
            }
        }
    }

    if result.is_empty() {
        vec![SubtitleChunk {
            start_sec: start,
            end_sec: end,
            text: text.to_string(),
            speaker_id: Some(speaker_id.to_string()),
            word_timestamps: Some(words.to_vec()),
        }]
    } else {
        result
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
        let words = seg.words.as_ref().map(|ws| {
            ws.iter()
                .filter_map(|w| {
                    let o = w.offsets.as_ref()?;
                    Some(crate::comm::WordTimestamp {
                        word: w.text.clone(),
                        start_sec: o.from as f64 / 1000.0,
                        end_sec: o.to as f64 / 1000.0,
                    })
                })
                .collect::<Vec<_>>()
        }).unwrap_or_default();
        let split_chunks = split_segment_by_words(&speaker_id, text, start, end, &words);
        subtitle_chunks.extend(split_chunks);
    }
if speaker_segments.is_empty() {
    return None;
}
    speaker_segments.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());
    // Спикер без сегмента длиннее MIN_CLONE_REF_SEC не является голосом —
    // присоединяем его к соседу, иначе TTS упадёт на клонировании.
    crate::comm::merge_unclonable_speakers(&mut speaker_segments, &mut subtitle_chunks);
    Some((speaker_segments, subtitle_chunks))
}
/// Один прогон CrispASR: parakeet ASR + VAD + end-to-end диаризация sortformer.
/// Возвращает None, если движок недоступен/не дал сегментов.
///
/// **Почему sortformer, а не сессионная кластеризация TitaNet.** Раньше здесь
/// стояло `--diarize-speakers --diarize-embedder auto`: движок резал речь на
/// локальные трёки (pyannote), а глобальные номера спикеров собирал отдельной
/// кластеризацией эмбеддингов TitaNet-Large. На тестовом ролике (двое
/// говорящих) она выдавала ТРИ кластера и, главное, **сливала женский голос с
/// мужским в один**: сегменты 14.7–21.8 с и 22.1–55.2 с несли `speaker 2` при
/// разговоре «ведущий → Мелоди → ведущий». Дальше цепочка исправна, но
/// `tts::pick_ref_segment` берёт самый вокализированный сегмент кластера
/// (мужские ~125 Гц), и женские реплики этого кластера озвучиваются мужским
/// референсом — «мужик озвучивает бабу». Замер: `Speaker_1 F->M=12`.
/// Подбор `--diarize-cluster-threshold` лечил бы симптом под один ролик
/// (core §1.12), а класс отказа остался бы.
///
/// sortformer (NVIDIA Nemotron-3-Diarization) решает задачу целиком и за один
/// глобальный проход: до 8 спикеров в порядке появления, 10 мс разрешение.
/// Число спикеров остаётся автоматическим — харкода нет. На том же ролике:
/// 2 кластера (как в видео), 224 сегмента против 129, 12.7 с против 31.1 с.
///
/// JSON на выходе у обоих методов одинаков по форме (`offsets`, `speaker`,
/// `text`, `words[].offsets`), поэтому парсер выше не менялся.
///
/// Флаг `--diarize-speakers` здесь НЕ оставлен: в `--help` движка он описан как
/// алиас «pyannote + сессионная кластеризация», то есть называет другой
/// алгоритм. Проверено прогоном, что под `--diarize-method sortformer` он
/// инертен и разметку не меняет — так что оставить его можно было бы, но
/// явный `--diarize` читается однозначно (см. tasks\12.10.26 Диаризация
/// sortformer вместо TitaNet.md, гейт Б).
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
        "Diarization: [CrispASR] запускаем parakeet + диаризацию sortformer \
         (end-to-end, авто-число спикеров до 8)..."
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
        .arg("--diarize")
        .arg("--diarize-method")
        .arg("sortformer")
        .arg("--auto-download")
        .arg("--verbose")
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
    // stderr движка НЕ удаляем: это единственный источник попутной разбивки
    // по стадиям (`--verbose`), и без него фазовые тайминги диаризации
    // нечем диагностировать после прогона. Путь пишется в лог.
    log::info!(
        "Diarization: отчёт движка сохранён — {}",
        stderr_log.display()
    );
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
