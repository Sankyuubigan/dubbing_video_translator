use crate::comm::{PipelineContext, SpeakerSegment, SubtitleChunk};
use anyhow::{Context, Result};
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig,
};
use std::path::{Path, PathBuf};
use std::process::Command;

const MODELS_SUBDIR: &str = "models\\diarization";

fn project_root() -> PathBuf {
    // В debug — напрямую от CARGO_MANIFEST_DIR
    if cfg!(debug_assertions) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        log::info!("Diarization: project_root (debug) = {}", root.display());
        return root;
    }

    // В release — ищем корень, поднимаясь от exe
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().unwrap();
        loop {
            if dir.join("test").join("for_test.mp4").exists()
                || dir.join("src").join("App.tsx").exists()
            {
                log::info!("Diarization: project_root (exe walk) = {}", dir.display());
                return dir.to_path_buf();
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }
    }

    let fallback = std::env::current_dir().unwrap_or_default();
    log::info!("Diarization: project_root (fallback) = {}", fallback.display());
    fallback
}

fn models_dir() -> PathBuf {
    if let Some(cfg_dir) = crate::config::load().diarization_models_dir {
        let p = PathBuf::from(cfg_dir);
        if p.is_absolute() {
            return p;
        }
        return project_root().join(&p);
    }
    project_root().join(MODELS_SUBDIR)
}

const EMBEDDING_MODEL: &str = "nemo_en_titanet_small.onnx";

fn ensure_models_downloaded(models_dir: &Path) -> Result<()> {
    let seg_model = models_dir
        .join("sherpa-onnx-pyannote-segmentation-3-0")
        .join("model.onnx");
    let emb_model = models_dir.join(EMBEDDING_MODEL);

    if seg_model.exists() && emb_model.exists() {
        log::info!("Diarization: модели уже загружены в {}", models_dir.display());
        return Ok(());
    }

    log::info!("Diarization: проверяем/скачиваем модели в {}", models_dir.display());
    std::fs::create_dir_all(models_dir)
        .context("Ошибка создания директории для моделей диаризации")?;

    if !seg_model.exists() {
        let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
        let tar_path = models_dir.join("sherpa-onnx-pyannote-segmentation-3-0.tar.bz2");
        log::info!("Diarization: скачиваем модель сегментации (5 MB)...");
        download_file(url, &tar_path)?;
        log::info!("Diarization: распаковываем модель сегментации...");
        extract_tar_bz2(&tar_path, models_dir)?;
        std::fs::remove_file(&tar_path).ok();
    }

    if !emb_model.exists() {
        let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx";
        log::info!("Diarization: скачиваем модель эмбеддингов NeMo TitaNet (EN) (16 MB)...");
        download_file(url, &emb_model)?;
    }

    log::info!("Diarization: все модели готовы");
    Ok(())
}

fn download_file(url: &str, path: &Path) -> Result<()> {
    // Попытка 1: PowerShell с TLS 1.2
    let ps_result = try_download_powershell(url, path);
    if ps_result.is_ok() {
        return Ok(());
    }
    let ps_err = ps_result.unwrap_err();
    log::warn!("Diarization: PowerShell download failed: {}", ps_err);

    // Попытка 2: curl (если установлен)
    let curl_result = try_download_curl(url, path);
    if curl_result.is_ok() {
        return Ok(());
    }
    let curl_err = curl_result.unwrap_err();
    log::warn!("Diarization: curl download failed: {}", curl_err);

    // Попытка 3: bitsadmin (встроено в Windows)
    let bits_result = try_download_bitsadmin(url, path);
    if bits_result.is_ok() {
        return Ok(());
    }
    let bits_err = bits_result.unwrap_err();
    log::warn!("Diarization: bitsadmin download failed: {}", bits_err);

    anyhow::bail!(
        "Не удалось скачать модель.\n\
         URL: {}\n\
         Путь: {}\n\
         PowerShell: {}\n\
         curl: {}\n\
         bitsadmin: {}\n\n\
         Скачайте модель вручную и положите в {}",
        url,
        path.display(),
        ps_err,
        curl_err,
        bits_err,
        path.display()
    );
}

fn try_download_powershell(url: &str, path: &Path) -> Result<()> {
    let script = format!(
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; \
         [Net.ServicePointManager]::ServerCertificateValidationCallback = {{$true}}; \
         $p = Invoke-WebRequest -Uri \"{url}\" -OutFile \"{path}\" -UseBasicParsing -PassThru; \
         if ($p.StatusCode -ne 200) {{ throw \"HTTP $($p.StatusCode)\" }}",
        url = url,
        path = path.to_string_lossy()
    );

    let output = Command::new("powershell")
        .arg("-NoProfile")
        .arg("-Command")
        .arg(&script)
        .output()
        .context("Ошибка запуска PowerShell")?;

    if output.status.success() {
        if path.exists() && std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) > 1000 {
            return Ok(());
        }
        anyhow::bail!("Файл скачан, но слишком мал или отсутствует");
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!("PowerShell: {}", stderr.trim())
}

fn try_download_curl(url: &str, path: &Path) -> Result<()> {
    // Проверяем, доступен ли curl
    let which = Command::new("where").arg("curl").output();
    match which {
        Ok(out) if out.status.success() => {}
        _ => anyhow::bail!("curl не найден"),
    }

    let output = Command::new("curl")
        .args(&[
            "-L",
            "-o",
            &path.to_string_lossy(),
            "-f",
            "--ssl-reqd",
            url,
        ])
        .output()
        .context("Ошибка запуска curl")?;

    if output.status.success() && path.exists() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!("curl: {}", stderr.trim())
}

fn try_download_bitsadmin(url: &str, path: &Path) -> Result<()> {
    let output = Command::new("bitsadmin")
        .args(&[
            "/transfer",
            "DiarizationDownload",
            url,
            &path.to_string_lossy(),
        ])
        .output()
        .context("Ошибка запуска bitsadmin")?;

    if output.status.success() && path.exists() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!("bitsadmin: {}", stderr.trim())
}

fn extract_tar_bz2(archive: &Path, dest: &Path) -> Result<()> {
    // Способ 1: Rust bzip2 декомпрессия + Windows tar
    let result = try_extract_rust_bzip2(archive, dest);
    if result.is_ok() {
        return Ok(());
    }
    let err1 = result.unwrap_err();
    log::warn!("Diarization: bzip2 extraction failed: {}", err1);

    // Способ 2: 7-Zip (часто установлен на Windows)
    let result = try_extract_7z(archive, dest);
    if result.is_ok() {
        return Ok(());
    }
    let err2 = result.unwrap_err();
    log::warn!("Diarization: 7z extraction failed: {}", err2);

    anyhow::bail!(
        "Не удалось распаковать {} в {}.\n\
         bzip2: {}\n\
         7z: {}\n\n\
         Скачайте архив вручную и распакуйте 7-Zip или WinRAR в {}",
        archive.display(),
        dest.display(),
        err1,
        err2,
        dest.display()
    );
}

fn try_extract_rust_bzip2(archive: &Path, dest: &Path) -> Result<()> {
    use std::fs::File;
    use std::io::Read;

    // Декомпрессия bz2 -> tar (в память)
    let file = File::open(archive)
        .context("Ошибка открытия bz2 архива")?;
    let mut decoder = bzip2::read::BzDecoder::new(file);
    let mut tar_bytes = Vec::new();
    decoder
        .read_to_end(&mut tar_bytes)
        .context("Ошибка декомпрессии bz2")?;
    drop(decoder);

    // Сохраняем tar во временный файл рядом с архивом
    let tar_path = archive.with_extension("tar");
    std::fs::write(&tar_path, &tar_bytes)
        .context("Ошибка записи временного tar-файла")?;

    // Извлекаем tar через Windows tar
    let output = Command::new("tar")
        .arg("-xf")
        .arg(tar_path.to_string_lossy().to_string())
        .arg("-C")
        .arg(dest.to_string_lossy().to_string())
        .output()
        .context("Ошибка запуска tar для распаковки")?;

    std::fs::remove_file(&tar_path).ok();

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tar: {}", err.trim());
    }
    Ok(())
}

fn try_extract_7z(archive: &Path, dest: &Path) -> Result<()> {
    let check = Command::new("where").arg("7z").output();
    match check {
        Ok(out) if out.status.success() => {}
        _ => anyhow::bail!("7z не найден"),
    }

    let output = Command::new("7z")
        .args(&["x", "-y", "-o", &dest.to_string_lossy()])
        .arg(archive.to_string_lossy().to_string())
        .output()
        .context("Ошибка запуска 7z")?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("7z: {}", err.trim());
    }
    Ok(())
}

fn load_audio(path: &str) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path).context("Ошибка открытия WAV")?;
    Ok(reader
        .into_samples::<i16>()
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / 32768.0)
        .collect())
}

/// Минимальное расстояние между двумя сегментами (0 если пересекаются).
fn segment_distance(a: &SpeakerSegment, b: &SpeakerSegment) -> f64 {
    if a.start_sec < b.end_sec && a.end_sec > b.start_sec {
        return 0.0;
    }
    if a.end_sec <= b.start_sec {
        b.start_sec - a.end_sec
    } else {
        a.start_sec - b.end_sec
    }
}

/// Пост-обработка сегментов диаризации — слияние спикеров-дубликатов.
///
/// Стратегия (в порядке приоритета):
/// 1. Если есть >= 3 спикеров — пытаемся склеить не-доминантных между собой
///    (они не пересекаются во времени → один реальный человек).
/// 2. Если после этого остаётся минорный спикер (<15% времени), не пересекающийся
///    с доминантным — склеиваем с ближайшим мажорным спикером по времени.
fn merge_interleaved_speakers(segments: &[SpeakerSegment]) -> Vec<SpeakerSegment> {
    let n_speakers = segments.iter()
        .map(|s| &s.speaker_id)
        .collect::<std::collections::HashSet<_>>()
        .len();
    if n_speakers < 2 {
        log::debug!("Diarization: merge_interleaved_speakers — всего {} спикер, пропускаем", n_speakers);
        return segments.to_vec();
    }

    // Группируем сегменты по спикеру
    let mut by_speaker: std::collections::HashMap<String, Vec<SpeakerSegment>> =
        std::collections::HashMap::new();
    for seg in segments {
        by_speaker.entry(seg.speaker_id.clone()).or_default().push(seg.clone());
    }
    for segs in by_speaker.values_mut() {
        segs.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());
    }

    let speaker_ids: Vec<String> = by_speaker.keys().cloned().collect();
    let total_duration: f64 = segments.iter().map(|s| s.end_sec - s.start_sec).sum();

    let mut duration_map: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for seg in segments {
        *duration_map.entry(&seg.speaker_id).or_insert(0.0) += seg.end_sec - seg.start_sec;
    }

    // Сортируем спикеров по убыванию длительности
    let mut sorted_speakers: Vec<&str> = speaker_ids.iter().map(|s| s.as_str()).collect();
    sorted_speakers.sort_by(|a, b| {
        duration_map.get(b).unwrap_or(&0.0)
            .partial_cmp(duration_map.get(a).unwrap_or(&0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    log::debug!(
        "Diarization: merge_interleaved_speakers — {} speakers (total={:.1}s), sorted: {:?}",
        sorted_speakers.len(), total_duration,
        sorted_speakers.iter().map(|s| format!("{}={:.1}s", s, duration_map.get(s).unwrap_or(&0.0))).collect::<Vec<_>>()
    );

    let mut result = segments.to_vec();

    // Шаг 1: склеиваем не-доминантных между собой
    if sorted_speakers.len() >= 3 {
        let dominant = sorted_speakers[0];
        let anchor = sorted_speakers[1];
        log::debug!(
            "Diarization: merge — dominant={}, anchor={}, candidates={:?}",
            dominant, anchor, &sorted_speakers[2..]
        );

        for &candidate in &sorted_speakers[2..] {
            let candidate_segs = match by_speaker.get(candidate) {
                Some(s) => s,
                None => continue,
            };
            let anchor_segs = match by_speaker.get(anchor) {
                Some(s) => s,
                None => continue,
            };

            let has_overlap = candidate_segs.iter().any(|cs| {
                anchor_segs.iter().any(|a_seg| {
                    cs.start_sec < a_seg.end_sec && cs.end_sec > a_seg.start_sec
                })
            });

            let can_dur = duration_map.get(candidate).unwrap_or(&0.0);
            if !has_overlap {
                log::info!(
                    "Diarization: merging {} ({:.1}s, {:.0}%) into {} (no overlap) — same person",
                    candidate, can_dur, can_dur / total_duration * 100.0,
                    anchor
                );
                for seg in result.iter_mut() {
                    if seg.speaker_id == candidate {
                        seg.speaker_id = anchor.to_string();
                    }
                }
            } else {
                log::info!(
                    "Diarization: NOT merging {} ({:.1}s) into {} (overlaps — might not be same person)",
                    candidate, can_dur, anchor
                );
            }
        }
    }

    // Обновляем группы после шага 1
    let mut by_speaker_after: std::collections::HashMap<String, Vec<SpeakerSegment>> =
        std::collections::HashMap::new();
    for seg in &result {
        by_speaker_after.entry(seg.speaker_id.clone()).or_default().push(seg.clone());
    }
    let remaining_speakers: Vec<String> = by_speaker_after.keys().cloned().collect();
    log::debug!(
        "Diarization: после шага 1 осталось спикеров: {} — {:?}",
        remaining_speakers.len(),
        remaining_speakers
    );

    // Шаг 2: если осталось > 2 спикеров — склеиваем минорного (<15%)
    // с ближайшим мажорным спикером по времени (не только с доминантным)
    if remaining_speakers.len() >= 3 {
        let mut dur_after: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        for seg in &result {
            *dur_after.entry(seg.speaker_id.clone()).or_insert(0.0) += seg.end_sec - seg.start_sec;
        }
        let mut sorted_after: Vec<String> = remaining_speakers.clone();
        sorted_after.sort_by(|a, b| {
            let da = dur_after.get(a).copied().unwrap_or(0.0);
            let db = dur_after.get(b).copied().unwrap_or(0.0);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Собираем мажорных спикеров (>15% времени)
        let major_speakers: Vec<&String> = sorted_after.iter()
            .filter(|s| dur_after.get(*s).copied().unwrap_or(0.0) > total_duration * 0.15)
            .collect();
        // Если все спикеры минорные — используем доминантного как fallback
        let major_speakers: Vec<&String> = if major_speakers.is_empty() {
            vec![&sorted_after[0]]
        } else {
            major_speakers
        };

        for minor in &sorted_after[1..] {
            let minor_dur = dur_after.get(minor).copied().unwrap_or(0.0);
            if minor_dur > total_duration * 0.15 {
                continue;
            }

            let minor_segs = match by_speaker_after.get(minor) {
                Some(s) => s,
                None => continue,
            };

            // Проверяем пересечение со ВСЕМИ мажорными спикерами
            let mut overlaps_with: Option<&String> = None;
            for &major in &major_speakers {
                let major_segs = match by_speaker_after.get(major) {
                    Some(s) => s,
                    None => continue,
                };
                let has_overlap = minor_segs.iter().any(|ms| {
                    major_segs.iter().any(|ds| {
                        ms.start_sec < ds.end_sec && ms.end_sec > ds.start_sec
                    })
                });
                if has_overlap {
                    overlaps_with = Some(major);
                    break;
                }
            }

            match overlaps_with {
                Some(major) => {
                    log::debug!("Diarization: шаг 2 — {} overlaps with {}, не склеиваем", minor, major);
                }
                None => {
                    // Не пересекается ни с кем — находим ближайшего мажорного спикера
                    let nearest: &String = major_speakers.iter()
                        .min_by(|&a, &b| {
                            let a_segs = by_speaker_after.get(a.as_str()).unwrap();
                            let b_segs = by_speaker_after.get(b.as_str()).unwrap();
                            let dist_a = minor_segs.iter()
                                .map(|ms| a_segs.iter()
                                    .map(|a_seg| segment_distance(ms, a_seg))
                                    .min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap_or(f64::MAX))
                                .min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                                .unwrap_or(f64::MAX);
                            let dist_b = minor_segs.iter()
                                .map(|ms| b_segs.iter()
                                    .map(|b_seg| segment_distance(ms, b_seg))
                                    .min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap_or(f64::MAX))
                                .min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                                .unwrap_or(f64::MAX);
                            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map_or(&sorted_after[0], |n| *n);

                    log::info!(
                        "Diarization: шаг 2 — merging {} ({:.1}s, {:.0}%) into {} (no overlap, nearest major)",
                        minor, minor_dur, minor_dur / total_duration * 100.0,
                        nearest
                    );
                    for seg in result.iter_mut() {
                        if seg.speaker_id == *minor {
                            seg.speaker_id = nearest.to_string();
                        }
                    }
                }
            }
        }
    }

    // Финальная пересборка — склеиваем соседние сегменты одного спикера
    let mut merged: Vec<SpeakerSegment> = Vec::new();
    for seg in result {
        if let Some(last) = merged.last_mut() {
            let gap = seg.start_sec - last.end_sec;
            if last.speaker_id == seg.speaker_id && gap <= 1.0 {
                last.end_sec = last.end_sec.max(seg.end_sec);
                continue;
            }
        }
        merged.push(seg);
    }

    let final_speakers: std::collections::HashSet<&str> = merged.iter().map(|s| s.speaker_id.as_str()).collect();
    log::info!(
        "Diarization: merge_interleaved_speakers — итого {} спикеров: {:?}",
        final_speakers.len(),
        final_speakers
    );

    merged
}

/// 3. Сливаем минорных спикеров (<5% общего времени) с ближайшим мажорным
/// 4. Гарантируем минимум 2 спикера, если их было >= 2 до слияния
/// 5. Перенумеровываем спикеров в порядке появления (Speaker_1, Speaker_2, ...)
fn postprocess_segments(segments: Vec<SpeakerSegment>) -> Vec<SpeakerSegment> {
    const MIN_DURATION: f64 = 0.5;
    const MERGE_GAP: f64 = 1.0;
    const MINOR_SPEAKER_RATIO: f64 = 0.05;

    // Шаг 1: фильтруем слишком короткие сегменты
    let filtered: Vec<SpeakerSegment> = segments
        .into_iter()
        .filter(|s| (s.end_sec - s.start_sec) >= MIN_DURATION)
        .collect();

    if filtered.is_empty() {
        return filtered;
    }

    // Шаг 2: объединяем соседние/перекрывающиеся сегменты одного спикера
    let mut merged: Vec<SpeakerSegment> = Vec::new();
    for seg in filtered {
        if let Some(last) = merged.last_mut() {
            let gap = seg.start_sec - last.end_sec;
            if last.speaker_id == seg.speaker_id && gap <= MERGE_GAP {
                last.end_sec = last.end_sec.max(seg.end_sec);
                continue;
            }
        }
        merged.push(seg);
    }

    // Считаем общую длительность каждого спикера
    let total_duration: f64 = merged.iter().map(|s| s.end_sec - s.start_sec).sum();
    let mut speaker_duration: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for seg in &merged {
        *speaker_duration.entry(seg.speaker_id.clone()).or_insert(0.0) += seg.end_sec - seg.start_sec;
    }

    // Определяем мажорных спикеров (>MINOR_SPEAKER_RATIO времени)
    let min_major_duration = total_duration * MINOR_SPEAKER_RATIO;
    let mut major_speakers: std::collections::HashSet<String> = speaker_duration
        .iter()
        .filter(|(_, &dur)| dur >= min_major_duration)
        .map(|(id, _)| id.clone())
        .collect();

    // Гарантируем минимум 2 мажорных спикера, если до слияния было >= 2 разных
    let distinct_before = merged.iter()
        .map(|s| &s.speaker_id)
        .collect::<std::collections::HashSet<_>>()
        .len();
    if distinct_before >= 2 && major_speakers.len() < 2 {
        let mut sorted: Vec<(String, f64)> = speaker_duration
            .iter()
            .filter(|(id, _)| !major_speakers.contains(*id))
            .map(|(id, dur)| (id.clone(), *dur))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (id, _) in &sorted {
            if major_speakers.len() >= 2 {
                break;
            }
            major_speakers.insert(id.clone());
        }
    }

    // Шаг 3: минорных спикеров с ближайшим мажорным по времени.
    // НЕ склеиваем, если сегмент находится МЕЖДУ двумя сегментами одного мажора
    // (turn-taking — реальный диалог, а не артефакт модели).
    let major_segs: Vec<&SpeakerSegment> = merged.iter()
        .filter(|s| major_speakers.contains(&s.speaker_id))
        .collect();
    // Группируем мажорные сегменты по спикеру для быстрого поиска "между"
    let mut major_by_speaker: std::collections::HashMap<&str, Vec<&SpeakerSegment>> = std::collections::HashMap::new();
    for s in &major_segs {
        major_by_speaker.entry(&s.speaker_id).or_default().push(s);
    }

    let mut result: Vec<SpeakerSegment> = Vec::new();
    for seg in &merged {
        if major_speakers.contains(&seg.speaker_id) {
            result.push(seg.clone());
        } else {
            if let Some(nearest) = major_segs.iter()
                .min_by(|a, b| {
                    let dist_a = (a.start_sec - seg.start_sec).abs()
                        .min((a.end_sec - seg.start_sec).abs());
                    let dist_b = (b.start_sec - seg.start_sec).abs()
                        .min((b.end_sec - seg.start_sec).abs());
                    dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                // Проверка на interleaved: если сегмент минора находится между
                // двумя сегментами ОДНОГО мажора — это turn-taking, не склеиваем
                let is_interleaved = if let Some(nearest_segs) = major_by_speaker.get(nearest.speaker_id.as_str()) {
                    let before = nearest_segs.iter()
                        .filter(|s| s.end_sec <= seg.start_sec)
                        .max_by(|a, b| a.end_sec.partial_cmp(&b.end_sec).unwrap_or(std::cmp::Ordering::Equal));
                    let after = nearest_segs.iter()
                        .filter(|s| s.start_sec >= seg.end_sec)
                        .min_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap_or(std::cmp::Ordering::Equal));
                    // С обеих сторон есть сегменты одного мажора с разумным зазором — interleaved
                    matches!((before, after), (Some(b), Some(a))
                        if (seg.start_sec - b.end_sec) <= 3.0
                        && (a.start_sec - seg.end_sec) <= 3.0)
                } else {
                    false
                };

                if is_interleaved {
                    log::debug!(
                        "Diarization: шаг 3 — НЕ склеиваем {} ({:.1}s–{:.1}s) с {} (interleaved, turn-taking)",
                        seg.speaker_id, seg.start_sec, seg.end_sec, nearest.speaker_id
                    );
                    result.push(seg.clone());
                } else {
                    let mut s = seg.clone();
                    s.speaker_id = nearest.speaker_id.clone();
                    result.push(s);
                }
            } else {
                result.push(seg.clone());
            }
        }
    }

    // Шаг 4: повторно объединяем соседние/перекрывающиеся сегменты одного спикера (после слияния)
    let mut final_segments: Vec<SpeakerSegment> = Vec::new();
    for seg in result {
        if let Some(last) = final_segments.last_mut() {
            let gap = seg.start_sec - last.end_sec;
            if last.speaker_id == seg.speaker_id && gap <= MERGE_GAP {
                last.end_sec = last.end_sec.max(seg.end_sec);
                continue;
            }
        }
        final_segments.push(seg);
    }

    // Шаг 4.5: умное слияние чередующихся непересекающихся спикеров
    final_segments = merge_interleaved_speakers(&final_segments);

    // Шаг 5: перенумеровываем спикеров в порядке появления
    let mut speaker_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut next_id = 1;
    for seg in &mut final_segments {
        let old_id = seg.speaker_id.clone();
        let new_num = *speaker_map.entry(old_id).or_insert_with(|| {
            let n = next_id;
            next_id += 1;
            n
        });
        seg.speaker_id = format!("Speaker_{}", new_num);
    }

    final_segments
}

pub fn diarize(ctx: PipelineContext) -> Result<PipelineContext> {
    let wav_path = match ctx.wav_path.as_deref() {
        Some(p) => p.to_string(),
        None => anyhow::bail!("Нет WAV файла"),
    };

    let models_dir = models_dir();
    ensure_models_downloaded(&models_dir)?;

    let seg_model = models_dir
        .join("sherpa-onnx-pyannote-segmentation-3-0")
        .join("model.onnx");
    let emb_model = models_dir.join(EMBEDDING_MODEL);

    if !seg_model.exists() {
        anyhow::bail!(
            "Модель сегментации не найдена: {}\n\
             Скачайте вручную:\n\
             https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2\n\
             и распакуйте в {}",
            seg_model.display(),
            models_dir.display()
        );
    }
    if !emb_model.exists() {
        anyhow::bail!(
            "Модель эмбеддингов не найдена: {}\n\
             Скачайте вручную:\n\
             https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx\n\
             в {}",
            emb_model.display(),
            models_dir.display()
        );
    }

    let threshold = ctx
        .config
        .diarization_threshold
        .unwrap_or(0.5);
    // По умолчанию 2 кластера — TitaNet эмбеддинги сами разделят мужской/женский голос.
    // -1 (авто) создаёт до 8 кластеров, потом мы их вручную склеиваем — это вносит ошибки.
    let num_clusters = ctx
        .config
        .diarization_num_speakers
        .unwrap_or(2);

    log::info!("Diarization: инициализация sherpa-onnx диаризации");
    log::info!("  seg_model: {}", seg_model.display());
    log::info!("  emb_model: {} (NeMo TitaNet EN)", emb_model.display());
    log::info!("  threshold={}, num_clusters={}", threshold, num_clusters);

    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(seg_model.to_string_lossy().to_string()),
            },
            num_threads: 4,
            ..Default::default()
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some(emb_model.to_string_lossy().to_string()),
            num_threads: 4,
            ..Default::default()
        },
        clustering: FastClusteringConfig {
            num_clusters,
            threshold: threshold as f32,
        },
        // min_duration_on: минимальная длительность речевого сегмента (сек)
        // min_duration_off: минимальная пауза между спикерами для смены (сек)
        //   Было 1.0 — слишком длинная пауза, проглатывались быстрые реплики собеседника
        min_duration_on: 0.3,
        min_duration_off: 0.5,
    };

    let sd = OfflineSpeakerDiarization::create(&config)
        .context("Ошибка создания OfflineSpeakerDiarization — проверьте целостность моделей")?;

    let audio = load_audio(&wav_path).context("Ошибка загрузки WAV")?;
    log::info!(
        "Diarization: обрабатываем {:.1}s аудио ({} сэмплов)",
        audio.len() as f64 / 16000.0,
        audio.len()
    );

    let result = match sd.process(&audio) {
        Some(r) => r,
        None => {
            log::warn!("Diarization: процесс не вернул результатов");
            return Ok(PipelineContext {
                speaker_segments: Some(Vec::new()),
                ..ctx
            });
        }
    };

    let n_speakers = result.num_speakers();
    let n_segments = result.num_segments();
    log::info!(
        "Diarization: найдено {} спикеров, {} сегментов",
        n_speakers,
        n_segments
    );

    let mut speaker_segments: Vec<SpeakerSegment> = result
        .sort_by_start_time()
        .into_iter()
        .map(|s| SpeakerSegment {
            start_sec: s.start as f64,
            end_sec: s.end as f64,
            speaker_id: format!("Speaker_{}", s.speaker + 1),
        })
        .collect();

    // Пост-обработка: фильтруем короткие сегменты и объединяем соседние одного спикера
    speaker_segments = postprocess_segments(speaker_segments);

    for seg in &speaker_segments {
        log::info!(
            "Diarization: {:.1}s–{:.1}s -> {}",
            seg.start_sec,
            seg.end_sec,
            seg.speaker_id
        );
    }

    if !speaker_segments.is_empty() {
        let unique: std::collections::HashSet<&str> = speaker_segments
            .iter()
            .map(|s| s.speaker_id.as_str())
            .collect();
        log::info!(
            "Diarization: после пост-обработки осталось {} сегментов, {} уникальных спикеров",
            speaker_segments.len(),
            unique.len()
        );
    }

    // Применяем спикеров к subtitle_chunks (если есть)
    let subtitle_chunks = match ctx.subtitle_chunks.as_ref() {
        Some(chunks) => {
            let updated: Vec<SubtitleChunk> = chunks.iter().map(|chunk| {
                let speaker = crate::comm::find_speaker_for_chunk(chunk, &speaker_segments);
                SubtitleChunk {
                    speaker_id: speaker.map(|s| s.to_string()),
                    word_timestamps: None,
                    ..chunk.clone()
                }
            }).collect();
            log::info!("Diarization: назначены спикеры для {} субтитров", updated.len());
            Some(updated)
        }
        None => None,
    };

    Ok(PipelineContext {
        speaker_segments: Some(speaker_segments),
        subtitle_chunks,
        ..ctx
    })
}
