use crate::comm::{PipelineContext, SubtitleChunk};
use anyhow::{Context, Result};
use std::os::windows::process::CommandExt;
use std::process::Command;

const CREATE_NO_WINDOW: u32 = 0x08000000;

pub fn mux(ctx: PipelineContext) -> Result<PipelineContext> {
    let ru_chunks = match ctx.translated_chunks.as_ref() {
        Some(c) => c,
        None => {
            log::error!("output: Нет переведённых чанков");
            anyhow::bail!("Нет переведённых чанков");
        }
    };
    let en_chunks = match ctx.subtitle_chunks.as_ref() {
        Some(c) => c,
        None => {
            log::error!("output: Нет оригинальных чанков");
            anyhow::bail!("Нет оригинальных чанков");
        }
    };

    let srt_ru = match write_srt(ru_chunks, "ru") {
        Ok(p) => p,
        Err(e) => {
            log::error!("output: Ошибка записи SRT (ru): {:#}", e);
            return Err(e);
        }
    };
    let srt_en = match write_srt(en_chunks, "en") {
        Ok(p) => p,
        Err(e) => {
            log::error!("output: Ошибка записи SRT (en): {:#}", e);
            return Err(e);
        }
    };

    let input = ctx.resolved_input_path.as_deref().unwrap_or(&ctx.config.input_path);

    if !std::path::Path::new(input).exists() {
        log::error!("output: Входной файл не найден: {}", input);
        anyhow::bail!("Входной файл не найден: {}", input);
    }

    let fmt = ctx.config.output_format.to_lowercase();

    let output_path = match try_mux(
        input,
        &srt_en,
        &srt_ru,
        &fmt,
        &ctx.config.ffmpeg_path,
        ctx.config.mix_volume,
        ctx.dubbed_audio_path.as_deref(),
    ) {
        Ok(path) => path,
        Err(e) => {
            log::error!("output: Muxing failed: {}", e);
            return Err(e);
        }
    };

    Ok(PipelineContext {
        output_path: Some(output_path),
        ..ctx
    })
}

fn try_mux(
    input: &str,
    srt_en: &str,
    srt_ru: &str,
    fmt: &str,
    ffmpeg_cfg: &Option<String>,
    mix_volume: f64,
    dubbed_audio: Option<&str>,
) -> Result<String> {
    let output_path = generate_output_path(input, fmt);
    log::info!("Маскинг: вшиваем субтитры (en + ru) в {}", output_path);

    let sub_codec = match fmt {
        "mkv" => "srt",
        _ => "mov_text",
    };

    let ffmpeg = crate::ffmpeg::resolve(ffmpeg_cfg);
    if let Some(dub_path) = dubbed_audio {
        run_ffmpeg_mux_with_dub(&ffmpeg, input, srt_en, srt_ru, &output_path, sub_codec, dub_path, mix_volume)?;
    } else {
        run_ffmpeg_mux(&ffmpeg, input, srt_en, srt_ru, &output_path, sub_codec)?;
    }
    Ok(output_path)
}

fn run_ffmpeg_mux(ffmpeg: &str, input: &str, srt_en: &str, srt_ru: &str, output: &str, sub_codec: &str) -> Result<()> {
    let mut cmd = Command::new(ffmpeg);
    cmd.creation_flags(CREATE_NO_WINDOW)
        .arg("-i").arg(input)
        .arg("-sub_charenc").arg("UTF-8")
        .arg("-i").arg(srt_en)
        .arg("-sub_charenc").arg("UTF-8")
        .arg("-i").arg(srt_ru)
        .arg("-map").arg("0:v")
        .arg("-map").arg("0:a")
        .arg("-map").arg("1")
        .arg("-map").arg("2")
        .arg("-c:v").arg("copy")
        .arg("-c:a").arg("copy")
        .arg("-c:s").arg(sub_codec)
        .arg("-metadata:s:s:0").arg("language=eng")
        .arg("-metadata:s:s:1").arg("language=rus")
        .arg("-y")
        .arg(output);

    log::info!("output: ffmpeg command: {:?}", cmd);

    let status = cmd.output()
        .map_err(|e| anyhow::anyhow!("FFmpeg не найден: {}", e))?;

    if !status.status.success() {
        let stderr = String::from_utf8_lossy(&status.stderr);
        log::error!("output: ffmpeg stderr:\n{}", stderr);
        anyhow::bail!("FFmpeg muxing error: {}", stderr);
    }
    log::info!("output: muxed -> {}", output);
    Ok(())
}

fn run_ffmpeg_mux_with_dub(
    ffmpeg: &str,
    input: &str,
    srt_en: &str,
    srt_ru: &str,
    output: &str,
    sub_codec: &str,
    dubbed_audio: &str,
    mix_volume: f64,
) -> Result<()> {
    let mut cmd = Command::new(ffmpeg);
    cmd.creation_flags(CREATE_NO_WINDOW)
        .arg("-i").arg(input)
        .arg("-i").arg(dubbed_audio)
        .arg("-sub_charenc").arg("UTF-8")
        .arg("-i").arg(srt_en)
        .arg("-sub_charenc").arg("UTF-8")
        .arg("-i").arg(srt_ru)
        .arg("-filter_complex")
        .arg(format!("[0:a]volume={}[orig];[orig][1:a]amix=inputs=2:duration=first:dropout_transition=2[aout]", mix_volume))
        .arg("-map").arg("0:v")
        .arg("-map").arg("[aout]")
        .arg("-map").arg("0:a")
        .arg("-map").arg("2")
        .arg("-map").arg("3")
        .arg("-c:v").arg("copy")
        .arg("-c:a:0").arg("aac")
        .arg("-c:a:1").arg("copy")
        .arg("-c:s").arg(sub_codec)
        .arg("-metadata:s:a:0").arg("language=rus")
        .arg("-metadata:s:a:1").arg("language=eng")
        .arg("-metadata:s:s:0").arg("language=eng")
        .arg("-metadata:s:s:1").arg("language=rus")
        .arg("-y")
        .arg(output);

    log::info!("output: ffmpeg command with dub: {:?}", cmd);

    let status = cmd.output()
        .map_err(|e| anyhow::anyhow!("FFmpeg не найден: {}", e))?;

    if !status.status.success() {
        let stderr = String::from_utf8_lossy(&status.stderr);
        log::error!("output: ffmpeg stderr:\n{}", stderr);
        anyhow::bail!("FFmpeg muxing error: {}", stderr);
    }
    log::info!("output: muxed with dub -> {}", output);
    Ok(())
}

/// Чистая функция: список чанков → содержимое SRT. Без файлов и без
/// побочных эффектов — её можно звать и из тест-бинарника, и из юнит-теста.
///
/// `SKIP_MARKER` фильтруется здесь, а не в `write_srt`, потому что это
/// инвариант формата, а не особенность записи на диск. `tts` делает то же
/// сравнение (`tts/mod.rs`), и раньше расходились только две из трёх точек
/// потребления, из-за чего служебный маркер `(-)` попадал в субтитры.
pub fn build_srt_content(chunks: &[SubtitleChunk]) -> String {
    let mut content = String::from("\u{FEFF}"); // UTF-8 BOM
    let mut idx = 0;

    for chunk in chunks.iter() {
        let text = chunk.text.trim();
        // Метка пропуска — не текст для показа. Чанк, не прошедший
        // верификацию, молчит в озвучке (tts) и не должен занимать экран.
        if text.is_empty() || text == crate::comm::SKIP_MARKER {
            continue;
        }
        idx += 1;
        let start = format_timestamp(chunk.start_sec);
        let end = format_timestamp(chunk.end_sec);
        let speaker = chunk
            .speaker_id
            .as_deref()
            .unwrap_or("Speaker");
        content.push_str(&format!("{}\n{} --> {}\n[{}] {}\n\n", idx, start, end, speaker, text));
    }

    content
}

fn write_srt(chunks: &[SubtitleChunk], suffix: &str) -> Result<String> {
    let srt_path = crate::paths::temp_file(&format!("deedub_subtitles_{}.srt", suffix))
        .to_string_lossy()
        .to_string();

    let content = build_srt_content(chunks);

    std::fs::write(&srt_path, content)
        .map_err(|e| { log::error!("output: Ошибка записи SRT: {:#}", e); e })
        .context("Ошибка записи SRT")?;
    Ok(srt_path)
}

fn format_timestamp(secs: f64) -> String {
    let total_ms = (secs * 1000.0 + 0.5) as u64;
    let hours = total_ms / 3_600_000;
    let mins = (total_ms % 3_600_000) / 60_000;
    let secs_remain = (total_ms % 60_000) / 1_000;
    let millis = total_ms % 1_000;
    format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs_remain, millis)
}

fn generate_output_path(input: &str, format: &str) -> String {
    let input_path = std::path::Path::new(input);
    let parent = input_path.parent().unwrap_or(std::path::Path::new("."));
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    parent
        .join(format!("{}_subbed.{}", stem, format))
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::SKIP_MARKER;

    fn chunk(start: f64, end: f64, text: &str) -> SubtitleChunk {
        SubtitleChunk {
            start_sec: start,
            end_sec: end,
            text: text.to_string(),
            speaker_id: Some("Speaker_4".to_string()),
            word_timestamps: None,
        }
    }

    /// Номера cue-блоков. Разбор, а не поиск подстроки: перед первым индексом
    /// стоит BOM, поэтому `"\n1\n"` не матчится.
    fn cue_indices(srt: &str) -> Vec<u32> {
        srt.lines()
            .filter_map(|l| {
                let t = l.trim().trim_start_matches('\u{FEFF}').trim();
                if !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()) {
                    t.parse().ok()
                } else {
                    None
                }
            })
            .collect()
    }

    /// Регрессия: `SKIP_MARKER` — служебная метка, а не текст. Чанк, не
    /// прошедший верификацию, молчит в TTS и не должен занимать экран.
    #[test]
    fn test_skip_marker_not_written_to_srt() {
        let srt = build_srt_content(&[
            chunk(108.44, 111.19, "Ну что ж, я заберу вас всех."),
            chunk(180.354, 191.840, SKIP_MARKER),
        ]);
        assert!(!srt.contains(SKIP_MARKER), "маркер попал в SRT: {}", srt);
        assert!(!srt.contains("00:03:00,354"), "тайминг маркера в SRT");
        assert!(srt.contains("Ну что ж, я заберу вас всех."));
    }

    /// Нумерация сплошная: `idx` растёт только для записанных cue, поэтому
    /// пропуск не оставляет дыры и не сдвигает остальные номера.
    #[test]
    fn test_indices_contiguous_after_skip() {
        let srt = build_srt_content(&[
            chunk(0.0, 1.0, "первый"),
            chunk(1.0, 2.0, SKIP_MARKER),
            chunk(2.0, 3.0, "второй"),
        ]);
        assert_eq!(cue_indices(&srt), vec![1, 2]);
    }

    /// Обрезка пробелов не должна обойти фильтр: `write_srt` триммит перед
    /// сравнением, и `"  (-)  "` тоже должен отфильтроваться.
    #[test]
    fn test_skip_marker_with_surrounding_spaces() {
        let srt = build_srt_content(&[chunk(0.0, 1.0, "  (-)  ")]);
        assert!(!srt.contains("-->"), "маркер в пробелах записан: {}", srt);
    }

    #[test]
    fn test_empty_text_still_filtered() {
        let srt = build_srt_content(&[chunk(0.0, 1.0, "   ")]);
        assert!(!srt.contains("-->"), "пустой чанк записан: {}", srt);
    }

    /// Рефакторинг вынес форматирование в чистую функцию — формат не должен
    /// поехать: BOM для `mov_text`, `,` в миллисекундах, `[Speaker]`.
    #[test]
    fn test_srt_format_preserved() {
        let srt = build_srt_content(&[chunk(0.24, 2.44, "Вот во что верят любители.")]);
        assert!(srt.starts_with('\u{FEFF}'), "потерян UTF-8 BOM");
        assert!(srt.contains("00:00:00,240 --> 00:00:02,440"), "{}", srt);
        assert!(srt.contains("[Speaker_4] Вот во что верят любители."), "{}", srt);
    }
}


