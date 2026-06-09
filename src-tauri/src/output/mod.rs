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

    let output_path = match try_mux(input, &srt_en, &srt_ru, &fmt, &ctx.config.ffmpeg_path) {
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

fn try_mux(input: &str, srt_en: &str, srt_ru: &str, fmt: &str, ffmpeg_cfg: &Option<String>) -> Result<String> {
    let output_path = generate_output_path(input, fmt);
    log::info!("Маскинг: вшиваем субтитры (en + ru) в {}", output_path);

    // MKV поддерживает SRT нативно, MP4/MOV требуют mov_text
    let sub_codec = match fmt {
        "mkv" => "srt",
        _ => "mov_text",
    };

    let ffmpeg = crate::ffmpeg::resolve(ffmpeg_cfg);
    run_ffmpeg_mux(&ffmpeg, input, srt_en, srt_ru, &output_path, sub_codec)?;
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

fn write_srt(chunks: &[SubtitleChunk], suffix: &str) -> Result<String> {
    let srt_path = std::env::temp_dir()
        .join(format!("dubvidtra_subtitles_{}.srt", suffix))
        .to_string_lossy()
        .to_string();

    let mut content = String::from("\u{FEFF}"); // UTF-8 BOM
    let mut idx = 0;

    for chunk in chunks.iter() {
        let text = chunk.text.trim();
        if text.is_empty() {
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


