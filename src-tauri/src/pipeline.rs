use crate::comm::{PipelineContext, ProgressUpdate};
use anyhow::Result;
use std::time::Instant;
use tauri::Emitter;

/// Прогресс пайплайна. Открыт наружу, чтобы длинные фазы (TTS) могли
/// сообщать внутрифазовый прогресс, а не visить на одной константе.
pub fn emit_progress(stage: &str, percent: f32) {
    if let Some(handle) = crate::APP_HANDLE.get() {
        let _ = handle.emit(
            "pipeline-progress",
            ProgressUpdate {
                stage: stage.to_string(),
                percent,
                result_path: None,
                error_message: None,
            },
        );
    }
}

macro_rules! timed_stage {
    ($name:expr, $percent:expr, $ctx:expr, $stage:expr) => {{
        if crate::is_cancelled() {
            anyhow::bail!("Pipeline отменён пользователем на этапе {}", $name);
        }
        emit_progress($name, $percent);
        let t = Instant::now();
        let result = $stage($ctx);
        log::info!("[timing] {}: {:.1}s", $name, t.elapsed().as_secs_f64());
        result
    }};
}

pub fn run(ctx: PipelineContext) -> Result<PipelineContext> {
    let t_total = Instant::now();

    let ctx = timed_stage!("extract", 5.0, ctx, crate::audio_extractor::extract)
        .map_err(|e| { log::error!("[pipeline] audio_extractor: {:#}", e); e })?;

    let ctx = timed_stage!("diarize", 20.0, ctx, crate::diarization::diarize)
        .map_err(|e| { log::error!("[pipeline] diarization: {:#}", e); e })?;

    let ctx = timed_stage!("stt", 40.0, ctx, crate::stt::transcribe)
        .map_err(|e| { log::error!("[pipeline] stt: {:#}", e); e })?;

    // `translate` и `verify` — два последовательных LLM-этапа. Они делят ОДНУ
    // сессию `llama-server.exe`: иначе верификация поднимала второй процесс
    // (5 с загрузки 6.5 ГиБ в VRAM) ради десятка запросов по бракованным
    // чанкам. Сессия умирает на выходе из `verify`, то есть строго ДО старта
    // TTS — VRAM освобождается по desktop §6.5.
    let (ctx, llm) = timed_stage!("translate", 60.0, ctx, crate::translation::translate_with_session)
        .map_err(|e| { log::error!("[pipeline] translation: {:#}", e); e })?;

    let ctx = timed_stage!("verify", 65.0, (ctx, llm), crate::verification::verify_with_session)
        .map_err(|e| { log::error!("[pipeline] verification: {:#}", e); e })?;

    let ctx = timed_stage!("tts", 70.0, ctx, crate::tts::dub)
        .map_err(|e| { log::error!("[pipeline] tts: {:#}", e); e })?;

    let ctx = timed_stage!("mux", 90.0, ctx, crate::output::mux)
        .map_err(|e| { log::error!("[pipeline] output: {:#}", e); e })?;

    log::info!("[timing] TOTAL pipeline: {:.1}s", t_total.elapsed().as_secs_f64());
    Ok(ctx)
}
