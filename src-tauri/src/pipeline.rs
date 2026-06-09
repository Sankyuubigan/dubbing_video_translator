use crate::comm::{PipelineContext, ProgressUpdate};
use anyhow::Result;
use std::time::Instant;
use tauri::Emitter;

fn emit(stage: &str, percent: f32) {
    if let Some(handle) = crate::APP_HANDLE.get() {
        let _ = handle.emit(
            "pipeline-progress",
            ProgressUpdate {
                stage: stage.to_string(),
                percent,
                result_path: None,
            },
        );
    }
}

macro_rules! timed_stage {
    ($name:expr, $percent:expr, $ctx:expr, $stage:expr) => {{
        emit($name, $percent);
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

    let ctx = timed_stage!("vad", 15.0, ctx, crate::vad_ffmpeg::detect)
        .map_err(|e| { log::error!("[pipeline] vad_ffmpeg: {:#}", e); e })?;

    let ctx = timed_stage!("stt", 30.0, ctx, crate::stt::transcribe)
        .map_err(|e| { log::error!("[pipeline] stt: {:#}", e); e })?;

    let ctx = timed_stage!("diarize", 50.0, ctx, crate::diarization::diarize)
        .map_err(|e| { log::error!("[pipeline] diarization: {:#}", e); e })?;

    let ctx = timed_stage!("translate", 70.0, ctx, crate::translation::translate)
        .map_err(|e| { log::error!("[pipeline] translation: {:#}", e); e })?;

    let ctx = timed_stage!("mux", 90.0, ctx, crate::output::mux)
        .map_err(|e| { log::error!("[pipeline] output: {:#}", e); e })?;

    log::info!("[timing] TOTAL pipeline: {:.1}s", t_total.elapsed().as_secs_f64());
    Ok(ctx)
}
