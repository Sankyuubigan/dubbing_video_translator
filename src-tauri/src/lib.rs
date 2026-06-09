mod audio_extractor;
mod comm;
mod config;
mod diarization;
mod ffmpeg;
mod output;
mod pipeline;
mod stt;
mod translation;
mod vad_ffmpeg;

use comm::{PipelineConfig, PipelineContext, ProgressUpdate};
use llama_cpp_2::llama_backend::LlamaBackend;
use log::{LevelFilter, Log, Metadata, Record};
use serde::Serialize;
use std::io::Write;
use std::panic;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use tauri::Emitter;

// ---- LlamaBackend (for translation module, init once) ----

static LLAMA_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static LLAMA_INIT_ERR: Mutex<Option<String>> = Mutex::new(None);

pub fn get_llama_backend() -> Result<&'static LlamaBackend, String> {
    if let Some(backend) = LLAMA_BACKEND.get() {
        return Ok(backend);
    }
    if let Some(err) = LLAMA_INIT_ERR.lock().unwrap().as_ref() {
        return Err(err.clone());
    }
    match LlamaBackend::init() {
        Ok(b) => {
            let _ = LLAMA_BACKEND.set(b);
            Ok(LLAMA_BACKEND.get().unwrap())
        }
        Err(e) => {
            let msg = format!("Ошибка инициализации LlamaBackend: {:#}", e);
            *LLAMA_INIT_ERR.lock().unwrap() = Some(msg.clone());
            Err(msg)
        }
    }
}

// ---- Pipeline busy flag (prevent double run) ----

static PIPELINE_BUSY: AtomicBool = AtomicBool::new(false);

#[tauri::command]
fn is_pipeline_busy() -> bool {
    PIPELINE_BUSY.load(Ordering::SeqCst)
}

struct PipelineGuard;

impl PipelineGuard {
    fn try_acquire() -> Option<Self> {
        if PIPELINE_BUSY.swap(true, Ordering::SeqCst) {
            None
        } else {
            Some(Self)
        }
    }
}

impl Drop for PipelineGuard {
    fn drop(&mut self) {
        PIPELINE_BUSY.store(false, Ordering::SeqCst);
    }
}

// ---- Global AppHandle (for log events) ----

static APP_HANDLE: OnceLock<tauri::AppHandle> = OnceLock::new();

// ---- Log buffer ----

#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub level: String,
    pub message: String,
}

static LOG_BUF: Mutex<Vec<LogEntry>> = Mutex::new(Vec::new());

fn log_paths() -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();

    // 1) AppData/Local/dubvidtra2/last_logs.log
    if let Some(data_dir) = dirs::data_local_dir() {
        paths.push(data_dir.join("dubvidtra2").join("last_logs.log"));
    }

    // 2) рядом с exe
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
    {
        paths.push(exe_dir.join("last_logs.log"));
    }

    // 3) рядом с CWD
    if let Ok(cwd) = std::env::current_dir() {
        if !paths.iter().any(|p| p.parent() == Some(&cwd)) {
            paths.push(cwd.join("last_logs.log"));
        }
    }

    // 4) Temp
    let tmp = std::env::temp_dir().join("dubvidtra2").join("last_logs.log");
    if !paths.contains(&tmp) {
        paths.push(tmp);
    }

    paths
}

fn log_to_file(msg: &str) {
    for path in &log_paths() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
        {
            let _ = writeln!(f, "{}", msg);
            let _ = f.flush();
        }
    }
}

fn timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let s = d.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        (s / 3600) % 24,
        (s / 60) % 60,
        s % 60
    )
}

// ---- Custom Logger (file + stdout + buffer + Tauri events) ----

struct AppLogger;

impl Log for AppLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        let ts = timestamp();
        let msg = format!("{} [{}] {}", ts, record.level(), record.args());
        let plain = format!("{}", record.args());

        // stderr (всегда работает)
        eprintln!("{}", msg);

        // file (все пути)
        log_to_file(&msg);

        // memory buffer
        let entry = LogEntry {
            level: record.level().to_string(),
            message: plain,
        };
        if let Ok(mut buf) = LOG_BUF.lock() {
            if buf.len() >= 5000 {
                buf.remove(0);
            }
            buf.push(entry.clone());
        }

        // emit to frontend
        if let Some(handle) = APP_HANDLE.get() {
            let _ = handle.emit("new-log", entry);
        }
    }

    fn flush(&self) {}
}

// ---- App State ----

struct AppState {
    config: Mutex<config::AppConfig>,
}

// ---- Commands ----

#[tauri::command]
fn process_video(
    input_path: String,
    gguf_model_path: Option<String>,
    output_format: Option<String>,
    vad_threshold_db: Option<String>,
    sherpa_onnx_dir: Option<String>,
    app_handle: tauri::AppHandle,
    state: tauri::State<AppState>,
) -> Result<String, String> {
    let _guard = match PipelineGuard::try_acquire() {
        Some(g) => g,
        None => return Err("Pipeline уже запущен".to_string()),
    };

    let app_cfg = state.config.lock().unwrap().clone();
    let onnx_dir = sherpa_onnx_dir.or(app_cfg.sherpa_onnx_dir);

    let cfg = PipelineConfig {
        input_path,
        output_format: output_format.unwrap_or_default(),
        gguf_model_path,
        ffmpeg_path: app_cfg.ffmpeg_path.clone(),
        vad_threshold_db,
        sherpa_onnx_dir: onnx_dir,
    };

    let handle = app_handle.clone();
    std::thread::spawn(move || {
        let ctx = PipelineContext::new(cfg);
        let result = pipeline::run(ctx);

        match result {
            Ok(result) => {
                let path = result.output_path.unwrap_or_default();
                log::info!("process_video: готово, output={}", path);
                handle
                    .emit(
                        "pipeline-progress",
                        ProgressUpdate {
                            stage: "done".to_string(),
                            percent: 100.0,
                            result_path: Some(path),
                        },
                    )
                    .ok();
            }
            Err(e) => {
                log::error!("process_video: {}", e);
                handle
                    .emit(
                        "pipeline-progress",
                        ProgressUpdate {
                            stage: "error".to_string(),
                            percent: 0.0,
                            result_path: None,
                        },
                    )
                    .ok();
            }
        }
    });

    Ok("started".to_string())
}

#[tauri::command]
fn get_config(state: tauri::State<AppState>) -> config::AppConfig {
    state.config.lock().unwrap().clone()
}

#[tauri::command]
fn save_config(state: tauri::State<AppState>, cfg: config::AppConfig) -> Result<(), String> {
    config::save(&cfg).map_err(|e| e.to_string())?;
    *state.config.lock().unwrap() = cfg;
    Ok(())
}

#[tauri::command]
fn get_logs() -> Vec<LogEntry> {
    LOG_BUF.lock().unwrap().clone()
}

#[tauri::command]
fn get_log_paths() -> Vec<String> {
    log_paths().into_iter().map(|p| p.to_string_lossy().to_string()).collect()
}

// ---- Entry ----

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Init panic hook FIRST — до всего остального
    panic::set_hook(Box::new(move |info| {
        let msg = format!("PANIC: {}", info);
        eprintln!("{}", msg);
        log_to_file(&msg);
        if let Some(location) = info.location() {
            let loc = format!("  at {}", location);
            eprintln!("{}", loc);
            log_to_file(&loc);
        }
    }));

    // Set up logger
    if let Err(e) = log::set_logger(&AppLogger) {
        let msg = format!("WARN: log::set_logger failed: {}", e);
        eprintln!("{}", msg);
        log_to_file(&msg);
    }
    log::set_max_level(LevelFilter::Info);

    // Write startup messages — теперь через работающий логгер
    log::info!("=== DubVidTra2 Start ===");
    for p in log_paths() {
        log::info!("Log file: {}", p.display());
    }

    let app_cfg = config::load();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_process::init())
        .manage(AppState {
            config: Mutex::new(app_cfg),
        })
        .setup(|app| {
            APP_HANDLE.set(app.handle().clone()).ok();
            log::info!("=== App initialized ===");

            let app_handle = app.handle().clone();
            if let Ok(video) = std::env::var("DUBVID_TEST_VIDEO") {
                if !video.is_empty() {
                    log::info!("=== AUTO: starting pipeline with {}", video);
                    std::thread::spawn(move || {
                        let _guard = match PipelineGuard::try_acquire() {
                            Some(g) => g,
                            None => {
                                log::warn!("AUTO: pipeline уже запущен, пропускаем");
                                return;
                            }
                        };
                        let cfg = config::load();
                        let sherpa_dir = cfg.sherpa_onnx_dir.clone().unwrap_or_default();
                        let translate = cfg.gguf_model_path.clone().unwrap_or_default();
                        if sherpa_dir.is_empty() || translate.is_empty() {
                            log::info!("AUTO: missing models in config, skipping (need sherpa_onnx_dir + gguf_model_path)");
                            return;
                        }
                        let pcfg = comm::PipelineConfig {
                            input_path: video,
                            output_format: "mp4".to_string(),
                            gguf_model_path: Some(translate),
                            ffmpeg_path: cfg.ffmpeg_path.clone(),
                            vad_threshold_db: cfg.vad_threshold_db.clone(),
                            sherpa_onnx_dir: cfg.sherpa_onnx_dir,
                        };
                        let ctx = comm::PipelineContext::new(pcfg);
                        log::info!("AUTO: running pipeline...");
                        let _ = app_handle.emit(
                            "pipeline-progress",
                            comm::ProgressUpdate {
                                stage: "started".to_string(),
                                percent: 0.0,
                                result_path: None,
                            },
                        );
                        let result = pipeline::run(ctx);
                        match result {
                            Ok(res) => {
                                let out = res.output_path.unwrap_or_default();
                                log::info!("AUTO: SUCCESS output={}", out);
                                let _ = app_handle.emit(
                                    "pipeline-progress",
                                    comm::ProgressUpdate {
                                        stage: "done".to_string(),
                                        percent: 100.0,
                                        result_path: Some(out),
                                    },
                                );
                            }
                            Err(e) => {
                                log::error!("AUTO: {:#}", e);
                                let _ = app_handle.emit(
                                    "pipeline-progress",
                                    comm::ProgressUpdate {
                                        stage: "error".to_string(),
                                        percent: 0.0,
                                        result_path: None,
                                    },
                                );
                            }
                        }
                    });
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            is_pipeline_busy,
            process_video,
            get_config,
            save_config,
            get_logs,
            get_log_paths,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::PipelineConfig;

    fn project_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn test_full_pipeline() {
        let video = project_root().join("test").join("for_test.mp4");
        assert!(video.exists(), "Тестовый файл не найден: {:?}", video);

        let cfg = config::load();
        let translate = cfg.gguf_model_path.clone()
            .expect("Нет gguf_model_path в конфиге!");

        eprintln!("[TEST] Video: {}", video.display());
        eprintln!("[TEST] Translate: {}", translate);
        eprintln!("[TEST] Sherpa-ONNX: {:?}", cfg.sherpa_onnx_dir);

        let pipeline_cfg = PipelineConfig {
            input_path: video.to_string_lossy().to_string(),
            output_format: "mp4".to_string(),
            gguf_model_path: Some(translate),
            ffmpeg_path: None,
            vad_threshold_db: None,
            sherpa_onnx_dir: cfg.sherpa_onnx_dir.clone(),
        };
        let ctx = PipelineContext::new(pipeline_cfg);

        let result = pipeline::run(ctx);
        if let Err(ref e) = result {
            eprintln!("[TEST] PIPELINE ERROR: {:#}", e);
        }
        assert!(result.is_ok(), "Pipeline failed: {:#}", result.err().unwrap());
        eprintln!("[TEST] SUCCESS: {:?}", result.unwrap().output_path);
    }
}
