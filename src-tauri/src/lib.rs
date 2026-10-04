mod audio_extractor;
pub mod comm;
pub mod config;
mod diarization;
mod ffmpeg;
mod llm;
pub mod output;
pub mod paths;
pub mod pipeline;
mod stt;
mod translation;
pub mod tts;
mod verification;

use comm::{PipelineConfig, PipelineContext, ProgressUpdate, SubtitleChunk};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tauri::Emitter;

// ---- Pipeline busy flag (prevent double run) & cancel ----

static PIPELINE_BUSY: AtomicBool = AtomicBool::new(false);
static PIPELINE_CANCEL: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Флаг отмены пайплайна. `Arc`, потому что движок LLM (плагин
/// `tauri-plugin-llama-engine`) принимает `Arc<AtomicBool>` и прерывает
/// генерацию по нему — один и тот же флаг видит и пайплайн, и движок.
pub fn cancel_flag() -> Arc<AtomicBool> {
    PIPELINE_CANCEL
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

#[tauri::command]
fn is_pipeline_busy() -> bool {
    PIPELINE_BUSY.load(Ordering::SeqCst)
}

#[tauri::command]
fn cancel_pipeline() -> bool {
    let was_set = cancel_flag().swap(true, Ordering::SeqCst);
    log::info!("Cancel: пользователь запросил отмену");
    !was_set // true если флаг был установлен сейчас
}

pub fn is_cancelled() -> bool {
    cancel_flag().load(Ordering::SeqCst)
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
        cancel_flag().store(false, Ordering::SeqCst);
    }
}

// ---- Global AppHandle (for log events) ----

static APP_HANDLE: OnceLock<tauri::AppHandle> = OnceLock::new();

/// Глобальный AppHandle. Устанавливается в `setup` приложения; в чистом
/// консольном запуске (без GUI) отсутствует → `None`.
pub fn app_handle() -> Option<&'static tauri::AppHandle> {
    APP_HANDLE.get()
}

// ---- Global async runtime (движки плагина — async: ensure/transcribe/speak) ----

/// Блокирующий прогон async-фьючи на глобальном tokio runtime.
pub fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    let rt = RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("failed to build tokio runtime")
    });
    rt.block_on(fut)
}

/// Подбирает исполняемый файл CrispASR: сохранённый в настройках backend,
/// иначе cuda13 → cuda → cpu (первый существующий).
pub fn pick_engine_exe() -> Result<String, String> {
    let app = APP_HANDLE
        .get()
        .ok_or_else(|| "AppHandle не инициализирован".to_string())?;
    let tss = tauri_plugin_speech::tts_settings::load(app);

    let mut candidates: Vec<String> = Vec::new();
    let saved = tss.engine_backend.trim();
    if !saved.is_empty() {
        candidates.push(saved.to_string());
    }
    for b in ["cuda13", "cuda", "cpu"] {
        if !candidates.iter().any(|c| c == b) {
            candidates.push(b.to_string());
        }
    }

    for b in &candidates {
        let p = tauri_plugin_speech::download::resolve_engine_exe(&tss.engine_dir, b);
        if p.exists() {
            return Ok(p.to_string_lossy().to_string());
        }
    }
    Err(format!(
        "crispasr.exe не найден (искал engines: {}). Проверьте TTS-настройки.",
        candidates.join(", ")
    ))
}

/// Ищет GGUF-модель STT. Приоритет:
/// 1. env `DEEDUB_STT_MODEL` (явный путь);
/// 2. известные пути в `D:\nn\models\stt`;
/// 3. поиск по `D:\nn\models\stt` с предпочтением имени "parakeet".
pub fn resolve_stt_model() -> Result<String, String> {
    if let Ok(p) = std::env::var("DEEDUB_STT_MODEL") {
        if !p.is_empty() {
            if std::path::Path::new(&p).exists() {
                log::info!("STT: модель из DEEDUB_STT_MODEL: {p}");
                return Ok(p);
            }
            return Err(format!(
                "STT: DEEDUB_STT_MODEL задан, но файл не существует: {p}"
            ));
        }
    }

    let candidates = [
        r"D:\nn\models\stt\parakeet-tdt-0.6b-v3\parakeet-tdt-0.6b-v3-q4_k.gguf",
        r"D:\nn\models\stt\parakeet-tdt-0.6b-v3\parakeet-tdt-0.6b-v3-fp16.gguf",
        r"D:\nn\models\stt\parakeet-tdt-0.6b-v3\parakeet-tdt-0.6b-v3-q8_0.gguf",
    ];
    for c in &candidates {
        if std::path::Path::new(c).exists() {
            log::info!("STT: модель найдена: {c}");
            return Ok(c.to_string());
        }
    }

    let root = std::path::Path::new(r"D:\nn\models\stt");
    if root.is_dir() {
        if let Some(p) = find_stt_gguf(root, "parakeet") {
            log::info!("STT: модель найдена (поиск): {}", p.display());
            return Ok(p.to_string_lossy().to_string());
        }
    }

    Err("STT: GGUF-модель не найдена. Скачайте parakeet-tdt-0.6b-v3-q4_k.gguf \
         в D:/nn/models/stt (или укажите DEEDUB_STT_MODEL)".to_string())
}

/// Одноуровневый поиск GGUF под `root`; файлы, чьё имя содержит `prefer`, берутся первыми.
fn find_stt_gguf(root: &std::path::Path, prefer: &str) -> Option<std::path::PathBuf> {
    let mut matches: Vec<std::path::PathBuf> = Vec::new();
    let dirs = std::fs::read_dir(root).ok()?;
    for entry in dirs.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if let Ok(sub) = std::fs::read_dir(&p) {
                for e in sub.flatten() {
                    let f = e.path();
                    if f.is_file() && f.extension().and_then(|x| x.to_str()) == Some("gguf") {
                        matches.push(f);
                    }
                }
            }
        } else if p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("gguf") {
            matches.push(p);
        }
    }
    if let Some(pos) = matches
        .iter()
        .position(|m| m.file_name().and_then(|n| n.to_str()).unwrap_or("").contains(prefer))
    {
        return Some(matches.remove(pos));
    }
    matches.into_iter().next()
}

// ---- App State ----

struct AppState {
    config: Mutex<config::AppConfig>,
}

// ---- Commands ----

#[tauri::command]
fn process_video(
    input_path: String,
    output_format: Option<String>,
    enable_dubbing: Option<bool>,
    app_handle: tauri::AppHandle,
    state: tauri::State<AppState>,
) -> Result<String, String> {
    let _guard = match PipelineGuard::try_acquire() {
        Some(g) => g,
        None => return Err("Pipeline уже запущен".to_string()),
    };

    let app_cfg = state.config.lock().unwrap().clone();

    let cfg = PipelineConfig {
        input_path,
        output_format: output_format.unwrap_or_default(),
        ffmpeg_path: app_cfg.ffmpeg_path.clone(),
        enable_dubbing: enable_dubbing.unwrap_or(false),
        mix_volume: app_cfg.mix_volume,
        prompt_style: Some(app_cfg.prompt_style.clone()),
    };

    let handle = app_handle.clone();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let ctx = PipelineContext::new(cfg);
            pipeline::run(ctx)
        }));

        match result {
            Ok(Ok(result)) => {
                let path = result.output_path.unwrap_or_default();
                log::info!("process_video: готово, output={}", path);
                handle
                    .emit(
                        "pipeline-progress",
                        ProgressUpdate {
                            stage: "done".to_string(),
                            percent: 100.0,
                            result_path: Some(path),
                            error_message: None,
                        },
                    )
                    .ok();
            }
            Ok(Err(e)) => {
                let err_msg = format!("{:#}", e);
                log::error!("process_video: {}", err_msg);
                handle
                    .emit(
                        "pipeline-progress",
                        ProgressUpdate {
                            stage: "error".to_string(),
                            percent: 0.0,
                            result_path: None,
                            error_message: Some(err_msg),
                        },
                    )
                    .ok();
            }
            Err(panic) => {
                let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                    format!("Внутренняя ошибка: {}", s)
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    format!("Внутренняя ошибка: {}", s)
                } else {
                    "Внутренняя ошибка: паника в pipeline".to_string()
                };
                log::error!("process_video: PANIC: {}", msg);
                handle
                    .emit(
                        "pipeline-progress",
                        ProgressUpdate {
                            stage: "error".to_string(),
                            percent: 0.0,
                            result_path: None,
                            error_message: Some(msg),
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

/// Делает `path` активной моделью перевода и возвращает её в списке моделей.
///
/// Пишет в конфиг плагина движка (`app_config.json`, ключ `last_model`) — тот же
/// файл, откуда модель читается при запуске (`llm::resolve_model_path`), и
/// поэтому единственный источник правды (core §2.1). Никакой второй
/// «выбранной модели» в нашем конфиге не появляется специально: иначе два
/// места правды разъедутся, и UI будет показывать не ту модель, что поедет.
///
/// Плагин не меняем: `load_config`/`save_config` у него и так `pub`, хост уже
/// пользуется ими кросс-кратом.
#[tauri::command]
fn set_translation_model(app: tauri::AppHandle, path: String) -> Result<String, String> {
    let path = path.trim().to_string();
    if path.is_empty() {
        return Err("Путь к модели перевода не указан".to_string());
    }
    if !std::path::Path::new(&path).is_file() {
        return Err(format!("Файл модели не найден: {path}"));
    }

    let mut cfg = tauri_plugin_llama_engine::engine::load_config(&app);
    if !cfg.models.iter().any(|m| m == &path) {
        // Файл на диске есть, но в реестре плагина его нет — значит модель
        // добавлена мимо панели (или панель её уже чистила). Регистрируем здесь
        // же: иначе выбор из UI нельзя было бы вернуть в реестр, а `remove_model`
        // при следующем открытии панели решит, что модели не существует.
        cfg.models.push(path.clone());
    }
    cfg.last_model = Some(path.clone());
    tauri_plugin_llama_engine::engine::save_config(&app, &cfg)
        .map_err(|e| format!("Не удалось сохранить выбор модели: {e}"))?;
    log::info!("LLM: активная модель перевода — {path}");
    Ok(path)
}

/// Список моделей перевода, доступных для выбора: пути из конфига плагина плюс
/// отметка активной. Отдельной команды плагина для этого нет — там есть
/// `get_engine_config`, но он отдаёт ещё и параметры сэмплирования, а здесь нужен
/// только список для селекта.
#[tauri::command]
fn list_translation_models(app: tauri::AppHandle) -> Result<Vec<ModelChoice>, String> {
    let cfg = tauri_plugin_llama_engine::engine::load_config(&app);
    let mut models: Vec<ModelChoice> = cfg
        .models
        .iter()
        .map(|p| ModelChoice {
            path: p.clone(),
            name: short_model_name(p),
            exists: std::path::Path::new(p).is_file(),
        })
        .collect();
    // Активная модель может отсутствовать в реестре (например, её добавили
    // до обновления конфига). Без неё селект показал бы «пусто» при реально
    // работающей модели — показываем и её.
    let active = cfg.last_model.clone().unwrap_or_default();
    if !active.is_empty() && !models.iter().any(|m| m.path == active) {
        models.insert(
            0,
            ModelChoice {
                path: active.clone(),
                name: short_model_name(&active),
                exists: std::path::Path::new(&active).is_file(),
            },
        );
    }
    Ok(models)
}

/// Строка для селекта: имя файла без пути и без расширения — длинные пути
/// `uncen\gemma-4-12B\gemma-4-12B-…gguf` в `<option>` нечитаемы.
fn short_model_name(path: &str) -> String {
    let file = path
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(path)
        .trim_end_matches(".gguf");
    if file.is_empty() {
        path.to_string()
    } else {
        file.to_string()
    }
}

#[derive(serde::Serialize)]
struct ModelChoice {
    path: String,
    name: String,
    /// Файл на месте? Реестр плагина может содержать удалённые пути, и селект
    /// обязан это показывать, иначе пользователь выберет модель и получит ошибку
    /// только в момент перевода.
    exists: bool,
}

// ---- Logger setup (shared between GUI and headless) ----

/// Логирование — плагин `tauri-plugin-logs` (core §2.5.2): он ставит
/// `log::Log`, пишет `deedub.log` рядом с exe, зеркалит в `test/last_logs.txt`
/// и шлёт строки в UI по событию `logs:message`. Хост только подключает
/// плагин в `base_builder` и вызывает `tauri_plugin_logs::early_init`
/// первой строкой входа (там же ловится паника до создания приложения).
pub const LOG_FILE_NAME: &str = "deedub.log";

// ---- Entry ----

/// Единый Builder: плагины + AppState + команды. Общий для GUI и headless.
fn base_builder(app_cfg: config::AppConfig) -> tauri::Builder<tauri::Wry> {
    tauri::Builder::default()
        // Логи первыми: остальные плагины пишут через `log::` (core §2.5.2).
        .plugin(tauri_plugin_logs::init())
        // Единый движок скачивания — обязателен для плагина движка LLM.
        .plugin(tauri_plugin_downloader::init())
        // Движок LLM: llama-server отдельным процессом, каталог моделей,
        // выбор активной модели (desktop §6.7).
        .plugin(tauri_plugin_llama_engine::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_speech::init())
        .manage(AppState {
            config: Mutex::new(app_cfg),
        })
        .invoke_handler(tauri::generate_handler![
            is_pipeline_busy,
            cancel_pipeline,
            process_video,
            get_config,
            save_config,
            set_translation_model,
            list_translation_models,
        ])
}

/// Общий для GUI и headless вход: логгер (с ловлей паники до создания
/// приложения) + папка конфига плагина движка LLM.
fn init_logging_and_engine_config() {
    tauri_plugin_logs::early_init(LOG_FILE_NAME);
    tauri_plugin_llama_engine::engine::config::set_app_data_dir_name(APP_DATA_DIR_NAME);
}

/// Имя папки app-data — единый источник для `tauri-plugin-llama-engine`
/// (совпадает с `identifier` в `tauri.conf.json`).
const APP_DATA_DIR_NAME: &str = "com.deedub.desktop";

fn run_pipeline_blocking(
    handle: tauri::AppHandle,
    video: String,
    enable_dubbing: bool,
) -> Result<(), String> {
    let _guard = PipelineGuard::try_acquire()
        .ok_or_else(|| "pipeline уже запущен".to_string())?;
    let cfg = config::load();
    let pcfg = comm::PipelineConfig {
        input_path: video,
        output_format: "mp4".to_string(),
        ffmpeg_path: cfg.ffmpeg_path.clone(),
        enable_dubbing,
        mix_volume: cfg.mix_volume,
        prompt_style: Some(cfg.prompt_style),
    };
    let ctx = comm::PipelineContext::new(pcfg);
    log::info!("run_pipeline_blocking: запуск пайплайна...");
    let _ = handle.emit(
        "pipeline-progress",
        comm::ProgressUpdate {
            stage: "started".to_string(),
            percent: 0.0,
            result_path: None,
            error_message: None,
        },
    );
    let result = pipeline::run(ctx);
    match result {
        Ok(res) => {
            let out = res.output_path.unwrap_or_default();
            log::info!("run_pipeline_blocking: SUCCESS output={}", out);
            let _ = handle.emit(
                "pipeline-progress",
                comm::ProgressUpdate {
                    stage: "done".to_string(),
                    percent: 100.0,
                    result_path: Some(out),
                    error_message: None,
                },
            );
            Ok(())
        }
        Err(e) => {
            let err_msg = format!("{:#}", e);
            log::error!("run_pipeline_blocking: {}", err_msg);
            let _ = handle.emit(
                "pipeline-progress",
                comm::ProgressUpdate {
                    stage: "error".to_string(),
                    percent: 0.0,
                    result_path: None,
                    error_message: Some(err_msg.clone()),
                },
            );
            Err(err_msg)
        }
    }
}

/// Запускает пайплайн отдельным потоком на полном tauri-приложении в фоне.
fn spawn_pipeline_thread(handle: tauri::AppHandle, video: String, enable_dubbing: bool) {
    std::thread::spawn(move || {
        let _ = run_pipeline_blocking(handle.clone(), video, enable_dubbing);
    });
}

/// Полноценное GUI-приложение Tauri.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_logging_and_engine_config();

    let app_cfg = config::load();

    base_builder(app_cfg)
        .setup(|app| {
            APP_HANDLE.set(app.handle().clone()).ok();
            log::info!("=== App initialized ===");

            let app_handle = app.handle().clone();
            if let Ok(video) = std::env::var("DEEDUB_TEST_VIDEO") {
                if !video.is_empty() {
                    log::info!("=== AUTO: starting pipeline with {}", video);
                    spawn_pipeline_thread(app_handle, video, false);
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// Headless-запуск: строит tauri-приложение (нужно для плагинов speech),
/// запускает пайплайн в фоновом потоке и блокируется до завершения.
///
/// ВАЖНО: event loop (tao на Windows) обязан жить на main-потоке — поэтому
/// `Builder::build()` + `app.run(...)` выполняются на вызывающем потоке,
/// а пайплайн крутится в `std::thread::spawn`. Код выхода пишется в атомик
/// и передаётся в `handle.exit(code)`; падение фонового потока → exit(1).
///
/// Возвращает: 0 — успех, 1 — ошибка пайплайна/падение.
pub fn run_headless(video_path: String, enable_dubbing: bool) -> i32 {
    init_logging_and_engine_config();

    let app_cfg = config::load();

    let app = base_builder(app_cfg)
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    APP_HANDLE.set(app.handle().clone()).ok();
    log::info!("=== Headless app initialized ===");

    let handle = app.handle().clone();
    let exit_code: Arc<AtomicI32> = Arc::new(AtomicI32::new(-1));
    let exit_code_inner = Arc::clone(&exit_code);

    std::thread::spawn(move || {
        let code = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            log::info!("=== HEADLESS: starting pipeline with {}", video_path);
            match run_pipeline_blocking(handle.clone(), video_path, enable_dubbing) {
                Ok(()) => 0,
                Err(_) => 1,
            }
        }))
        .unwrap_or_else(|_| {
            log::error!("HEADLESS: паника в потоке пайплайна");
            1
        });
        exit_code_inner.store(code, Ordering::SeqCst);
        std::process::exit(code);
    });

    app.run(|_app_handle, _event| {});

    let code = exit_code.load(Ordering::SeqCst);
    if code < 0 {
        log::error!("HEADLESS: приложение завершилось без кода выхода");
        1
    } else {
        code
    }
}

/// A/B прогон ТОЛЬКО перевода на фиксированном входе: обе модели получают
/// один и тот же список чанков, один и тот же промпт и один и тот же код
/// (`translation::translate_with_session`). Диаризация/STT/TTS не запускаются —
/// иначе вход плавает между прогонами (парakeet+VAD недетерминированы, число
/// спикеров гуляет 4↔5) и сравнение переводов становится бессмысленным.
///
/// Модель выбирается env `DEEDUB_LLM_MODEL` (см. `llm::resolve_model_path`),
/// иначе берётся из конфига плагина.
pub fn run_mt_ab(video_path: String, json_path: String) -> i32 {
    init_logging_and_engine_config();

    let app_cfg = config::load();
    let app = base_builder(app_cfg)
        .build(tauri::generate_context!())
        .expect("error while building tauri application");
    APP_HANDLE.set(app.handle().clone()).ok();
    let _keep_app = app; // держим Wry-приложение живым до конца прогона

    let code = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let raw = match std::fs::read_to_string(&json_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("MT-AB: не читается {}: {e}", json_path);
                return 2;
            }
        };
        let src: Vec<SubtitleChunk> = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("MT-AB: невалидный JSON {}: {e}", json_path);
                return 2;
            }
        };
        log::info!(
            "MT-AB: вход {} чанков из {}",
            src.len(),
            json_path
        );

        // Формат промпта берём из тех же настроек, что и обычный запуск: иначе
        // харнесс мерил бы Index-Translate в одном формате, а приложение в
        // другом, и A/B показывал бы разницу, которой нет в проде. Env
        // `DEEDUB_LLM_PROMPT` всё равно приоритетнее (см. `PromptStyle::resolve`).
        let cfg = config::load();
        let mut ctx = PipelineContext::new(PipelineConfig {
            input_path: video_path.clone(),
            output_format: "mp4".to_string(),
            ffmpeg_path: cfg.ffmpeg_path.clone(),
            enable_dubbing: false,
            mix_volume: 1.0,
            prompt_style: Some(cfg.prompt_style),
        });
        ctx.subtitle_chunks = Some(src);

        // Модель показываем, но сессию НЕ открываем: `translate_with_session`
        // открывает её сам и возвращает живой сессией. Вторая сессия = второй
        // llama-server и 6.5 ГиБ в VRAM (пик 14.3 ГиБ -> просадка скорости).
        match std::env::var("DEEDUB_LLM_MODEL") {
            Ok(p) if !p.trim().is_empty() => eprintln!("MT-AB: модель (env) {p}"),
            _ => eprintln!(
                "MT-AB: модель из настроек плагина (DEEDUB_LLM_MODEL не задан)"
            ),
        }

        let t0 = std::time::Instant::now();
        let (out, _sess) = match crate::translation::translate_with_session(ctx) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("MT-AB: перевод провалился: {e:?}");
                return 4;
            }
        };
        let secs = t0.elapsed().as_secs_f64();
        eprintln!("MT-AB: {} чанков за {:.1} с", out.translated_chunks.as_ref().map_or(0, |v| v.len()), secs);
        log::info!("MT-AB: translate {:.1} с", secs);

        let chunks = out.translated_chunks.clone().unwrap_or_default();
        let out_json = serde_json::to_string_pretty(&chunks).unwrap_or_default();
        if let Err(e) =
            std::fs::write(crate::paths::temp_dir().join("mt_ab_out.json"), out_json)
        {
            eprintln!("MT-AB: не записал вывод: {e}");
            return 4;
        }
        0
    }))
    .unwrap_or_else(|_| {
        eprintln!("MT-AB: паника");
        1
    });
    code
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

    fn run_pipeline(cfg: &config::AppConfig, label: &str) {
        let video = project_root().join("test").join("for_test.mp4");
        assert!(video.exists(), "Тестовый файл не найден: {:?}", video);

        eprintln!("[TEST {}] Video: {}", label, video.display());

        let pipeline_cfg = PipelineConfig {
            input_path: video.to_string_lossy().to_string(),
            output_format: "mp4".to_string(),
            ffmpeg_path: cfg.ffmpeg_path.clone(),
            enable_dubbing: false,
            mix_volume: 1.0,
            prompt_style: Some(cfg.prompt_style.clone()),
        };
        let ctx = PipelineContext::new(pipeline_cfg);

        let result = pipeline::run(ctx);
        if let Err(ref e) = result {
            eprintln!("[TEST {}] PIPELINE ERROR: {:#}", label, e);
        }
        let result = result.expect(&format!("[TEST {}] Pipeline failed", label));
        let output_path = result.output_path.expect("Нет output_path");
        eprintln!("[TEST {}] SUCCESS: {:?}", label, output_path);

        // Читаем сгенерированные субтитры
        let en_srt = output_path.replace(".mp4", "_en.srt");
        let ru_srt = output_path.replace(".mp4", "_ru.srt");
        for srt_path in &[&en_srt, &ru_srt] {
            // Ищем SRT рядом с выходным файлом (если есть) или во временной папке
            let alt_path = std::path::Path::new(&output_path)
                .parent()
                .map(|p| {
                    let name = if srt_path.contains("_en.srt") { "subtitles_en.srt" } else { "subtitles_ru.srt" };
                    p.join(name)
                });

            let srt_content = std::fs::read_to_string(srt_path)
                .or_else(|_| alt_path.map_or(Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no alt")), |p| std::fs::read_to_string(p)))
                .or_else(|_| std::fs::read_to_string(std::path::Path::new(&output_path).with_extension("srt")));

            if let Ok(content) = srt_content {
                if content.trim().is_empty() {
                    eprintln!("[TEST {}] {} — пустой!", label, srt_path);
                } else {
                    let lines: Vec<&str> = content.lines().collect();
                    eprintln!("[TEST {}] {} — {} строк:", label, srt_path, lines.len());
                    for chunk in lines.chunks(4) {
                        let text = chunk.join(" | ");
                        eprintln!("  {}", text);
                    }
                }
            } else {
                eprintln!("[TEST {}] {} — не найден (ищем рядом с output)", label, srt_path);
            }
        }
    }

    #[test]
    #[ignore] // требует запуска внутри tauri-приложения (движок STT — через plugin)
    fn test_pipeline() {
        let cfg = config::load();
        run_pipeline(&cfg, "Pipeline");
    }
}
