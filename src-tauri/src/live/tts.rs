//! Озвучка live-режима (шаг 4 PRP): RU-фразы → WAV через CosyVoice3.
//!
//! Один движок на live-сессию (cosyvoice3-tts-rl), голос — клон-референс WAV,
//! который юзер выбирает на вкладке (у пресета voice_type=clone, референс
//! обязателен; один запуск движка = один голос — tts/mod.rs PASS 1). Движок
//! поднимается в фоне; синтез блокирует ~1–2 с на фразу — очередь mpsc.
//!
//! Кто триггерит синтез: ФРОНТ (правило №7 PRP — очередь дубляжа на фронте).
//! Когда playhead подходит к фразе (t0 − LEAD), фронт зовёт команду
//! `live_synth`; готовый WAV отдаётся событием `live-dub {id, t0, dur, url}`,
//! фронт ставит его в свою очередь воспроизведения с skip-правилом и даком.
//! Файлы `tts_<id>.wav` лежат в папке сессии и раздаются loopback-сервером
//! (/tts/<name>); удаляются через delay+30 с (гонка с воспроизведением).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use tauri::{Emitter, Manager};

/// Публичный счётчик синтезированных реплик (для env-селфтеста).
pub static DUB_N: AtomicU64 = AtomicU64::new(0);

/// Запрос на синтез (от команды live_synth).
pub struct TtsJob {
    pub id: u64,
    pub t0: f64,
    pub t1: f64,
    pub ru: String,
}

/// Запуск TTS-воркера. `voice_wav` пуст → озвучка выключена (tts_state="off",
/// канал жив, live_synth отвечает ошибкой — фронт показывает подсказку).
/// `delay_s` — задержка показа: файлы живут delay+30 c, потом удаляются.
pub fn start(
    app: tauri::AppHandle,
    cancel: Arc<AtomicBool>,
    session_dir: PathBuf,
    voice_wav: String,
    delay_s: f64,
) -> mpsc::Sender<TtsJob> {
    // Без ограничения ёмкости: очередь дренажится в самом воркере (устаревшие
    // фразы файлового режима выбрасываются), лимит здесь только мешал бы.
    let (tx, rx) = mpsc::channel::<TtsJob>();
    let _ = std::thread::Builder::new().name("live-tts".into()).spawn(move || {
        worker(app, cancel, rx, session_dir, voice_wav, delay_s);
    });
    tx
}

fn worker(
    app: tauri::AppHandle,
    cancel: Arc<AtomicBool>,
    rx: mpsc::Receiver<TtsJob>,
    session_dir: PathBuf,
    voice_wav: String,
    delay_s: f64,
) {
    let set_state = |s: &str| crate::live::status_slot().lock().unwrap().tts_state = s.into();

    if voice_wav.trim().is_empty() {
        log::info!("Live-TTS: голос не выбран — озвучка выключена (сабы работают)");
        set_state("off");
        // Канал жив до teardown: live_synth отвечает ошибкой, фронт подскажет.
        while !cancel.load(Ordering::SeqCst) {
            match rx.recv_timeout(Duration::from_millis(500)) {
                Ok(_) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
        return;
    }

    set_state("loading");
    if let Err(e) = ensure_engine(&app, &voice_wav) {
        log::error!("Live-TTS: движок не поднялся: {e}");
        set_state(&format!("failed: {e}"));
        return;
    }
    if cancel.load(Ordering::SeqCst) {
        return;
    }
    set_state("ready");
    log::info!("Live-TTS: движок готов (голос {voice_wav})");

    let mut pending_delete: Vec<(String, Instant)> = Vec::new();
    let lifetime = Duration::from_secs((delay_s + 30.0).max(30.0) as u64);

    loop {
        // Удаление озвученных файлов (не блокирует очередь).
        let now = Instant::now();
        let expired: Vec<String> = pending_delete
            .iter()
            .filter(|(_, dl)| now >= *dl)
            .map(|(f, _)| f.clone())
            .collect();
        pending_delete.retain(|(_, dl)| now < *dl);
        for f in expired {
            if let Err(e) = std::fs::remove_file(session_dir.join(&f)) {
                log::warn!("Live-TTS: не удалить {f}: {e}");
            }
        }

        let job = match rx.recv_timeout(Duration::from_millis(300)) {
            Ok(j) => j,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if cancel.load(Ordering::SeqCst) {
                    return;
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        };
        if cancel.load(Ordering::SeqCst) {
            return;
        }

        // Дренаж устаревших задач: файловый режим идёт в 30–100× реалтайма и
        // накапливает фразы, которые к моменту синтеза уже не успеют в свои
        // окна воспроизведения. Синтезируем только самую свежую, остальные
        // отдаём с пустым url (фронт помечает failed — RU-сабы остаются).
        let mut job = job;
        loop {
            match rx.try_recv() {
                Ok(next) => {
                    log::info!("Live-TTS: пропуск устаревшей фразы [{:.2}] (дренаж)", job.t0);
                    emit_dub_empty(&app, &job);
                    job = next;
                }
                Err(_) => break,
            }
        }

        // Цифры → слова (CosyVoice3 цифры не читает; промпт-правило уже было,
        // конвертер ловит хвост — PRP шаг 4).
        let text = super::digits::digits_to_words(&job.ru);
        let t_start = Instant::now();
        let state = app.state::<plugin_speech_gen_engine::PluginState>();
        let speak_fut = state.tts.speak(&text, "", "", "", 1.0, true, "ru", "en", None);
        let wav = match crate::block_on(speak_fut) {
            Ok((bytes, _timing)) => bytes,
            Err(e) => {
                log::warn!("Live-TTS: синтез '{}' не удался: {e}", job.ru);
                continue;
            }
        };
        let dt = t_start.elapsed().as_secs_f64();
        let dur = crate::tts::wav_sr_and_duration_bytes(&wav)
            .map(|(_, d)| d)
            .unwrap_or(0.0);

        let fname = format!("tts_{}.wav", job.id);
        if let Err(e) = std::fs::write(session_dir.join(&fname), &wav) {
            log::error!("Live-TTS: не записать {fname}: {e}");
            continue;
        }
        DUB_N.fetch_add(1, Ordering::SeqCst);
        pending_delete.push((fname.clone(), Instant::now() + lifetime));
        log::info!(
            "Live-TTS: [{:.2}] '{}…' → {} ({:.1} с аудио, {:.1} с синтез)",
            job.t0,
            text.chars().take(40).collect::<String>(),
            fname,
            dur,
            dt
        );

        // URL файла на loopback-сервере: порт берём из статуса (он уже
        // установлен — TTS-воркер стартует после server::spawn).
        let port = crate::live::status_slot().lock().unwrap().port;
        if port == 0 {
            log::warn!("Live-TTS: порт сервера ещё 0 — событие live-dub не отправлено");
            continue;
        }
        if let Err(e) = app.emit(
            "live-dub",
            serde_json::json!({
                "id": job.id,
                "t0": job.t0,
                "t1": job.t1,
                "dur": dur,
                "url": format!("http://127.0.0.1:{port}/tts/{fname}"),
            }),
        ) {
            log::warn!("Live-TTS: не отправить live-dub: {e}");
        }
    }
}

/// Отдача задачи без озвучки (дренаж): пустой url → фронт помечает failed.
fn emit_dub_empty(app: &tauri::AppHandle, job: &TtsJob) {
    let port = crate::live::status_slot().lock().unwrap().port;
    if port == 0 {
        return;
    }
    let _ = app.emit(
        "live-dub",
        serde_json::json!({
            "id": job.id,
            "t0": job.t0,
            "t1": job.t1,
            "dur": 0.0,
            "url": "",
        }),
    );
}

/// Подъём TTS-движка: resolve пресета/моделей как в файловом пайплайне
/// (tts/mod.rs:640-680), затем ensure плагина speech с клон-референсом.
fn ensure_engine(app: &tauri::AppHandle, voice_wav: &str) -> Result<(), String> {
    let state = app.state::<plugin_speech_gen_engine::PluginState>();
    let tss = plugin_speech_gen_engine::tts_settings::load(app);
    let models_dir = if tss.models_dir.trim().is_empty() {
        plugin_speech_gen_engine::download::default_models_dir()
    } else {
        std::path::PathBuf::from(tss.models_dir.clone())
    };
    let engine_exe = crate::pick_engine_exe()?;
    let preset_id = std::env::var("DEEDUB_TTS_PRESET")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            let s = tss.preset.trim();
            if s.is_empty() { None } else { Some(s.to_string()) }
        })
        .unwrap_or_else(|| "cosyvoice3-tts-rl".to_string());
    let tts_backend = plugin_speech_gen_engine::download::preset_backend(&preset_id)
        .map(|b| b.to_string())
        .unwrap_or_else(|| preset_id.clone());
    log::info!(
        "Live-TTS: движок {} / модели {} / пресет {} (backend {})",
        engine_exe,
        models_dir.display(),
        preset_id,
        tts_backend
    );
    crate::block_on(state.tts.ensure(
        app,
        &engine_exe,
        &tts_backend,
        &models_dir.to_string_lossy().to_string(),
        &preset_id,
        voice_wav,
    ))
    .map_err(|e| format!("{e}"))
}
