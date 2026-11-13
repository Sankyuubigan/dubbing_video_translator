//! Перевод фраз live-режима (шаг 3 PRP): финалы STT → RU-текст.
//!
//! Один LlmSession на live-сессию (как в файловом пайплайне: translate и
//! verify жили в одной сессии ради отсутствия повторной загрузки модели).
//! Движок поднимается в фоне: live_start не ждёт его, фразы уходят в очередь
//! std::sync::mpsc и переводятся по мере готовности (0.3–1 с на фразу,
//! скорость MT >> темпа речи — очередь не растёт).
//!
//! Промпт — ССОТ выбранного формата (`prompt_style` из config.toml, тот же
//! механизм, что у файлового пайплайна): chat (system + предыдущие пары
//! EN/RU, гендер «unknown») или insttrans (канонический формат семейства
//! Index-Translate). Ответ прогоняется через `translation::clean_output`
//! — та же постобработка, что в пайплайне (CoT-блоки, префиксы, цензурные
//! маски `*`).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use tauri::Emitter;

/// Публичный счётчик переведённых фраз (для env-селфтеста live-режима).
pub static RU_N: AtomicU64 = AtomicU64::new(0);

/// Фраза на перевод (t0/t1 — та же шкала, что в live-subs: эпоха для
/// HLS-источников с PDT, секунды файла для file/VOD-источников).
pub struct PhraseJob {
    pub t0: f64,
    pub t1: f64,
    pub en: String,
}

/// Запуск переводчика: открывает LlmSession в фоне, возвращает очередь.
/// `cancel` — флаг teardown'а сессии: между фразами и в ожидании очереди
/// поток выходит, при выходе Drop у LlmSession гасит llama-server (§6.5).
pub fn start(
    app: tauri::AppHandle,
    cancel: Arc<AtomicBool>,
) -> mpsc::Sender<PhraseJob> {
    // Без ограничения ёмкости: MT (~0.3-1 с/фраза) быстрее речи, а в
    // файловом режиме STT гонит фразы в 30-100× реалтайма — очередь сама
    // рассасывается, и ограничитель только ломал бы порядок.
    let (tx, rx) = mpsc::channel::<PhraseJob>();
    let _ = std::thread::Builder::new().name("live-mt".into()).spawn(move || {
        worker(app, cancel, rx);
    });
    tx
}

fn worker(
    app: tauri::AppHandle,
    cancel: Arc<AtomicBool>,
    rx: mpsc::Receiver<PhraseJob>,
) {
    {
        let st = crate::live::status_slot();
        st.lock().unwrap().mt_state = "loading".into();
    }

    // LlmSession::open блокирует на десятки секунд (грузит GGUF) — сам поток
    // и есть «spawn_blocking», tokio не занят.
    let session = match crate::llm::LlmSession::open(&app) {
        Ok(s) => s,
        Err(e) => {
            log::error!("Live-MT: сессия LLM не открылась: {e}");
            crate::live::status_slot().lock().unwrap().mt_state = format!("failed: {e}");
            return;
        }
    };
    if cancel.load(Ordering::SeqCst) {
        return;
    }

    let cfg = crate::config::load();
    let style = crate::translation::PromptStyle::resolve(
        Some(cfg.prompt_style.as_str()),
        session.model_path(),
    );
    log::info!(
        "Live-MT: переводчик готов ({:?}, модель {})",
        style,
        session.model_path()
    );
    crate::live::status_slot().lock().unwrap().mt_state = "ready".into();

    // Контекст: последние 6 пар EN/RU (как в пайплайне; source-side контекст
    // важнее целевого — research_tower_plus_9b).
    let mut history: Vec<(String, String)> = Vec::new();

    loop {
        if cancel.load(Ordering::SeqCst) {
            log::info!("Live-MT: остановлен");
            return;
        }
        let job = match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(j) => j,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::info!("Live-MT: очередь закрыта — поток завершён");
                return;
            }
        };

        let messages = match style {
            crate::translation::PromptStyle::Chat => {
                crate::translation::build_messages(&job.en, &history, &[], "unknown")
            }
            crate::translation::PromptStyle::InstTrans => {
                crate::translation::build_messages_insttrans(&job.en, "unknown", &[])
            }
        };

        let t_start = std::time::Instant::now();
        let ru = match session.generate(&messages, None) {
            Ok(res) => crate::translation::clean_output(&res.text),
            Err(e) => {
                log::warn!("Live-MT: фраза не переведена ({e}) — RU-саб пропущен");
                continue;
            }
        };
        let dt = t_start.elapsed().as_secs_f64();

        if ru.trim().is_empty() {
            log::warn!("Live-MT: пустой перевод фразы '{}' — пропущен", job.en);
            continue;
        }
        RU_N.fetch_add(1, Ordering::SeqCst);
        log::info!(
            "Live-MT: [{:.2}..{:.2}] {:.1}мс {} → {}",
            job.t0,
            job.t1,
            dt * 1000.0,
            job.en,
            ru
        );
        if let Err(e) = app.emit(
            "live-ru",
            serde_json::json!({ "t0": job.t0, "t1": job.t1, "ru": ru }),
        ) {
            log::warn!("Live-MT: не отправить live-ru: {e}");
        }

        history.push((job.en, ru));
        if history.len() > 6 {
            history.remove(0);
        }
    }
}
