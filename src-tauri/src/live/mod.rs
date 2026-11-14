//! Live-режим: HLS-конвейер Twitch с задержкой показа D секунд (PRP
//! «11.11.26 Live-режим Twitch.md» v2, шаг 1 — без STT/MT/TTS).
//!
//! Архитектура (поправки специалиста v2): даунлоадер качает сегменты в момент
//! их появления в weaver-плейлисте, а плеер показывает их через D секунд —
//! рассинхрон невозможен структурно, т.к. весь запас времени уходит на буфер.
//! Скользящее окно = D + 20 с запаса (rebuffer + рестарт hls.js).
//!
//! SSOT: один Session (static SESSION), один STATUS (static). Рекламные
//! сегменты (is_ad) качаются и показываются плееру, но на шаге 2+ не пойдут в
//! STT/MT/TTS. Любая сетевая ошибка логируется с доменом (net_log).

pub mod audio;
pub mod digits;
pub mod engine;
mod hls;
pub mod mt;
pub mod net_log;
pub mod playlist;
pub mod server;
pub mod tts;
pub mod ytdlp;

use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::Emitter;

use playlist::ServedSeg;

/// Задержка показа по умолчанию (слайдер 10-60 с).
const DEFAULT_DELAY_SECS: f64 = 20.0;
/// Запас окна сзади за playhead: rebuffer + рестарт hls.js.
const BACK_MARGIN_SECS: f64 = 20.0;
/// Сколько ошибок опроса подряд → состояние degraded (+ событие live-error).
const DEGRADED_AFTER_FAILS: u32 = 10;
/// Сколько ошибок подряд → попытка перерезолвить поток (usher-токен протух).
const RERESOLVE_AFTER_FAILS: u32 = 30;
/// Нет новых сегментов N секунд → состояние offline (стрим кончился).
const OFFLINE_AFTER_SECS: u64 = 30;
/// Удалять файл сегмента через N секунд после выхода из окна (гонка 404 у
/// hls.js: он мог запросить сегмент уже после выпадения из плейлиста).
const DELETE_LAG_SECS: u64 = 10;

/// Снимок состояния live-сессии для фронта (поля snake_case в JSON).
#[derive(Clone, Debug, Serialize)]
pub struct LiveStatus {
    pub running: bool,
    /// starting | playing | degraded | offline | stopped
    pub state: String,
    pub port: u16,
    pub channel: String,
    pub variant: String,
    pub delay_s: f64,
    pub segments_downloaded: u64,
    pub segments_failed: u64,
    /// Суммарная длительность сегментов в скользящем окне.
    pub served_window_secs: f64,
    pub target_duration: f64,
    /// EXT-X-PROGRAM-DATE-TIME последнего скачанного сегмента (правда времени
    /// потока; замер задержки и будущий маппинг озвучки — по нему).
    pub live_edge_pdt: String,
    /// Состояние STT-движка шага 2: off | loading | ready | failed: …
    pub stt_state: String,
    /// Источник сессии (шаг 5 PRP): twitch | url | file.
    pub source_kind: String,
    /// Режим сессии (13.11.26): "dub" — полный комбайн (сабы+перевод+озвучка),
    /// "watch" — только плеер (медиа-сервер + прокси, ноль движков).
    pub mode: String,
    /// Состояние переводчика (шаг 3): off | loading | ready | failed: …
    pub mt_state: String,
    /// Состояние озвучки (шаг 4): off | loading | ready | failed: …
    pub tts_state: String,
    /// Включён ли дубляж (голос-референс выбран).
    pub dub_enabled: bool,
    /// Тип плеера фронтенда: "hls" (hls.js на /live.m3u8) или "file"
    /// (<video src=/media>; file-источник и YouTube-VOD).
    pub player_kind: String,
    /// Удалённый источник с раздельным аудио: фронт синхронизирует второй
    /// <audio> с /audio (оригинальный звук VOD).
    pub remote_audio: bool,
    pub last_error: String,
}

impl Default for LiveStatus {
    fn default() -> Self {
        Self {
            running: false,
            state: "stopped".into(),
            port: 0,
            channel: String::new(),
            variant: String::new(),
            delay_s: DEFAULT_DELAY_SECS,
            segments_downloaded: 0,
            segments_failed: 0,
            served_window_secs: 0.0,
            target_duration: 0.0,
            live_edge_pdt: String::new(),
            stt_state: "off".into(),
            source_kind: String::new(),
            mode: "dub".into(),
            mt_state: "off".into(),
            tts_state: "off".into(),
            dub_enabled: false,
            player_kind: "hls".into(),
            remote_audio: false,
            last_error: String::new(),
        }
    }
}

struct Session {
    cancel: Arc<AtomicBool>,
    dir: PathBuf,
    tasks: Vec<tauri::async_runtime::JoinHandle<()>>,
    /// STT-движок live-сессии (шаг 2): хранится здесь, teardown гасит
    /// процесс через Drop — VRAM освобождается (desktop §6.5).
    engine: Option<engine::SttEngine>,
    /// Очередь озвучки (шаг 4): live_synth кладёт фразы, TTS-воркер
    /// синтезирует. None = дубляж выключен (голос не выбран). Дроп сессии
    /// закрывает канал — воркер выходит.
    tts_tx: Option<std::sync::mpsc::Sender<tts::TtsJob>>,
}

fn session_slot() -> &'static Mutex<Option<Session>> {
    static S: OnceLock<Mutex<Option<Session>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// Единственный источник правды статуса: живёт пережизни Session, чтобы
/// live_status отвечал и после остановки (running=false).
pub(crate) fn status_slot() -> Arc<Mutex<LiveStatus>> {
    static S: OnceLock<Arc<Mutex<LiveStatus>>> = OnceLock::new();
    S.get_or_init(|| Arc::new(Mutex::new(LiveStatus::default()))).clone()
}

fn emit_live_error(app: &tauri::AppHandle, message: &str) {
    if let Err(e) = app.emit("live-error", serde_json::json!({ "message": message })) {
        log::warn!("Live: не отправить событие live-error: {e}");
    }
}

/// `https://twitch.tv/Chan_1?x` / `  Chan_1  ` → `chan_1`. Пустой → Err.
fn sanitize_channel(input: &str) -> Result<String, String> {
    let s = input.trim();
    let s = match s.to_ascii_lowercase().find("twitch.tv/") {
        Some(i) => &s[i + "twitch.tv/".len()..],
        None => s,
    };
    let s = s.split(['/', '?', '#']).next().unwrap_or(s);
    let cleaned: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<String>()
        .to_ascii_lowercase();
    if cleaned.is_empty() {
        Err("Укажите имя канала Twitch".into())
    } else {
        Ok(cleaned)
    }
}

/// Имя папки live-сессии: channel_display бывает ЛЮБОЙ (канал twitch, имя
/// файла, сырая URL). URL содержит `:/?&=` — `create_dir_all` на Windows
/// падает os error 123 (13.11.26: юзер вбил YouTube-ссылку, папка сессии не
/// создалась). Оставляем буквы/цифры/`-_.`, прочее → `_`, повторные `_`
/// схлопываем, длина ≤ 48, хвостовые точки/пробелы срезаем (Windows-квирк),
/// пусто → «session».
fn session_slug(input: &str) -> String {
    let mapped: String = input
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let mut out = String::with_capacity(mapped.len());
    let mut prev_underscore = false;
    for c in mapped.chars() {
        let u = c == '_';
        if !(u && prev_underscore) {
            out.push(c);
        }
        prev_underscore = u;
    }
    let mut out: String = out.chars().take(48).collect();
    while out.ends_with('.') || out.ends_with(' ') {
        out.pop();
    }
    if out.is_empty() {
        "session".into()
    } else {
        out
    }
}

#[tauri::command]
pub async fn live_start(
    app: tauri::AppHandle,
    channel: String,
    delay_s: Option<f64>,
    quality: Option<String>,
    proxy: Option<String>,
    oauth_token: Option<String>,
    // Шаг 5 PRP: источник (twitch|url|file), ссылка/файл, голос дубляжа.
    source_kind: Option<String>,
    url: Option<String>,
    file_path: Option<String>,
    voice_wav: Option<String>,
    dub_enabled: Option<bool>,
    // Режим сессии (13.11.26): "watch" — только плеер (медиа-сервер + прокси
    // /media /audio, ноль движков: ни llama-server, ни CrispASR); "dub" —
    // полный комбайн (сабы EN/RU + перевод + озвучка). Дефолт "dub" —
    // селфтесты и старые вызовы работают без правок.
    mode: Option<String>,
) -> Result<serde_json::Value, String> {
    let source_kind = source_kind.unwrap_or_else(|| "twitch".into());
    if !matches!(source_kind.as_str(), "twitch" | "url" | "file") {
        return Err(format!(
            "Неизвестный источник live: {source_kind} (twitch|url|file)"
        ));
    }
    let mode = mode.unwrap_or_else(|| "dub".into());
    if !matches!(mode.as_str(), "watch" | "dub") {
        return Err(format!("Неизвестный режим live: {mode} (watch|dub)"));
    }
    let watch = mode == "watch";
    let delay_s = delay_s.unwrap_or(DEFAULT_DELAY_SECS).clamp(10.0, 60.0);
    let quality = quality.unwrap_or_else(|| "source".into());
    let voice = voice_wav.unwrap_or_default().trim().to_string();
    // Дубляж: явный флаг ИЛИ сам факт выбора голоса; без клон-референса
    // клон-пресет невозможен (PRP шаг 4). Watch-режим: голос и флаг дубляжа
    // игнорируются — движков у сессии нет.
    let dub = !watch && dub_enabled.unwrap_or(true) && !voice.is_empty();

    if session_slot().lock().unwrap().is_some() {
        return Err("Live-сессия уже запущена — сначала нажмите Стоп".into());
    }

    let mut client = hls::build_client(proxy.as_deref())?;

    // Резолв media-плейлиста по источнику (шаг 5): twitch — GQL/usher,
    // url — yt-dlp -J (live/VOD известны до fetch), file — плейлиста нет.
    let mut media_url = String::new();
    let mut variant = String::new();
    let mut oauth: Option<String> = None;
    let mut file: Option<PathBuf> = None;
    // YouTube-VOD (12.11.26): прямой медиа-URL вместо HLS-плейлиста.
    let mut vod: Option<ytdlp::UrlInfo> = None;
    let channel_display = match source_kind.as_str() {
        "twitch" => {
            let ch = sanitize_channel(&channel)?;
            oauth = oauth_token
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty());
            let r = hls::resolve_with_quality(&client, &ch, &quality, oauth.as_deref()).await?;
            media_url = r.media_url;
            variant = r.variant;
            ch
        }
        "url" => {
            let u = url
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .ok_or("Укажите ссылку на поток/видео")?;
            let info = ytdlp::resolve_url_info(&app, &u, proxy.as_deref()).await?;
            // Заголовки yt-dlp (если есть) — во все запросы этой сессии:
            // fetch манифеста и сегментов идут тем же клиентом.
            if !info.headers.is_empty() {
                client = hls::build_client_with(proxy.as_deref(), &info.headers)?;
            }
            if info.is_live {
                let direct = info
                    .hls_url
                    .clone()
                    .ok_or("Живой стрим вернулся без HLS-манифеста")?;
                let head = hls::fetch_text(&client, "url-resolve", &direct).await?;
                if head.contains("#EXT-X-STREAM-INF") {
                    // Master-плейлист: yt-dlp вернул список качеств — выбрать
                    // вариант; URL может быть относительным (Url::join).
                    let variants = playlist::parse_master(&head);
                    let v = playlist::pick_variant(&variants, &quality)
                        .ok_or("В master-плейлисте нет подходящего варианта")?;
                    media_url = url_resolve(&direct, &v.url);
                    variant = v.name.clone();
                } else {
                    media_url = direct;
                }
            } else {
                // VOD: fetch_text на прямом mp4 тянул бы гигабайты в память —
                // не делаем. Аудио пойдёт в ffmpeg-читалку, видео/звук —
                // через reverse-proxy (см. VOD-ветку ниже).
                vod = Some(info);
            }
            u
        }
        _ => {
            let p = file_path
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .ok_or("Укажите файл")?;
            let pb = PathBuf::from(&p);
            if !pb.is_file() {
                return Err(format!("Файл не найден: {p}"));
            }
            file = Some(pb.clone());
            pb.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or(p)
        }
    };

    // Резервирование слота до асинхронного старта сервера: вторая попытка
    // live_start не проскочит, а при ошибке ниже — самоочистка.
    let live_root = crate::paths::temp_dir().join("live");
    let _ = std::fs::create_dir_all(&live_root);
    clean_old_sessions(&live_root);

    let dir = live_root.join(format!(
        "{}_{}",
        chrono::Utc::now().format("%Y%m%d_%H%M%S%3f"),
        session_slug(&channel_display)
    ));
    std::fs::create_dir_all(&dir).map_err(|e| format!("Не создать папку сессии: {e}"))?;

    let cancel = Arc::new(AtomicBool::new(false));
    let reserved = {
        let mut slot = session_slot().lock().unwrap();
        if slot.is_some() {
            let _ = std::fs::remove_dir_all(&dir);
            return Err("Live-сессия уже запущена — сначала нажмите Стоп".into());
        }
        let shared = Arc::new(server::ServerShared {
            playlist: tokio::sync::RwLock::new(String::new()),
            dir: dir.clone(),
            ready: tokio::sync::Notify::new(),
            local_file: tokio::sync::RwLock::new(file.clone()),
            remote: tokio::sync::RwLock::new(None),
            http: client.clone(),
        });
        let st = status_slot();
        *st.lock().unwrap() = LiveStatus {
            running: true,
            state: "starting".into(),
            port: 0,
            channel: channel_display.clone(),
            variant: variant.clone(),
            delay_s,
            source_kind: source_kind.clone(),
            mode: mode.clone(),
            dub_enabled: dub,
            ..LiveStatus::default()
        };
        *slot = Some(Session {
            cancel: cancel.clone(),
            dir: dir.clone(),
            tasks: Vec::new(),
            engine: None,
            tts_tx: None,
        });
        shared
    };

    let port = match server::spawn(reserved.clone()).await {
        Ok(p) => p,
        Err(e) => {
            cleanup_aborted(&cancel, &dir);
            return Err(e);
        }
    };
    // Порт в статус НЕ ставится до готовности манифеста: опросер фронта на
    // смонтированной вкладке цепляет плеер по running+port (Live.tsx), и
    // ранний порт давал гонку — hls.js получал ещё пустой манифест →
    // fatal manifestParsingError («no EXTM3U delimiter»), без восстановления.

    // Шаг 3 (PRP): переводчик на сессию — сабы ядро, заводится всегда; шаг 4:
    // TTS-воркер только с голосом (dub). Канал закрывается дропом Session.
    // Watch-режим (13.11.26): ноль движков — ни MT-очереди, ни TTS-воркера;
    // llama-server в VRAM не поднимается вовсе.
    let (mt_tx, tts_tx) = if watch {
        (None, None)
    } else {
        let mt_tx = mt::start(app.clone(), cancel.clone());
        let tts_tx = if dub {
            Some(tts::start(
                app.clone(),
                cancel.clone(),
                dir.clone(),
                voice.clone(),
                delay_s,
            ))
        } else {
            None
        };
        (Some(mt_tx), tts_tx)
    };
    {
        let mut slot = session_slot().lock().unwrap();
        if let Some(s) = slot.as_mut() {
            s.tts_tx = tts_tx;
        }
    }

    // Шаг 2 (PRP): STT-движок поднимается в фоне — не блокирует старт плеера;
    // аудио-конвейер получает сегменты от даунлоадера и пишет фразы в
    // live-subs, финалы уходят мостом в очередь MT (шаг 3).
    // Watch-режим (13.11.26): ни STT-движка, ни аудио-конвейера — подача
    // аудио в STT не нужна, stt_state остаётся "off", CrispASR не стартует.
    let (audio_pipe, engine_handle) = if watch {
        (None, None)
    } else {
        let engine_port = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let audio_pipe = audio::AudioPipeline::start(app.clone(), engine_port.clone(), mt_tx);
        {
            let st = status_slot();
            st.lock().unwrap().stt_state = "loading".into();
        }
        let status_for_engine = status_slot();
        let engine_handle = tauri::async_runtime::spawn_blocking(move || {
            match engine::SttEngine::spawn() {
                Ok(eng) => {
                    engine_port.store(eng.ws_port as u64, std::sync::atomic::Ordering::SeqCst);
                    status_for_engine.lock().unwrap().stt_state = "ready".into();
                    // Движок хранится в Session: teardown погасит его через Drop.
                    // Если сессия уже остановлена (слот пуст) — eng просто
                    // дропнется здесь, VRAM освободится.
                    if let Some(s) = session_slot().lock().unwrap().as_mut() {
                        s.engine = Some(eng);
                    }
                }
                Err(e) => {
                    status_for_engine.lock().unwrap().stt_state = format!("failed: {e}");
                    log::error!("{e}");
                }
            }
        });
        (Some(audio_pipe), Some(engine_handle))
    };

    if let Some(info) = vod {
        // YouTube-VOD (12.11.26): как файловый режим, но медиа удалённое —
        // ffmpeg тянет звук с бэкпрешером (30–100× реалтайм), видео/ориг.
        // звук плееру отдают reverse-proxy /media и /audio.
        let video = info.video_url.clone().ok_or("yt-dlp не вернул видео-поток")?;
        let audio_sep = match info.audio_url.as_ref() {
            Some(a) => *a != video,
            None => false,
        };
        *reserved.remote.write().await = Some(server::RemoteMedia {
            video_url: video,
            audio_url: if audio_sep { info.audio_url.clone() } else { None },
            headers: info.headers.clone(),
        });
        // Аудио-конвейер (STT/MT/TTS) — только в dub-режиме; в watch прокси
        // /media /audio уже настроены выше, ориг. звук идёт плееру.
        if let Some(pipe) = &audio_pipe {
            let audio_src = info
                .audio_url
                .clone()
                .unwrap_or_else(|| info.video_url.clone().unwrap_or_default());
            pipe.open_url(audio_src, info.headers.clone()).await;
        }
        {
            let slot = status_slot();
            let mut st = slot.lock().unwrap();
            st.port = port;
            st.state = "playing".into();
            st.player_kind = "file".into();
            st.remote_audio = audio_sep;
        }
        log::info!(
            "Live: VOD {} — плеер /media{} (режим: {mode})",
            channel_display,
            if audio_sep { " + /audio (ориг. звук)" } else { "" }
        );
        return Ok(serde_json::json!({
            "port": port,
            "channel": channel_display,
            "variant": "",
            "mode": mode,
        }));
    }

    if let Some(fpath) = &file {
        // Файл (шаг 5 PRP): манифеста нет — порт и «playing» сразу; плеер
        // фронтенда идёт на /media, аудио-конвейер тянет файл с бэкпрешером
        // (30–100× реалтайм → STT/MT естественным образом отстают на минуты,
        // что и есть «буфер» файлового режима). В watch-режиме конвейер не
        // заводится: файл отдаёт /media-прокси из shared.local_file.
        if let Some(pipe) = &audio_pipe {
            pipe.open_file(fpath.clone()).await;
        }
        {
            let slot = status_slot();
            let mut st = slot.lock().unwrap();
            st.port = port;
            st.state = "playing".into();
            st.player_kind = "file".into();
        }
        log::info!(
            "Live: файл {} — плеер http://127.0.0.1:{port}/media (режим: {mode})",
            fpath.display()
        );
        return Ok(serde_json::json!({
            "port": port,
            "channel": channel_display,
            "variant": "",
            "mode": mode,
        }));
    }

    let ctx = LoopCtx {
        app: app.clone(),
        client,
        media_url,
        channel: channel_display.clone(),
        quality: quality.clone(),
        oauth: oauth.clone(),
        delay_s,
        cancel: cancel.clone(),
        status: status_slot(),
        shared: reserved.clone(),
        dir: dir.clone(),
        audio: audio_pipe,
        source_kind: source_kind.clone(),
    };
    let handle = tauri::async_runtime::spawn(downloader_loop(ctx));

    {
        let mut slot = session_slot().lock().unwrap();
        if let Some(s) = slot.as_mut() {
            s.tasks.push(handle);
            // Watch-режим (13.11.26): STT-движок не спавнился — задачи нет.
            if let Some(h) = engine_handle {
                s.tasks.push(h);
            }
        }
    }

    // Ждём первый собранный плейлист: плеер, получив пустой манифест, падает
    // с manifestParsingError и не восстанавливается. На таймауте — teardown.
    if tokio::time::timeout(Duration::from_secs(30), reserved.ready.notified())
        .await
        .is_err()
    {
        teardown_session();
        return Err(
            "Поток не поднялся за 30 с — проверьте сеть; домены и причины ошибок в логе \
             (gql.twitch.tv, usher.ttvnw.net, video-weaver.*.hls.ttvnw.net)"
                .into(),
        );
    }

    // Порт в статус — только ПОСЛЕ готовности манифеста: опросер фронта
    // цепляет плеер по running+port, ранний порт давал гонку (hls.js получал
    // ещё пустой манифест → fatal manifestParsingError, без восстановления).
    status_slot().lock().unwrap().port = port;

    log::info!(
        "Live: стартовал {source_kind} «{channel_display}» (вариант {variant}), режим {mode}, задержка {delay_s} с, плейлист http://127.0.0.1:{port}/live.m3u8"
    );
    Ok(serde_json::json!({ "port": port, "channel": channel_display, "variant": variant, "mode": mode }))
}

/// Синтез фразы дубляжа (шаг 4 PRP): триггерит ФРОНТ, когда playhead дошёл до
/// t0−LEAD (правило №7 — очередь дубляжа на фронте). Готовый WAV прилетит
/// событием `live-dub {id, t0, dur, url}`.
#[tauri::command]
pub fn live_synth(id: u64, t0: f64, t1: f64, ru: String) -> Result<(), String> {
    let tx = {
        let slot = session_slot().lock().unwrap();
        let s = slot.as_ref().ok_or("Live-сессия не запущена")?;
        s.tts_tx
            .as_ref()
            .ok_or("Озвучка выключена — выберите голос-референс WAV на вкладке Live")?
            .clone()
    };
    tx.send(tts::TtsJob { id, t0, t1, ru })
        .map_err(|e| format!("Очередь озвучки недоступна ({e})"))
}

/// Относительный URI сегмента/MAP → абсолютный URL (master-варианты yt-dlp и
/// часть CDN ссылаются относительно плейлиста). Абсолютный вход — как есть.
pub fn url_resolve(base: &str, uri: &str) -> String {
    if uri.contains("://") {
        return uri.to_string();
    }
    match reqwest::Url::parse(base) {
        Ok(b) => b
            .join(uri)
            .map(|u| u.to_string())
            .unwrap_or_else(|_| uri.to_string()),
        Err(_) => uri.to_string(),
    }
}

#[tauri::command]
pub async fn live_stop() -> Result<(), String> {
    teardown_session();
    let status = status_slot();
    let mut st = status.lock().unwrap();
    *st = LiveStatus { running: false, state: "stopped".into(), ..LiveStatus::default() };
    Ok(())
}

/// Гасит сессию: отмена, аборт задач, удаление папки (с ретраями на Windows-лок).
fn teardown_session() {
    let taken = { session_slot().lock().unwrap().take() };
    if let Some(s) = taken {
        s.cancel.store(true, Ordering::SeqCst);
        for t in &s.tasks {
            t.abort();
        }
        // Файл мог быть захвачен убитой записью — 3 попытки с паузой.
        for i in 0..3 {
            match std::fs::remove_dir_all(&s.dir) {
                Ok(_) => break,
                Err(e) if i < 2 => {
                    log::warn!("Live: папка {} ещё занята ({e}), повтор", s.dir.display());
                    std::thread::sleep(Duration::from_millis(300));
                }
                Err(e) => log::warn!(
                    "Live: не удалить папку {} — подчистит следующий запуск: {e}",
                    s.dir.display()
                ),
            }
        }
        log::info!("Live: сессия остановлена");
    }
}

#[tauri::command]
pub fn live_status() -> LiveStatus {
    status_slot().lock().unwrap().clone()
}

fn cleanup_aborted(cancel: &Arc<AtomicBool>, dir: &PathBuf) {
    cancel.store(true, Ordering::SeqCst);
    if let Err(e) = std::fs::remove_dir_all(dir) {
        log::warn!("Live: не удалить папку {} при откате: {e}", dir.display());
    }
}

fn clean_old_sessions(live_root: &std::path::Path) {
    if let Ok(rd) = std::fs::read_dir(live_root) {
        for e in rd.flatten() {
            if e.path().is_dir() {
                if let Err(err) = std::fs::remove_dir_all(e.path()) {
                    log::warn!("Live: не подчистить старую сессию {}: {err}", e.path().display());
                }
            }
        }
    }
}

/// Всё, что нужно циклу даунлоадера (двигается by value в async move).
struct LoopCtx {
    app: tauri::AppHandle,
    client: reqwest::Client,
    media_url: String,
    channel: String,
    quality: String,
    oauth: Option<String>,
    delay_s: f64,
    cancel: Arc<AtomicBool>,
    status: Arc<Mutex<LiveStatus>>,
    shared: Arc<server::ServerShared>,
    dir: PathBuf,
    /// Шаг 2: аудио-конвейер (ffmpeg-pipe → кольцо → crispasr → live-subs).
    /// None в watch-режиме (13.11.26): сегменты качаются только для плеера,
    /// аудио в STT не подаётся.
    audio: Option<audio::AudioPipeline>,
    /// Шаг 5: источник (twitch|url) — GQL-перерезолв только для twitch.
    source_kind: String,
}

/// Даунлоадер: опрос weaver-плейлиста, докачка новых сегментов, скользящее
/// окно W = D + 20 с, инкремент MEDIA-SEQUENCE, отложенное удаление файлов.
/// Сетевые ошибки — через net_log (домен+причина), падение poll'ов → degraded,
/// тишина 30 с → offline, протухший usher-токен → перерезолв.
async fn downloader_loop(mut ctx: LoopCtx) {
    let window_secs = ctx.delay_s + BACK_MARGIN_SECS;
    let mut seen: HashSet<String> = HashSet::new();
    let mut window: VecDeque<ServedSeg> = VecDeque::new();
    let mut served_seq: u64 = 0;
    let mut media_sequence: u64 = 0;
    let mut target_dur: f64 = 2.0;
    let mut force_disc_next = false;
    let mut consecutive_fails: u32 = 0;
    let mut degraded_announced = false;
    let mut offline_announced = false;
    let mut last_new_at = Instant::now();
    let mut live_edge_pdt = String::new();
    let mut segments_failed: u64 = 0;
    let mut pending_delete: Vec<(String, Instant)> = Vec::new();
    // fMP4/CMAF: активный init (EXT-X-MAP) и его локальный файл. Смена MAP —
    // новый init-файл + разрыв перед следующим сегментом.
    let mut cur_map_uri: Option<String> = None;
    let mut cur_map_file: Option<String> = None;
    let mut cur_map_bytes: Option<Arc<Vec<u8>>> = None;
    let mut init_seq: u64 = 0;
    // Шаг 2: открыт ли аудио-блок (init подаётся в пайп первым — без него
    // fMP4-проба не распознает moov, и блок умрёт на первом сегменте).
    let mut audio_block_open = false;
    // PDT-инференс: последний явный PDT + сумма длительностей сегментов после
    // него. Явный PDT → ресинк; без него якорь достраивается (после рекламы
    // первый контентный сегмент может не иметь явного PDT — прогон 02:55).
    let mut last_pdt: Option<f64> = None;
    let mut pdt_acc = 0.0f64;

    while !ctx.cancel.load(Ordering::SeqCst) {
        let text = match hls::fetch_text(&ctx.client, "weaver-media", &ctx.media_url).await {
            Ok(t) => {
                consecutive_fails = 0;
                degraded_announced = false;
                t
            }
            Err(e) => {
                consecutive_fails += 1;
                ctx.status.lock().unwrap().last_error = e.clone();
                if consecutive_fails == DEGRADED_AFTER_FAILS && !degraded_announced {
                    degraded_announced = true;
                    ctx.status.lock().unwrap().state = "degraded".into();
                    emit_live_error(
                        &ctx.app,
                        &format!("Связь с Twitch прерывается ({consecutive_fails} ошибок опроса подряд): {e}"),
                    );
                }
                if consecutive_fails >= RERESOLVE_AFTER_FAILS
                    && consecutive_fails % RERESOLVE_AFTER_FAILS == 0
                    && ctx.source_kind == "twitch"
                {
                    // usher-токен живёт часами, но не вечно: перерезолв раз в
                    // ~минуту. Только twitch (url-источники рестартом сессии).
                    match hls::resolve_with_quality(
                        &ctx.client,
                        &ctx.channel,
                        &ctx.quality,
                        ctx.oauth.as_deref(),
                    )
                    .await
                    {
                        Ok(r) => {
                            log::info!("Live: поток перерезолвлен после {consecutive_fails} ошибок");
                            ctx.media_url = r.media_url;
                            consecutive_fails = 0;
                        }
                        Err(_) => log::warn!("Live: перерезолв не удался (попытка через минуту)"),
                    }
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };

        let (td, segs) = playlist::parse_media(&text);
        if td > 0.0 {
            target_dur = td;
        }
        let mut new_segs = Vec::new();
        for s in segs {
            if !seen.contains(&s.uri) {
                new_segs.push(s);
            }
        }
        if !new_segs.is_empty() {
            last_new_at = Instant::now();
            offline_announced = false;
        }

        for s in new_segs {
            // Init для fMP4/CMAF: у сегмента свой EXT-X-MAP (или первый сегмент).
            // (Some(uri), prev) покрывает и prev=None (первый MAP) — отдельная
            // рука (Some(_), None) была недостижима.
            let need_init = match (&s.map_uri, &cur_map_uri) {
                (Some(uri), prev) => prev.as_deref() != Some(uri.as_str()),
                (None, None) => false,
                (None, Some(_)) => true,
            };
            let mut seg_map_file: Option<String> = None;
            if need_init {
                match &s.map_uri {
                    Some(uri) => {
                        let fname = format!("init_{init_seq:06}.m4s");
                        let uri_abs = url_resolve(&ctx.media_url, uri);
                        match hls::fetch_bytes(&ctx.client, "cdn-init", &uri_abs).await {
                            Ok(bytes) => {
                                if let Err(e) =
                                    tokio::fs::write(ctx.dir.join(&fname), &bytes).await
                                {
                                    log::error!("Live: не записать init {fname}: {e}");
                                    continue; // сегменты без init бесполезны
                                }
                                log::info!(
                                    "Live: fMP4 init скачан ({fname}, {} КБ)",
                                    bytes.len() / 1024
                                );
                                init_seq += 1;
                                cur_map_uri = Some(uri.clone());
                                cur_map_file = Some(fname.clone());
                                cur_map_bytes = Some(Arc::new(bytes));
                                seg_map_file = Some(fname);
                                force_disc_next = true;
                            }
                            Err(e) => {
                                // Init не скачался: эти сегменты в окно не берём;
                                // НЕ помечаем seen — ретрай на следующем опросе.
                                segments_failed += 1;
                                ctx.status.lock().unwrap().last_error = e;
                                continue;
                            }
                        }
                    }
                    None => {
                        // Плейлист перестал объявлять MAP (TS-режим?) — просто
                        // разрыв, плеер перейдёт на TS-демуксер.
                        cur_map_uri = None;
                        cur_map_file = None;
                        cur_map_bytes = None;
                        seg_map_file = None;
                        force_disc_next = true;
                    }
                }
            }

            let bytes = match hls::fetch_bytes(&ctx.client, "cdn-segment", &url_resolve(&ctx.media_url, &s.uri)).await {
                Ok(bytes) => bytes,
                Err(e) => {
                    // Дыра в окне — следующий сегмент начинаем с разрыва,
                    // иначе декодер плеера рвётся на пропавшем куске.
                    seen.insert(s.uri.clone());
                    force_disc_next = true;
                    segments_failed += 1;
                    ctx.status.lock().unwrap().last_error = e;
                    continue;
                }
            };
            // Расширение по сниффу: MPEG-TS начинается с 0x47, fMP4 —
            // с ftyp/styp (ISO-BMFF). hls.js/ffmpeg сниффят контент,
            // но корректное расширение упрощает дебаг ffprobe'ом.
            let ext = if bytes.first() == Some(&0x47) { "ts" } else { "m4s" };
            let fname = format!("seg_{served_seq:06}.{ext}");
            if let Err(e) = tokio::fs::write(ctx.dir.join(&fname), &bytes).await {
                log::error!("Live: не записать сегмент {fname}: {e}");
                seen.insert(s.uri.clone());
                continue;
            }
            seen.insert(s.uri.clone());
            let disc = s.discontinuity || force_disc_next;
            force_disc_next = false;

            // PDT-инференс: тег EXT-X-PROGRAM-DATE-TIME стоит не на каждом
            // сегменте — якорь блока достраивается: явный PDT → ресинк, иначе
            // последний явный + сумма длительностей прошедших сегментов.
            // (Заметено на прогоне 02:55: второй блок открылся с pdt0=None.)
            let anchor: Option<f64> =
                if let Some(p) = s.pdt.as_deref().and_then(audio::pdt_to_epoch_secs) {
                    last_pdt = Some(p);
                    pdt_acc = s.duration;
                    Some(p)
                } else if let Some(p) = last_pdt {
                    let a = p + pdt_acc;
                    pdt_acc += s.duration;
                    Some(a)
                } else {
                    None
                };

            // Шаг 2: аудио-конвейер (dub-режим; в watch его нет — сегменты
            // качаются только для плеера). Разрыв/реклама/дыра закрывают блок,
            // следующий не-рекламный сегмент открывает новый — с init впереди
            // (без moov проба ffmpeg не распознает fMP4, блок умрёт).
            if let Some(audio) = &ctx.audio {
                if disc || s.is_ad {
                    audio_block_open = false;
                    audio.restart().await;
                }
                if !s.is_ad {
                    if !audio_block_open {
                        // Якорь блока = PDT (явный или достроенный) первого МЕДИА-
                        // сегмента; у init PDT нет, init пишется следом (порядок в
                        // пайпе обязателен: без moov проба ffmpeg не распознает fMP4).
                        audio.open_block(anchor).await;
                        if let Some(init) = &cur_map_bytes {
                            audio.feed(Arc::clone(init), None).await;
                        }
                        audio_block_open = true;
                    }
                    audio.feed(Arc::new(bytes), anchor).await;
                }
            }

            window.push_back(ServedSeg {
                seq: served_seq,
                file: fname,
                duration: s.duration,
                discontinuity: disc,
                is_ad: s.is_ad,
                map_file: seg_map_file,
                // PDT якоря блока: явный или достроенный (см. выше) — уходит
                // в плейлист тегом EXT-X-PROGRAM-DATE-TIME для playingDate.
                pdt: anchor,
            });
            served_seq += 1;
            if let Some(p) = &s.pdt {
                live_edge_pdt = p.clone();
            }
            if s.is_ad {
                log::info!("Live: рекламный сегмент скачан (в STT/MT/TTS не пойдёт)");
            }
        }

        // Обрезка окна: держим новейшие сегменты суммарно ≥ window_secs.
        let mut acc = 0.0f64;
        let mut keep_start = window.len();
        for (i, s) in window.iter().enumerate().rev() {
            acc += s.duration;
            keep_start = i;
            if acc >= window_secs {
                break;
            }
        }
        let drop = keep_start;
        for _ in 0..drop {
            let s = window.pop_front().expect("окно не пусто, пока drop > 0");
            media_sequence += 1;
            pending_delete.push((s.file, Instant::now() + Duration::from_secs(DELETE_LAG_SECS)));
        }
        let now = Instant::now();
        let expired: Vec<String> = pending_delete
            .iter()
            .filter(|(_, dl)| now >= *dl)
            .map(|(f, _)| f.clone())
            .collect();
        pending_delete.retain(|(_, dl)| now < *dl);
        for f in expired {
            if let Err(e) = tokio::fs::remove_file(ctx.dir.join(&f)).await {
                log::warn!("Live: не удалить сегмент {f}: {e}");
            }
        }

        let win_secs: f64 = window.iter().map(|s| s.duration).sum();
        let flat: Vec<ServedSeg> = window.iter().cloned().collect();
        let m3u8 = playlist::build_media_m3u8(
            &flat,
            media_sequence,
            target_dur.ceil().max(1.0) as u64,
        );
        *ctx.shared.playlist.write().await = m3u8;
        // Разрешаем живой_start: манифест готов (notify_one копит один
        // разрешающий билет — повторные вызовы его не ломают).
        ctx.shared.ready.notify_one();

        let offline = last_new_at.elapsed() > Duration::from_secs(OFFLINE_AFTER_SECS);
        {
            let mut st = ctx.status.lock().unwrap();
            st.segments_downloaded = served_seq;
            st.segments_failed = segments_failed;
            st.served_window_secs = win_secs;
            st.target_duration = target_dur;
            st.live_edge_pdt = live_edge_pdt.clone();
            st.state = if offline {
                "offline".into()
            } else {
                "playing".into()
            };
        }
        if offline && !offline_announced {
            offline_announced = true;
            emit_live_error(
                &ctx.app,
                "Новые сегменты не приходят 30 с — трансляция, вероятно, завершилась",
            );
        }

        let sleep = (target_dur / 2.0).clamp(0.8, 3.0);
        tokio::time::sleep(Duration::from_secs_f64(sleep)).await;
    }
    log::info!("Live: цикл даунлоадера завершён");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_channel_forms() {
        assert_eq!(sanitize_channel("Papich").unwrap(), "papich");
        assert_eq!(sanitize_channel("  https://twitch.tv/Chan_1?x ").unwrap(), "chan_1");
        assert_eq!(sanitize_channel("www.twitch.tv/some_name/").unwrap(), "some_name");
        assert!(sanitize_channel("!!!").is_err());
        assert!(sanitize_channel("").is_err());
    }

    #[test]
    fn session_slug_makes_valid_windows_name() {
        // YouTube-ссылка (реальный кейс 13.11.26): `:/?&=` недопустимы в имени
        // папки Windows — os error 123.
        let slug = session_slug("https://www.youtube.com/watch?v=q1v4IuOoWiI");
        assert!(!slug.contains([':', '/', '?', '&', '=']));
        assert!(slug.len() <= 48);
        assert!(!slug.is_empty());
        // Схема схлопнулась, вопрос-параметры заменены.
        assert!(slug.starts_with("https_"));
        // Имя файла и канал не искажаются.
        assert_eq!(session_slug("test_TTS_dubbing.mp4"), "test_TTS_dubbing.mp4");
        assert_eq!(session_slug("bogur"), "bogur");
        // Хвостовые точки запрещены Windows; пусто → дефолт.
        assert_eq!(session_slug("name..."), "name");
        assert_eq!(session_slug("///"), "session");
        assert_eq!(session_slug(""), "session");
    }
}
