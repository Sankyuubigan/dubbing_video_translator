//! Loopback HTTP-сервер для плеера вкладки Live (поправка специалиста №4:
//! axum на 127.0.0.1:0 вместо кастомного URI-протокола — WebView2 глючит на
//! медиа через WebResourceRequested: Range/206, CORS preflight, IPC-мост).
//!
//! Отдаёт:
//! - `/live.m3u8` — текущий скользящий плейлист (генерирует даунлоадер);
//! - `/seg_NNNNNN.ts` — сегменты из папки сессии.
//!
//! CORS `*` обязателен: hls.js делает XHR из webview-ориджина (dev —
//! http://localhost:5173, prod — http://tauri.localhost) на 127.0.0.1:порт.

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Удалённый медиа-источник (YouTube-VOD): ffmpeg тянет с него звук для
/// сабов/озвучки, а плеер получает видео/звук через наш reverse-proxy с Range.
#[derive(Debug, Clone)]
pub struct RemoteMedia {
    /// Видео (мукс, если есть, иначе video-only поток).
    pub video_url: String,
    /// Отдельный аудио-поток (None, когда видео мукс со звуком).
    pub audio_url: Option<String>,
    /// Заголовки от yt-dlp (UA и пр.) — проксируются насквозь.
    pub headers: Vec<(String, String)>,
}

pub struct ServerShared {
    /// Текст текущего скользящего плейлиста; пишет даунлоадер, читает сервер.
    pub playlist: RwLock<String>,
    /// Папка сессии temp/live/<session>/ с файлами seg_*.ts.
    pub dir: PathBuf,
    /// Сигнал «первый плейлист собран»: live_start ждёт его, чтобы не отдать
    /// фронту пустой манифест (hls.js на пустом ответе даёт
    /// manifestParsingError и не восстанавливается).
    pub ready: tokio::sync::Notify,
    /// Файловый источник (шаг 5 PRP): путь к локальному видео для /media.
    /// None → /media отвечает 404 (twitch/url сессии).
    pub local_file: RwLock<Option<PathBuf>>,
    /// YouTube-VOD (12.11.26): прокси-цели для /media и /audio.
    pub remote: RwLock<Option<RemoteMedia>>,
    /// HTTP-клиент прокси (прокси-настройки юзера учтены при сборке).
    pub http: reqwest::Client,
}

pub async fn spawn(shared: Arc<ServerShared>) -> Result<u16, String> {
    let app = Router::new()
        .route("/live.m3u8", get(playlist_handler))
        .route("/seg/{file}", get(seg_handler))
        .route("/tts/{file}", get(tts_handler))
        .route("/audio", get(audio_proxy_handler))
        .route("/media", get(media_handler))
        .with_state(shared);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| format!("Не занять порт для loopback-сервера: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("Не узнать порт loopback-сервера: {e}"))?
        .port();

    tauri::async_runtime::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            log::error!("Live: loopback-сервер упал: {e}");
        }
    });
    log::info!("Live: loopback-сервер на 127.0.0.1:{port}");
    Ok(port)
}

async fn playlist_handler(State(sh): State<Arc<ServerShared>>) -> Response {
    let text = sh.playlist.read().await.clone();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .header(header::CONTENT_TYPE, "application/vnd.apple.mpegurl")
        .header(header::CACHE_CONTROL, "no-store")
        .body(axum::body::Body::from(text))
        .unwrap_or_else(|_| empty_500())
}

async fn seg_handler(State(sh): State<Arc<ServerShared>>, Path(file): Path<String>) -> Response {
    // Санитизация: только наши имена — seg_NNNNNN.ts / seg_NNNNNN.m4s (сегменты,
    // расширение по сниффу контента: fMP4 Twitch сохраняется как .m4s) и
    // init_NNNNNN.m4s (fMP4/CMAF init из EXT-X-MAP). Никакого path traversal.
    let ok = is_media_file_name(&file);
    if !ok {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::from("bad segment name"))
            .unwrap_or_else(|_| empty_500());
    }
    let content_type = if file.ends_with(".m4s") {
        "video/iso.segment"
    } else {
        "video/mp2t"
    };
    match tokio::fs::read(sh.dir.join(&file)).await {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .header(header::CONTENT_TYPE, content_type)
            .body(axum::body::Body::from(bytes))
            .unwrap_or_else(|_| empty_500()),
        Err(_) => {
            // Сегмент мог быть удалён скользящим окном (крайний случай гонки) —
            // это нормальный 404 для плеера, не ошибка лога.
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(axum::body::Body::from("segment gone"))
                .unwrap_or_else(|_| empty_500())
        }
    }
}

/// `seg_NNNNNN.ts` / `seg_NNNNNN.m4s` / `init_NNNNNN.m4s` — имена файлов сессии.
pub fn is_media_file_name(f: &str) -> bool {
    let (stem, ext) = match (f.strip_suffix(".ts"), f.strip_suffix(".m4s")) {
        (Some(s), _) => (s, "ts"),
        (_, Some(s)) => (s, "m4s"),
        _ => return false,
    };
    let prefix = match stem.strip_prefix("seg_") {
        Some(s) => s,
        None => match stem.strip_prefix("init_") {
            Some(s) if ext == "m4s" => s,
            _ => return false,
        },
    };
    prefix.len() == 6 && prefix.bytes().all(|c| c.is_ascii_digit())
}

/// `tts_<id>.wav` — озвучка дубляжа (шаг 4): только цифры после префикса.
pub fn is_tts_file_name(f: &str) -> bool {
    let Some(stem) = f.strip_suffix(".wav") else {
        return false;
    };
    match stem.strip_prefix("tts_") {
        Some(num) => !num.is_empty() && num.bytes().all(|c| c.is_ascii_digit()),
        None => false,
    }
}

/// WAV-файл озвучки: `http://…/tts/tts_<id>.wav` для <audio> дубляжа.
async fn tts_handler(State(sh): State<Arc<ServerShared>>, Path(file): Path<String>) -> Response {
    if !is_tts_file_name(&file) {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::from("bad tts name"))
            .unwrap_or_else(|_| empty_500());
    }
    match tokio::fs::read(sh.dir.join(&file)).await {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .header(header::CONTENT_TYPE, "audio/wav")
            .header(header::CACHE_CONTROL, "no-store")
            .body(axum::body::Body::from(bytes))
            .unwrap_or_else(|_| empty_500()),
        Err(_) => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::from("tts gone"))
            .unwrap_or_else(|_| empty_500()),
    }
}

/// Размер чанка /media: плеер просит «bytes=N-» (до конца файла), в память
/// целиком не годится (многочасовые подкасты) — отдаём частями по 2 МиБ
/// (RFC 7233: сервер вправе закрыть диапазон раньше с корректным
/// Content-Range, браузер запросит продолжение).
const MEDIA_CHUNK: u64 = 2 * 1024 * 1024;

/// Полная цепочка причин ошибки (reqwest: to_string() даёт только верх
/// «error sending request», причина DNS/TLS/connect — в source()-цепочке;
/// без неё NET-FAIL в логе не диагностируется — баг 12.11.26).
fn err_chain(e: &reqwest::Error) -> String {
    let mut msg = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(s) = src {
        msg.push_str(&format!(" | причина: {s}"));
        src = s.source();
    }
    msg
}

/// Локальный файл источника «file»: <video src=…/media> с поддержкой Range
/// (браузер гуляет по файлу при старте/сике — без 206 плеер не живёт).
async fn media_handler(
    State(sh): State<Arc<ServerShared>>,
    headers: axum::http::HeaderMap,
) -> Response {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};

    let cors = [(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")];
    // YouTube-VOD (12.11.26): проксируем удалённое видео с Range до проверки
    // локального файла (взаимоисключающие источники).
    if let Some(rm) = sh.remote.read().await.clone() {
        let ctype = if rm.video_url.contains("mime=audio") { "audio/mp4" } else { "video/mp4" };
        return proxy_range(&sh, rm.video_url, rm.headers, &headers, ctype).await;
    }
    let Some(path) = sh.local_file.read().await.clone() else {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::from("media not available"))
            .unwrap_or_else(|_| empty_500());
    };
    let Ok(meta) = tokio::fs::metadata(&path).await else {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::from("media gone"))
            .unwrap_or_else(|_| empty_500());
    };
    let len = meta.len();

    // Range: только bytes=N-[M]; суффиксный «-N» и прочее — с начала.
    let mut start: u64 = 0;
    let mut ranged = false;
    if let Some(rv) = headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
        if let Some(spec) = rv.strip_prefix("bytes=") {
            let first = spec.split('-').next().unwrap_or("").trim().to_string();
            if let Ok(a) = first.parse::<u64>() {
                if a < len {
                    start = a;
                    ranged = true;
                }
            }
        }
    }
    if start >= len {
        return Response::builder()
            .status(StatusCode::RANGE_NOT_SATISFIABLE)
            .header(header::CONTENT_RANGE, format!("bytes */{len}"))
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::empty())
            .unwrap_or_else(|_| empty_500());
    }
    let end = len.saturating_sub(1).min(start + MEDIA_CHUNK - 1);

    let mut f = match tokio::fs::File::open(&path).await {
        Ok(f) => f,
        Err(e) => {
            log::warn!("Live: /media не открыть {}: {e}", path.display());
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(axum::body::Body::from("media read error"))
                .unwrap_or_else(|_| empty_500());
        }
    };
    if f.seek(std::io::SeekFrom::Start(start)).await.is_err() {
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::from("media seek error"))
            .unwrap_or_else(|_| empty_500());
    }
    let mut buf = vec![0u8; (end - start + 1) as usize];
    if let Err(e) = f.read_exact(&mut buf).await {
        log::warn!("Live: /media недочитал чанк {}: {e}", path.display());
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::from("media read error"))
            .unwrap_or_else(|_| empty_500());
    }

    let ctype = video_content_type(&path);
    let mut rb = Response::builder()
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .header(header::CONTENT_TYPE, ctype)
        .header(header::CACHE_CONTROL, "no-store");
    rb = if ranged {
        rb.status(StatusCode::PARTIAL_CONTENT).header(
            header::CONTENT_RANGE,
            format!("bytes {start}-{end}/{len}"),
        )
    } else {
        rb.status(StatusCode::OK)
    };
    rb.body(axum::body::Body::from(buf)).unwrap_or_else(|_| empty_500())
}

/// Оригинальный звук YouTube-VOD: прокси на отдельный аудио-поток
/// (видео-мукс без звука не нужен — тогда /audio не запрашивается).
async fn audio_proxy_handler(
    State(sh): State<Arc<ServerShared>>,
    headers: axum::http::HeaderMap,
) -> Response {
    let Some(rm) = sh.remote.read().await.clone() else {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::from("audio not available"))
            .unwrap_or_else(|_| empty_500());
    };
    let Some(url) = rm.audio_url else {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::from("audio not available"))
            .unwrap_or_else(|_| empty_500());
    };
    proxy_range(&sh, url, rm.headers, &headers, "audio/mp4").await
}

/// Range-прокси на удалённый медиа-поток: плеер просит bytes=N-, мы запрашиваем
/// у источника тот же диапазон чанком MEDIA_CHUNK и отдаём 206 насквозь
/// (googlevideo Range поддерживает — проверено curl 206 на свежем URL).
async fn proxy_range(
    sh: &Arc<ServerShared>,
    url: String,
    hdrs: Vec<(String, String)>,
    in_headers: &axum::http::HeaderMap,
    ctype: &'static str,
) -> Response {
    let cors = [(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")];

    // Диапазон плеера: только bytes=N- (суффиксный/парный — с начала).
    let mut start: u64 = 0;
    if let Some(rv) = in_headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
    {
        if let Some(spec) = rv.strip_prefix("bytes=") {
            if let Ok(a) = spec.split('-').next().unwrap_or("").trim().parse::<u64>() {
                start = a;
            }
        }
    }

    let end = start.saturating_add(MEDIA_CHUNK - 1);
    let mut req = sh
        .http
        .get(&url)
        .header(header::RANGE, format!("bytes={start}-{end}"));
    for (k, v) in &hdrs {
        req = req.header(k.as_str(), v.as_str());
    }
    let up = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            crate::live::net_log::log_net_fail("media-proxy", &url, &err_chain(&e));
            return Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(axum::body::Body::from(format!("upstream error: {e}")))
                .unwrap_or_else(|_| empty_500());
        }
    };
    if !up.status().is_success() {
        let st = up.status();
        return Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(axum::body::Body::from(format!("upstream HTTP {st}")))
            .unwrap_or_else(|_| empty_500());
    }
    let cr = up
        .headers()
        .get(header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = match up.bytes().await {
        Ok(b) => b,
        Err(e) => {
            crate::live::net_log::log_net_fail("media-proxy-body", &url, &err_chain(&e));
            return Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(axum::body::Body::from("upstream body error"))
                .unwrap_or_else(|_| empty_500());
        }
    };

    let mut rb = Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .header(header::CONTENT_TYPE, ctype)
        .header(header::CACHE_CONTROL, "no-store");
    rb = match cr {
        Some(v) => rb.header(header::CONTENT_RANGE, v),
        None => rb.header(
            header::CONTENT_RANGE,
            format!("bytes {start}-{}/*", start + bytes.len() as u64 - 1),
        ),
    };
    rb.body(axum::body::Body::from(bytes))
        .unwrap_or_else(|_| empty_500())
}

/// Content-Type по расширению локального файла (браузерному <video> он
/// нужен; неизвестное расширение — mp4 как наиболее вероятный контейнер).
fn video_content_type(p: &FsPath) -> &'static str {
    match p
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("mp4") | Some("m4v") | Some("mov") | Some("m4a") => "video/mp4",
        Some("webm") => "video/webm",
        Some("mkv") => "video/x-matroska",
        Some("avi") => "video/x-msvideo",
        Some("ts") => "video/mp2t",
        _ => "video/mp4",
    }
}

fn empty_500() -> Response {
    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .body(axum::body::Body::empty())
        .expect("пустой ответ 500 всегда собирается")
}
