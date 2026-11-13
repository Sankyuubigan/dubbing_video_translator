//! Резолвер ссылок для live-источника «URL» (шаг 5 PRP) через yt-dlp.
//!
//! ССОТ пути к yt-dlp — плагин `tauri-plugin-ytdlp`: YtdlpConfig.ytdlp_dir →
//! get_exe_path, иначе default_dir рядом с exe, иначе «yt-dlp» из PATH
//! (на машине юзера это D:\Programs\yt-dlp\yt-dlp.exe). Прокси передаётся
//! насквозь. Проба 12.11.26: twitch через yt-dlp -g отдаёт ГОТОВЫЙ
//! media-плейлист (PDT на сегментах); master тоже возможен (определяется по
//! `#EXT-X-STREAM-INF`).
//!
//! 12.11.26 (VOD-фикс): классификация live/VOD по `-J` ДО любого fetch —
//! fetch_text на прямом mp4 YouTube-VOD тянул бы 1 ГиБ тела.
//!
//! 12.11.26 (вечер, PO-token): без JS-рантайма yt-dlp возвращает URL класса
//! ANDROID_VR, которые googlevideo отдаёт только первые ~2-4 МиБ (дальше 403
//! на любой HTTP-клиент — curl/ffmpeg/reqwest, любой UA, любой Range). Лечится
//! обновлением yt-dlp (≥2026.10.26) + `--js-runtimes node`: challenge-провайдер
//! решает BotGuard, URL качаются целиком (206/200 на любой диапазон). Вдобавок
//! `-headers` ffmpeg требует CRLF (audio.rs).

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tauri::AppHandle;

/// Разрешает ссылку в прямую HLS-ссылку (`yt-dlp -g`, первая строка вывода).
/// Блокирует до 90 с — вызывается через spawn_blocking.
pub async fn resolve_hls_url(
    app: &AppHandle,
    url: &str,
    proxy: Option<&str>,
) -> Result<String, String> {
    let url = url.to_string();
    let proxy = proxy.map(str::to_string);
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || resolve_blocking(&app, &url, proxy.as_deref()))
        .await
        .map_err(|e| format!("yt-dlp поток: {e}"))?
}

/// Путь к yt-dlp (та же логика, что у команд плагина): конфиг → папка рядом
/// с exe → «yt-dlp» из PATH.
/// Убийство yt-dlp, переживающего TerminateProcess (PyInstaller-обёртка):
/// kill → ждём до 5 с → если жив — taskkill /F /T (дерево). Замер 20:06:
/// child.kill() не убил PID, wait() висел 5 мин до внешнего taskkill.
fn kill_ytdlp_tree(child: &mut std::process::Child) {
    let pid = child.id();
    let _ = child.kill();
    for _ in 0..25 {
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return,
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
        }
    }
    log::warn!("LIVE: yt-dlp (PID {pid}) пережил kill — taskkill /F /T");
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F", "/T"])
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = child.wait();
}

fn ytdlp_exe(app: &AppHandle) -> PathBuf {
    use tauri::Manager;

    let cfg = tauri_plugin_ytdlp::engine::load_config(app);
    let dir = match cfg.ytdlp_dir.as_deref() {
        Some(p) if !p.trim().is_empty() => PathBuf::from(p),
        _ => {
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."));
            tauri_plugin_ytdlp::engine::ytdlp_manager::default_dir(&exe_dir)
        }
    };
    let p = tauri_plugin_ytdlp::engine::ytdlp_manager::get_exe_path(&dir);
    if p.exists() {
        return p;
    }
    // Фолбэк: yt-dlp в PATH.
    PathBuf::from("yt-dlp")
}

fn resolve_blocking(app: &AppHandle, url: &str, proxy: Option<&str>) -> Result<String, String> {
    let exe = ytdlp_exe(app);
    let mut cmd = Command::new(&exe);
    cmd.arg("-g");
    if let Some(p) = proxy.filter(|s| !s.trim().is_empty()) {
        cmd.args(["--proxy", p]);
    }
    cmd.arg(url);

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Не запустить yt-dlp ({}): {e}", exe.display()))?;

    // stdout и stderr сливаем ПАРАЛЛЕЛЬНО: вывод -J ~100 КиБ не влезает в
    // буфер пайпа — без дренажа ребёнок блокируется на записи и не выходит
    // (замер 20.11: ручной -J 2.4 с, в приложении — вечно).
    let mut stdout_pipe = child.stdout.take().unwrap();
    let stdout_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stdout_pipe.read_to_string(&mut buf);
        buf
    });
    let mut stderr_pipe = child.stderr.take().unwrap();
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr_pipe.read_to_string(&mut buf);
        buf
    });

    // Ждём выхода с таймаутом (yt-dlp может зависнуть на живом стриме).
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {
                if started.elapsed() > Duration::from_secs(90) {
                    kill_ytdlp_tree(&mut child);
                    let _ = stderr_thread.join();
                    let _ = stdout_thread.join();
                    return Err("yt-dlp не ответил за 90 с — источник недоступен".into());
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => return Err(format!("yt-dlp wait: {e}")),
        }
    };

    let stdout = stdout_thread.join().unwrap_or_default();
    let stderr = stderr_thread.join().unwrap_or_default();

    if !status.success() {
        let tail = stderr.lines().rev().take(3).collect::<Vec<_>>();
        let tail = tail.into_iter().rev().collect::<Vec<_>>().join("; ");
        return Err(format!(
            "yt-dlp не смог разрешить ссылку (exit {}): {}",
            status.code().unwrap_or(-1),
            if tail.is_empty() {
                "нет диагностики".to_string()
            } else {
                tail
            }
        ));
    }

    // yt-dlp может вернуть несколько строк (видео+аудио раздельно) — для
    // HLS-источника нужен один манифест: берём первую непустую.
    let first = stdout
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .ok_or("yt-dlp вернул пустой вывод")?
        .to_string();
    Ok(first)
}

/// Результат `-J`-резолва: тип источника известен ДО любого fetch.
#[derive(Debug, Clone)]
pub struct UrlInfo {
    pub is_live: bool,
    /// Живой стрим: HLS-манифест (muxed-вариант). None для VOD.
    pub hls_url: Option<String>,
    /// VOD: аудио-поток (или мукс, если раздельного нет).
    pub audio_url: Option<String>,
    /// VOD: видео-поток (мукс, если есть; иначе лучшее видео).
    pub video_url: Option<String>,
    /// Заголовки от yt-dlp (браузерный UA и пр.) — для ffmpeg-читалки и
    /// reverse-proxy. Часто пусты: googlevideo web-класса отвечает и без
    /// них (проверено curl 206 на свежем URL).
    pub headers: Vec<(String, String)>,
}

/// JSON-ответ `yt-dlp -J` (нужны только эти поля).
#[derive(serde::Deserialize)]
struct JInfo {
    #[serde(default)]
    is_live: bool,
    #[serde(default)]
    protocol: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    vcodec: Option<String>,
    #[serde(default)]
    acodec: Option<String>,
    #[serde(default)]
    http_headers: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    requested_formats: Vec<JFormat>,
    /// Полный список форматов (yt-dlp сортирует от худшего к лучшему —
    /// «последний подходящий» и есть лучший).
    #[serde(default)]
    formats: Vec<JFormat>,
}

#[derive(serde::Deserialize)]
struct JFormat {
    #[serde(default)]
    url: String,
    #[serde(default)]
    protocol: String,
    #[serde(default)]
    vcodec: Option<String>,
    #[serde(default)]
    acodec: Option<String>,
    #[serde(default)]
    http_headers: std::collections::BTreeMap<String, String>,
}

fn hdrs(m: &std::collections::BTreeMap<String, String>) -> Vec<(String, String)> {
    // Только то, что реально нужно серверу/ffmpeg. Cookie сознательно не
    // пробрасываем: содержимое аккаунта юзера не должно гулять по прокси.
    const KEEP: [&str; 5] = ["User-Agent", "Referer", "Accept", "Accept-Language", "Sec-Fetch-Mode"];
    KEEP.iter()
        .filter_map(|k| m.get(*k).map(|v| (k.to_string(), v.clone())))
        .collect()
}

fn has_video(f: &JFormat) -> bool {
    matches!(f.vcodec.as_deref(), Some(v) if v != "none")
}
fn has_audio(f: &JFormat) -> bool {
    matches!(f.acodec.as_deref(), Some(v) if v != "none")
}

/// Разрешает ссылку в UrlInfo: live → HLS-манифест, VOD → пара аудио/видео.
/// PO-токен эпоха YouTube (замер 12.11.26): без JS-рантайма yt-dlp возвращает
/// URL, качающиеся только первые ~2-4 МиБ (дальше googlevideo 403) — поэтому
/// `--js-runtimes node` (challenge-провайдер; node есть почти на любой машине
/// с приложением). Старая версия yt-dlp флаг не знает → повтор без него.
pub async fn resolve_url_info(
    app: &AppHandle,
    url: &str,
    proxy: Option<&str>,
) -> Result<UrlInfo, String> {
    let url = url.to_string();
    let proxy = proxy.map(str::to_string);
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let raw = match run_ytdlp_json(&app, &url, proxy.as_deref(), true) {
            Ok(v) => v,
            Err(e)
                if e.contains("unrecognized")
                    || e.contains("no such option")
                    || e.contains("no such argument") =>
            {
                log::warn!("Live: yt-dlp не знает --js-runtimes ({e}) — повтор без флага");
                run_ytdlp_json(&app, &url, proxy.as_deref(), false)?
            }
            Err(e) => return Err(e),
        };
        url_info_from_json(&raw)
    })
    .await
    .map_err(|e| format!("yt-dlp поток: {e}"))?
}

fn fmt_of(j: &JInfo) -> JFormat {
    JFormat {
        url: j.url.clone(),
        protocol: j.protocol.clone(),
        vcodec: j.vcodec.clone(),
        acodec: j.acodec.clone(),
        http_headers: j.http_headers.clone(),
    }
}

fn url_info_from_json(raw: &str) -> Result<UrlInfo, String> {
    let j: JInfo =
        serde_json::from_str(raw).map_err(|e| format!("yt-dlp -J: некорректный JSON: {e}"))?;

    if j.is_live {
        // Живой стрим: нужен ОДИН muxed-манифест. Раздельные video/audio
        // манифесты наш HLS-даунлоадер не умеет — честный отказ вместо
        // тихо пустых сабов.
        // ВАЖНО: гейт только по is_live. У YouTube-VOD топ-левел protocol
        // бывает "m3u8_native+https" (замер 13.11.26, q1v4IuOoWiI) при
        // is_live=false — проверка protocol.contains("m3u8") ошибочно
        // отправляла VOD в эту ветку с «раздельные манифесты не
        // поддерживаются».
        if j.protocol.contains("m3u8") && !j.url.is_empty() {
            let f = fmt_of(&j);
            if has_video(&f) && has_audio(&f) {
                return Ok(UrlInfo {
                    is_live: true,
                    hls_url: Some(j.url.clone()),
                    audio_url: None,
                    video_url: None,
                    headers: hdrs(&j.http_headers),
                });
            }
        }
        if let Some(m) = j
            .requested_formats
            .iter()
            .find(|f| has_video(f) && has_audio(f) && f.protocol.contains("m3u8"))
        {
            return Ok(UrlInfo {
                is_live: true,
                hls_url: Some(m.url.clone()),
                audio_url: None,
                video_url: None,
                headers: hdrs(&m.http_headers),
            });
        }
        if !j.requested_formats.is_empty() {
            return Err(
                "Живой стрим отдаёт раздельные видео/аудио манифесты — источник «URL» для \
                 этого сайта пока не поддерживается (Twitch работает)"
                    .into(),
            );
        }
        return Err("Живой стрим вернулся без HLS-манифеста".into());
    }

    // VOD: только ПРЯМЫЕ https-форматы — Range-прокси /media отдаёт байты,
    // а m3u8-манифест это текст плейлиста, медианой быть не может (замер
    // 13.11.26: у YouTube-VOD все форматы либо m3u8_native, либо https,
    // муксных https (itag 18/22) может не быть вовсе — значит, нужны
    // раздельные видео+аудио https).
    let mut muxed: Option<&JFormat> = None;
    let mut audio: Option<&JFormat> = None;
    let mut video: Option<&JFormat> = None;
    for f in &j.formats {
        if f.protocol != "https" {
            continue;
        }
        let v = has_video(f);
        let a = has_audio(f);
        if v && a {
            muxed = Some(f);
        }
        if v && !a {
            video = Some(f);
        }
        if !v && a {
            audio = Some(f);
        }
    }
    // Связка локальной копии топ-уровня: Option<&JFormat> не может заимствовать
    // временный результат fmt_of (E0515).
    let single = fmt_of(&j);
    let (audio_url, video_url, headers) = if let Some(m) = muxed {
        (
            Some(m.url.clone()),
            Some(m.url.clone()),
            hdrs(&m.http_headers),
        )
    } else if let (Some(v), Some(a)) = (video, audio) {
        let h = if v.http_headers.is_empty() {
            hdrs(&a.http_headers)
        } else {
            hdrs(&v.http_headers)
        };
        (Some(a.url.clone()), Some(v.url.clone()), h)
    } else if !j.url.is_empty() && j.protocol == "https" {
        // Фолбэк: экстрактор вернул один мукс-формат топ-уровня (старые
        // YouTube / прочие сайты).
        (Some(single.url.clone()), Some(single.url.clone()), hdrs(&single.http_headers))
    } else {
        // Прямых https-потоков нет (например, Twitch-VOD — весь список m3u8):
        // Range-прокси такому источнику не поможет, честный отказ.
        return Err(
            "VOD отдаёт только HLS/DASH-манифесты — прямых медиа-URL для прокси нет \
             (YouTube поддерживается)"
                .into(),
        );
    };
    if audio_url.is_none() && video_url.is_none() {
        return Err("yt-dlp не вернул ни одного потока".into());
    }
    Ok(UrlInfo {
        is_live: false,
        hls_url: None,
        audio_url,
        video_url,
        headers,
    })
}

/// `yt-dlp -J --no-warnings [--js-runtimes node] [--proxy P] URL` → JSON.
fn run_ytdlp_json(
    app: &AppHandle,
    url: &str,
    proxy: Option<&str>,
    js_node: bool,
) -> Result<String, String> {
    let exe = ytdlp_exe(app);
    let mut cmd = Command::new(&exe);
    cmd.arg("-J").arg("--no-warnings");
    if js_node {
        cmd.args(["--js-runtimes", "node"]);
    }
    if let Some(p) = proxy.filter(|s| !s.trim().is_empty()) {
        cmd.args(["--proxy", p]);
    }
    cmd.arg(url);

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Не запустить yt-dlp ({}): {e}", exe.display()))?;

    // Дренаж обоих пайпов параллельно (см. resolve_blocking: без него ребёнок
    // блокируется на записи stdout и мой kill не спасает).
    let mut stdout_pipe = child.stdout.take().unwrap();
    let stdout_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stdout_pipe.read_to_string(&mut buf);
        buf
    });
    let mut stderr_pipe = child.stderr.take().unwrap();
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr_pipe.read_to_string(&mut buf);
        buf
    });

    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {
                if started.elapsed() > Duration::from_secs(90) {
                    kill_ytdlp_tree(&mut child);
                    let _ = stderr_thread.join();
                    let _ = stdout_thread.join();
                    return Err("yt-dlp не ответил за 90 с — источник недоступен".into());
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => return Err(format!("yt-dlp wait: {e}")),
        }
    };

    let stdout = stdout_thread.join().unwrap_or_default();
    let stderr = stderr_thread.join().unwrap_or_default();

    if !status.success() {
        let tail = stderr.lines().rev().take(3).collect::<Vec<_>>();
        let tail = tail.into_iter().rev().collect::<Vec<_>>().join("; ");
        return Err(format!(
            "yt-dlp не смог разобрать ссылку (exit {}): {}",
            status.code().unwrap_or(-1),
            if tail.is_empty() {
                "нет диагностики".to_string()
            } else {
                tail
            }
        ));
    }
    Ok(stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// YouTube-VOD 2026 (замер q1v4IuOoWiI): is_live=false, но топ-левел
    /// protocol="m3u8_native+https", муксных https нет — только раздельные.
    /// Парсер не должен попасть в live-ветку и обязан взять https-пару.
    #[test]
    fn vod_with_hls_top_protocol_uses_https_pair() {
        let raw = r#"{
            "is_live": false,
            "protocol": "m3u8_native+https",
            "url": "",
            "formats": [
                {"url": "https://v/low.mp4", "protocol": "https", "vcodec": "vp9", "acodec": "none", "http_headers": {"User-Agent": "ua1"}},
                {"url": "https://manifest.googlevideo.com/hls.m3u8", "protocol": "m3u8_native", "vcodec": "vp9", "acodec": "none"},
                {"url": "https://a/opus.webm", "protocol": "https", "vcodec": "none", "acodec": "opus", "http_headers": {"User-Agent": "ua2"}},
                {"url": "https://v/high.mp4", "protocol": "https", "vcodec": "avc1", "acodec": "none", "http_headers": {"User-Agent": "ua3"}}
            ]
        }"#;
        let info = url_info_from_json(raw).expect("VOD должен разобраться");
        assert!(!info.is_live);
        // Последний подходящий = лучший (сортировка yt-dlp от худшего к лучшему)
        assert_eq!(info.video_url.as_deref(), Some("https://v/high.mp4"));
        assert_eq!(info.audio_url.as_deref(), Some("https://a/opus.webm"));
        // Заголовки с видео-потока (не пустые)
        assert!(info.headers.iter().any(|(k, v)| k == "User-Agent" && v == "ua3"));
    }

    /// Муксный https (itag 18) закрывает оба потока одним URL.
    #[test]
    fn vod_muxed_https_preferred() {
        let raw = r#"{
            "is_live": false,
            "formats": [
                {"url": "https://a/opus.webm", "protocol": "https", "vcodec": "none", "acodec": "opus"},
                {"url": "https://v/high.mp4", "protocol": "https", "vcodec": "avc1", "acodec": "none"},
                {"url": "https://m/mux.mp4", "protocol": "https", "vcodec": "avc1", "acodec": "mp4a"}
            ]
        }"#;
        let info = url_info_from_json(raw).expect("VOD должен разобраться");
        assert_eq!(info.video_url.as_deref(), Some("https://m/mux.mp4"));
        assert_eq!(info.audio_url.as_deref(), Some("https://m/mux.mp4"));
    }

    /// Настоящий живой стрим с мукс-манифестом → hls_url (ветка is_live).
    #[test]
    fn live_muxed_manifest_goes_hls() {
        let raw = r#"{
            "is_live": true,
            "protocol": "m3u8_native",
            "url": "https://manifest.googlevideo.com/live.m3u8",
            "vcodec": "avc1", "acodec": "mp4a"
        }"#;
        let info = url_info_from_json(raw).expect("live должен разобраться");
        assert!(info.is_live);
        assert_eq!(info.hls_url.as_deref(), Some("https://manifest.googlevideo.com/live.m3u8"));
        assert!(info.video_url.is_none());
    }

    /// VOD без единого прямого https (Twitch-VOD: весь список m3u8) — честный
    /// отказ вместо URL m3u8-манифеста в video_url.
    #[test]
    fn manifest_only_vod_is_honest_error() {
        let raw = r#"{
            "is_live": false,
            "protocol": "m3u8_native",
            "url": "https://twitch-cdn/vod.m3u8",
            "vcodec": "avc1", "acodec": "mp4a",
            "formats": [
                {"url": "https://twitch-cdn/vod.m3u8", "protocol": "m3u8_native", "vcodec": "avc1", "acodec": "mp4a"}
            ]
        }"#;
        let err = url_info_from_json(raw).unwrap_err();
        assert!(err.contains("прямых медиа-URL"), "неожиданная ошибка: {err}");
    }
}
