//! Резолвер HLS-потока Twitch: GQL (анонимный playback access token) → usher
//! master-плейлист → выбор варианта. Сегменты качает наш код (контроль буфера
//! задержки D), внешние резолверы (streamlink/yt-dlp) подключаются на шаге 5.
//!
//! Известные риски (специалист, §8 в.1): антибот Twitch может потребовать
//! Client-Integrity токен (чаще с VPN/прокси IP) — тогда GQL вернёт ошибку
//! access_token, и пользователь может ввести свой OAuth auth-token в настройках
//! Live-вкладки (с ним Twitch почти не требует integrity). Все сетевые ошибки
//! логируются через net_log с доменом и причиной (требование §2.4).

use std::fmt::Display;

use serde_json::json;

use super::net_log;
use super::playlist::{parse_master, pick_variant};

/// Публичный client_id веб-клиента Twitch (используется streamlink/yt-dlp
/// сообществом для анонимного доступа к playback access token).
const GQL_CLIENT_ID: &str = "kimne78kx3ncx6brgo4mv6wki5h1ko";
const GQL_URL: &str = "https://gql.twitch.tv/gql";
/// Известный sha256 persisted-запроса PlaybackAccessToken (streamlink).
/// Twitch периодически меняет хэши — при ошибке PersistedQueryNotFound
/// пользователь увидит понятную ошибку с доменом, а лог подскажет причину.
const PERSISTED_QUERY_HASH: &str =
    "0828119ded1c13477966434e15800ff57ddacf13ba1911c129dc2200705b0712";

/// HTTP-клиент live-сессии: таймаут + опциональный прокси (SOCKS5/HTTP —
/// совет специалиста §8 в.9: значительная часть РФ-пользователей гонит трафик
/// через локальный Sing-box/Nekoray порт).
pub fn build_client(proxy: Option<&str>) -> Result<reqwest::Client, String> {
    let mut b = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) DeeDub/1.0");
    if let Some(p) = proxy.map(str::trim).filter(|s| !s.is_empty()) {
        let pr = reqwest::Proxy::all(p).map_err(|e| {
            net_log::log_net_fail("proxy", p, &e);
            format!("Некорректный прокси {p}: {e}")
        })?;
        b = b.proxy(pr);
        log::info!("Live: сетевые запросы через прокси {p}");
    }
    b.build().map_err(|e| {
        net_log::log_net_fail("client", "http://client-build", &e);
        format!("Не удалось создать HTTP-клиент: {e}")
    })
}

/// Клиент с заголовками от yt-dlp (браузерный UA и пр.) — для url-сессий:
/// googlevideo web-класса отвечает и без них, но заголовки не мешают, а для
/// прочих сайтов они обязательны.
pub fn build_client_with(
    proxy: Option<&str>,
    headers: &[(String, String)],
) -> Result<reqwest::Client, String> {
    let mut b = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) DeeDub/1.0");
    if let Some(p) = proxy.map(str::trim).filter(|s| !s.is_empty()) {
        let pr = reqwest::Proxy::all(p).map_err(|e| {
            net_log::log_net_fail("proxy", p, &e);
            format!("Некорректный прокси {p}: {e}")
        })?;
        b = b.proxy(pr);
        log::info!("Live: сетевые запросы через прокси {p}");
    }
    if !headers.is_empty() {
        let mut hm = reqwest::header::HeaderMap::new();
        for (k, v) in headers {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                reqwest::header::HeaderValue::from_str(v),
            ) {
                hm.insert(name, val);
            }
        }
        b = b.default_headers(hm);
    }
    b.build().map_err(|e| {
        net_log::log_net_fail("client", "http://client-build", &e);
        format!("Не удалось создать HTTP-клиент: {e}")
    })
}

pub struct ResolvedStream {
    /// URL media-плейлиста выбранного варианта (video-weaver).
    pub media_url: String,
    /// Имя выбранного варианта (для статуса).
    pub variant: String,
}

/// Полный резолв с явным выбором качества: GQL-токен → usher → master →
/// вариант. Ошибка содержит домен и причину — она же уходит в UI.
/// Наличие канала и оффлайн определяются по usher (404), т.к. этот
/// persisted-запрос отдаёт только токен, без данных о пользователе.
pub async fn resolve_with_quality(
    client: &reqwest::Client,
    channel: &str,
    quality: &str,
    oauth_token: Option<&str>,
) -> Result<ResolvedStream, String> {
    let (value, signature) = fetch_access_token(client, channel, oauth_token).await?;
    let usher_url = build_usher_url(channel, &value, &signature);
    let master = match fetch_text(client, "usher-master", &usher_url).await {
        Ok(t) => t,
        Err(e) if e.contains("HTTP 404") => {
            return Err(format!(
                "Канал '{channel}' не найден на Twitch или сейчас оффлайн (usher вернул 404)"
            ));
        }
        Err(e) => return Err(e),
    };
    let variants = parse_master(&master);
    if variants.is_empty() {
        return Err(format!(
            "usher не вернул вариантов качества (домен: {})",
            net_log::domain_of(&usher_url)
        ));
    }
    let v = pick_variant(&variants, quality)
        .ok_or_else(|| "usher вернул пустой список вариантов".to_string())?;
    log::info!(
        "Live: канал {channel} — вариант '{}' ({} кбит/с) из {} доступных",
        v.name,
        v.bandwidth / 1000,
        variants.len()
    );
    Ok(ResolvedStream { media_url: v.url.clone(), variant: v.name.clone() })
}

async fn fetch_access_token(
    client: &reqwest::Client,
    channel: &str,
    oauth_token: Option<&str>,
) -> Result<(String, String), String> {
    let body = json!({
        "operationName": "PlaybackAccessToken",
        "variables": {
            "login": channel,
            "isLive": true,
            "vodID": "",
            "isVod": false,
            "playerType": "site"
        },
        "extensions": {
            "persistedQuery": {
                "version": 1,
                "sha256Hash": PERSISTED_QUERY_HASH
            }
        }
    });
    let mut req = client.post(GQL_URL).header("Client-ID", GQL_CLIENT_ID);
    if let Some(tok) = oauth_token.map(str::trim).filter(|s| !s.is_empty()) {
        // Токен пользователя в лог НЕ пишем никогда.
        log::info!("Live: используется OAuth-токен пользователя");
        req = req.header("Authorization", format!("OAuth {tok}"));
    } else {
        log::info!("Live: анонимный GQL-запрос токена (канал {channel})");
    }
    let resp = req.json(&body).send().await;
    let resp = resp.map_err_ctx("gql", GQL_URL)?;
    parse_gql_response(resp, channel).await
}

async fn parse_gql_response(
    resp: reqwest::Response,
    _channel: &str,
) -> Result<(String, String), String> {
    let status = resp.status();
    if !status.is_success() {
        net_log::log_http_status("gql", GQL_URL, status.as_u16());
        let hint = if status.as_u16() == 403 {
            " — возможно, потребуется OAuth-токен в настройках Live"
        } else {
            ""
        };
        return Err(format!("GQL вернул HTTP {}{hint}", status.as_u16()));
    }
    let v: serde_json::Value = resp.json().await.map_err_ctx("gql-json", GQL_URL)?;
    if let Some(errs) = v.get("errors") {
        let msg = errs.to_string();
        net_log::log_http_status("gql-errors", GQL_URL, 400);
        let hint = if msg.contains("PersistedQueryNotFound") {
            " — Twitch сменил хэш persisted-запроса; укажите OAuth-токен или обновите приложение"
        } else {
            ""
        };
        return Err(format!("GQL ошибка: {msg}{hint}"));
    }
    // Ответ этого persisted-запроса: data.streamPlaybackAccessToken.{value,signature}
    // (НЕ data.user.stream — путь из другого варианта GQL-запроса, у нас его нет).
    let pat = v.pointer("/data/streamPlaybackAccessToken");
    let value = pat
        .and_then(|p| p.get("value"))
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string();
    let signature = pat
        .and_then(|p| p.get("signature"))
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string();
    if value.is_empty() || signature.is_empty() {
        return Err(
            "GQL не отдал playbackAccessToken (возможно, требуется Client-Integrity) \
             — попробуйте OAuth-токен в настройках Live"
                .to_string(),
        );
    }
    Ok((value, signature))
}

fn build_usher_url(channel: &str, token: &str, sig: &str) -> String {
    let mut u = reqwest::Url::parse(&format!(
        "https://usher.ttvnw.net/api/channel/hls/{channel}.m3u8"
    ))
    .expect("usher URL всегда валиден");
    {
        let mut qp = u.query_pairs_mut();
        qp.append_pair("token", token);
        qp.append_pair("sig", sig);
        qp.append_pair("allow_source", "true");
        qp.append_pair("allow_audio_only", "true");
    }
    u.to_string()
}

/// GET текстового ресурса (плейлисты). Лог с доменом на любой ошибке.
pub(super) async fn fetch_text(
    client: &reqwest::Client,
    ctx: &str,
    url: &str,
) -> Result<String, String> {
    let resp = client.get(url).send().await;
    let resp = resp.map_err_ctx(ctx, url)?;
    let status = resp.status();
    if !status.is_success() {
        net_log::log_http_status(ctx, url, status.as_u16());
        return Err(format!("[{ctx}] HTTP {}", status.as_u16()));
    }
    resp.text().await.map_err_ctx(ctx, url)
}

/// GET бинарного сегмента → Vec<u8>.
pub(super) async fn fetch_bytes(
    client: &reqwest::Client,
    ctx: &str,
    url: &str,
) -> Result<Vec<u8>, String> {
    let resp = client.get(url).send().await;
    let resp = resp.map_err_ctx(ctx, url)?;
    let status = resp.status();
    if !status.is_success() {
        net_log::log_http_status(ctx, url, status.as_u16());
        return Err(format!("[{ctx}] HTTP {}", status.as_u16()));
    }
    resp.bytes().await.map(|b| b.to_vec()).map_err_ctx(ctx, url)
}

/// Лог + контекст для Result сетевых операций (единый формат ошибок).
trait ErrCtx<T> {
    fn map_err_ctx(self, ctx: &str, url: &str) -> Result<T, String>;
}

impl<T, E: Display> ErrCtx<T> for Result<T, E> {
    fn map_err_ctx(self, ctx: &str, url: &str) -> Result<T, String> {
        self.map_err(|e| {
            net_log::log_net_fail(ctx, url, &e);
            format!("[{ctx}] {e}")
        })
    }
}
