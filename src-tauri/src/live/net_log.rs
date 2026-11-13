//! Сетевое логирование live-модуля (требование «Отчет для специалиста» §2.4).
//!
//! Пользователь в РФ: Twitch/YouTube деградируют/блокируются РКН, домены
//! вносятся вручную в сторонний DPI-bypass инструмент. Поэтому ЛЮБАЯ сетевая
//! ошибка live-режима обязана попасть в лог с полным доменом назначения и
//! классификацией причины: Timeout / TCP RST / TLS EOF — похоже на DPI-блок,
//! HTTP 403 — антибот самого сервиса. Это разные проблемы с разными решениями.

use std::fmt::Display;

/// Домен из URL: `https://usher.ttvnw.net/api/...` → `usher.ttvnw.net`,
/// `http://127.0.0.1:8123/x` → `127.0.0.1` (порт срезается, IPv6 в скобках
/// остаётся в форме `[::1]`). Некорректный URL → строка «<unparsable>».
pub fn domain_of(url: &str) -> String {
    let after_scheme = match url.find("://") {
        Some(i) => &url[i + 3..],
        None => url,
    };
    let host = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    let host = match host.rsplit_once('@') {
        Some((_userinfo, h)) => h, // на случай user:pass@host
        None => host,
    };
    // Порт: последняя ':' и только цифры после неё (IPv6 литерал в скобках не трогаем).
    let host = if let Some(i) = host.rfind(':') {
        let (h, p) = host.split_at(i);
        let p = &p[1..];
        if !h.is_empty() && !h.contains(':') && p.chars().all(|c| c.is_ascii_digit()) {
            h
        } else {
            host
        }
    } else {
        host
    };
    if host.is_empty() {
        "<unparsable>".to_string()
    } else {
        host.to_string()
    }
}

/// Классификация причины по тексту ошибки. Строки — из Display reqwest/std
/// («operation timed out», «connection reset by peer», «connection closed
/// before message completed» и т.п.), плюс наши собственные `HTTP <code>`
/// для не-200 ответов.
pub fn classify(err_text: &str) -> &'static str {
    let t = err_text.to_ascii_lowercase();
    if t.contains("403") {
        "HTTP 403 — антибот сервиса (не DPI)"
    } else if t.contains("timeout") || t.contains("timed out") {
        "Timeout — возможен DPI-блок"
    } else if t.contains("connection reset") {
        "TCP RST — возможен DPI-блок"
    } else if t.contains("connection refused") {
        "Connection refused — сервис не отвечает"
    } else if t.contains("eof") || t.contains("closed before message") {
        "TLS/HTTP EOF — возможен DPI-блок"
    } else if t.contains("dns")
        || t.contains("resolve")
        || t.contains("nodename")
        || t.contains("lookup address")
        || t.contains("getaddrinfo")
    {
        "DNS — домен не резолвится"
    } else if t.contains("404") {
        "HTTP 404 — ресурс не найден"
    } else {
        "прочая сетевая ошибка"
    }
}

/// Единая точка лога сетевой ошибки: домен + причина + исходный текст.
pub fn log_net_fail(ctx: &str, url: &str, err: &impl Display) {
    let text = err.to_string();
    log::error!(
        "NET-FAIL [{}] {} (домен: {}, причина: {}): {}",
        ctx,
        url,
        domain_of(url),
        classify(&text),
        text
    );
}

/// Лог не-200 HTTP-ответа (у reqwest .error_for_status() даёт текст со статусом,
/// но явный вызов читаемее: сразу видно код и домен).
pub fn log_http_status(ctx: &str, url: &str, status: u16) {
    let text = format!("HTTP {status}");
    log::error!(
        "NET-FAIL [{}] {} (домен: {}, причина: {})",
        ctx,
        url,
        domain_of(url),
        classify(&text)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_of_basic() {
        assert_eq!(domain_of("https://usher.ttvnw.net/api/channel/hls/x.m3u8"), "usher.ttvnw.net");
        assert_eq!(domain_of("http://127.0.0.1:8080/live.m3u8"), "127.0.0.1");
        assert_eq!(domain_of("https://user:pw@host.example.com/a?b=c"), "host.example.com");
        assert_eq!(domain_of("nonsense"), "nonsense");
    }

    #[test]
    fn classify_kinds() {
        assert!(classify("HTTP 403 Forbidden").contains("антибот"));
        assert!(classify("operation timed out").contains("DPI"));
        assert!(classify("failed to lookup address information").contains("DNS"));
    }
}
