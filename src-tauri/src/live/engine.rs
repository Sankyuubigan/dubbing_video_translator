//! Жизненный цикл STT-движка crispasr для live-режима (шаг 2 PRP).
//!
//! Один процесс на live-сессию: `--server --ws-port P` + parakeet (EN-сабы).
//! Поднимается в фоне после live_start (не блокирует старт плеера), гасится
//! teardown'ом сессии — VRAM освобождается (desktop §6.5). Раскладка аргументов
//! повторяет проверенный спавн плагина speech (stt.rs::ensure) и диаризации:
//! `--backend` — тип ASR-модели, НЕ cpu/cuda (handoff §Технические находки 1).
//!
//! Раскладка портов (проба 0.8.43): HTTP-сервер на `--port P`, «стриминговый»
//! WS плагина — по факту мёртв (GET / → 404), рабочий вход — vLLM Realtime
//! `ws://...:P+1/v1/realtime` (dedicated флага нет; `--ws-port 0` = «port+1»
//! подтверждает схему). Отказ «failed to start vLLM Realtime API» наблюдён
//! разово, когда P+1 был занят чем-то другим — поэтому резервируем ПАРУ портов
//! и после /health проверяем, что realtime реально слушает P+1.
//!
//! Нарезка по паузам (правило №6 PRP): `--vad` + `--stream-vad-merge-gap-ms`
//! 150 мс + `--stream-final-on-silence-ms 500` → финалы по границам фраз.
//! Замер на реальном аудио q1v4IuOoWiI (3 мин): gap 350 давал 8 финалов со
//! средней 21.6 с (худшая 63 с / 1104 символа — суб-«простыня»); gap 150 +
//! финал 500 → 18 финалов со средней 9.2 с. `--vad-max-speech-duration-s`
//! в стриминге НЕ работает (проверено: 8 → 8 финалов) — длинную непрерывную
//! речь без пауз движок всё равно склеит, остаток дробит audio.rs (кап фразы).
//! `--stream-final-mode redecode` (дефолт) собирает финальный текст по всей
//! фразе заново — лучшее качество.

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Таймаут готовности движка: parakeet-q4_k грузится секунд десять-двадцать.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(120);

pub struct SttEngine {
    pub child: Child,
    pub ws_port: u16,
}

impl SttEngine {
    /// Спавн движка в серверном режиме + ожидание /health. Блокирующе до
    /// готовности — вызывать из фонового таска, не из команды live_start.
    pub fn spawn() -> Result<Self, String> {
        let exe = crate::pick_engine_exe()?;
        let model = crate::resolve_stt_model()?;
        let ws_port = pick_free_ports()?;

        let mut args = vec![
            "--server".to_string(),
            "--backend".to_string(),
            "parakeet".to_string(),
            "-m".to_string(),
            model,
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            ws_port.to_string(),
            "--ws-port".to_string(),
            ws_port.to_string(),
            // Нарезка по паузам: финал после 500 мс тишины, слияние только
            // зазоров < 150 мс (замер: gap 350 давал фразы по 20-60 с).
            "--vad".to_string(),
            "--stream-vad-merge-gap-ms".to_string(),
            "150".to_string(),
            "--stream-final-on-silence-ms".to_string(),
            "500".to_string(),
            // Финал пересобирается по всей фразе (лучше качество, дефолт движка).
            "--stream-final-mode".to_string(),
            "redecode".to_string(),
            // Отметки происхождения отключаем, как у всех вызовов движка.
            "--no-watermark".to_string(),
            "--accept-marking-responsibility".to_string(),
            "--no-spoken-disclaimer".to_string(),
        ];

        log::info!(
            "Live-STT: стартует crispasr (HTTP :{ws_port}, realtime ws :{}): {} {}",
            ws_port + 1,
            exe,
            args.join(" ")
        );

        let mut cmd = Command::new(&exe);
        cmd.args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = cmd.spawn().map_err(|e| format!("Live-STT: не запустить crispasr: {e}"))?;

        // stderr → общий лог (движок пишет загрузку модели и ошибки туда).
        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    log::info!("Live-STT: [crispasr] {line}");
                }
            });
        }

        Self::wait_health(&mut child, ws_port)?;
        log::info!("Live-STT: движок готов (ws-port {ws_port})");
        Ok(Self { child, ws_port })
    }

    /// GET /health до успеха или выхода процесса (как stt.rs::ensure плагина).
    /// Готовность = /health отвечает И realtime-WS слушает ws_port+1 (проба
    /// 0.8.43: «failed to start vLLM Realtime API» бывает при занятом P+1 —
    /// тогда сабы бы молча не пошли).
    fn wait_health(child: &mut Child, ws_port: u16) -> Result<(), String> {
        let started = Instant::now();
        loop {
            if started.elapsed() > HEALTH_TIMEOUT {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Live-STT: движок не поднялся за 120 с (лог crispasr — в test/last_logs.txt)".into());
            }
            if let Ok(Some(st)) = child.try_wait() {
                return Err(format!(
                    "Live-STT: crispasr завершился сразу (exit={:?}) — проверьте модель parakeet",
                    st.code()
                ));
            }
            if health_ok(ws_port) && tcp_ok(ws_port + 1) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(300));
        }
    }
}

impl Drop for SttEngine {
    fn drop(&mut self) {
        // kill + wait: VRAM освобождается, движок не остаётся сиротой.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// GET /health по сырому TCP. Спавн живёт в spawn_blocking — async-reqwest
/// там не нужен; HTTP/1.0 достаточно, сервер отвечает строкой статуса.
fn health_ok(ws_port: u16) -> bool {
    use std::io::{Read, Write};
    let addr = format!("127.0.0.1:{ws_port}");
    let Ok(mut stream) = std::net::TcpStream::connect(&addr) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let req = b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n";
    if stream.write_all(req).is_err() {
        return false;
    }
    let mut buf = [0u8; 512];
    let mut head = Vec::new();
    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                head.extend_from_slice(&buf[..n]);
                if head.len() > 4096 {
                    break;
                }
            }
        }
    }
    // Только строка статуса — в теле JSON « 200» мог бы совпасть случайно.
    let line = String::from_utf8_lossy(&head);
    let status = line.split("\r\n").next().unwrap_or("");
    status.starts_with("HTTP/") && status.contains("200")
}

/// Свободная ПАРА смежных портов (P, P+1) ВНЕ эфемерного диапазона: realtime
/// движка биндится на P+1 через сотни мс после спавна (warmup), и в машине с
/// динамическим диапазоном 10000-39999 исходящий коннект за это время успевает
/// забрать P+1 (наблюдено 2/2 прогонов: «failed to start vLLM Realtime API on
/// port 14817»). bind(:0) показывает голову аллокатора; пара ищется в полосе
/// НАД диапазоном, если голова ниже 40000, или НИЖЕ — если голова в 45000+.
/// Гонка с исходящими коннектами устранена принципиально (вне диапазона их нет).
fn pick_free_ports() -> Result<u16, String> {
    // Голова эфемерного аллокатора: порт, который bind(:0) выдаёт сейчас.
    let probe = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Live-STT: не найти свободный порт: {e}"))?;
    let head = probe
        .local_addr()
        .map_err(|e| format!("Live-STT: не узнать порт: {e}"))?
        .port();
    drop(probe);
    // Полоса поиска пары: над головой эфемерного диапазона, если он ниже 40000
    // (наша машина: 10000-39999 → полоса 45000+), или под ним — если эфемерный
    // диапазон лежит в 45000+ (дефолт Windows 49152-65535 → полоса 10000-39000).
    let (lo, hi) = if head < 40000 {
        (45500u16, 63999u16)
    } else {
        (10051u16, 39000u16)
    };
    log::info!("Live-STT: голова эфемерных портов ~{head}, ищу пару в {lo}..{hi}");
    for p in lo..=hi {
        if p == 54288 || p + 1 == 54288 {
            continue; // исключённый диапазон этой машины (netsh excludedportrange)
        }
        let l1 = TcpListener::bind(("127.0.0.1", p));
        let l2 = TcpListener::bind(("127.0.0.1", p + 1));
        match (l1, l2) {
            (Ok(a), Ok(b)) => {
                drop(a);
                drop(b);
                return Ok(p);
            }
            (Ok(a), Err(_)) => {
                drop(a);
                continue; // P свободен, P+1 занят — пара не годится
            }
            (Err(_), _) => continue, // P занят — следующий
        }
    }
    Err("Live-STT: не найти свободную пару портов (P, P+1)".into())
}

/// TCP-соединение прошло — кто-то слушает порт (realtime-сервер не отвечает
/// на GET /health, но bound-порт принимает соединение).
fn tcp_ok(port: u16) -> bool {
    std::net::TcpStream::connect(format!("127.0.0.1:{port}")).is_ok()
}
