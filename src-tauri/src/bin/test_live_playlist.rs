//! Проверка парсинга/сборки HLS-плейлистов и сетевого логгера live-режима.
//!
//! Отдельный бинарник, а не `cargo test`: харнесс на этой машине не стартует
//! (0xc0000139 STATUS_ENTRYPOINT_NOT_FOUND). Зовём production-функции
//! `live::playlist` и `live::net_log` напрямую — тест не расходится с кодом.

use deedub_lib::live::audio::{
    f32_to_s16_le_bytes, le_bytes_to_f32, parse_rt_event, pdt_to_epoch_secs, RtEvent, Ring,
};
use deedub_lib::live::digits::digits_to_words;
use deedub_lib::live::net_log::{classify, domain_of};
use deedub_lib::live::playlist::{
    build_media_m3u8, parse_master, parse_media, pick_variant, ServedSeg,
};
use deedub_lib::live::server::is_tts_file_name;
use deedub_lib::live::url_resolve;
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};

static FAILED: AtomicBool = AtomicBool::new(false);

fn check(name: &str, ok: bool, detail: String) {
    if ok {
        println!("  ok   — {name}");
    } else {
        println!("  FAIL — {name}: {detail}");
        FAILED.store(true, Ordering::Relaxed);
    }
}

/// Twitch-плейлист с рекламным блоком и разрывами (упрощённый РЕАЛЬНЫЙ ответ
/// video-weaver 12.11.26: fMP4/CMAF — EXT-X-MAP init, PDT, DATERANGE-реклама,
/// DISCONTINUITY, PREFETCH).
const WEAVER_SAMPLE: &str = "#EXTM3U\n\
    #EXT-X-VERSION:6\n\
    #EXT-X-TARGETDURATION:2\n\
    #EXT-X-MEDIA-SEQUENCE:120\n\
    #EXT-X-MAP:URI=\"https://cdn/init.mp4\"\n\
    #EXT-X-PROGRAM-DATE-TIME:2026-11-12T10:00:00.000Z\n\
    #EXTINF:2.000,\n\
    https://video-weaver.b01.hls.ttvnw.net/v1/playlist/abc.mp4\n\
    #EXT-X-DISCONTINUITY\n\
    #EXT-X-DATERANGE:ID=\"stitched-ad-1\",CLASS=\"twitch-stitched-ad\",START-DATE=\"2026-11-12T10:00:02.000Z\",X-TV-TWITCH-AD-LENGTH=\"30.0\"\n\
    #EXT-X-PROGRAM-DATE-TIME:2026-11-12T10:00:02.000Z\n\
    #EXTINF:2.000,\n\
    https://video-weaver.b01.hls.ttvnw.net/v1/playlist/sand-1.mp4\n\
    #EXT-X-TWITCH-PREFETCH:https://video-weaver.b01.hls.ttvnw.net/v1/playlist/prefetch.mp4\n\
    #EXT-X-DISCONTINUITY\n\
    #EXT-X-PROGRAM-DATE-TIME:2026-11-12T10:00:32.000Z\n\
    #EXTINF:2.000,\n\
    https://video-weaver.b01.hls.ttvnw.net/v1/playlist/def.mp4\n";

fn main() {
    println!("test-live-playlist: HLS-плейлисты + сетевые логи live-режима\n");

    // --- 1. Master-плейлист usher: варианты и выбор качества ---------------
    let master = "#EXTM3U\n\
        #EXT-X-STREAM-INF:BANDWIDTH=6500000,RESOLUTION=1920x1080,CODECS=\"avc1,mp4a\",NAME=\"chunked\"\n\
        https://weaver/chunked.m3u8\n\
        #EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1280x720,CODECS=\"avc1,mp4a\",NAME=\"720p60\"\n\
        https://weaver/720p60.m3u8\n\
        #EXT-X-STREAM-INF:BANDWIDTH=160000,CODECS=\"mp4a\",NAME=\"audio_only\"\n\
        https://weaver/audio_only.m3u8\n";
    let variants = parse_master(master);
    check("master: три варианта", variants.len() == 3, format!("got {}", variants.len()));
    check(
        "master: имя и bandwidth первого",
        variants[0].name == "chunked" && variants[0].bandwidth == 6500000,
        format!("{:?}", variants[0]),
    );
    check(
        "pick source → chunked",
        pick_variant(&variants, "source").map(|v| v.name.clone()).as_deref() == Some("chunked"),
        "не выбрал chunked".into(),
    );
    check(
        "pick 720p → 720p60",
        pick_variant(&variants, "720p").map(|v| v.name.clone()).as_deref() == Some("720p60"),
        "не выбрал 720p60".into(),
    );
    check(
        "pick audio_only",
        pick_variant(&variants, "audio_only").map(|v| v.name.clone()).as_deref()
            == Some("audio_only"),
        "не выбрал audio_only".into(),
    );
    check(
        "pick неизвестного качества → фолбэк на максимальный не-audio",
        pick_variant(&variants, "480p").map(|v| v.name.clone()).as_deref() == Some("chunked"),
        "фолбэк сломан".into(),
    );

    // --- 1b. Живой формат usher (захвачен 12.11.26): NAME нет в STREAM-INF,
    //         имена в #EXT-X-MEDIA, ссылка через VIDEO="group" -----------------
    let master_real = "#EXTM3U\n\
        #EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID=\"1080p60\",NAME=\"1080p60\",AUTOSELECT=YES,DEFAULT=YES\n\
        #EXT-X-STREAM-INF:BANDWIDTH=8042999,RESOLUTION=1920x1080,CODECS=\"avc1,mp4a\",VIDEO=\"1080p60\",FRAME-RATE=60.000\n\
        https://weaver/1080p60.m3u8\n\
        #EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID=\"720p60\",NAME=\"720p60\",AUTOSELECT=YES,DEFAULT=YES\n\
        #EXT-X-STREAM-INF:BANDWIDTH=3322199,RESOLUTION=1280x720,CODECS=\"avc1,mp4a\",VIDEO=\"720p60\",FRAME-RATE=60.000\n\
        https://weaver/720p60.m3u8\n\
        #EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID=\"audio_only\",NAME=\"audio_only\",AUTOSELECT=NO,DEFAULT=NO\n\
        #EXT-X-STREAM-INF:BANDWIDTH=160000,CODECS=\"mp4a\",VIDEO=\"audio_only\"\n\
        https://weaver/audio_only.m3u8\n";
    let real = parse_master(master_real);
    check(
        "usher-реал: имена из EXT-X-MEDIA",
        real.len() == 3
            && real[0].name == "1080p60"
            && real[1].name == "720p60"
            && real[2].name == "audio_only",
        format!("{:?}", real.iter().map(|v| v.name.clone()).collect::<Vec<_>>()),
    );
    check(
        "usher-реал: source → 1080p60 (фолбэк по bandwidth)",
        pick_variant(&real, "source").map(|v| v.name.clone()).as_deref() == Some("1080p60"),
        "не выбрал 1080p60".into(),
    );
    check(
        "usher-реал: 720p → 720p60",
        pick_variant(&real, "720p").map(|v| v.name.clone()).as_deref() == Some("720p60"),
        "не выбрал 720p60".into(),
    );

    // --- 2. Media-плейлист: реклама, разрывы, PREFETCH, PDT ----------------
    let (target, segs) = parse_media(WEAVER_SAMPLE);
    check("target duration = 2.0", target == 2.0, format!("got {target}"));
    check(
        "3 сегмента (PREFETCH не сегмент)",
        segs.len() == 3,
        format!("got {}", segs.len()),
    );
    check(
        "первый сегмент — контент без разрыва, PDT и fMP4 init есть",
        !segs[0].is_ad
            && !segs[0].discontinuity
            && segs[0].pdt.is_some()
            && segs[0].map_uri.as_deref() == Some("https://cdn/init.mp4"),
        format!("{:?}", segs[0]),
    );
    check(
        "второй сегмент помечен рекламой (DATERANGE stitched-ad + sand- в URI)",
        segs[1].is_ad && segs[1].discontinuity,
        format!("{:?}", segs[1]),
    );
    check(
        "третий — контент после разрыва, реклама закрылась",
        !segs[2].is_ad && segs[2].discontinuity,
        format!("{:?}", segs[2]),
    );

    // --- 3. Скользящая сборка: MEDIA-SEQUENCE, разрывы, fMP4 init, без ENDLIST
    let window = vec![
        ServedSeg {
            seq: 1,
            file: "seg_000001.m4s".into(),
            duration: 2.0,
            discontinuity: false,
            is_ad: false,
            map_file: Some("init_000000.m4s".into()),
            pdt: Some(1794506995.29),
        },
        ServedSeg { seq: 2, file: "seg_000002.m4s".into(), duration: 2.0, discontinuity: true, is_ad: true, map_file: None, pdt: None },
        ServedSeg { seq: 3, file: "seg_000003.m4s".into(), duration: 2.0, discontinuity: true, is_ad: false, map_file: Some("init_000001.m4s".into()), pdt: Some(1794507032.0) },
    ];
    let m3u8 = build_media_m3u8(&window, 41, 2);
    check(
        "заголовок с MEDIA-SEQUENCE:41, TARGETDURATION:2, VERSION:6",
        m3u8.contains("#EXT-X-MEDIA-SEQUENCE:41")
            && m3u8.contains("#EXT-X-TARGETDURATION:2")
            && m3u8.contains("#EXT-X-VERSION:6"),
        format!("{m3u8}"),
    );
    check(
        "init первого сегмента окна объявлен EXT-X-MAP",
        m3u8.contains("#EXT-X-MAP:URI=\"seg/init_000000.m4s\"\n#EXT-X-PROGRAM-DATE-TIME:"),
        format!("{m3u8}"),
    );
    check(
        "PDT-тег есть (playingDate фронта без него null) и отформатирован RFC3339",
        m3u8.matches("#EXT-X-PROGRAM-DATE-TIME:").count() == 2
            && m3u8.contains("#EXT-X-PROGRAM-DATE-TIME:2026-11-"),
        format!("{m3u8}"),
    );
    check(
        "разрыв и смена init на месте (маршрут /seg/{file})",
        m3u8.contains("#EXT-X-DISCONTINUITY\n#EXT-X-MAP:URI=\"seg/init_000001.m4s\""),
        format!("{m3u8}"),
    );
    check(
        "ENDLIST нет — живой поток",
        !m3u8.contains("ENDLIST"),
        "ENDLIST появился".into(),
    );
    check(
        "все 3 EXTINF",
        m3u8.matches("EXTINF").count() == 3,
        format!("got {}", m3u8.matches("EXTINF").count()),
    );

    // --- 4. Окно из одного сегмента: MEDIA-SEQUENCE не двигается ------------
    let single = vec![ServedSeg {
        seq: 0,
        file: "seg_000000.ts".into(),
        duration: 2.0,
        discontinuity: false,
        is_ad: false,
        map_file: Some("init_000000.m4s".into()),
        pdt: None,
    }];
    let m3u8 = build_media_m3u8(&single, 0, 2);
    check(
        "одиночное окно собирается с sequence 0",
        m3u8.contains("#EXT-X-MEDIA-SEQUENCE:0") && m3u8.matches("EXTINF").count() == 1,
        format!("{m3u8}"),
    );

    // --- 5. net_log: домен и классификация ---------------------------------
    check(
        "domain_of usher",
        domain_of("https://usher.ttvnw.net/api/channel/hls/x.m3u8") == "usher.ttvnw.net",
        format!("got {}", domain_of("https://usher.ttvnw.net/api/channel/hls/x.m3u8")),
    );
    check(
        "domain_of loopback",
        domain_of("http://127.0.0.1:8123/live.m3u8") == "127.0.0.1",
        format!("got {}", domain_of("http://127.0.0.1:8123/live.m3u8")),
    );
    check(
        "HTTP 403 → антибот, не DPI",
        classify("HTTP 403 Forbidden").contains("антибот"),
        format!("got {}", classify("HTTP 403 Forbidden")),
    );
    check(
        "timeout → возможен DPI-блок",
        classify("operation timed out").contains("DPI"),
        format!("got {}", classify("operation timed out")),
    );
    check(
        "DNS-ошибка классифицируется",
        classify("failed to lookup address information").contains("DNS"),
        format!("got {}", classify("failed to lookup address information")),
    );

    // --- 6. audio.rs: PDT→эпоха, LE-байты→f32, кольцо сэмплов ---------------
    let p0 = pdt_to_epoch_secs("2026-11-12T10:00:00.000Z");
    let p2 = pdt_to_epoch_secs("2026-11-12T10:00:02.000Z");
    let poff = pdt_to_epoch_secs("2026-11-12T12:00:00.000+02:00");
    check(
        "pdt_to_epoch_secs: Z и +02:00 — одна точка, дельта 2 с",
        p0.is_some() && p2.unwrap() - p0.unwrap() == 2.0 && poff == p0,
        format!("{p0:?} {p2:?} {poff:?}"),
    );
    check(
        "pdt_to_epoch_secs: мусор → None",
        pdt_to_epoch_secs("не время").is_none(),
        "распарсил мусор".into(),
    );

    let f32_bytes: Vec<u8> = [1.5f32, -2.25, 0.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    check(
        "le_bytes_to_f32: рантрип + хвост <4 байт отброшен",
        le_bytes_to_f32(&f32_bytes) == vec![1.5, -2.25, 0.0]
            && le_bytes_to_f32(&[0u8, 0, 0]).is_empty(),
        format!("{:?}", le_bytes_to_f32(&f32_bytes)),
    );

    let ring = Ring::default();
    let mut consumed = 0u64;
    ring.push(&[1.0, 2.0, 3.0]);
    let got = ring.take_new(&mut consumed, 10);
    check(
        "кольцо: пуш → выдача → исчерпание",
        got == vec![1.0, 2.0, 3.0]
            && consumed == 3
            && ring.take_new(&mut consumed, 10).is_empty(),
        format!("{got:?} consumed={consumed}"),
    );
    ring.push(&[4.0, 5.0]);
    let g1 = ring.take_new(&mut consumed, 1);
    let g2 = ring.take_new(&mut consumed, 10);
    check(
        "кольцо: лимит max=1 и продолжение",
        g1 == vec![4.0] && g2 == vec![5.0] && consumed == 5,
        format!("{g1:?} {g2:?} consumed={consumed}"),
    );
    // Подрезка спереди: 960_001 сэмплов (cap 60 с × 16 кГц = 960_000) →
    // base сдвинулся, отставший потребитель перепрыгивает на актуальное
    // место по абсолютным индексам — без паники и без утечки.
    let big = vec![0.0f32; 960_001];
    ring.push(&big);
    let mut lagged = 0u64;
    let got = ring.take_new(&mut lagged, 10);
    // buf: 5 старых + 960_001 − 6 подрезанных = 960_000, base = 6 → lagged = 16.
    check(
        "кольцо: подрезка переживает абсолютные индексы",
        got.len() == 10 && got.iter().all(|&s| s == 0.0) && lagged == 16,
        format!("len={} lagged={lagged}", got.len()),
    );
    ring.set_eof();
    check(
        "кольцо: eof флаг",
        ring.is_eof(),
        "eof не выставился".into(),
    );

    // --- 7. audio.rs: f32→s16le (PCM16 realtime-протокола), разбор событий --
    // 0.5→16383.5→16384 (round от нуля), 1.5/-1.5 зажимаются в [-1..1].
    let s16 = f32_to_s16_le_bytes(&[0.5, -0.5, 1.5, -1.5, 0.0]);
    let s16_vals: Vec<i16> = s16
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    check(
        "f32_to_s16: масштаб/зажим/round",
        s16_vals == vec![16384, -16384, 32767, -32767, 0],
        format!("{s16_vals:?}"),
    );
    check(
        "f32_to_s16: длина 2 байта/сэмпл",
        s16.len() == 10,
        format!("len={}", s16.len()),
    );

    let ev = parse_rt_event(&serde_json::json!({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "Yeah.",
        "audio_duration_ms": 1000
    }));
    check(
        "parse_rt_event: completed с transcript и длительностью",
        ev == RtEvent::Completed { transcript: "Yeah.".into(), audio_duration_ms: 1000.0 },
        format!("{ev:?}"),
    );
    check(
        "parse_rt_event: speech_started/stopped",
        parse_rt_event(&serde_json::json!({"type": "input_audio_buffer.speech_started"}))
            == RtEvent::SpeechStarted
            && parse_rt_event(&serde_json::json!({"type": "input_audio_buffer.speech_stopped"}))
                == RtEvent::SpeechStopped,
        "started/stopped не распознаны".into(),
    );
    check(
        "parse_rt_event: delta и session.created → Other",
        parse_rt_event(&serde_json::json!({"type": "conversation.item.input_audio_transcription.delta", "delta": "I"}))
            == RtEvent::Other
            && parse_rt_event(&serde_json::json!({"type": "session.created"})) == RtEvent::Other,
        "побочные события не Other".into(),
    );
    let ev = parse_rt_event(&serde_json::json!({
        "type": "conversation.item.input_audio_transcription.completed"
    }));
    check(
        "parse_rt_event: completed без полей → пусто/0.0",
        ev == RtEvent::Completed { transcript: String::new(), audio_duration_ms: 0.0 },
        format!("{ev:?}"),
    );

    // ── Цифры → слова перед TTS (live/digits.rs, шаг 4) ─────────────────────
    println!("digits_to_words:");
    check(
        "целое число",
        digits_to_words("Сезон 10 стартует") == "Сезон десять стартует",
        format!("{:?}", digits_to_words("Сезон 10 стартует")),
    );
    check(
        "дробная часть по цифрам",
        digits_to_words("рейтинг 9.5") == "рейтинг девять и пять",
        format!("{:?}", digits_to_words("рейтинг 9.5")),
    );
    check(
        "запятая как разделитель дроби",
        digits_to_words("2,25 метра") == "два и два пять метра",
        format!("{:?}", digits_to_words("2,25 метра")),
    );
    check(
        "миллионы",
        digits_to_words("продано 1500000 копий") == "продано один миллион пятьсот тысяч копий",
        format!("{:?}", digits_to_words("продано 1500000 копий")),
    );
    check(
        "женские тысячи (одна/две тысячи)",
        digits_to_words("2000 и 2001") == "две тысячи и две тысячи один",
        format!("{:?}", digits_to_words("2000 и 2001")),
    );
    check(
        "ноль",
        digits_to_words("осталось 0") == "осталось ноль",
        format!("{:?}", digits_to_words("осталось 0")),
    );
    check(
        "точка без следующей цифры — не дробь",
        digits_to_words("конец 1.") == "конец один.",
        format!("{:?}", digits_to_words("конец 1.")),
    );
    check(
        "больше 9 цифр — вернуть как было",
        digits_to_words("id 1234567890123") == "id 1234567890123",
        format!("{:?}", digits_to_words("id 1234567890123")),
    );
    check(
        "текст без цифр не меняется",
        digits_to_words("привет мир") == "привет мир",
        format!("{:?}", digits_to_words("привет мир")),
    );

    // ── Относительные URI CDN (live/mod.rs url_resolve, шаг 5) ─────────────
    println!("url_resolve:");
    check(
        "абсолютный URI — как есть",
        url_resolve("https://a.com/v1/playlist/x.m3u8", "https://cdn.example.com/seg.ts")
            == "https://cdn.example.com/seg.ts",
        "прошло".into(),
    );
    check(
        "относительный — join к базе",
        url_resolve("https://a.com/v1/playlist/x.m3u8", "seg_000001.ts")
            == "https://a.com/v1/playlist/seg_000001.ts",
        url_resolve("https://a.com/v1/playlist/x.m3u8", "seg_000001.ts"),
    );
    check(
        "корневой /path — от хоста",
        url_resolve("https://a.com/v1/playlist/x.m3u8", "/init.mp4")
            == "https://a.com/init.mp4",
        url_resolve("https://a.com/v1/playlist/x.m3u8", "/init.mp4"),
    );
    check(
        "битая база — URI как есть",
        url_resolve("not a url", "seg_000002.ts") == "seg_000002.ts",
        "прошло".into(),
    );

    // ── Раздача озвучки /tts/{file} (live/server.rs, шаг 4) ────────────────
    println!("is_tts_file_name:");
    check(
        "tts_1.wav — да",
        is_tts_file_name("tts_1.wav"),
        "прошло".into(),
    );
    check(
        "tts_12345.wav — да",
        is_tts_file_name("tts_12345.wav"),
        "прошло".into(),
    );
    check(
        "tts_.wav (пустой id) — нет",
        !is_tts_file_name("tts_.wav"),
        "прошло".into(),
    );
    check(
        "tts_abc.wav — нет",
        !is_tts_file_name("tts_abc.wav"),
        "прошло".into(),
    );
    check(
        "имя чужого маршрута — нет",
        !is_tts_file_name("seg_000001.ts") && !is_tts_file_name("tts_1.mp3"),
        "прошло".into(),
    );

    println!();
    if FAILED.load(Ordering::Relaxed) {
        println!("TEST FAILED");
        exit(1);
    }
    println!("ALL PASS");
    println!("TEST OK");
}
