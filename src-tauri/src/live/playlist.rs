//! Парсинг и сборка HLS-плейлистов для live-режима (чистые функции, без сети
//! и файлов — проверяются тест-бинарником `test_live_playlist`).
//!
//! Ключевые решения по поправкам специалиста (ai-studio-chat §Грабли 2, 3):
//! - реклама Twitch (`twitch-stitched-ad` / `stitched-ad` в URI) помечается
//!   флагом и НЕ должна попадать в STT/MT/TTS (шаг 2+), но в плеер идёт —
//!   иначе на ad-break у зрителя дыра в видео;
//! - `EXT-X-DISCONTINUITY` переносится в наш скользящий плейлист как есть —
//!   без него декодер рвётся на смене профиля кодирования (реклама);
//! - скользящая сборка (EXT-X-MEDIA-SEQUENCE + окно) вместо бесконечного
//!   файла: браузерный currentTime привязан к PTS внутри .ts, а не к номеру
//!   строки плейлиста.

use chrono::TimeZone;

/// Вариант качества из master-плейлиста.
#[derive(Clone, Debug)]
pub struct Variant {
    pub name: String,
    pub bandwidth: u64,
    pub url: String,
}

/// Парс master-плейлиста (ответ usher). Имя варианта:
/// - NAME="chunked" в самом `#EXT-X-STREAM-INF` — старый формат;
/// - у живого usher NAME в STREAM-INF НЕТ — имя лежит в предшествующем
///   `#EXT-X-MEDIA:TYPE=VIDEO,...,GROUP-ID="720p60",NAME="720p60"`, а STREAM-INF
///   ссылается на группу атрибутом `VIDEO="720p60"`. Сопоставляем так.
pub fn parse_master(text: &str) -> Vec<Variant> {
    let mut out = Vec::new();
    let mut group_names: std::collections::HashMap<String, String> = Default::default();
    // (bandwidth, name из STREAM-INF, video-группа)
    let mut cur: Option<(u64, String, String)> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if let Some(attrs) = line.strip_prefix("#EXT-X-MEDIA:") {
            let mut group = String::new();
            let mut name = String::new();
            for kv in split_attrs(attrs) {
                if let Some((k, v)) = kv.split_once('=') {
                    let k = k.trim();
                    let v = v.trim().trim_matches('"');
                    match k {
                        "GROUP-ID" => group = v.to_string(),
                        "NAME" => name = v.to_string(),
                        _ => {}
                    }
                }
            }
            if !group.is_empty() && !name.is_empty() {
                group_names.insert(group, name);
            }
        } else if let Some(attrs) = line.strip_prefix("#EXT-X-STREAM-INF:") {
            let mut bandwidth = 0u64;
            let mut name = String::new();
            let mut video = String::new();
            for kv in split_attrs(attrs) {
                if let Some((k, v)) = kv.split_once('=') {
                    let k = k.trim();
                    let v = v.trim().trim_matches('"');
                    match k {
                        "BANDWIDTH" => bandwidth = v.parse().unwrap_or(0),
                        "NAME" => name = v.to_string(),
                        "VIDEO" => video = v.to_string(),
                        _ => {}
                    }
                }
            }
            cur = Some((bandwidth, name, video));
        } else if !line.is_empty() && !line.starts_with('#') {
            if let Some((bw, name, video)) = cur.take() {
                let name = if !name.is_empty() {
                    name
                } else if !video.is_empty() {
                    group_names.get(&video).cloned().unwrap_or(video)
                } else {
                    String::new()
                };
                out.push(Variant { name, bandwidth: bw, url: line.to_string() });
            }
        }
    }
    out
}

/// Разбиение атрибутов HLS-тега по запятым с учётом кавычек.
fn split_attrs(attrs: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in attrs.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                cur.push(ch);
            }
            ',' if !in_quotes => {
                parts.push(std::mem::take(&mut cur));
            }
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

/// Сегмент media-плейлиста с признаками, нужными пайплайну.
#[derive(Clone, Debug)]
pub struct MediaSegment {
    pub uri: String,
    pub duration: f64,
    /// EXT-X-PROGRAM-DATE-TIME сегмента (абсолютное время потока) — источник
    /// правды времени для замера задержки и будущего маппинга озвучки.
    pub pdt: Option<String>,
    /// Сегмент начинает новый блок (перед ним был разрыв кодека/таймлайна).
    pub discontinuity: bool,
    /// Рекламная вставка Twitch: в плеер идёт, в STT/MT/TTS не пойдёт.
    pub is_ad: bool,
    /// URI EXT-X-MAP, активный у этого сегмента (fMP4/CMAF init-сегмент;
    /// Twitch с 2025 отдаёт fMP4, не MPEG-TS). Действует до следующей MAP-строки.
    pub map_uri: Option<String>,
}

/// Парс media-плейлиста (ответ video-weaver). Возвращает
/// (EXT-X-TARGETDURATION или вычисленный максимум, сегменты в порядке плейлиста).
/// PREFETCH-строки Twitch (EXT-X-TWITCH-PREFETCH) игнорируются — это подсказки
/// о будущих сегментах, не сегменты.
pub fn parse_media(text: &str) -> (f64, Vec<MediaSegment>) {
    let mut target: f64 = 0.0;
    let mut segs: Vec<MediaSegment> = Vec::new();
    let mut cur_dur: Option<f64> = None;
    let mut cur_pdt: Option<String> = None;
    let mut disc = false;
    let mut ad_class_active = false;
    let mut cur_map: Option<String> = None;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(v) = line.strip_prefix("#EXT-X-TARGETDURATION:") {
            target = v.trim().parse().unwrap_or(0.0);
        } else if line.starts_with("#EXT-X-TWITCH-PREFETCH") {
            continue;
        } else if line.starts_with("#EXT-X-DISCONTINUITY") {
            disc = true;
            ad_class_active = false; // разрыв закрывает предыдущий DATERANGE-блок
        } else if let Some(v) = line.strip_prefix("#EXT-X-MAP:") {
            // fMP4/CMAF init: применяется к сегментам до следующей MAP-строки.
            cur_map = extract_attr(v, "URI");
        } else if let Some(attrs) = line.strip_prefix("#EXT-X-DATERANGE:") {
            // Признак рекламы Twitch: CLASS="twitch-stitched-ad" (специалист,
            // Грабли №2). DATERANGE относится к сегментам, идущим следом.
            if attrs.contains("twitch-stitched-ad") || attrs.contains("stitched-ad") {
                ad_class_active = true;
            }
        } else if let Some(v) = line.strip_prefix("#EXT-X-PROGRAM-DATE-TIME:") {
            cur_pdt = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("#EXTINF:") {
            let d = v.split(',').next().unwrap_or(v).trim().parse().unwrap_or(0.0);
            cur_dur = Some(d);
        } else if !line.starts_with('#') {
            // строка URI сегмента
            let dur = cur_dur.take().unwrap_or(0.0);
            let is_ad = ad_class_active
                || line.contains("stitched-ad")
                || line.contains("/sand-");
            segs.push(MediaSegment {
                uri: line.to_string(),
                duration: dur,
                pdt: cur_pdt.take(),
                discontinuity: disc,
                is_ad,
                map_uri: cur_map.clone(),
            });
            disc = false;
            ad_class_active = false;
        }
    }

    let max_dur = segs.iter().map(|s| s.duration).fold(0.0f64, f64::max);
    if target <= 0.0 {
        target = max_dur.ceil();
    }
    (target, segs)
}

/// Значение атрибута `KEY="VALUE"` из тела HLS-тега (после префикса `#EXT-X-…:`).
fn extract_attr(body: &str, key: &str) -> Option<String> {
    split_attrs(body).into_iter().find_map(|kv| {
        kv.split_once('=')
            .filter(|(k, _)| k.trim() == key)
            .map(|(_, v)| v.trim().trim_matches('"').to_string())
    })
}

/// Сегмент в нашем скользящем плейлисте.
#[derive(Clone, Debug)]
pub struct ServedSeg {
    /// Локальный порядковый номер (растёт с каждого нового сегмента).
    pub seq: u64,
    /// Имя файла в папке сессии (отдаётся как относительный URL).
    pub file: String,
    pub duration: f64,
    pub discontinuity: bool,
    pub is_ad: bool,
    /// Имя init-файла (fMP4/CMAF), который надо объявить `#EXT-X-MAP` перед
    /// этим сегментом: Some у первого сегмента окна и у каждого сегмента после
    /// смены MAP. None — активен init предыдущего сегмента окна.
    pub map_file: Option<String>,
    /// PDT сегмента (эпоха, с) — уходит в наш плейлист тегом
    /// EXT-X-PROGRAM-DATE-TIME. Без него hls.js `playingDate` = null, и фронт
    /// не может сопоставить фразы (шкала эпох) с позицией плеера — сабы и
    /// дубляж на twitch не показываются никогда (баг 12.11.26, вечерний
    /// прогон юзера). None — сегмент без якоря (тег не пишется).
    pub pdt: Option<f64>,
}

/// Эпоха (с) → RFC3339 UTC с миллисекундами (`2026-11-12T10:00:00.000Z`) —
/// формат тега EXT-X-PROGRAM-DATE-TIME. Вне диапазона chrono → None.
pub fn epoch_to_pdt_rfc3339(secs: f64) -> Option<String> {
    let floor = secs.floor();
    let nsec = ((secs - floor) * 1_000_000_000.0).round() as u32;
    chrono::Utc
        .timestamp_opt(floor as i64, nsec)
        .single()
        .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string())
}

/// Сборка скользящего live-плейлиста из окна сегментов (поправка специалиста
/// №3: окно + инкремент MEDIA-SEQUENCE, без ENDLIST — это живой поток).
/// URI сегментов — `seg/<файл>`: маршрут loopback-сервера `/seg/{file}`.
pub fn build_media_m3u8(segs: &[ServedSeg], media_sequence: u64, target_duration: u64) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("#EXTM3U\n#EXT-X-VERSION:6\n");
    s.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", target_duration.max(1)));
    s.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", media_sequence));
    for seg in segs {
        if seg.discontinuity {
            s.push_str("#EXT-X-DISCONTINUITY\n");
        }
        if let Some(mf) = &seg.map_file {
            s.push_str(&format!("#EXT-X-MAP:URI=\"seg/{mf}\"\n"));
        }
        // PDT обязателен: hls.js по нему даёт playingDate — единственный
        // мост между шкалой эпох фраз (live-subs/live-ru) и плеером.
        if let Some(p) = seg.pdt {
            if let Some(tag) = epoch_to_pdt_rfc3339(p) {
                s.push_str(&format!("#EXT-X-PROGRAM-DATE-TIME:{tag}\n"));
            }
        }
        s.push_str(&format!("#EXTINF:{:.3},\nseg/{}\n", seg.duration, seg.file));
    }
    s
}

/// Выбор варианта по строке качества UI. `source` → NAME "chunked" (имя
/// source-варианта у Twitch) либо максимальный bandwidth; `audio_only` →
/// NAME "audio_only"; числовые («720p») → имя начинается с префикса.
/// Вариант не найден → максимальный не-audio (чтобы плеер всё равно работал).
pub fn pick_variant<'a>(variants: &'a [Variant], quality: &str) -> Option<&'a Variant> {
    if variants.is_empty() {
        return None;
    }
    let q = quality.trim();
    let audio_only = |v: &&Variant| v.name == "audio_only";
    let found = if q == "audio_only" {
        variants.iter().find(audio_only)
    } else if q == "source" {
        variants
            .iter()
            .find(|v| v.name == "chunked" && !audio_only(v))
            .or_else(|| variants.iter().filter(|v| !audio_only(v)).max_by_key(|v| v.bandwidth))
    } else {
        variants
            .iter()
            .find(|v| v.name.to_ascii_lowercase().starts_with(&q.to_ascii_lowercase()))
            .or_else(|| variants.iter().filter(|v| !audio_only(v)).max_by_key(|v| v.bandwidth))
    };
    Some(found.unwrap_or_else(|| variants.first().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_master_twitch() {
        let master = "#EXTM3U\n\
            #EXT-X-STREAM-INF:BANDWIDTH=6500000,RESOLUTION=1920x1080,CODECS=\"avc1,mp4a\",NAME=\"chunked\"\n\
            https://weaver/chunked.m3u8\n\
            #EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1280x720,CODECS=\"avc1,mp4a\",NAME=\"720p60\"\n\
            https://weaver/720p60.m3u8\n\
            #EXT-X-STREAM-INF:BANDWIDTH=160000,CODECS=\"mp4a\",NAME=\"audio_only\"\n\
            https://weaver/audio_only.m3u8\n";
        let v = parse_master(master);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].name, "chunked");
        assert_eq!(v[0].bandwidth, 6500000);
        assert_eq!(v[2].url, "https://weaver/audio_only.m3u8");
        assert_eq!(pick_variant(&v, "source").unwrap().name, "chunked");
        assert_eq!(pick_variant(&v, "720p").unwrap().name, "720p60");
        assert_eq!(pick_variant(&v, "audio_only").unwrap().name, "audio_only");
        assert_eq!(pick_variant(&v, "480p").unwrap().name, "chunked"); // фолбэк
    }

    /// Живой ответ usher (захвачен 12.11.26, канал bogur): NAME в STREAM-INF
    /// НЕТ, имена вариантов живут в #EXT-X-MEDIA, ссылка — атрибут VIDEO.
    #[test]
    fn parse_master_usher_real_shape() {
        let master = "#EXTM3U\n\
            #EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID=\"1080p60\",NAME=\"1080p60\",AUTOSELECT=YES,DEFAULT=YES\n\
            #EXT-X-STREAM-INF:BANDWIDTH=8042999,RESOLUTION=1920x1080,CODECS=\"avc1,mp4a\",VIDEO=\"1080p60\",FRAME-RATE=60.000\n\
            https://weaver/1080p60.m3u8\n\
            #EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID=\"720p60\",NAME=\"720p60\",AUTOSELECT=YES,DEFAULT=YES\n\
            #EXT-X-STREAM-INF:BANDWIDTH=3322199,RESOLUTION=1280x720,CODECS=\"avc1,mp4a\",VIDEO=\"720p60\",FRAME-RATE=60.000\n\
            https://weaver/720p60.m3u8\n\
            #EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID=\"audio_only\",NAME=\"audio_only\",AUTOSELECT=NO,DEFAULT=NO\n\
            #EXT-X-STREAM-INF:BANDWIDTH=160000,CODECS=\"mp4a\",VIDEO=\"audio_only\"\n\
            https://weaver/audio_only.m3u8\n";
        let v = parse_master(master);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].name, "1080p60");
        assert_eq!(v[1].name, "720p60");
        assert_eq!(v[2].name, "audio_only");
        assert_eq!(pick_variant(&v, "source").unwrap().name, "1080p60"); // фолбэк по bandwidth
        assert_eq!(pick_variant(&v, "720p").unwrap().name, "720p60");
        assert_eq!(pick_variant(&v, "audio_only").unwrap().name, "audio_only");
    }

    #[test]
    fn parse_media_ads_and_discontinuity() {
        let media = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-VERSION:6\n\
            #EXT-X-MEDIA-SEQUENCE:100\n\
            #EXT-X-MAP:URI=\"https://cdn/init.mp4\"\n\
            #EXT-X-PROGRAM-DATE-TIME:2026-11-12T10:00:00.000Z\n\
            #EXTINF:2.000,\nhttps://cdn/seg1.mp4\n\
            #EXTINF:2.000,\nhttps://cdn/seg2.mp4\n\
            #EXT-X-DISCONTINUITY\n\
            #EXT-X-DATERANGE:ID=\"stitched-ad-1\",CLASS=\"twitch-stitched-ad\",START-DATE=\"2026-11-12T10:00:04.000Z\",X-TV-TWITCH-AD-LENGTH=\"30.0\"\n\
            #EXT-X-PROGRAM-DATE-TIME:2026-11-12T10:00:04.000Z\n\
            #EXTINF:2.000,\nhttps://cdn/adsand-1.mp4\n\
            #EXT-X-DISCONTINUITY\n\
            #EXT-X-PROGRAM-DATE-TIME:2026-11-12T10:00:34.000Z\n\
            #EXTINF:2.000,\nhttps://cdn/seg3.mp4\n";
        let (target, segs) = parse_media(media);
        assert_eq!(target, 2.0);
        assert_eq!(segs.len(), 4);
        assert!(!segs[0].is_ad && !segs[0].discontinuity);
        assert_eq!(segs[0].pdt.as_deref(), Some("2026-11-12T10:00:00.000Z"));
        assert!(
            segs[0].map_uri.as_deref() == Some("https://cdn/init.mp4"),
            "fMP4 init должен быть захвачен: {:?}",
            segs[0].map_uri
        );
        assert!(segs[2].is_ad, "рекламный блок должен быть помечен");
        assert!(segs[2].discontinuity);
        assert!(!segs[3].is_ad, "после разрыва реклама кончилась");
        assert!(segs[3].discontinuity);
        assert_eq!(segs[3].pdt.as_deref(), Some("2026-11-12T10:00:34.000Z"));
    }

    /// Смена EXT-X-MAP в середине плейлиста: сегменты после неё получают новый
    /// init (рекламные вставки могут идти с собственным init-сегментом).
    #[test]
    fn parse_media_map_change() {
        let media = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n\
            #EXT-X-MAP:URI=\"https://cdn/initA.mp4\"\n\
            #EXTINF:2.000,\nhttps://cdn/a1.mp4\n\
            #EXT-X-MAP:URI=\"https://cdn/initB.mp4\"\n\
            #EXTINF:2.000,\nhttps://cdn/b1.mp4\n";
        let (_, segs) = parse_media(media);
        assert_eq!(segs[0].map_uri.as_deref(), Some("https://cdn/initA.mp4"));
        assert_eq!(segs[1].map_uri.as_deref(), Some("https://cdn/initB.mp4"));
    }

    #[test]
    fn parse_media_prefetch_ignored() {
        let media = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n\
            #EXTINF:2.000,\nhttps://cdn/seg1.ts\n\
            #EXT-X-TWITCH-PREFETCH:https://cdn/seg2.ts\n";
        let (_, segs) = parse_media(media);
        assert_eq!(segs.len(), 1, "PREFETCH — не сегмент");
    }

    #[test]
    fn build_media_sliding() {
        let segs = vec![
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
        let text = build_media_m3u8(&segs, 41, 2);
        assert!(text.starts_with("#EXTM3U"));
        assert!(text.contains("#EXT-X-VERSION:6"), "fMP4/CMAF требует версию 6");
        assert!(text.contains("#EXT-X-MEDIA-SEQUENCE:41"));
        assert!(text.contains("#EXT-X-TARGETDURATION:2"));
        assert!(
            text.contains("#EXT-X-MAP:URI=\"seg/init_000000.m4s\"\n#EXT-X-PROGRAM-DATE-TIME:"),
            "init + PDT первого сегмента окна объявлены: {text}"
        );
        assert!(
            text.contains("#EXT-X-PROGRAM-DATE-TIME:2026-11-12T"),
            "PDT отформатирован RFC3339 UTC: {text}"
        );
        assert!(
            text.matches("#EXT-X-PROGRAM-DATE-TIME:").count() == 2,
            "PDT только у сегментов с якорем: {text}"
        );
        assert!(
            text.contains("#EXT-X-DISCONTINUITY\n#EXT-X-MAP:URI=\"seg/init_000001.m4s\""),
            "после разрыва сменился init: {text}"
        );
        assert!(!text.contains("ENDLIST"), "живой поток без ENDLIST");
        assert_eq!(text.matches("EXTINF").count(), 3);
    }

    #[test]
    fn epoch_to_pdt_rfc3339_shapes() {
        let t = epoch_to_pdt_rfc3339(1794506995.29).unwrap();
        assert!(
            t.starts_with("2026-11-") && t.ends_with("Z") && t.contains('.'),
            "формат RFC3339 UTC с дробью: {t}"
        );
        // Целые секунды — без хвостовых нанос-артефактов.
        assert_eq!(epoch_to_pdt_rfc3339(0.0).unwrap(), "1970-01-01T00:00:00.000Z");
        // Вне диапазона chrono (до эпохи) — None, тег не пишется.
        assert!(epoch_to_pdt_rfc3339(-1e18).is_none());
    }
}
