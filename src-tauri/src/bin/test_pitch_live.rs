//! Сравнение детекторов F0 на РЕАЛЬНОМ аудио с метками спикеров.
//!
//! Зачем отдельный бинарник: синтетические синусы тут врут. YIN на синусах
//! даёт точность 2 Гц, а на живой речи с шумом и реверберацией срабатывает
//! на 1.9% кадров — то есть почти не работает. Проверять детектор надо на
//! записи с реальными голосами, где известен правильный ответ по тексту:
//! в диалоге реплики-вопросы и реплики-ответы чередуются.
//!
//! Печатает по каждому спикеру: покрытие (доля окон с определённым тоном),
//! медиану и распределение — для КАЖДОГО из детекторов. По этим числам
//! выбирается метод, а не на глаз и не по «красиво на синусах».
//!
//! Запуск: cargo run --bin test-pitch-live --release -- <wav> <engine.json>

use std::collections::HashMap;

const MIN_F: f64 = 50.0;
const MAX_F: f64 = 500.0;
const FRAME: usize = 1024;
const HOP: usize = 512;
/// Порог пола из продакшн-кода.
const GENDER_HZ: f64 = 160.0;

fn load_wav(path: &str) -> Option<(Vec<f32>, u32)> {
    let r = hound::WavReader::open(path).ok()?;
    let sr = r.spec().sample_rate;
    let s: Vec<f32> = r
        .into_samples::<i16>()
        .filter_map(|v| v.ok())
        .map(|v| v as f32 / 32768.0)
        .collect();
    Some((s, sr))
}

/// Текущий продовский метод: глобальный максимум нормированной автокорреляции.
/// Возвращает все тона кадра (медиана берётся вызывающим).
fn autocorr_pitches(frame: &[f32], sr: u32) -> Vec<f64> {
    let min_lag = (sr as f64 / MAX_F) as usize;
    let max_lag = ((sr as f64 / MIN_F) as usize).min(frame.len() / 2);
    let energy: f32 = frame.iter().map(|x| x * x).sum();
    if energy < 1e-6 || max_lag <= min_lag {
        return Vec::new();
    }
    let mut out = Vec::new();
    for lag in min_lag..=max_lag {
        let mut corr = 0.0f32;
        for i in 0..(frame.len() - lag) {
            corr += frame[i] * frame[i + lag];
        }
        let lag_energy: f32 = frame[lag..].iter().map(|x| x * x).sum();
        let denom = (energy * lag_energy).sqrt();
        if denom > 1e-8 && corr / denom > 0.3 {
            out.push(sr as f64 / lag as f64);
        }
    }
    out
}

/// Пик автокорреляции + проверка на октаву: если на lag/2 корреляция почти
/// такая же, истинный период вдвое короче. Именно это даёт женские 220 Гц,
/// которые читаются как 110 Гц.
fn autocorr_octave(frame: &[f32], sr: u32, tol: f32) -> Vec<f64> {
    let min_lag = (sr as f64 / MAX_F) as usize;
    let max_lag = ((sr as f64 / MIN_F) as usize).min(frame.len() / 2);
    let energy: f32 = frame.iter().map(|x| x * x).sum();
    if energy < 1e-6 || max_lag <= min_lag {
        return Vec::new();
    }
    let mut corr = vec![0.0f32; max_lag + 1];
    let mut best = 0.0f32;
    let mut best_lag = min_lag;
    for lag in min_lag..=max_lag {
        let mut c = 0.0f32;
        for i in 0..(frame.len() - lag) {
            c += frame[i] * frame[i + lag];
        }
        let lag_energy: f32 = frame[lag..].iter().map(|x| x * x).sum();
        let denom = (energy * lag_energy).sqrt();
        if denom > 1e-8 {
            c /= denom;
        }
        corr[lag] = c;
        if c > best {
            best = c;
            best_lag = lag;
        }
    }
    if best <= 0.3 {
        return Vec::new();
    }
    let mut lag = best_lag;
    while lag / 2 >= min_lag {
        let half = lag / 2;
        let is_peak = corr[half] >= best * tol
            && (half == min_lag || corr[half] >= corr[half - 1])
            && corr[half] >= corr[half + 1];
        if !is_peak {
            break;
        }
        lag = half;
    }
    vec![sr as f64 / lag as f64]
}

/// Медиана по окнам сегмента — так же, как это делает прод.
fn segment_pitch(
    samples: &[f32],
    sr: u32,
    a: usize,
    b: usize,
    f: impl Fn(&[f32], u32) -> Vec<f64>,
) -> Option<f64> {
    let mut all = Vec::new();
    let mut pos = a;
    while pos + FRAME <= b {
        all.extend(f(&samples[pos..pos + FRAME], sr));
        pos += HOP;
    }
    median(all)
}

/// Разностная функция YIN (straightforward, без инкрементальной формулы —
/// та завязана на другую индексацию окна и даёт мусор).
fn yin_pitches(frame: &[f32], sr: u32, accept: f32, fallback: f32) -> Vec<f64> {
    let min_tau = (sr as f64 / MAX_F) as usize;
    let max_tau = ((sr as f64 / MIN_F) as usize).min(frame.len() / 2);
    let n = frame.len();
    if n < max_tau * 2 + 2 || max_tau <= min_tau {
        return Vec::new();
    }
    let mut d = vec![0.0f32; max_tau + 1];
    for (tau, slot) in d.iter_mut().enumerate().take(max_tau + 1).skip(1) {
        let mut acc = 0.0f32;
        for j in 0..(n - tau) {
            let diff = frame[j] - frame[j + tau];
            acc += diff * diff;
        }
        *slot = acc;
    }
    let mut cmnd = vec![1.0f32; max_tau + 1];
    let mut running = 0.0f32;
    for tau in 1..=max_tau {
        running += d[tau];
        if running > 1e-12 {
            cmnd[tau] = d[tau] * tau as f32 / running;
        }
    }
    for tau in min_tau..max_tau {
        if cmnd[tau] < accept && cmnd[tau] < cmnd[tau - 1] && cmnd[tau] < cmnd[tau + 1] {
            return vec![sr as f64 / tau as f64];
        }
    }
    let mut best = min_tau;
    for tau in min_tau..=max_tau {
        if cmnd[tau] < cmnd[best] {
            best = tau;
        }
    }
    if cmnd[best] < fallback {
        vec![sr as f64 / best as f64]
    } else {
        Vec::new()
    }
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(v[v.len() / 2])
}

fn segment_span(sr: u32, t0: f64, t1: f64) -> (usize, usize) {
    ((t0 * sr as f64) as usize, (t1 * sr as f64) as usize)
}

fn main() {
    // Продовский детектор — та же функция, что в пайплайне, а не её копия.
    let prod = deedub_lib::diarization::estimate_median_pitch;
    let mut args = std::env::args().skip(1);
    let wav = args.next().unwrap_or_else(|| "temp/deedub_audio.wav".into());
    let js = args.next().unwrap_or_default();

    let (samples, sr) = match load_wav(&wav) {
        Some(v) => v,
        None => {
            eprintln!("не открыть WAV: {wav}");
            std::process::exit(1);
        }
    };
    println!("аудио {wav}: {} сэмплов @ {sr} Гц", samples.len());

    // Сегменты движка: считаем F0 по каждому окну внутри сегмента.
    let raw = std::fs::read_to_string(&js).expect("читать json");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("парсить json");
    let empty = vec![];
    let segs = parsed
        .get("transcription")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);

    struct Acc {
        ac: Vec<f64>,
        oct: Vec<f64>,
        yin: Vec<f64>,
        windows: usize,
    }
    let mut per: HashMap<String, Acc> = HashMap::new();
    let mut total_windows = 0usize;

    for s in segs {
        let spk = s.get("speaker").and_then(|v| v.as_str()).unwrap_or("?").to_string();
        let off = match s.get("offsets") {
            Some(o) => o,
            None => continue,
        };
        let t0 = off.get("from").and_then(|v| v.as_i64()).unwrap_or(0) as f64 / 1000.0;
        let t1 = off.get("to").and_then(|v| v.as_i64()).unwrap_or(0) as f64 / 1000.0;
        let (a, b) = segment_span(sr, t0, t1);
        if b <= a + FRAME || b > samples.len() {
            continue;
        }
        let acc = per.entry(spk).or_insert(Acc {
            ac: Vec::new(),
            oct: Vec::new(),
            yin: Vec::new(),
            windows: 0,
        });
        let mut pos = a;
        while pos + FRAME <= b {
            let fr = &samples[pos..pos + FRAME];
            acc.windows += 1;
            total_windows += 1;
            let ac = autocorr_pitches(fr, sr);
            // медиана окна = текущее продовое поведение
            if let Some(m) = median(ac.clone()) {
                acc.ac.push(m);
            }
            acc.oct.extend(autocorr_octave(fr, sr, 0.85));
            acc.yin.extend(yin_pitches(fr, sr, 0.15, 0.6));
            pos += HOP;
        }
    }

    println!("всего окон: {total_windows}\n");
    println!(
        "{:<12}{:>7}{:>10}{:>10}{:>10}   {}",
        "спикер", "окон", "покр. ac", "покр. oct", "покр. yin", "медианы ac / oct / yin (Гц → пол)"
    );
    for (spk, acc) in &per {
        let n = acc.windows.max(1);
        let (m_ac, m_oct, m_yin) = (
            median(acc.ac.clone()),
            median(acc.oct.clone()),
            median(acc.yin.clone()),
        );
        let fmt = |m: Option<f64>| match m {
            Some(v) => format!("{:.0}{}", v, if v < GENDER_HZ { "M" } else { "F" }),
            None => "-".to_string(),
        };
        println!(
            "{:<12}{:>7}{:>9.0}%{:>9.0}%{:>9.0}%   {} / {} / {}",
            spk,
            acc.windows,
            acc.ac.len() as f64 / n as f64 * 100.0,
            acc.oct.len() as f64 / n as f64 * 100.0,
            acc.yin.len() as f64 / n as f64 * 100.0,
            fmt(m_ac),
            fmt(m_oct),
            fmt(m_yin),
        );
    }

    // --- Сверка прода с эталонной октавной поправкой ---
    // СЕЙЧАС это справочная таблица, а не проверка: в проде октавной поправки
    // нет (она чинит Melanie и ломает cecelia — см. комментарий в
    // `estimate_median_pitch`). Столбец `oct` показывает, сколько спикеров
    // изменило бы пол при её включении, чтобы решение принимали по числам.
    println!("\n=== справочно: что изменила бы октавная поправка ===");
    let mut changed = 0usize;
    let mut checked = 0usize;
    for s in segs {
        let spk = s.get("speaker").and_then(|v| v.as_str()).unwrap_or("?");
        let off = match s.get("offsets") {
            Some(o) => o,
            None => continue,
        };
        let t0 = off.get("from").and_then(|v| v.as_i64()).unwrap_or(0) as f64 / 1000.0;
        let t1 = off.get("to").and_then(|v| v.as_i64()).unwrap_or(0) as f64 / 1000.0;
        let (a, b) = segment_span(sr, t0, t1);
        if b <= a + FRAME || b > samples.len() || b - a > 20 * sr as usize {
            continue;
        }
        let expected =
            segment_pitch(&samples, sr, a, b, |fr, s| autocorr_octave(fr, s, 0.85));
        let got = prod(&samples[a..b], sr);
        if let (Some(e), Some(g)) = (expected, got) {
            checked += 1;
            if (e < GENDER_HZ) != (g < GENDER_HZ) {
                changed += 1;
                if changed <= 5 {
                    println!(
                        "  {spk} {t0:.1}-{t1:.1}s: прод {g:.0} ({}) против поправки {e:.0} ({})",
                        if g < GENDER_HZ { "M" } else { "F" },
                        if e < GENDER_HZ { "M" } else { "F" }
                    );
                }
            }
        }
    }
    println!("сегментов сверено: {checked}, пол изменился бы у: {changed}");
}