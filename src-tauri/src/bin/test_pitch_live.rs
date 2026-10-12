//! Сверка продового определения тона с НЕЗАВИСИМЫМ ground truth на реальном аудио.
//!
//! Зачем отдельный бинарник: харнесс `cargo test` на этой машине не стартует,
//! а «посмотреть на видео глазами» — не проверка. Бинарник берёт реальную
//! запись и JSON диаризации движка и для каждого сегмента считает тон ДВАЖДЫ:
//!
//! - продовым [`deedub_lib::comm::voice::voice_track`] — тем же кодом, что
//!   работает в пайплайне;
//! - эталонным методом по **расстоянию между соседними пиками спектра**
//!   внутри гармонической полосы. Этот метод трекером не является и октавной
//!   ошибки в нём быть не может по построению: шаг между гармониками равен F0.
//!
//! Совпадение пола по обоим методам и есть критерий. Расхождение означает, что
//! продовый детектор врёт — и где именно врёт, видно по таблице.
//!
//! Запуск: test-pitch-live <wav> <engine.json>
//! Пример: test-pitch-live temp\deedub_audio.wav temp\mel20.json

use std::collections::HashMap;

use deedub_lib::comm::voice;

const HARMONIC_N: usize = 4096;
const HARMONIC_HOP: usize = HARMONIC_N / 2;
const PEAK_MIN_HZ: f64 = 70.0;
const PEAK_MAX_HZ: f64 = 1200.0;
const PEAK_REL_LEVEL: f64 = 0.12;
const MIN_SPIKES: usize = 4;
const ENERGY_FLOOR_RMS: f64 = 0.015;

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

/// Эталонный F0: медиана расстояний между соседними пиками спектра.
fn harmonic_pitch(samples: &[f32], sr: u32) -> Option<f64> {
    let filtered = voice::highpass(samples, sr, PEAK_MIN_HZ);
    let frames: Vec<&[f32]> = (0..filtered.len().saturating_sub(HARMONIC_N))
        .step_by(HARMONIC_HOP)
        .map(|i| &filtered[i..i + HARMONIC_N])
        .collect();
    if frames.is_empty() {
        return None;
    }
    let rms_of = |fr: &[f32]| -> f64 {
        fr.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / fr.len() as f64
    };
    let rms: Vec<f64> = frames.iter().map(|f| rms_of(f).sqrt()).collect();
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let gate = sorted[sorted.len() / 2] * 0.7; // громкие 30% кадров

    let mut ests: Vec<f64> = Vec::new();
    for (fr, r) in frames.iter().zip(rms.iter()) {
        if *r < gate.max(ENERGY_FLOOR_RMS) {
            continue;
        }
        let mut win: Vec<f64> = fr.iter().map(|v| *v as f64).collect();
        let n = win.len();
        for (i, v) in win.iter_mut().enumerate() {
            let hann =
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());
            *v *= hann;
        }
        let spectrum = real_fft(&win);
        let bins_per_hz = HARMONIC_N as f64 / sr as f64;
        let mut peaks: Vec<usize> = Vec::new();
        for i in 1..spectrum.len() - 1 {
            if spectrum[i] > spectrum[i - 1] && spectrum[i] >= spectrum[i + 1] {
                let hz = i as f64 / bins_per_hz;
                if (PEAK_MIN_HZ..=PEAK_MAX_HZ).contains(&hz) {
                    peaks.push(i);
                }
            }
        }
        if peaks.len() < MIN_SPIKES {
            continue;
        }
        let max = peaks
            .iter()
            .map(|&i| spectrum[i])
            .fold(f64::MIN, f64::max);
        peaks.retain(|&i| spectrum[i] > max * PEAK_REL_LEVEL);
        if peaks.len() < MIN_SPIKES {
            continue;
        }
        let mut d: Vec<f64> = peaks
            .windows(2)
            .map(|w| (w[1] - w[0]) as f64 / bins_per_hz)
            .filter(|d| (70.0..400.0).contains(d))
            .collect();
        if d.len() < 2 {
            continue;
        }
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ests.push(d[d.len() / 2]);
    }
    if ests.is_empty() {
        return None;
    }
    ests.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(ests[ests.len() / 2])
}

/// БПФ с вещественными входными данными (длина — степень двойки).
fn real_fft(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    // БПФ Кули-Тьюки: упаковка двух вещественных последовательностей в одну
    // комплексную. Для длин кадра 4096 этого хватает с большим запасом.
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    for i in 0..n / 2 {
        let a = input[2 * i];
        let b = input[2 * i + 1];
        re[i] = a;
        im[i] = b;
    }
    fft_inplace(&mut re, &mut im);
    // Восстановление спектра исходной последовательности
    let mut out = vec![0.0f64; n / 2 + 1];
    for k in 0..=n / 2 {
        let kr = (k * 2) % n;
        let ki = (n - k * 2 + 1) % n;
        let zr = re[kr] + re[ki];
        let zi = im[kr] - im[ki];
        let er = 0.5 * zr;
        let ei = -0.5 * zi;
        out[k] = (er * er + ei * ei).sqrt();
    }
    out
}

fn fft_inplace(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let wr = ang.cos();
        let wi = ang.sin();
        let mut i = 0usize;
        while i < n {
            let mut cr = 1.0f64;
            let mut ci = 0.0f64;
            for k in 0..len / 2 {
                let ur = re[i + k];
                let ui = im[i + k];
                let vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                let vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

fn gender(f0: Option<f64>) -> &'static str {
    match f0 {
        Some(hz) if hz >= voice::GENDER_MALE_BELOW_HZ => "F",
        Some(_) => "M",
        None => "-",
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let wav = args
        .next()
        .unwrap_or_else(|| "temp/deedub_audio.wav".into());
    let js = args.next().unwrap_or_else(|| "temp/mel20.json".into());

    let (samples, sr) = match load_wav(&wav) {
        Some(v) => v,
        None => {
            eprintln!("не открыть WAV: {wav}");
            std::process::exit(1);
        }
    };
    println!("аудио {wav}: {} сэмплов @ {sr} Гц", samples.len());
    println!(
        "пороги продового детектора: ФВЧ {:.0} Гц, гейт p{:.0}, приёмка {:.2}\n",
        voice::HP_CUTOFF_HZ,
        voice::VOICING_ENERGY_PERCENTILE,
        voice::YIN_ACCEPT
    );

    let raw = std::fs::read_to_string(&js).expect("читать json диаризации");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("парсить json");
    let empty = vec![];
    let segs = parsed
        .get("transcription")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);

    let mut per: HashMap<String, (Vec<f64>, Vec<f64>, usize, usize)> = HashMap::new();
    let mut agree = 0usize;
    let mut compared = 0usize;

    for s in segs {
        let spk = s
            .get("speaker")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let off = match s.get("offsets") {
            Some(o) => o,
            None => continue,
        };
        let t0 = off.get("from").and_then(|v| v.as_i64()).unwrap_or(0) as f64 / 1000.0;
        let t1 = off.get("to").and_then(|v| v.as_i64()).unwrap_or(0) as f64 / 1000.0;
        let a = (t0 * sr as f64) as usize;
        let b = ((t1 * sr as f64) as usize).min(samples.len());
        if a >= b {
            continue;
        }
        let track = voice::voice_track(&samples[a..b], sr);
        let truth = harmonic_pitch(&samples[a..b], sr);
        let acc = per
            .entry(spk.clone())
            .or_insert_with(|| (Vec::new(), Vec::new(), 0, 0));
        acc.3 += 1;
        if let Some(f) = track.f0_median {
            acc.0.push(f);
        }
        if let Some(f) = truth {
            acc.1.push(f);
            acc.2 += 1;
        }
        if let (Some(p), Some(t)) = (track.f0_median, truth) {
            compared += 1;
            if gender(Some(p)) == gender(Some(t)) {
                agree += 1;
            } else {
                println!(
                    "  расхождение {spk} {t0:.1}-{t1:.1}с: прод {:.0} ({}) против \
                     эталона по гармоникам {:.0} ({})",
                    p,
                    gender(Some(p)),
                    t,
                    gender(Some(t))
                );
            }
        }
    }

    println!(
        "\n{:<12}{:>8}{:>12}{:>12}   {}",
        "спикер", "сегм.", "прод Гц", "эталон Гц", "пол прод/эталон"
    );
    for (spk, (prod, truth, with_truth, total)) in &per {
        let mp = median(prod.clone());
        let mt = median(truth.clone());
        println!(
            "{:<12}{:>8}{:>12}{:>12}   {} / {}",
            spk,
            total,
            mp.map(|v| format!("{:.0}", v)).unwrap_or("-".into()),
            mt.map(|v| format!("{:.0}", v)).unwrap_or("-".into()),
            gender(mp),
            gender(mt),
        );
        let _ = with_truth;
    }
    println!(
        "\nсегментов с эталоном: {}, пол совпал у {} ({:.0}%)",
        compared,
        agree,
        if compared == 0 {
            0.0
        } else {
            100.0 * agree as f64 / compared as f64
        }
    );
}