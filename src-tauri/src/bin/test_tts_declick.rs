//! Регресс-тест деклика на РЕАЛЬНЫХ фикстурах (test/tts_declick_fixtures/).
//!
//! Фикстуры — это файлы temp/deedub_tts_final_<N>.wav из последнего прогона:
//! сигнал ПОСЛЕ stretch+resample (44.1k), ДО declick_spikes. Идеальные входы
//! для теста: никакой зависимости от ffmpeg/rubberband.
//!
//! Метрика: локальный перепад |s[i]-s[i-1]| на 44.1k. Клики документированы
//! (tasks/16.09.26): ч.3 — V-спайк @5.0785s (max delta 11952 = 0.365 FS),
//! ч.27 — срыв @2.3016s (max delta 8734 = 0.267 FS). Вектор ожидания:
//! после declick локальный перепад в окне клика должен упасть ниже порога,
//! при этом речь (control-фикстуры) не должна быть тронута.

use std::path::{Path, PathBuf};

const FS: f32 = 1.0; // full scale = 1.0 в f32/32768
const PASS_MAX_DELTA: f32 = 0.20; // после declick локальный перепад в окне клика
const CONTROL_CHANGED_OK: f32 = 0.30; // % изменённых сэмплов в контроле не выше

struct Fixture {
    name: &'static str,
    click_win_start_sec: Option<f64>,
    click_win_end_sec: Option<f64>,
    is_control: bool,
}

const FIXTURES: &[Fixture] = &[
    // Клик-фикстуры: окна вокруг задокументированных артефактов (в секундах,
    // финальная 44.1k шкала).
    Fixture {
        name: "ch03_click.wav",
        click_win_start_sec: Some(5.03),
        click_win_end_sec: Some(5.13),
        is_control: false,
    },
    Fixture {
        name: "ch27_click.wav",
        click_win_start_sec: Some(2.24),
        click_win_end_sec: Some(2.38),
        is_control: false,
    },
    // Контроль: чистая речь (без кликов) — declick обязан её не трогать.
    Fixture {
        name: "ch01_speech.wav",
        click_win_start_sec: None,
        click_win_end_sec: None,
        is_control: true,
    },
    Fixture {
        name: "ch05_speech.wav",
        click_win_start_sec: None,
        click_win_end_sec: None,
        is_control: true,
    },
    Fixture {
        name: "ch14_speech.wav",
        click_win_start_sec: None,
        click_win_end_sec: None,
        is_control: true,
    },
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fixtures_dir = parse_arg(&args, "--fixtures").map(|s| PathBuf::from(s))
        .unwrap_or_else(|| project_root().join("test/tts_declick_fixtures"));

    let mut failures = 0usize;

    for fx in FIXTURES {
        let p = fixtures_dir.join(fx.name);
        if !p.exists() {
            eprintln!("FAIL [{}]: fixture missing: {}", fx.name, p.display());
            failures += 1;
            continue;
        }
        let (sr, i16s) = match read_wav_i16(&p) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("FAIL [{}]: read error: {}", fx.name, e);
                failures += 1;
                continue;
            }
        };
        let orig: Vec<f32> = i16s.iter().map(|&s| s as f32 / 32768.0).collect();
        let mut cleaned = orig.clone();
        deedub_lib::tts::declick_spikes(&mut cleaned);

        if fx.is_control {
            let n_changed = orig
                .iter()
                .zip(cleaned.iter())
                .filter(|(a, b)| (**a - **b).abs() > 0.0)
                .count();
            let pct = n_changed as f32 / orig.len().max(1) as f32 * 100.0;
            if pct <= CONTROL_CHANGED_OK {
                println!("PASS [{}] control: {:.2}% samples touched", fx.name, pct);
            } else {
                eprintln!(
                    "FAIL [{}] control: {:.2}% samples touched (> {:.1}%) — declick портит речь",
                    fx.name, pct, CONTROL_CHANGED_OK
                );
                failures += 1;
            }
            continue;
        }

        // Клик-фикстура: max локальный перепад в окне ДО и ПОСЛЕ.
        let (ws, we) = (
            fx.click_win_start_sec.unwrap(),
            fx.click_win_end_sec.unwrap(),
        );
        let i0 = (ws * sr as f64) as usize;
        let i1 = (we * sr as f64) as usize;
        let (delta_before, delta_after) = max_delta_in(&orig, i0, i1, sr as usize)
            .zip(max_delta_in(&cleaned, i0, i1, sr as usize))
            .unwrap();
        let before_norm = delta_before / FS;
        let after_norm = delta_after / FS;
        let ok = after_norm < PASS_MAX_DELTA;
        println!(
            "[{}] sr={} win=[{:.3};{:.3}] delta {:.3} -> {:.3} FS {}",
            fx.name,
            sr,
            ws,
            we,
            before_norm,
            after_norm,
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            eprintln!(
                "  `declick_spikes` не убрал клик: {:.3} FS >= порог {:.3} FS",
                after_norm, PASS_MAX_DELTA
            );
            failures += 1;
        }
    }

    eprintln!("\n=== {} ===", if failures == 0 { "ALL PASS" } else { "FAILURES" });
    std::process::exit(if failures == 0 { 0 } else { 1 });
}

fn max_delta_in(s: &[f32], i0: usize, i1: usize, _sr: usize) -> Option<f32> {
    let hi = i1.min(s.len());
    let lo = i0.max(1);
    if hi <= lo {
        return None;
    }
    (lo..hi)
        .map(|i| (s[i] - s[i - 1]).abs())
        .fold(0.0f32, f32::max)
        .into()
}

fn read_wav_i16(p: &Path) -> Result<(u32, Vec<i16>), String> {
    let r = hound::WavReader::open(p).map_err(|e| e.to_string())?;
    let sr = r.spec().sample_rate;
    let samples = r
        .into_samples::<i16>()
        .filter_map(|s| s.ok())
        .collect::<Vec<_>>();
    Ok((sr, samples))
}

fn parse_arg(args: &[String], name: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            if let Some(val) = iter.next() {
                return Some(val.clone());
            }
        }
    }
    None
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}