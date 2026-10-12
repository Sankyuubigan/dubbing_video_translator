//! Определение пола спикера на реальных фикстурах.
//!
//! Харнесс `cargo test` на этой машине не стартует (0xc0000139), поэтому
//! проверка живёт отдельным бинарником — как `test-srt-filter` и остальные.
//!
//! Проверяется ровно одно свойство: для каждой выдержки пол, вычисленный
//! продовым кодом `comm::voice`, совпадает с полом по независимому эталону из
//! `fixtures.json`. Эталон — F0 по расстоянию между пиками спектра; трекером
//! он не измеряется, поэтому расхождение означает реальную ошибку детектора,
//! а не спорный порог.
//!
//! Файл ожиданий — `test/fixtures/gender/fixtures.json`; сами WAV лежат рядом.
//! Запуск: test-gender-fixtures [каталог_фикстур]

use serde::Deserialize;
use std::path::PathBuf;
use std::process::exit;

use deedub_lib::comm::voice;

#[derive(Deserialize)]
struct Manifest {
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    file: String,
    truth_f0_hz: f64,
    expected_gender: String,
}

fn load_wav(path: &PathBuf) -> Option<(Vec<f32>, u32)> {
    let r = hound::WavReader::open(path).ok()?;
    let sr = r.spec().sample_rate;
    let s: Vec<f32> = r
        .into_samples::<i16>()
        .filter_map(|v| v.ok())
        .map(|v| v as f32 / 32768.0)
        .collect();
    Some((s, sr))
}

fn main() {
    // Путь к фикстурам — от корня проекта (`paths::test_dir`), а не от cwd:
    // бинарник запускается из `src-tauri`, и относительный путь от рабочей
    // папки здесь означал бы другой набор файлов при запуске из корня.
    let dir: PathBuf = std::env::args().nth(1).map_or_else(
        || deedub_lib::paths::test_dir().join("fixtures").join("gender"),
        PathBuf::from,
    );

    let manifest_path = dir.join("fixtures.json");
    let raw = match std::fs::read_to_string(&manifest_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[FAIL] не прочитать {}: {e}", manifest_path.display());
            exit(1);
        }
    };
    let manifest: Manifest = match serde_json::from_str(&raw) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[FAIL] не разобрать {}: {e}", manifest_path.display());
            exit(1);
        }
    };

    println!("фикстур: {}", manifest.rows.len());
    println!(
        "пороги продового детектора: ФВЧ {:.0} Гц, гейт p{:.0}, приёмка {:.2}, пол >= {:.0} Гц\n",
        voice::HP_CUTOFF_HZ,
        voice::VOICING_ENERGY_PERCENTILE,
        voice::YIN_ACCEPT,
        voice::GENDER_MALE_BELOW_HZ
    );

    let mut failed = 0usize;
    println!(
        "{:<20}{:>10}{:>12}{:>12}{:>10}",
        "фикстура", "эталон", "прод F0", "вокал%", "пол"
    );
    for row in &manifest.rows {
        let path = dir.join(&row.file);
        let (samples, sr) = match load_wav(&path) {
            Some(v) => v,
            None => {
                println!("{:<20}{:>10}  [FAIL] не открыть WAV", row.file, "-");
                failed += 1;
                continue;
            }
        };
        let track = voice::voice_track(&samples, sr);
        let got = voice::gender_of(track.f0_median);
        let ok = got == Some(row.expected_gender.as_str());
        if !ok {
            failed += 1;
        }
        println!(
            "{:<20}{:>9.0}{:>11}{:>11.0}%{:>10}  {}",
            row.file,
            row.truth_f0_hz,
            track
                .f0_median
                .map(|v| format!("{:.0}", v))
                .unwrap_or_else(|| "-".into()),
            track.voiced_ratio * 100.0,
            got.unwrap_or("-"),
            if ok { "OK" } else { "FAIL" }
        );
    }

    println!();
    if failed == 0 {
        println!("[OK] все {} фикстур классифицированы верно", manifest.rows.len());
    } else {
        println!("[FAIL] расхождений: {failed} из {}", manifest.rows.len());
        exit(1);
    }
}