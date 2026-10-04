//! Проверка того, что служебный `SKIP_MARKER` не попадает в SRT.
//!
//! Отдельный бинарник, а не `cargo test`: харнесс на этой машине не стартует
//! (0xc0000139 STATUS_ENTRYPOINT_NOT_FOUND, ошибка загрузчика ДО выполнения
//! тестов). Зовём production-функцию `output::build_srt_content` напрямую,
//! поэтому тест не расходится с кодом при правке копии.

use deedub_lib::comm::{SubtitleChunk, SKIP_MARKER};
use deedub_lib::output::build_srt_content;
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};

static FAILED: AtomicBool = AtomicBool::new(false);

fn chunk(start: f64, end: f64, text: &str) -> SubtitleChunk {
    SubtitleChunk {
        start_sec: start,
        end_sec: end,
        text: text.to_string(),
        speaker_id: Some("Speaker_4".to_string()),
        word_timestamps: None,
    }
}

fn check(name: &str, ok: bool, detail: String) {
    if ok {
        println!("  ok   — {name}");
    } else {
        println!("  FAIL — {name}: {detail}");
        FAILED.store(true, Ordering::Relaxed);
    }
}

/// Номера cue-блоков в порядке появления. Именно разбор, а не поиск
/// подстроки: перед первым индексом стоит BOM, поэтому `"\n1\n"` не матчится.
fn cue_indices(srt: &str) -> Vec<u32> {
    srt.lines()
        .filter_map(|l| {
            let t = l.trim().trim_start_matches('\u{FEFF}').trim();
            if !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()) {
                t.parse().ok()
            } else {
                None
            }
        })
        .collect()
}

fn main() {
    println!("test-srt-filter: SKIP_MARKER не должен попадать в SRT\n");

    // --- 1. Реальный случай из прогона: последний чанк упал в верификации ---
    let real = vec![
        chunk(108.44, 111.19, "Ну что ж, я заберу вас всех."),
        chunk(111.14, 112.24, "Просто даю тебе знать о событиях."),
        // Этот чанк не прошёл верификацию (digits_present на «9.5») →
        // verification проставил SKIP_MARKER.
        chunk(180.354, 191.840, SKIP_MARKER),
    ];
    let srt = build_srt_content(&real);
    check(
        "маркер не попадает в SRT",
        !srt.contains(SKIP_MARKER),
        format!("SRT содержит {SKIP_MARKER:?}"),
    );
    check(
        "соседние чанки на месте",
        srt.contains("Ну что ж, я заберу вас всех.")
            && srt.contains("Просто даю тебе знать о событиях."),
        "потерян обычный текст".to_string(),
    );
    check(
        "тайминг упавшего чанка не занимает экран",
        !srt.contains("00:03:00,354"),
        "тайминг SKIP_MARKER попал в SRT".to_string(),
    );
    // Нумерация должна быть сплошной: idx растёт только для записанных cue.
    check(
        "нумерация сплошная без дыр",
        cue_indices(&srt) == vec![1, 2],
        format!("индексы {:?}\n{srt}", cue_indices(&srt)),
    );

    // --- 2. SKIP_MARKER с обрезкой пробелов --------------------------------
    let padded = vec![chunk(0.0, 1.0, "  (-)  ")];
    let srt = build_srt_content(&padded);
    check(
        "SKIP_MARKER в пробелах тоже отфильтрован",
        !srt.contains("-->"),
        format!("SRT не пуст:\n{srt}"),
    );

    // --- 3. Пустой текст по-прежнему пропускается --------------------------
    let empty = vec![chunk(0.0, 1.0, "   ")];
    check(
        "пустой текст отфильтрован",
        !build_srt_content(&empty).contains("-->"),
        "пустой чанк записан".to_string(),
    );

    // --- 4. Отсутствие маркера не должно ломать остальное ------------------
    let clean = vec![
        chunk(0.24, 2.44, "Вот во что верят любители."),
        chunk(2.88, 10.388, "Знаешь, мне кажется, они сами себя захватили."),
    ];
    let srt = build_srt_content(&clean);
    check(
        "без маркера оба чанка записаны",
        srt.contains("[Speaker_4] Вот во что верят любители.")
            && srt.contains("[Speaker_4] Знаешь, мне кажется, они сами себя захватили."),
        format!("SRT:\n{srt}"),
    );

    // --- 5. Формат таймштампа не сломан рефакторингом ----------------------
    check(
        "формат времени SRT корректен",
        srt.contains("00:00:00,240 --> 00:00:02,440"),
        format!("SRT:\n{srt}"),
    );
    check(
        "BOM сохранён (кириллица в mov_text)",
        srt.starts_with('\u{FEFF}'),
        format!("SRT не начинается с BOM: {:?}", &srt[..srt.len().min(8)]),
    );

    println!();
    if FAILED.load(Ordering::Relaxed) {
        println!("TEST FAILED");
        exit(1);
    }
    println!("ALL PASS");
    println!("TEST OK");
}