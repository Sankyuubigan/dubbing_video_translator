//! Проверка слияния коротких спикеров (`comm::merge_unclonable_speakers`).
//!
//! Отдельный бинарник, а не `#[test]`, потому что харнесс cargo test на этой
//! машине не стартует ни в debug, ни в release (0xc0000139
//! STATUS_ENTRYPOINT_NOT_FOUND — ошибка загрузчика ДО выполнения тестов).
//! Рабочие точки проверки TTS-логики в проекте уже устроены так же
//! (`test_retry_tracker`, `test_tts_declick`).
//!
//! Сценарии повторяют юнит-тесты в `comm::tests` — здесь они нужны как
//! исполняемая проверка. Расхождение между двумя копиями означает, что
//! тесты в `comm` устарели.

use deedub_lib::comm::{
    merge_unclonable_speakers, SpeakerSegment, SubtitleChunk, MIN_CLONE_REF_SEC,
};
use std::collections::HashSet;

fn seg(spk: &str, start: f64, end: f64) -> SpeakerSegment {
    SpeakerSegment {
        start_sec: start,
        end_sec: end,
        speaker_id: spk.to_string(),
        gender: None,
    }
}

fn chunk(spk: &str, start: f64, end: f64, text: &str) -> SubtitleChunk {
    SubtitleChunk {
        start_sec: start,
        end_sec: end,
        text: text.to_string(),
        speaker_id: Some(spk.to_string()),
        word_timestamps: None,
    }
}

fn speakers(segments: &[SpeakerSegment]) -> HashSet<String> {
    segments
        .iter()
        .map(|s| s.speaker_id.clone())
        .collect()
}

fn every_speaker_has_reference(segments: &[SpeakerSegment]) -> bool {
    speakers(segments).iter().all(|spk| {
        segments
            .iter()
            .filter(|s| &s.speaker_id == spk)
            .map(|s| s.end_sec - s.start_sec)
            .fold(0.0f64, f64::max)
            >= MIN_CLONE_REF_SEC
    })
}

static mut FAILED: usize = 0;

fn check(name: &str, cond: bool) {
    if cond {
        println!("  OK   {name}");
    } else {
        println!("  FAIL {name}");
        unsafe { FAILED += 1 };
    }
}

fn main() {
    println!("MIN_CLONE_REF_SEC = {MIN_CLONE_REF_SEC}\n");

    // 1. Короткий спикер присоединяется к ближайшему по времени.
    {
        println!("short_speaker_merges_into_nearest_by_time");
        // Геометрия намеренно асимметричная: шумовой кластер (центр 5.4)
        // стоит в 2.9 с от Speaker_1 и в 8.1 с от Speaker_3 — при равных
        // расстояниях тест проверял бы случайность, а не правило выбора.
        let mut segs = vec![
            seg("Speaker_1", 0.0, 5.0),
            seg("Speaker_2", 5.2, 5.6),
            seg("Speaker_3", 12.0, 15.0),
        ];
        let mut chunks = vec![
            chunk("Speaker_1", 0.0, 5.0, "raz"),
            chunk("Speaker_2", 5.2, 5.6, "e-e-e"),
            chunk("Speaker_3", 12.0, 15.0, "dva"),
        ];
        merge_unclonable_speakers(&mut segs, &mut chunks);
        check("осталось 2 голоса", speakers(&segs).len() == 2);
        check("у каждого есть референс", every_speaker_has_reference(&segs));
        // Сравниваем с МЕТКОЙ ЦЕЛИ, а не с её прежним именем: после слияния
        // renumber_speakers перевыдаёт номера, поэтому «Speaker_3» на выходе
        // вполне может стать «Speaker_2». Проверять надо, что шумовой кластер
        // оказался в одной группе с соседом, а не с соседом другого.
        let label_at = |segs: &[SpeakerSegment], start: f64| {
            segs.iter()
                .find(|s| (s.start_sec - start).abs() < 0.01)
                .map(|s| s.speaker_id.clone())
                .unwrap_or_default()
        };
        check(
            "присоединён к соседу слева",
            label_at(&segs, 5.2) == label_at(&segs, 0.0) && label_at(&segs, 5.2) != label_at(&segs, 12.0),
        );

        // Обратный случай: тот же кластер, но ближе к третьему голосу.
        let mut segs = vec![
            seg("Speaker_1", 0.0, 2.0),
            seg("Speaker_2", 3.0, 3.4),
            seg("Speaker_3", 3.6, 6.0),
        ];
        let mut chunks = vec![chunk("Speaker_1", 0.0, 2.0, "raz")];
        merge_unclonable_speakers(&mut segs, &mut chunks);
        check(
            "при сдвиге вправо присоединён к соседу справа",
            label_at(&segs, 3.0) == label_at(&segs, 3.6) && label_at(&segs, 3.0) != label_at(&segs, 0.0),
        );
        println!();
    }

    // 2. Нумерация без пропусков.
    {
        println!("numbering_has_no_gaps_after_merge");
        let mut segs = vec![
            seg("Speaker_1", 0.0, 4.0),
            seg("Speaker_2", 4.2, 4.5),
            seg("Speaker_3", 5.0, 8.0),
        ];
        let mut chunks = vec![chunk("Speaker_1", 0.0, 4.0, "raz")];
        merge_unclonable_speakers(&mut segs, &mut chunks);
        let mut nums: Vec<usize> = speakers(&segs)
            .iter()
            .map(|s| s.trim_start_matches("Speaker_").parse().unwrap())
            .collect();
        nums.sort();
        check("номера сплошные [1,2]", nums == vec![1, 2]);
        println!();
    }

    // 3. Речь короткого кластера сохраняется и получает голос соседа.
    {
        println!("subtitles_survive_merge_and_follow_voice");
        let mut segs = vec![
            seg("Speaker_1", 0.0, 4.0),
            seg("Speaker_2", 4.2, 4.5),
            seg("Speaker_3", 12.0, 15.0),
        ];
        let mut chunks = vec![
            chunk("Speaker_1", 0.0, 4.0, "raz"),
            chunk("Speaker_2", 4.2, 4.5, "aga"),
            chunk("Speaker_3", 12.0, 15.0, "dva"),
        ];
        merge_unclonable_speakers(&mut segs, &mut chunks);
        check("все 3 субтитра на месте", chunks.len() == 3);
        let aga = chunks.iter().find(|c| c.text == "aga").and_then(|c| c.speaker_id.clone());
        let raz = chunks.iter().find(|c| c.text == "raz").and_then(|c| c.speaker_id.clone());
        check("речь кластера не потеряна", aga.is_some());
        check("субтитр и сегмент получили один голос", aga == raz);
        println!();
    }

    // 4. Один спикер — ничего не трогаем.
    {
        println!("single_speaker_case_is_untouched");
        let mut segs = vec![seg("Speaker_1", 0.0, 4.0), seg("Speaker_1", 5.0, 6.0)];
        let mut chunks = vec![chunk("Speaker_1", 0.0, 4.0, "raz")];
        merge_unclonable_speakers(&mut segs, &mut chunks);
        check("2 сегмента на месте", segs.len() == 2);
        check("1 голос", speakers(&segs).len() == 1);
        check(
            "субтитр сохранил Speaker_1",
            chunks[0].speaker_id.as_deref() == Some("Speaker_1"),
        );
        println!();
    }

    // 5. Короткий, но частый спикер всё равно сливается (важен max, не сумма).
    {
        println!("short_but_frequent_speaker_still_merges");
        let mut segs = vec![seg("Speaker_1", 0.0, 4.0)];
        for i in 0..10 {
            segs.push(seg("Speaker_2", 5.0 + i as f64 * 2.0, 5.6 + i as f64 * 2.0));
        }
        let mut chunks = vec![chunk("Speaker_1", 0.0, 4.0, "raz")];
        merge_unclonable_speakers(&mut segs, &mut chunks);
        check("остался 1 голос", speakers(&segs).len() == 1);
        check("у него есть референс", every_speaker_has_reference(&segs));
        println!();
    }

    // 6. Вырожденный случай: референса нет ни у кого — данные не мутируются.
    {
        println!("degenerate_all_short_is_left_alone");
        let mut segs = vec![seg("Speaker_1", 0.0, 0.5), seg("Speaker_2", 1.0, 1.7)];
        let mut chunks = vec![chunk("Speaker_1", 0.0, 0.5, "a")];
        merge_unclonable_speakers(&mut segs, &mut chunks);
        check(
            "оба кластера остались на месте",
            speakers(&segs).len() == 2
                && segs
                    .iter()
                    .all(|s| s.speaker_id == "Speaker_1" || s.speaker_id == "Speaker_2"),
        );
        println!();
    }

    let failed = unsafe { FAILED };
    if failed == 0 {
        println!("ALL PASS");
    } else {
        println!("FAILED: {failed}");
        std::process::exit(1);
    }
}