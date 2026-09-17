//! Прогон RetryTracker по всем задокументированным сценариям обрезания TTS.
//!
//! Ground truth чанка 7 «у вас уже 2 предупреждения»:
//!   - текст содержит 11 гласных → ожидаемая длительность ≈ 11/4.4 = 2.50с;
//!   - обе попытки реального прогона дали 1.48с → 1.48/2.50 = 0.59 — это
//!     ОБРЕЗАННЫЙ хвост (cosyvoice3 недо-генерация), а не полная фраза;
//!   - трекер обязан НЕ останавливаться после двух 1.48с и продолжать retry.
//!
//! Историческая причина бага: RetryTracker не знал ожидаемую длительность и
//! считал консенсус двух коротких «близнецов» (near_best>=2) подтверждением
//! полного прочтения. Фикс: остановка только при максимуме ≥ TTS_FULL_FLOOR×expected.
//!
//! Этот бинарь существует, потому что юнит-тест-харнесс на этой машине не
//! запускается (STATUS_ENTRYPOINT_NOT_FOUND — CUDA DLL при загрузке), а
//! автономные [[bin]] бинари работают.

use app_lib::tts::RetryTracker;

/// Прогоняет tracker по реальным длинам попыток и возвращает
/// (лучшая длина, число попыток до stop).
fn simulate(tracker: &mut RetryTracker, gens: &[f64]) -> (f64, usize) {
    let mut best = 0.0f64;
    for (i, &g) in gens.iter().enumerate() {
        let v = tracker.observe(g);
        if v.is_best {
            best = g;
        }
        if v.stop {
            return (best, i + 1);
        }
    }
    (best, gens.len())
}

fn main() {
    // Чанк 7: два одинаково коротких «близнеца» — НЕ подтверждение полного прочтения.
    let mut t = RetryTracker::new(2.50);
    let v0 = t.observe(1.48);
    println!("ch7  attempt 1: gen=1.48 -> is_best={} stop={}", v0.is_best, v0.stop);
    assert!(v0.is_best, "первая попытка всегда новая лучшая");
    assert!(!v0.stop, "первая попытка не должна останавливать");
    let v1 = t.observe(1.48);
    println!("ch7  attempt 2: gen=1.48 -> is_best={} stop={}", v1.is_best, v1.stop);
    assert!(!v1.stop, "2 коротких генерации (1.48с ≪ 2.50с) не должны останавливать retry");
    let v2 = t.observe(2.50);
    println!("ch7  attempt 3: gen=2.50 -> is_best={} stop={}", v2.is_best, v2.stop);
    assert!(v2.is_best, "полная генерация 2.50с должна стать новой лучшей");
    println!("ch7  OK: дождались полной генерации после коротких близнецов\n");

    // Чанк 27: полная 4.12с, обрезанные 3.20/3.56с → выбирается самая длинная.
    let mut t = RetryTracker::new(4.5);
    let (best, attempts) = simulate(&mut t, &[3.20, 3.56, 3.20, 4.12]);
    println!("ch27 best={best:.2} attempts={attempts} (ожид: best=4.12 attempts=4)");
    assert_eq!(attempts, 4, "полная генерация найдена на 4-й попытке");
    assert!((best - 4.12).abs() < 1e-9, "best={best}");
    println!("ch27 OK\n");

    // Чанк 34: обрезанная 2.28с + полные 3.04с (подтверждение 3.00с) → stop на 3-й.
    let mut t = RetryTracker::new(3.0);
    let (best, attempts) = simulate(&mut t, &[2.28, 3.04, 3.00]);
    println!("ch34 best={best:.2} attempts={attempts} (ожид: best=3.04 attempts=3)");
    assert_eq!(attempts, 3);
    assert!((best - 3.04).abs() < 1e-9, "best={best}");
    println!("ch34 OK\n");

    // Чанк 35 (длинный): полные 7.36-7.60с → stop на 3-й, не гоняя все 6.
    let mut t = RetryTracker::new(7.5);
    let (best, attempts) = simulate(&mut t, &[7.36, 7.60, 7.55]);
    println!("ch35 best={best:.2} attempts={attempts} (ожид: best=7.60 attempts=3)");
    assert_eq!(attempts, 3);
    assert!((best - 7.60).abs() < 1e-9, "best={best}");
    println!("ch35 OK\n");

    // Чанк 38 (66 гласных, ожидание ~15с): не гонять все 6.
    let mut t = RetryTracker::new(15.0);
    let (best, attempts) = simulate(&mut t, &[12.68, 11.04, 12.68]);
    println!("ch38 best={best:.2} attempts={attempts} (ожид: best=12.68 attempts<=4)");
    assert!(attempts <= 4, "должен остановиться раньше 6 попыток: {attempts}");
    assert!((best - 12.68).abs() < 1e-9, "best={best}");
    println!("ch38 OK\n");

    // Чанк 36: 2.04с не подтверждает 3.28с и не останавливает.
    let mut t = RetryTracker::new(3.3);
    let v0 = t.observe(3.28);
    let v1 = t.observe(2.04);
    println!("ch36 v0={{is_best={} stop={}}} v1={{is_best={} stop={}}}", v0.is_best, v0.stop, v1.is_best, v1.stop);
    assert!(v0.is_best);
    assert!(!v1.stop, "2.04с не подтверждает 3.28с");
    assert!(!v1.is_best);
    println!("ch36 OK\n");

    // Монотонный рост: нельзя останавливаться раньше времени.
    let mut t = RetryTracker::new(4.5);
    let (best, attempts) = simulate(&mut t, &[3.0, 3.5, 3.8, 4.0, 4.4]);
    println!("monotonic best={best:.2} attempts={attempts} (ожид: best=4.4 attempts=5)");
    assert_eq!(attempts, 5);
    assert!((best - 4.4).abs() < 1e-9, "best={best}");
    println!("monotonic OK\n");

    println!("ALL RETRY SCENARIOS PASSED");
}