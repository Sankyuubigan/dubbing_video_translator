use crate::comm::{PipelineContext, SpeakerSegment, SubtitleChunk};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;
use tauri::Manager;

const CREATE_NO_WINDOW: u32 = 0x08000000;

// --- Референсы голосового клона ---

/// Выбирает сегмент спикера для клонирования голоса.
/// Приоритет: длительность в диапазоне [3, 10] с ближайшей к 7 с;
/// иначе — самый длинный сегмент (но не короче 1 с).
fn pick_ref_segment(speaker: &str, segments: &[SpeakerSegment]) -> Option<(f64, f64)> {
    let mine: Vec<&SpeakerSegment> = segments
        .iter()
        .filter(|s| s.speaker_id == speaker)
        .collect();
    if mine.is_empty() {
        return None;
    }

    let mut best: Option<(f64, f64, f64)> = None;
    for s in &mine {
        let d = s.end_sec - s.start_sec;
        if (3.0..=10.0).contains(&d) {
            let diff = (d - 7.0).abs();
            if best.map_or(true, |b| diff < b.2) {
                best = Some((s.start_sec, s.end_sec, diff));
            }
        }
    }
    if let Some(b) = best {
        return Some((b.0, b.1));
    }

    let longest = mine
        .iter()
        .max_by(|a, b| {
            let da = a.end_sec - a.start_sec;
            let db = b.end_sec - b.start_sec;
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })?;
    let d = longest.end_sec - longest.start_sec;
    if d >= 1.0 {
        Some((longest.start_sec, longest.end_sec))
    } else {
        None
    }
}

fn safe_speaker(speaker: &str) -> String {
    let sanitized: String = speaker
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "speaker".to_string()
    } else {
        sanitized
    }
}

// --- Время и длительность WAV ---

/// Длительность WAV из байтов ответа движка (без записи на диск).
fn wav_sr_and_duration_bytes(bytes: &[u8]) -> Result<(u32, f64)> {
    let r = hound::WavReader::new(std::io::Cursor::new(bytes))
        .context("TTS: разбор WAV из байтов")?;
    let sr = r.spec().sample_rate;
    let dur = r.duration() as f64 / sr as f64;
    Ok((sr, dur))
}

// --- Адаптивный retry обрезанного хвоста (cosyvoice3) ---

/// Seed'ы для повтора синтеза. cosyvoice3 с cross-lingual клоном иногда
/// недо-генерирует реплику (LM выбрасывает reference-токены) и «обрезает»
/// финальные слоги. RAS-семплер seed-зависим: разные seed дают разные обрезки,
/// часть из них — полные фразы (проверено на движке 0.8.32).
const TTS_RETRY_SEEDS: [u64; 5] = [7, 42, 123, 777, 31337];

/// Всего попыток на чанк: исходная генерация (seed=None) + 5 контрольных.
/// Лимит НЕ растёт: щелчки убираются пост-обработкой (declick_spikes),
/// а не перегенерацией, поэтому искать «чистый» seed не нужно.
const TTS_MAX_ATTEMPTS: usize = 1 + TTS_RETRY_SEEDS.len();

/// Ожидаемая длительность озвучки по числу гласных в русском тексте.
/// Калибровка по замеренным генерациям cosyvoice3 RU (38 чанков):
/// темп ≈4.4 гласных/сек (mean), медиана 4.62, p25=4.1, p75=5.1.
/// Старая константа 3.0 завышала ожидание в ~1.5× → порог 0.8×expected
/// почти никогда не достигался, и все 6 попыток гонялись впустую.
const TTS_VOWEL_RATE: f64 = 4.4;
fn expected_spoken_secs(text: &str) -> f64 {
    let vowels = text
        .chars()
        .filter(|c| matches!(c, 'а'|'е'|'ё'|'и'|'о'|'у'|'ы'|'э'|'ю'|'я'|'А'|'Е'|'Ё'|'И'|'О'|'У'|'Ы'|'Э'|'Ю'|'Я'))
        .count();
    vowels as f64 / TTS_VOWEL_RATE
}

/// Минимальная доля от ожидаемой длительности, при которой консенсус двух
/// попыток считается ПОЛНЫМ прочтением.
///
/// Калибровка: самый быстрый реальный темп речи 6.9 гласных/сек → законная
/// полная фраза может звучать от expected×(4.4/6.9) = expected×0.638.
/// Обрезанный хвост ч.7 «у вас уже 2 предупреждения» (1.48с при ожидании
/// 2.50с = 0.592×expected) лежит НИЖЕ этого физического минимума — это
/// гарантированно недо-генерация cosyvoice3, а не быстрая речь.
///
/// Порог 0.62 < 0.638 (не отбрасывает ни одно возможное полное чтение),
/// но > 0.592 (отсекает задокументированный баг ч.7).
const TTS_FULL_FLOOR: f64 = 0.62;

/// Результат `RetryTracker::observe` для одной попытки синтеза.
#[derive(Debug, Clone, Copy)]
pub struct RetryVerdict {
    /// Эта попытка — новая лучшая (наибольшая полнота) генерация.
    pub is_best: bool,
    /// Найдено стабильно повторяющееся полное прочтение → можно прекращать.
    pub stop: bool,
}

/// Следит за длинами генераций и решает когда прекращать retry.
///
/// КАЛИБРОВКА: обрезанные (недо-генерированные) хвосты у cosyvoice3 короче
/// полного прочтения в ~0.72–0.86× (данные из ч.27: 3.2с/4.12с, ч.34: 2.48с/3.0с).
/// Абсолютный порог от expected НЕ используется напрямую как в старом коде
/// (V/3.0 завышал ожидание в ~1.5×, из-за чего ВСЕ длинные чанки гоняли
/// 6 попыток впустую). Вместо этого expected участвует только как НИЖНЯЯ
/// граница: подтверждённый максимум физически не может быть короче
/// TTS_FULL_FLOOR×expected (см. константу).
///
/// Правило остановки: полное прочтение — это самая длинная генерация. Когда
/// две независимые попытки (разные seed) дают почти одну длину (в пределах
/// 3% от максимума) И этот максимум ≥ TTS_FULL_FLOOR×expected — это
/// натуральная полная длина фразы, дальше не ищем. Если же два «близнеца»
/// короче ожидаемого (ч.7: 1.48с при ожидаемых 2.50с = 0.59×expected) —
/// это ОБРЕЗАННЫЙ хвост, и трекер обязан продолжать retry.
/// Щелчки к трекеру отношения не имеют: они вычищаются из готовых сэмплов
/// пост-обработкой (см. declick_spikes).
#[derive(Debug)]
pub struct RetryTracker {
    best: f64,
    near_best_count: u32,
    expected: f64,
}

impl RetryTracker {
    pub fn new(expected: f64) -> Self {
        Self { best: 0.0, near_best_count: 0, expected }
    }

    pub fn observe(&mut self, gen: f64) -> RetryVerdict {
        let is_best;
        if gen > self.best {
            self.best = gen;
            self.near_best_count = 1;
            is_best = true;
        } else if gen >= self.best * 0.97 {
            self.near_best_count += 1;
            is_best = false;
        } else {
            is_best = false;
        }
        // Полное прочтение подтверждено двумя независимыми попытками —
        // НО только если максимум достижим физически (не обрезанный хвост).
        let stop = self.near_best_count >= 2 && self.best >= self.expected * TTS_FULL_FLOOR;
        RetryVerdict { is_best, stop }
    }
}

/// Убирает из финальных сэмплов одиночные импульсные выбросы (щелчки).
///
/// Щелчок cosyvoice3 — это 1-2 сэмпла, резко «выстреливающие» из локального
/// контекста (примеры: 5908→−7815→180 в ч.3, резкий подъём −14443→5802→23306
/// в ч.27). Сглаживание — КЛАССИЧЕСКИЙ declick, один проход O(n):
///  - двойной пик: s[i] прыгнула от ОБОИХ соседей больше SPIKE → заменяем
///    средним двух соседей (выброс исчезает, тональный сигнал не тронут);
///  - одиночный обрыв |s[i]-s[i-1]| больше STEP → приближаем s[i] к середине
///    соседей, уступ превращается в плавную ступень.
/// Каждая правка — 1 сэмпл из 24000/с (0.04 мс), на слух незаметна; NORMAL
/// громкая атака речи (~плато) этим правилом почти не задевается (пороги
/// заметно выше типичных перепадов и не требуют идеального детекта — мы
/// ЧИНИМ выброс, а не решаем «клик или нет»).
pub fn declick_spikes(samples: &mut [f32]) {
    const SPIKE: f32 = 0.22;
    const STEP: f32 = 0.45;
    let n = samples.len();
    if n < 3 {
        return;
    }
    for i in 1..n - 1 {
        let before = samples[i - 1];
        let cur = samples[i];
        let after = samples[i + 1];
        let d1 = (cur - before).abs();
        let d2 = (after - cur).abs();
        if d1 > SPIKE && d2 > SPIKE {
            samples[i] = (before + after) / 2.0;
        } else if d1 > STEP {
            samples[i] = (before + after) / 2.0;
        }
    }

    // После ресемпла 24k→44.1k RAS-импульс превращается НЕ в одиночный пик,
    // а в ПЛАТО из 2–3 почти равных сэмплов (внутренний перепад ≈0), зажатое
    // двумя резкими ВНЕШНИМИ перепадами > SPIKE. Одиночное правило выше его
    // не видит — d2 внутри плато мал.
    //
    // ВАЖНО: внешние края плато [i, i+L) — это s[i-1]→s[i] и s[i+L-1]→s[i+L],
    // а НЕ внутренний перепад (старая версия детекта сверяла с внутренним
    // перепадом на первой же итерации и поэтому НИКОГДА не срабатывала).
    //
    // Два пути обработки:
    //   1) тихий контекст с обеих сторон — клик на тихом хвосте/паузе;
    //   2) громкая речь БЕЗ гейта тишины: клик сидит ПОСРЕДИ live-голоса
    //      (RMS контекста ±5 мс 0.08–0.20), тихого окружения у него нет.
    //      Ложные кандидаты отсекаются КВАДРАТНОЙ формой: ровно 2 сэмпла,
    //      внутренний перепад <0.03, оба внешних края >SPIKE — натуральная
    //      речь на 44.1k такой формы не порождает (замер по 192-с файлу:
    //      37 кандидатов, все — один и тот же RAS-артефакт), плюс амплитудный
    //      пол для надёжности.
    const PLATEAU_CTX_RADIUS: usize = 220; // ~5 мс на 44.1k
    const PLATEAU_CTX_LOUD: f32 = 0.03; // локальный RMS «речи» слева/справа (путь 1)
    const PLATEAU_LOUD_MIN: f32 = 0.1; // мин. амплитуда плато для пути 2
    const PLATEAU_FLAT_MAX: f32 = 0.03; // максимум внутреннего перепада плато
    const PLATEAU_MAX_LEN: usize = 3;
    let side_quiet = |xs: &[f32], lo: usize, hi: usize| -> bool {
        if hi <= lo {
            return true;
        }
        let rms =
            (xs[lo..hi].iter().map(|s| s * s).sum::<f32>() / (hi - lo) as f32).sqrt();
        rms < PLATEAU_CTX_LOUD
    };
    let mut i = 1usize;
    while i + PLATEAU_MAX_LEN + 1 < n {
        // Ищем плато [i, i+len), len = 2..3 с резкими внешними краями.
        let mut best_len = 0usize;
        for len in 2..=PLATEAU_MAX_LEN {
            if i + len + 1 >= n {
                break;
            }
            let d_left = (samples[i] - samples[i - 1]).abs();
            let d_right = (samples[i + len - 1] - samples[i + len]).abs();
            if d_left < SPIKE || d_right < SPIKE {
                continue;
            }
            let mut flat = true;
            for k in 0..len - 1 {
                if (samples[i + k + 1] - samples[i + k]).abs() > PLATEAU_FLAT_MAX {
                    flat = false;
                    break;
                }
            }
            if !flat {
                continue;
            }
            best_len = len;
            break;
        }
        if best_len > 0 && i + best_len + 1 < n {
            let peak = (0..best_len)
                .map(|k| samples[i + k].abs())
                .fold(0f32, f32::max);
            let left_ok = side_quiet(samples, i.saturating_sub(PLATEAU_CTX_RADIUS), i);
            let right_ok = side_quiet(
                samples,
                i + best_len + 1,
                (i + best_len + 1 + PLATEAU_CTX_RADIUS).min(n),
            );
            let loud_short_plateau = best_len == 2 && peak > PLATEAU_LOUD_MIN;
            if (left_ok && right_ok) || loud_short_plateau {
                let s0 = samples[i - 1];
                let s1 = samples[i + best_len];
                for k in 0..best_len {
                    let frac = (k + 1) as f32 / (best_len + 1) as f32;
                    samples[i + k] = s0 + (s1 - s0) * frac;
                }
            }
            i += best_len + 1;
            continue;
        }
        i += 1;
    }
}

/// Вырезает сегмент [start_sec, end_sec) из preloaded-сэмплов и пишет mono WAV.
fn write_segment_wav(out: &str, sample_rate: u32, samples: &[i16]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out, spec)
        .with_context(|| format!("TTS: создание {}", out))?;
    for &s in samples {
        writer.write_sample(s).ok();
    }
    writer.finalize().ok();
    Ok(())
}

// --- FFmpeg помощники ---

/// Применяет time-stretch через FFmpeg rubberband-фильтр.
/// rubberband сохраняет pitch (голос не становится бурундуком),
/// поддерживает ratio [0.25, 4.0].
fn time_stretch_wav(
    input_wav: &str,
    output_wav: &str,
    target_duration_sec: f64,
    current_duration_sec: f64,
    ffmpeg_path: &Option<String>,
) -> Result<()> {
    if current_duration_sec <= 0.0 || target_duration_sec <= 0.0 {
        std::fs::copy(input_wav, output_wav).ok();
        return Ok(());
    }
    let ratio = current_duration_sec / target_duration_sec;
    let clamped = ratio.clamp(0.25, 4.0);
    if (clamped - 1.0).abs() < 0.02 {
        std::fs::copy(input_wav, output_wav)
            .context("Ошибка копирования WAV")?;
        return Ok(());
    }
    log::info!(
        "TTS: rubberband ratio={:.2} (gen={:.1}s, target={:.1}s)",
        clamped,
        current_duration_sec,
        target_duration_sec
    );
    let ffmpeg = crate::ffmpeg::resolve(ffmpeg_path);
    let output = Command::new(&ffmpeg)
        .creation_flags(CREATE_NO_WINDOW)
        .arg("-y")
        .arg("-i")
        .arg(input_wav)
        .arg("-filter:a")
        .arg(format!("rubberband=tempo={:.4}", clamped))
        .arg(output_wav)
        .output()
        .context("Ошибка запуска FFmpeg для rubberband")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("FFmpeg rubberband error: {}", stderr);
    }
    Ok(())
}

fn resample_wav(
    input_wav: &str,
    output_wav: &str,
    out_sample_rate: u32,
    ffmpeg_path: &Option<String>,
) -> Result<()> {
    let ffmpeg = crate::ffmpeg::resolve(ffmpeg_path);
    let output = Command::new(&ffmpeg)
        .creation_flags(CREATE_NO_WINDOW)
        .arg("-y")
        .arg("-i")
        .arg(input_wav)
        .arg("-ar")
        .arg(out_sample_rate.to_string())
        .arg("-ac")
        .arg("1")
        .arg(output_wav)
        .output()
        .context("Ошибка запуска FFmpeg для ресемплинга")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("FFmpeg resample error: {}", stderr);
    }
    Ok(())
}

// --- Планировщик таймлайна (two-pass TTS) ---

/// Потолок сжатия rubberband, при котором речь ещё остаётся разборчивой:
/// фазовый вокодер начинает размывать финальные слоги уже при ~1.4–1.5x,
/// поэтому хвосты слов «съедаются» и фразы звучат обрезанными.
const MAX_STRETCH_RATIO: f64 = 1.3;
/// Минимальная пауза между соседними фразами (фразы не липнут встык).
const MIN_GAP_SEC: f64 = 0.1;
/// Аварийный предел «перетекания» фразы за конец окна (последний рубеж
/// перед обрезкой): лучше запоздавшая фраза, чем сжатая до неразборчивости.
const OVERFLOW_BORROW_SEC: f64 = 2.5;
/// Короткий fade на атаке — гладкое начало речи без клика.
const FADE_IN_SEC: f64 = 0.02;
/// Лёгкий fade только на последних мс хвоста. Натуральное затухание CosyVoice3
/// (~200 мс) должно доигрывать ПОЛНОСТЬЮ, поэтому постоянный fade не может
/// быть длинным — 0.03 «проглатывал» финальный гласный. 0.008 лишь гарантирует,
/// что последний сэмпл каждой фразы → 0 (ни одного клика на шве).
const FADE_OUT_SEC: f64 = 0.008;
/// Полноценное затухание — только для обрезанных (TRUNC) чанков.
const FADE_OUT_TRUNCATED_SEC: f64 = 0.12;
/// Запас между натуральным хвостом фразы и атакой следующей: хвост не доходит
/// до fade-in головы следующего чанка, клея голосов не возникает.
const TAIL_SAFETY_SEC: f64 = 0.03;
/// Предел обрезки «предзапуска» TTS — защита от полного проглатывания паузы.
const HEAD_TRIM_LIMIT_SEC: f64 = 0.5;

#[derive(Debug, Clone)]
struct PlannedChunk {
    index: usize,
    has_audio: bool,
    placed_start: f64,
    placed_duration: f64,
    stretch_ratio: f64,
    truncated: bool,
}

/// Распределяет чанкам финальные окна так, чтобы НИ ОДИН чанк не
/// перекрывался, не урезался соседями и не «уезжал» раньше оригинала.
///
/// Стратегия (разборчивость слов приоритетна, липсинк — по началу фразы):
///  1. Чанк СТАРТУЕТ ровно в своём start_sec (никогда раньше), и минимум
///     MIN_GAP_SEC после предыдущей фразы.
///  2. Фраза короче окна — звучит на натуральной скорости и заканчивает
///     раньше; между соседними фразами возникает естественная пауза.
///  3. Фраза не влезает в окно — сжимаем до потолка разборчивости
///     MAX_STRETCH_RATIO. Если и при этом не влезает, фраза «перетекает»
///     за конец окна (спилл до OVERFLOW_BORROW_SEC), а последующие чанки
///     каскадно сдвигаются вправо. Слова при этом НЕ режутся.
///  4. Последний рубеж — обрезка с fade-out (TRUNC), только когда спилл
///     упёрся в OVERFLOW_BORROW_SEC.
fn plan_timeline(chunks: &[SubtitleChunk], durations: &[f64]) -> Vec<PlannedChunk> {
    let mut plan = Vec::with_capacity(chunks.len());
    let mut cursor = -1.0f64;

    for (i, chunk) in chunks.iter().enumerate() {
        let gen = durations[i];
        if gen <= 0.0 {
            plan.push(PlannedChunk {
                index: i,
                has_audio: false,
                placed_start: chunk.start_sec,
                placed_duration: 0.0,
                stretch_ratio: 1.0,
                truncated: false,
            });
            continue;
        }

        let orig_s = chunk.start_sec;
        let orig_e = chunk.end_sec.max(orig_s);
        // Якорь: не раньше оригинала и не ближе MIN_GAP к предыдущему чанку.
        let placed_s = orig_s.max(cursor + MIN_GAP_SEC);
        let window = (orig_e - orig_s).max(0.1);

        let (placed_duration, stretch_ratio, truncated) =
            if gen <= window {
                // Натуральная скорость — влезает само окно.
                (gen, 1.0, false)
            } else {
                // Не влезает без сжатия: остаток окна до orig_e, зона спилла
                // — тишина ПОСЛЕ окна (до OVERFLOW_BORROW_SEC).
                let d_window = (orig_e - placed_s).max(0.05);
                let r_window = gen / d_window;
                if r_window <= MAX_STRETCH_RATIO {
                    // Умеренное сжатие — разборчиво сохраняется и влезает.
                    (d_window, r_window, false)
                } else {
                    // Сжатие до потолка разборчивости + перетекание за окно.
                    let spill_dur = gen / MAX_STRETCH_RATIO;
                    let spill_ceiling = orig_e + OVERFLOW_BORROW_SEC;
                    if placed_s + spill_dur <= spill_ceiling {
                        (spill_dur, MAX_STRETCH_RATIO, false)
                    } else {
                        // Последний рубеж — обрезка с затуханием.
                        let d = (spill_ceiling - placed_s).max(0.1).min(gen);
                        (d, gen / d, true)
                    }
                }
            };

        let placed_end = placed_s + placed_duration;
        cursor = placed_end;

        plan.push(PlannedChunk {
            index: i,
            has_audio: true,
            placed_start: placed_s,
            placed_duration,
            stretch_ratio,
            truncated,
        });
    }

    plan
}

/// Убирает «предзапуск» TTS: модель часто выдаёт 60–90 мс нулевой амплитуды
/// до первой гласной, из-за чего фраза на холсте начинается с мёртвой дыры.
/// Режем ТОЛЬКО голову (с пределом, чтобы не проглотить живую паузу), а хвост
/// трогать нельзя: CosyVoice3 затухает естественно ~200 мс, и обрезка хвоста
/// по тишине безвозвратно «съедает» последний гласный в быстрых референсах.
/// Возвращает число слитых сэмплов головы — вызывающий сдвигает offset на эту
/// величину, чтобы КОНТЕНТ фразы остался в исходной абсолютной позиции
/// (иначе весь хвост «уезжал» влево на drained и финальный гласный терялся).
fn trim_edges_silence(samples: &mut Vec<f32>, sr: usize) -> usize {
    const THRESH: f32 = 0.002;
    let step = ((sr / 1000) * 2).max(8);
    let head_limit = (HEAD_TRIM_LIMIT_SEC * sr as f64) as usize;
    let mut start = 0usize;
    while start + step < samples.len() && start < head_limit {
        let seg = &samples[start..start + step];
        let rms = (seg.iter().map(|s| s * s).sum::<f32>() / seg.len() as f32).sqrt();
        if rms >= THRESH {
            break;
        }
        start += step;
    }
    // Хвост НЕ трогаем: CosyVoice3 естественно затухает ~200 мс, обрезка хвоста
    // по тишине «съедает» последний гласный. Предел таймлайна применяется
    // ТОЛЬКО в trim_tail_to_cap — при реальном наезде на соседа.
    if start > 0 {
        samples.drain(0..start);
    }
    start
}

/// Если samples длиннее can_keep — гасит «перетянутый» хвост fade-ом и
/// обрезает ровно до can_keep. Направление fade: у точки среза (хвоста)
/// амплитуда → 0, вглубь хвоста → 1.0. Это гарантирует, что после truncate
/// сигнал у границы начинается с ~нуля и не «щёлкает» на стыке с次の фразой.
fn trim_tail_to_cap(samples: &mut Vec<f32>, can_keep: usize, out_sr: usize) {
    let excess = samples.len().saturating_sub(can_keep);
    if excess == 0 {
        return;
    }
    let tail_fade = (0.03 * out_sr as f64) as usize;
    let fade_len = tail_fade.min(excess.max(1));
    for i in 0..fade_len {
        // Сэмплы идём с конца вглубь: at = can_keep-1-i.
        let at = can_keep.saturating_sub(i + 1);
        // У среза (i=0) — ноль; вглубь (i=fade_len-1) — почти полная амплитуда.
        let frac = i as f32 / fade_len as f32;
        if let Some(s) = samples.get_mut(at) {
            *s *= frac;
        }
    }
    samples.truncate(can_keep);
}

/// Пишет samples в canvas со смещением offset, с fade-in/fade-out
/// и никогда не выходя за границы холста.
fn place_samples(
    canvas: &mut [f32],
    samples: &[f32],
    offset: usize,
    fade_in_samples: usize,
    fade_out_samples: usize,
) {
    if offset >= canvas.len() || samples.is_empty() {
        return;
    }
    let end = (offset + samples.len()).min(canvas.len());
    if end <= offset {
        return;
    }
    let n = end - offset;
    for i in 0..n {
        let mut s = samples[i];
        if fade_in_samples > 0 && i < fade_in_samples {
            s *= i as f32 / fade_in_samples as f32;
        }
        let from_end = n - 1 - i;
        if fade_out_samples > 0 && from_end < fade_out_samples {
            s *= from_end as f32 / fade_out_samples as f32;
        }
        canvas[offset + i] = s;
    }
}

/// Результат pass 1: сырые байты WAV + длительность синтеза.
struct SynthesizedChunk {
    index: usize,
    wav: Vec<u8>,
    gen_duration: f64,
    gen_sr: u32,
}

// --- Основной пайплайн озвучки ---

pub fn dub(mut ctx: PipelineContext) -> Result<PipelineContext> {
    if !ctx.config.enable_dubbing {
        log::info!("TTS: озвучка отключена, пропускаем");
        return Ok(ctx);
    }

    let mut translated = match std::mem::take(&mut ctx.translated_chunks) {
        Some(c) => c,
        None => {
            log::warn!("TTS: нет переведённых чанков");
            return Ok(ctx);
        }
    };
    let speaker_segments = match ctx.speaker_segments.as_ref() {
        Some(s) => s,
        None => {
            log::warn!("TTS: нет сегментов спикеров");
            return Ok(ctx);
        }
    };
    let wav_path = match ctx.wav_path.as_ref() {
        Some(p) => p,
        None => {
            log::warn!("TTS: нет WAV файла для определения длительности");
            return Ok(ctx);
        }
    };

    // Инициализация движка CrispASR (cosyvoice3-tts, GPU).
    let app = crate::app_handle()
        .ok_or_else(|| anyhow::anyhow!("TTS: AppHandle не инициализирован (запуск вне tauri?)"))?;
    let state = app.state::<tauri_plugin_speech::PluginState>();
    let tss = tauri_plugin_speech::tts_settings::load(app);
    let models_dir = if tss.models_dir.trim().is_empty() {
        tauri_plugin_speech::download::default_models_dir()
    } else {
        PathBuf::from(tss.models_dir.clone())
    };
    let engine_exe = crate::pick_engine_exe().map_err(|e| anyhow::anyhow!(e))?;

    // Выбранная TTS-модель: env `DUBVID_TTS_PRESET` (headless-прогоны/A-B), иначе
    // сохранённый пресет в настройках, иначе cosyvoice3-tts-rl (стабильная RL).
    let preset_id = std::env::var("DUBVID_TTS_PRESET")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            let s = tss.preset.trim();
            if s.is_empty() { None } else { Some(s.to_string()) }
        })
        .unwrap_or_else(|| "cosyvoice3-tts-rl".to_string());
    let tts_backend = tauri_plugin_speech::download::preset_backend(&preset_id)
        .map(|b| b.to_string())
        .unwrap_or_else(|| preset_id.clone());
    log::info!(
        "TTS: движок {} / модели {} / модель {} (backend {})",
        engine_exe,
        models_dir.display(),
        preset_id,
        tts_backend
    );

    // Движок стартует ЛЕНИВО, на каждый спикер отдельно: cosyvoice3 синтезирует
    // WAV-клон ТОЛЬКО из голоса, поданного при старте через `--voice <ref>`
    // (имена, зарегистрированные через /v1/voices, бэкенд не резолвит —
    // «voice not found (have 8)»). При смене спикера ensure() перезапускает движок
    // с новым референсом.

    // Preload исходный WAV (16 кГц mono) для вырезания референсов.
    let reader = hound::WavReader::open(wav_path)
        .with_context(|| format!("TTS: открытие {}", wav_path))?;
    let src_sr = reader.spec().sample_rate;
    let src_samples: Vec<i16> = reader
        .into_samples::<i16>()
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("TTS: чтение сэмплов: {e}"))?;
    let total_duration_sec = src_samples.len() as f64 / src_sr as f64;

    // Холст для финальной озвучки (44.1 кГц — полная полоса до 20 кГц).
    let out_sr: usize = 44100;
    let canvas_len = (total_duration_sec * out_sr as f64).ceil() as usize;
    let mut canvas: Vec<f32> = vec![0.0; canvas_len];

    let tmp_dir = crate::paths::temp_dir();
    let ffmpeg_path = ctx.config.ffmpeg_path.clone();

    // Карта спикера → путь к стартовому референсу (24кГц).
    let mut voice_map: HashMap<String, PathBuf> = HashMap::new();
    let mut current_startup_voice = String::new();

    // ---- PASS 1: синтез ВСЕХ чанков, без записи в холст ----
    let mut synthesized: Vec<SynthesizedChunk> = Vec::new();
    let mut durations: Vec<f64> = vec![0.0; translated.len()];

    for (idx, chunk) in translated.iter().enumerate() {
        let text = chunk.text.trim();
        if text.is_empty() || text == crate::comm::SKIP_MARKER {
            continue;
        }
        let speaker_id = chunk
            .speaker_id
            .as_deref()
            .unwrap_or("Speaker_1");

        let ref24 = match voice_map.get(speaker_id) {
            Some(v) => v.clone(),
            None => {
                // Создаём клон-референс по голосу спикера (16кГц → 24кГц для --voice).
                let seg = pick_ref_segment(speaker_id, speaker_segments)
                    .ok_or_else(|| anyhow::anyhow!("TTS: нет референсного сегмента для {}", speaker_id))?;
                let start = (seg.0 * src_sr as f64) as usize;
                let end = ((seg.1 * src_sr as f64) as usize).min(src_samples.len());
                if start >= end {
                    anyhow::bail!("TTS: пустой референс для {}", speaker_id);
                }
                let safe = safe_speaker(speaker_id);
                let ref16 = tmp_dir.join(format!("dubvidtra_tts_ref_{}.wav", safe));
                write_segment_wav(
                    &ref16.to_string_lossy(),
                    src_sr,
                    &src_samples[start..end],
                )?;
                let ref24 = tmp_dir.join(format!("dubvidtra_tts_ref_{}_24k.wav", safe));
                resample_wav(
                    &ref16.to_string_lossy(),
                    &ref24.to_string_lossy(),
                    24000,
                    &ffmpeg_path,
                )?;

                log::info!(
                    "TTS: спикер {} → стартовый голос {} (референс {:.1}–{:.1}с)",
                    speaker_id,
                    ref24.display(),
                    seg.0,
                    seg.1
                );
                voice_map.insert(speaker_id.to_string(), ref24.clone());
                ref24
            }
        };

        // Перезапуск движка с референсом спикера, если он сменился.
        let startup_str = ref24.to_string_lossy().to_string();
        if startup_str != current_startup_voice {
            crate::block_on(state.tts.ensure(
                app,
                &engine_exe,
                &tts_backend,
                &models_dir.to_string_lossy().to_string(),
                &preset_id,
                &startup_str,
            ))
            .map_err(|e| anyhow::anyhow!("TTS: запуск движка: {e}"))?;
            current_startup_voice = startup_str;
        }

        log::info!(
            "TTS: [{}/{}] ({:.1}s–{:.1}s) [{}] {}",
            idx + 1,
            translated.len(),
            chunk.start_sec,
            chunk.end_sec,
            speaker_id,
            text,
        );

        // Cross-lingual clone: EN-референс (startup voice) → RU-синтез.
        //
        // Адаптивный retry (костыль) применяется ТОЛЬКО к base cosyvoice3-tts:
        // его cross-lingual LM выбрасывает reference-токены и иногда
        // недо-генерирует реплику — финальные слоги «обрезаются». RAS-семплер
        // seed-зависим, поэтому пробуем несколько seed'ов. Условие выбора
        // финальной генерации — ПОЛНОТА: реальный темп речи варьируется
        // 3.9–6.9 гласных/сек, а обрезанные генерации короче полной в
        // ~0.72–0.86× — поэтому останавливаемся, когда ДВЕ попытки сходятся
        // на длине полной фразы.
        //
        // RL-модель (cosyvoice3-tts-rl, RE-постобучен на стабильность) этот
        // костыль НЕ требует: синтезируем один раз без сидов.
        let expected = expected_spoken_secs(text);
        let mut best: Option<(Vec<u8>, u32, f64)> = None;
        let attempt_count = if tts_backend == "cosyvoice3-tts" {
            TTS_MAX_ATTEMPTS
        } else {
            1
        };
        let mut tracker = RetryTracker::new(expected);
        for attempt in 0..attempt_count {
            let seed = match attempt {
                0 => None,
                n => Some(TTS_RETRY_SEEDS[n - 1]),
            };
            let (wav_bytes, _timing) = crate::block_on(state.tts.speak(
                text,
                "",
                "",
                "",
                1.0,
                true,
                "ru",
                "en",
                seed,
            ))
            .map_err(|e| {
                anyhow::anyhow!("TTS: синтез [{}] '{}': {}", idx + 1, text, e)
            })?;

            let (gen_sr, gen_duration) = wav_sr_and_duration_bytes(&wav_bytes)?;
            let v = tracker.observe(gen_duration);
            if v.is_best {
                best = Some((wav_bytes, gen_sr, gen_duration));
            }

            log::info!(
                "TTS: [{}] попытка {}/{} (seed={}) gen={:.2}s (ожид≈{:.2}s) ok{}",
                idx + 1,
                attempt + 1,
                attempt_count,
                seed.map_or_else(|| "-".to_string(), |s| s.to_string()),
                gen_duration,
                expected,
                if v.stop { " →стабильно" } else { "" },
            );

            if v.stop {
                break;
            }
        }

        let (wav_bytes, gen_sr, gen_duration) = best
            .expect("TTS: ни одна попытка не дала генерации");
        durations[idx] = gen_duration;
        synthesized.push(SynthesizedChunk {
            index: idx,
            wav: wav_bytes,
            gen_duration,
            gen_sr,
        });
    }

    // ---- PASS 2: планирование таймлайна без пересечений ----
    let plan = plan_timeline(&translated, &durations);
    for p in &plan {
        if p.has_audio {
            log::info!(
                "TTS: план [{}] окно {:.1}s–{:.1}s (dur={:.1}s ratio={:.2}{})",
                p.index + 1,
                p.placed_start,
                p.placed_start + p.placed_duration,
                p.placed_duration,
                p.stretch_ratio,
                if p.truncated { " TRUNC" } else { "" },
            );
        }
    }

    // ---- PASS 3: размещение + обновление таймингов RU ----
    // Капы: натуральный хвост фразы живёт до старта следующего звучащего
    // чанка (минус TAIL_SAFETY_SEC), а не обрезается по окну — текстура
    // фразы сохраняется, «предупреждения» не превращается в «предупрежdeny».
    let tail_safety_samples = (TAIL_SAFETY_SEC * out_sr as f64) as usize;
    let mut cap_next: usize = canvas.len();
    let mut tail_caps: Vec<usize> = vec![canvas.len(); translated.len()];
    for p in plan.iter().rev() {
        tail_caps[p.index] = cap_next;
        if p.has_audio {
            let start_samples = (p.placed_start * out_sr as f64) as usize;
            cap_next = cap_next.min(start_samples.saturating_sub(tail_safety_samples));
        }
    }
    for p in &plan {
        if !p.has_audio {
            continue;
        }
        let s = match synthesized.iter().find(|s| s.index == p.index) {
            Some(s) => s,
            None => continue,
        };
        let idx = p.index;

        let raw_wav = tmp_dir.join(format!("dubvidtra_tts_raw_{}.wav", idx));
        std::fs::write(&raw_wav, &s.wav)
            .with_context(|| format!("TTS: запись raw WAV {}", raw_wav.display()))?;

        let stretched_wav = tmp_dir.join(format!("dubvidtra_tts_stretched_{}.wav", idx));
        if p.stretch_ratio > 1.02 {
            time_stretch_wav(
                &raw_wav.to_string_lossy(),
                &stretched_wav.to_string_lossy(),
                p.placed_duration,
                s.gen_duration,
                &ffmpeg_path,
            )?;
        } else {
            std::fs::copy(&raw_wav, &stretched_wav).ok();
        }

        let resampled_wav = tmp_dir.join(format!("dubvidtra_tts_final_{}.wav", idx));
        resample_wav(
            &stretched_wav.to_string_lossy(),
            &resampled_wav.to_string_lossy(),
            out_sr as u32,
            &ffmpeg_path,
        )?;

        // Читаем обработанный WAV.
        let mut final_samples: Vec<f32> = {
            let r = hound::WavReader::open(&resampled_wav)
                .context("TTS: чтение финального WAV")?;
            r.into_samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect()
        };

        // Щелчки (импульсные выбросы RAS-семплера, до сих пор прохожили через
        // stretch/resample): сглаживаем до укладки — детерминированно, без
        // лишних перегенераций. Каждая правка ≤1 сэмпла из 24000/с.
        declick_spikes(&mut final_samples);

        // Речь стартует с атаки первого слова, без «предзапуска» TTS.
        // Возвращает число слитых сэмплов головы: сдвигаем offset на drained,
        // чтобы КОНТЕНТ остался в исходной абсолютной позиции — иначе весь
        // хвост фразы «уезжал» влево на drained и финальный гласный пропадал.
        let drained = trim_edges_silence(&mut final_samples, out_sr);
        let offset = (p.placed_start * out_sr as f64) as usize + drained;
        let fade_in = (FADE_IN_SEC * out_sr as f64) as usize;
        let fade_out = if p.truncated {
            (FADE_OUT_TRUNCATED_SEC * out_sr as f64) as usize
        } else {
            (FADE_OUT_SEC * out_sr as f64) as usize
        };

        // Хвост не режем по окну: если ресемплер слегка перетянул за кап
        // (старт следующего чанка), лишнее гасим плавным 30 мс fade — клик
        // невозможен, а сам хвост в штатном случае остаётся целиком.
        let cap = tail_caps[idx];
        let can_keep = final_samples.len().min(cap.saturating_sub(offset));
        trim_tail_to_cap(&mut final_samples, can_keep, out_sr);
        place_samples(&mut canvas, &final_samples, offset, fade_in, fade_out);

        // RU SRT должен отражать фактическую позицию дубляжа — включая
        // сохранённый натуральный хвост (а не синтетическую placed_duration).
        let placed_end = (offset + final_samples.len()).min(canvas.len()) as f64
            / out_sr as f64;
        translated[idx].start_sec = p.placed_start;
        translated[idx].end_sec = placed_end;

        log::info!(
            "TTS: [{}] размещено @{:.1}s (gen={:.1}s @{}k, dur={:.1}s, ratio={:.2})",
            idx + 1,
            p.placed_start,
            s.gen_duration,
            s.gen_sr / 1000,
            p.placed_duration,
            p.stretch_ratio,
        );

        // Чистим временные файлы чанка (диагностика: DUBVID_KEEP_TTS_WAV=1).
        if std::env::var_os("DUBVID_KEEP_TTS_WAV").is_none() {
            std::fs::remove_file(&raw_wav).ok();
            std::fs::remove_file(&stretched_wav).ok();
            std::fs::remove_file(&resampled_wav).ok();
        }
    }

    // Чистим референсные WAV.
    for key in voice_map.keys() {
        let safe = safe_speaker(key);
        std::fs::remove_file(tmp_dir.join(format!("dubvidtra_tts_ref_{}.wav", safe))).ok();
        std::fs::remove_file(tmp_dir.join(format!("dubvidtra_tts_ref_{}_24k.wav", safe))).ok();
    }

    // Останавливаем движок.
    crate::block_on(state.tts.stop());

    // Сохраняем финальную дорожку озвучки.
    let dubbed_path = tmp_dir
        .join("dubvidtra_dubbed.wav")
        .to_string_lossy()
        .to_string();
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: out_sr as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&dubbed_path, spec)
            .context("Ошибка создания финальной WAV дорожки")?;
        for &s in &canvas {
            let clamped = s.clamp(-1.0, 1.0);
            writer
                .write_sample((clamped * 32767.0) as i16)
                .ok();
        }
        writer.finalize().ok();
    }

    log::info!("TTS: дорожка озвучки сохранена: {}", dubbed_path);

    Ok(PipelineContext {
        translated_chunks: Some(translated),
        dubbed_audio_path: Some(dubbed_path),
        ..ctx
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(start: f64, end: f64) -> SubtitleChunk {
        SubtitleChunk {
            start_sec: start,
            end_sec: end,
            text: "test".to_string(),
            speaker_id: None,
            word_timestamps: None,
        }
    }

    fn no_overlaps(plan: &[PlannedChunk]) -> (bool, Option<(usize, f64, f64)>) {
        let mut prev_end = -1.0f64;
        for p in plan {
            if !p.has_audio {
                continue;
            }
            if p.placed_start < prev_end - 1e-6 {
                return (false, Some((p.index, p.placed_start, prev_end)));
            }
            prev_end = p.placed_start + p.placed_duration;
        }
        (true, None)
    }

    #[test]
    fn plan_natural_speed_without_borrow() {
        // Оба чанка помещаются в свои окна на натуральной скорости.
        let chunks = vec![chunk(0.0, 10.0), chunk(12.0, 20.0)];
        let durations = [4.0, 5.0];
        let plan = plan_timeline(&chunks, &durations);
        assert!(plan[0].has_audio && plan[1].has_audio);
        assert_eq!(plan[0].stretch_ratio, 1.0);
        assert_eq!(plan[1].stretch_ratio, 1.0);
        assert!((plan[0].placed_start - 0.0).abs() < 1e-6);
        assert!((plan[1].placed_start - 12.0).abs() < 1e-6);
        let (ok, bad) = no_overlaps(&plan);
        assert!(ok, "overlap: {:?}", bad);
    }

    #[test]
    fn plan_does_not_start_earlier_than_original() {
        // Фраза длиннее окна НЕ должна уезжать влево: она стартует в своём
        // start_sec и сжимается, влезая в окно.
        let chunks = vec![chunk(5.0, 10.0)];
        let durations = [6.5];
        let plan = plan_timeline(&chunks, &durations);
        let p = &plan[0];
        assert!(p.has_audio);
        assert!((p.placed_start - 5.0).abs() < 1e-6);
        assert!((p.placed_duration - 5.0).abs() < 1e-6);
        assert!((p.stretch_ratio - 1.3).abs() < 1e-6);
        assert!(!p.truncated);
        let (ok, bad) = no_overlaps(&plan);
        assert!(ok, "overlap: {:?}", bad);
    }

    #[test]
    fn plan_spills_after_window_instead_of_overcompressing() {
        // gen=8 в окне 5с: сжатие до потолка разборчивости 1.3 → 6.15s.
        // Фраза «перетекает» за конец окна (spill), следующая — каскадно вправо.
        let chunks = vec![chunk(0.0, 5.0), chunk(6.0, 9.0)];
        let durations = [8.0, 3.0];
        let plan = plan_timeline(&chunks, &durations);
        let p0 = &plan[0];
        assert!(p0.has_audio);
        assert!((p0.stretch_ratio - MAX_STRETCH_RATIO).abs() < 1e-6);
        assert!(
            p0.placed_start + p0.placed_duration > 5.0 + 1e-6,
            "фраза должна перетекать за конец окна, а не сжиматься сильнее"
        );
        assert!(!p0.truncated);
        let p1 = &plan[1];
        assert!(
            p1.placed_start >= p0.placed_start + p0.placed_duration + MIN_GAP_SEC - 1e-6,
            "следующая фраза сдвигается вправо (каскад)"
        );
        let (ok, bad) = no_overlaps(&plan);
        assert!(ok, "overlap: {:?}", bad);
    }

    #[test]
    fn plan_moderate_compression_stays_in_window() {
        // gen=6 в окне 5с: ratio 1.2 ≤ потолка → сжимается и влезает без spill.
        let chunks = vec![chunk(0.0, 5.0)];
        let durations = [6.0];
        let plan = plan_timeline(&chunks, &durations);
        let p = &plan[0];
        assert!(p.has_audio);
        assert!((p.placed_duration - 5.0).abs() < 1e-6);
        assert!((p.stretch_ratio - 1.2).abs() < 1e-6);
        assert!(!p.truncated);
    }

    #[test]
    fn plan_no_overlap_with_cascade() {
        // Плотная последовательность длинных фраз: никаких пересечений.
        let chunks: Vec<SubtitleChunk> = (0..10)
            .map(|i| chunk(i as f64 * 3.0, i as f64 * 3.0 + 2.0))
            .collect();
        let durations: Vec<f64> = (0..10).map(|i| 3.0 + (i % 3) as f64).collect();
        let plan = plan_timeline(&chunks, &durations);
        let (ok, bad) = no_overlaps(&plan);
        assert!(ok, "overlap: {:?}", bad);
        for p in &plan {
            assert!(p.has_audio);
            assert!(p.stretch_ratio > 0.0);
        }
    }

    #[test]
    fn plan_skips_chunks_without_audio() {
        let chunks = vec![chunk(0.0, 5.0), chunk(7.0, 9.0)];
        let durations = [0.0, 2.0];
        let plan = plan_timeline(&chunks, &durations);
        assert!(!plan[0].has_audio);
        assert!(plan[1].has_audio);
        let (ok, bad) = no_overlaps(&plan);
        assert!(ok, "overlap: {:?}", bad);
    }

    #[test]
    fn plan_truncation_last_resort() {
        // gen=20 в окне 3с: даже hard-stretch + overflow borrow не влезают → trunc.
        let chunks = vec![chunk(0.0, 3.0)];
        let durations = [20.0];
        let plan = plan_timeline(&chunks, &durations);
        let p = &plan[0];
        assert!(p.has_audio);
        assert!(p.truncated);
        assert!(p.placed_duration > 0.0);
    }

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

    #[test]
    fn retry_chunk27_picks_full_generation() {
        // Чанк 27: полная генерация 4.12с, обрезанные 3.20/3.56с.
        // Трекер должен выбрать 4.12с (самую длинную) и не остановиться на недо-версиях.
        let mut t = RetryTracker::new(4.5);
        let gens = [3.20, 3.56, 3.20, 4.12];
        let (best, attempts) = simulate(&mut t, &gens);
        assert_eq!(attempts, 4, "полная генерация найдена на 4-й попытке");
        assert!((best - 4.12).abs() < 1e-9, "best={best}");
    }

    #[test]
    fn retry_chunk34_stops_after_short_under_attempt() {
        // Чанк 34: обрезанная 2.28с + полные 3.04с (и подтверждение 3.00с в пределах 3%).
        let mut t = RetryTracker::new(3.0);
        let gens = [2.28, 3.04, 3.00];
        let (best, attempts) = simulate(&mut t, &gens);
        assert_eq!(attempts, 3);
        assert!((best - 3.04).abs() < 1e-9, "best={best}");
        // Первый (короткий) прогон — не лучший, но останавливаться рано нельзя.
    }

    #[test]
    fn retry_chunk35_accepts_on_first_confirmed_try() {
        // Чанк 35 (длинный, полные генерации ~7.36-7.60с): две попытки в пределах 3%
        // от максимума → можно заканчивать, не гоняя все 6.
        let mut t = RetryTracker::new(7.5);
        let gens = [7.36, 7.60, 7.55];
        let (best, attempts) = simulate(&mut t, &gens);
        assert_eq!(attempts, 3);
        assert!((best - 7.60).abs() < 1e-9, "best={best}");
    }

    #[test]
    fn retry_chunk38_does_not_run_all_six() {
        // Чанк 38 (66 гласных, ожидание ~15с): реальные генерации 11-12.7с.
        // Старый порог от ожид≈22с никогда не достигался → все 6 попыток впустую.
        // Трекер должен остановиться после подтверждения максимума.
        let mut t = RetryTracker::new(15.0);
        let gens = [12.68, 11.04, 12.68];
        let (best, attempts) = simulate(&mut t, &gens);
        assert!(attempts <= 4, "должен остановиться раньше 6 попыток: {attempts}");
        assert!((best - 12.68).abs() < 1e-9, "best={best}");
    }

    #[test]
    fn retry_chunk36_short_not_confirmed() {
        // Чанк 36: shorts 2.04-3.28с; 2.04с явно недо-генерация относительно 3.28с —
        // не подтверждает максимум и не останавливает.
        let mut t = RetryTracker::new(3.3);
        let v0 = t.observe(3.28);
        assert!(v0.is_best);
        let v1 = t.observe(2.04);
        assert!(!v1.stop, "2.04с не подтверждает 3.28с");
        assert!(!v1.is_best);
    }

    #[test]
    fn retry_chunk7_short_twins_not_confirmed_red() {
        // Чанк 7 «у вас уже 2 предупреждения»: обе попытки дали 1.48с при
        // ожидаемом ≈2.50с (11 гласных / 4.4). 1.48/2.50 = 0.59 — это ОБРЕЗАННЫЙ
        // хвост (полная фраза должна звучать как «...предупреждения» целиком).
        // Два одинаково коротких «близнеца» НЕ являются подтверждением полного
        // прочтения: трекер обязан продолжать retry (другие seed).
        // RED: текущий код останавливается после второго 1.48с (near_best=2).
        let mut t = RetryTracker::new(2.50);
        let v0 = t.observe(1.48);
        assert!(v0.is_best);
        assert!(!v0.stop, "первая попытка не должна останавливать");
        let v1 = t.observe(1.48);
        assert!(
            !v1.stop,
            "2 одинаково КОРОТКИХ генерации (1.48с ≪ ожидаемых 2.50с) не должны "
        );
    }

    #[test]
    fn retry_monotonic_increase_no_premature_stop() {
        // Последовательный рост длины (разные seed дают всё более полные версии):
        // останавливаться раньше времени нельзя.
        let mut t = RetryTracker::new(4.5);
        let gens = [3.0, 3.5, 3.8, 4.0, 4.4];
        let (best, attempts) = simulate(&mut t, &gens);
        assert_eq!(attempts, 5);
        assert!((best - 4.4).abs() < 1e-9, "best={best}");
    }

    #[test]
    fn tail_fade_ramps_down_towards_cut() {
        // Инверсия fade: у точки среза амплитуда → 0, вглубь хвоста → 1.0.
        let sr = 44100usize;
        let mut samples = vec![0.9f32; 5000];
        // can_keep = 4000 → 1000 сэмплов лишнего (~23 мс > fade 30 мс не должен короче).
        trim_tail_to_cap(&mut samples, 4000, sr);
        assert_eq!(samples.len(), 4000);
        // Последние ~3 сэмпла должны быть ~0 (гаснут у среза).
        for k in 0..4usize {
            assert!(
                samples[samples.len() - 1 - k] < 0.05,
                "хвост {k} должен быть ~0, got {}",
                samples[samples.len() - 1 - k]
            );
        }
        // Далеко от среза — амплитуда не тронута (0.9).
        assert!((samples[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn tail_fade_not_inverted_into_click() {
        // Регрессия: раньше у среза оставалась ПОЛНАЯ амплитуда (frac=1.0),
        // что давало щелчок. Проверяем: амплитуда последнего сэмпла ≪ первого.
let sr = 44100usize;
        let mut samples = vec![0.8f32; 3000];
        let cut = 2500usize;
        let excess = samples.len().saturating_sub(cut);
        assert!(excess > 0);
        trim_tail_to_cap(&mut samples, cut, sr);
        assert_eq!(samples.len(), cut);
        let last = samples[samples.len() - 1];
        let first = samples[0];
        assert!(
            last < first * 0.05,
            "последний сэмпл должен быть ~0: last={last} first={first}"
        );
    }

    #[test]
    fn declick_smooths_real_click_morphology() {
        // Реальные морфологии кликов cosyvoice3 (to f32/-32768):
        // ч.3: 5908 → -7815 → 180 (V-провал) и ч.27: -14443 → 5802 → 23306 (срыв).
        let mut s = vec![0.01f32; 200];
        s[100] = -7815.0 / 32768.0;
        s[99] = 5908.0 / 32768.0;
        s[101] = 180.0 / 32768.0;
        declick_spikes(&mut s);
        assert!(
            (s[100] - (s[99] + s[101]) / 2.0).abs() < 1e-6,
            "пик заменён средним соседей: {}",
            s[100]
        );
        assert_eq!(s.len(), 200);
    }

    #[test]
    fn declick_smooths_single_sample_spike() {
        let mut s = vec![0.1f32; 200];
        s[100] = 0.85;
        declick_spikes(&mut s);
        assert!((s[100] - 0.1).abs() < 1e-6, "пик убран: {}", s[100]);
    }

    #[test]
    fn declick_leaves_smooth_signal_untouched() {
        // Пилообразный тон: перепады ~0.08 ≪ порога SPIKE=0.22 — ничего не трогаем.
        let s: Vec<f32> = (0..2000).map(|i| ((i % 80) as f32) * 0.001 - 0.04).collect();
        let mut t = s.clone();
        declick_spikes(&mut t);
        assert_eq!(t, s, "плавный сигнал не меняется");
    }

    #[test]
    fn declick_short_buffer_no_panic() {
        let mut a: Vec<f32> = vec![];
        declick_spikes(&mut a);
        let mut b = vec![0.5f32; 2];
        declick_spikes(&mut b);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn declick_fixes_plateau_in_loud_context() {
        // Регрессия пути 2: RAS-импульс посреди ГРОМКОЙ речи — тихого
        // окружения нет, старый гейт «тихо слева И справа» пропускал такие
        // (37 остаточных кандидатов на 192-с файл). Плато: 2 сэмпла −0.25
        // на фоне 0.15; края 0.4 попадают в (0.22; 0.45) — выше SPIKE, но
        // ниже порога STEP, чтобы одиночное правило не вмешалось первым.
        let mut s = vec![0.15f32; 5000];
        s[2500] = -0.25;
        s[2501] = -0.25;
        declick_spikes(&mut s);
        assert_eq!(s[2500], 0.15, "плато в громком контексте заменено: {}", s[2500]);
        assert_eq!(s[2501], 0.15, "плато в громком контексте заменено: {}", s[2501]);
        // Хвост и голова не тронуты.
        assert_eq!(s[0], 0.15);
        assert_eq!(s[4999], 0.15);
    }

    #[test]
    fn trim_head_returns_drained_and_keeps_tail() {
        // Голова — ровно 100 шагов тишины, затем громкий тон. trim возвращает
        // число слитых сэмплов и НЕ трогает хвост (обрезка хвоста «съедает»
        // финальный гласный — регрессия).
        let sr = 44100usize;
        let step = ((sr / 1000) * 2).max(8);
        let head = step * 100;
        let mut s = vec![0.0f32; head + sr];
        for v in s.iter_mut().skip(head) {
            *v = 0.9;
        }
        let len_before = s.len();
        let drained = trim_edges_silence(&mut s, sr);
        assert_eq!(drained, head, "предзапуск слит целиком");
        assert_eq!(s.len(), len_before - drained, "хвост не должен резаться");
        assert!((s[0] - 0.9).abs() < 1e-4, "звук начинается с первого сэмпла");
        assert!((s[s.len() - 1] - 0.9).abs() < 1e-4, "хвост сохранил амплитуду");
    }
}