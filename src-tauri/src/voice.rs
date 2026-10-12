//! Измерение голоса по аудио: основной тон (F0) и непрерывность речи.
//!
//! Единственный источник измерений тона для всего проекта. Потребители:
//! `diarization` (определение пола) и `tts` (выбор клон-референса) —
//! они получают его через `comm::voice`, а не своей копией кода.
//!
//! ## Почему именно такая обработка
//!
//! Каждый параметр ниже выбран абляцией на семи реальных выдержках
//! (`test/fixtures/gender/`) против независимого ground truth, который трекером
//! не измеряется вовсе: F0 берётся как расстояние между соседними пиками
//! спектра, и октавной ошибки в нём быть не может по построению.
//!
//! ```
//! fixture               TRUTH     base     gate gate+hp70 gate+hp100
//! cluster0_0.wav (Мел)   163      130      229      200      222
//! cluster1_0.wav          86      n/a      n/a      131      144
//! cluster1_1.wav         109      262      262      127      215
//! cluster2_0.wav (вед)   114      132      131      132      133
//! cluster2_1.wav         111      129      128      129      131
//! cluster2_2.wav          82      102      107      117      117
//! cluster2_3.wav          98       95      n/a      125      147
//! ошибок определения пола:  base 2/6   gate 1/5   gate+hp70 0/7   gate+hp100 1/7
//! ```
//!
//! Что каждая ступень убирает:
//!
//! - **ФВЧ [`HP_CUTOFF_HZ`]** — в записи есть низкочастотный гул с пиком 47 Гц.
//!   Пока он не срезан, он тянет минимум CMND вправо, к большим лагам, и тон
//!   читается вдвое ниже: Мелоди 208 Гц -> 130 Гц, и пол определяется верно
//!   только по счастливой случайности. Частота среза 70 Гц, а не 100: мужской
//!   баз этой записи лежит на 82–137 Гц, и срез 100 Гц срезает его начало
//!   (в абляции 1 ошибка из 7 против 0 у 70 Гц).
//! - **Гейт по энергии [`VOICING_ENERGY_PERCENTILE`]** — паузы, согласные и
//!   шум не имеют тона, но попадали в медиану и тянули её вниз.
//! - **YIN с приёмкой [`YIN_ACCEPT`]** — берётся первый локальный минимум CMND,
//!   а не глобальный максимум корреляции: максимум на сильной второй гармонике
//!   уезжает на 2T.

/// Частота среза ФВЧ, убирающего низкочастотный гул записи.
pub const HP_CUTOFF_HZ: f64 = 70.0;

/// Кадры тише этого перцентиля энергии сегмента считаются неречевыми.
pub const VOICING_ENERGY_PERCENTILE: f64 = 70.0;

/// Порог приёмки локального минимума CMND в YIN.
pub const YIN_ACCEPT: f64 = 0.30;

/// Порог глобального минимума, если первый локальный не найден.
pub const YIN_FALLBACK: f64 = 0.60;

/// Нижняя граница поиска периода: ниже считаем не голосом.
pub const PITCH_SEARCH_MIN_HZ: f64 = 70.0;

/// Верхняя граница поиска периода: выше считаем не голосом.
pub const PITCH_SEARCH_MAX_HZ: f64 = 450.0;

/// Оценки вне диапазона отбрасываются как октавные сбои.
pub const KEEP_MIN_HZ: f64 = 65.0;
pub const KEEP_MAX_HZ: f64 = 400.0;

/// Длина кадра анализа. 4096 сэмплов при 16 кГц = 256 мс, два периода F0.
pub const FRAME_SIZE: usize = 4096;

/// Шаг кадра: каждый второй, чтобы не считать одно и то же дважды.
pub const HOP_SIZE: usize = 1024;

/// Граница пола по основному тону: ниже — male, выше — female.
///
/// Порог не подбирался на ролике, а взят из границ мужского и женского
/// регистров ([`KEEP_MAX_HZ`] — 400 Гц) и проверен абляцией: конфигурация
/// [`HP_CUTOFF_HZ`] + [`VOICING_ENERGY_PERCENTILE`] + [`YIN_ACCEPT`] даёт 0
/// ошибок из 7 фикстур, где базовая конфигурация даёт 2 из 6.
pub const GENDER_MALE_BELOW_HZ: f64 = 160.0;

/// Результат измерения одного отрезка аудио.
#[derive(Debug, Clone, Default)]
pub struct VoiceTrack {
    /// Медиана F0 по принятым кадрам. `None` — тона не нашлось.
    pub f0_median: Option<f64>,
    /// Доля кадров, в которых тон найден. Непрерывность речи: насколько
    /// отрезок пригоден как референс для клона, где тембровый эмбеддинг
    /// строится по одному непрерывному utterance.
    pub voiced_ratio: f64,
    /// Всего кадров проанализировано.
    pub frames: usize,
    /// Тон каждого принятого кадра. Нужен, чтобы сложить медиану по всему
    /// говорящему, а не по median-of-medians отдельных реплик: у одного
    /// человека тон гуляет между репликами, и median-of-medians искажает
    /// распределение, а кадровая медиана его не искажает.
    pub pitches: Vec<f64>,
}

impl VoiceTrack {
    fn empty() -> Self {
        Self::default()
    }
}

/// Пол по измеренному тону. `None` — тон не измерен, пол неизвестен.
///
/// `None` возвращается наружу, а не подменяется дефолтом: подставить «male»
/// означало бы сообщить пользователю ложь (core §2.2).
pub fn gender_of(f0_median: Option<f64>) -> Option<&'static str> {
    f0_median.map(|hz| {
        if hz >= GENDER_MALE_BELOW_HZ {
            "female"
        } else {
            "male"
        }
    })
}

/// Биквадратный сектор ФВЧ (RBJ cookbook), нормированный на `a0`.
#[derive(Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

/// Два сектора Butterworth 4-го порядка из двух секций 2-го порядка.
fn highpass_sections(sample_rate: u32, cutoff_hz: f64) -> Vec<Biquad> {
    // Q для полюсов Butterworth: 0.5412, 1.3066
    [0.5411961f64, 1.3065629]
        .iter()
        .map(|&q| {
            let w0 = 2.0 * std::f64::consts::PI * cutoff_hz / sample_rate as f64;
            let cw = w0.cos();
            let alpha = w0.sin() / (2.0 * q);
            let a0 = 1.0 + alpha;
            Biquad {
                b0: (1.0 + cw) / 2.0 / a0,
                b1: -(1.0 + cw) / a0,
                b2: (1.0 + cw) / 2.0 / a0,
                a1: -2.0 * cw / a0,
                a2: (1.0 - alpha) / a0,
            }
        })
        .collect()
}

/// ФВЧ 4-го порядка. Прямая форма I с транспонированным состоянием.
///
/// Если частота среза не помещается под Найквист, сигнал возвращается без
/// изменения: проектировать такой фильтр нельзя, а молча пропускать гулевой
/// полкосы — значило бы вернуть заведомо неверный тон.
pub fn highpass(samples: &[f32], sample_rate: u32, cutoff_hz: f64) -> Vec<f32> {
    let fs = sample_rate as f64;
    if sample_rate == 0 || !(fs > 2.2 * cutoff_hz) {
        log::warn!(
            "Voice: ФВЧ {} Гц не проектируется при {} Гц оцифровки — сигнал оставлен как есть",
            cutoff_hz,
            sample_rate
        );
        return samples.to_vec();
    }
    let sections = highpass_sections(sample_rate, cutoff_hz);
    let mut state: Vec<(f64, f64)> = sections.iter().map(|_| (0.0, 0.0)).collect();
    let mut out = Vec::with_capacity(samples.len());
    for &xv in samples {
        let mut x = xv as f64;
        for (i, s) in sections.iter().enumerate() {
            let y = s.b0 * x + state[i].0;
            state[i].0 = s.b1 * x - s.a1 * y + state[i].1;
            state[i].1 = s.b2 * x - s.a2 * y;
            x = y;
        }
        out.push(x as f32);
    }
    out
}

/// Перцентиль (0..100) по возрастанию.
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (((v.len() - 1) as f64) * p / 100.0).round() as usize;
    v[idx.min(v.len() - 1)]
}

/// Тон одного кадра через YIN (кумулятивная нормированная разностная функция).
///
/// Возвращает `None`, если в кадре нет устойчивого периода.
fn frame_pitch(frame: &[f64], sample_rate: u32) -> Option<f64> {
    let max_lag = (sample_rate as f64 / PITCH_SEARCH_MIN_HZ) as usize;
    let hi_lag = max_lag.min(FRAME_SIZE / 2);
    let min_lag = (sample_rate as f64 / PITCH_SEARCH_MAX_HZ) as usize;
    if hi_lag <= min_lag + 1 || min_lag == 0 {
        return None;
    }
    let mut d = vec![0.0f64; hi_lag + 1];
    for tau in 1..=hi_lag {
        let mut acc = 0.0f64;
        for j in 0..(frame.len() - tau) {
            let diff = frame[j] - frame[j + tau];
            acc += diff * diff;
        }
        d[tau] = acc;
    }
    let mut cmnd = vec![1.0f64; hi_lag + 1];
    let mut running = 0.0f64;
    for tau in 1..=hi_lag {
        running += d[tau];
        if running > 1e-12 {
            cmnd[tau] = d[tau] * tau as f64 / running;
        }
    }
    let mut period: Option<usize> = None;
    for tau in min_lag..=hi_lag {
        if cmnd[tau] < YIN_ACCEPT
            && cmnd[tau] < cmnd[tau - 1]
            && (tau == hi_lag || cmnd[tau] <= cmnd[tau + 1])
        {
            period = Some(tau);
            break;
        }
    }
    if period.is_none() {
        let mut best = min_lag;
        for tau in min_lag..=hi_lag {
            if cmnd[tau] < cmnd[best] {
                best = tau;
            }
        }
        if cmnd[best] < YIN_FALLBACK {
            period = Some(best);
        }
    }
    period.map(|p| sample_rate as f64 / p as f64)
}

/// Полное измерение отрезка: тон и непрерывность речи.
pub fn voice_track(samples: &[f32], sample_rate: u32) -> VoiceTrack {
    if samples.len() < FRAME_SIZE || sample_rate == 0 {
        return VoiceTrack::empty();
    }
    let filtered = highpass(samples, sample_rate, HP_CUTOFF_HZ);
    let frames = 1 + (filtered.len() - FRAME_SIZE) / HOP_SIZE;

    let mut rms = Vec::with_capacity(frames);
    for i in 0..frames {
        let s = &filtered[i * HOP_SIZE..i * HOP_SIZE + FRAME_SIZE];
        let e: f64 = s.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        rms.push((e / FRAME_SIZE as f64).sqrt());
    }
    let threshold = percentile(&rms, VOICING_ENERGY_PERCENTILE);

    let mut pitches: Vec<f64> = Vec::new();
    let mut frame = vec![0.0f64; FRAME_SIZE];
    for i in 0..frames {
        if rms[i] < threshold {
            continue;
        }
        let src = &filtered[i * HOP_SIZE..i * HOP_SIZE + FRAME_SIZE];
        let mean: f64 = src.iter().map(|v| *v as f64).sum::<f64>() / FRAME_SIZE as f64;
        for (dst, s) in frame.iter_mut().zip(src) {
            *dst = *s as f64 - mean;
        }
        if let Some(f0) = frame_pitch(&frame, sample_rate) {
            if (KEEP_MIN_HZ..=KEEP_MAX_HZ).contains(&f0) {
                pitches.push(f0);
            }
        }
    }
    if pitches.is_empty() {
        return VoiceTrack { pitches, frames, ..VoiceTrack::default() };
    }
    pitches.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let f0_median = Some(pitches[pitches.len() / 2]);
    VoiceTrack {
        f0_median,
        voiced_ratio: pitches.len() as f64 / frames as f64,
        frames,
        pitches,
    }
}

/// Медиана по объединённым кадрам нескольких отрезков.
///
/// Считается по кадрам, а не медиана медиан: у говорящего тон гуляет от
/// реплики к реплике, и median-of-medians смещает результат в сторону
/// реплик, попавших в выборку. На тестовом ролике разница видна прямо —
/// Мелоди по медиане реплик давала 140 Гц (мужской диапазон), а по кадрам
/// 229 Гц (женский); эталон по гармоникам — 208 Гц.
pub fn median_pitch(pitches: &mut Vec<f64>) -> Option<f64> {
    if pitches.is_empty() {
        return None;
    }
    pitches.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(pitches[pitches.len() / 2])
}

/// Измерение тона для произвольного окна `[start_sec, end_sec)` исходного WAV.
pub fn voice_track_window(
    samples: &[f32],
    sample_rate: u32,
    start_sec: f64,
    end_sec: f64,
) -> VoiceTrack {
    let start = (start_sec * sample_rate as f64).max(0.0) as usize;
    let end = ((end_sec * sample_rate as f64) as usize).min(samples.len());
    if start >= end {
        return VoiceTrack::empty();
    }
    voice_track(&samples[start..end], sample_rate)
}

/// То же окно, но прямо из сэмплов WAV (`hound::WavReader` отдаёт `i16`).
///
/// Нормализация и нарезка живут здесь, а не в вызывающем модуле: иначе вызов
/// из TTS и вызов из тест-бинарника разойдутся в масштабе амплитуды, а гейт
/// по энергии — величина относительная, и сдвиг амплитуды его молча меняет.
pub fn voice_track_window_i16(
    samples: &[i16],
    sample_rate: u32,
    start_sec: f64,
    end_sec: f64,
) -> VoiceTrack {
    let start = (start_sec * sample_rate as f64).max(0.0) as usize;
    let end = ((end_sec * sample_rate as f64) as usize).min(samples.len());
    if start >= end {
        return VoiceTrack::empty();
    }
    let window: Vec<f32> = samples[start..end]
        .iter()
        .map(|v| *v as f32 / 32768.0)
        .collect();
    voice_track(&window, sample_rate)
}