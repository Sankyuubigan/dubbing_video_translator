use serde::{Deserialize, Serialize};

/// Маркер мусорного перевода: озвучка его не зачитывает, но в субтитрах
/// видно, что здесь пропуск. Ставят верификация/перевод.
pub const SKIP_MARKER: &str = "(-)";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSegment {
    pub start_sec: f64,
    pub end_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start_sec: f64,
    pub end_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleChunk {
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
    pub speaker_id: Option<String>,
    pub word_timestamps: Option<Vec<WordTimestamp>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub input_path: String,
    pub output_format: String,
    pub ffmpeg_path: Option<String>,
    pub enable_dubbing: bool,
    pub mix_volume: f64,
    /// Формат промпта перевода из настроек (`auto` | `insttrans` | `chat`).
    ///
    /// Живёт здесь, а не в глобальном состоянии или в `env::var` посреди
    /// пайплайна: `translation` и `verification` получают одно и то же значение
    /// из контекста, поэтому разойтись они не могут физически. Разойтись
    /// могли — и тогда ретраи верификации собирали бы промпт в чужом формате и
    /// чинили один дефект, ломая остальные.
    ///
    /// `None` означает «не задано» (headless-прогоны): тогда работает env
    /// `DEEDUB_LLM_PROMPT`, а после него определение по модели.
    #[serde(default)]
    pub prompt_style: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineContext {
    pub config: PipelineConfig,
    pub resolved_input_path: Option<String>,
    pub wav_path: Option<String>,
    pub subtitle_chunks: Option<Vec<SubtitleChunk>>,
    pub speaker_segments: Option<Vec<SpeakerSegment>>,
    pub translated_chunks: Option<Vec<SubtitleChunk>>,
    pub dubbed_audio_path: Option<String>,
    pub output_path: Option<String>,
}

impl PipelineContext {
    pub fn new(config: PipelineConfig) -> Self {
        let output_format = if config.output_format.is_empty() {
            let ext = std::path::Path::new(&config.input_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("mp4")
                .to_string();
            ext
        } else {
            config.output_format.clone()
        };

        Self {
            config: PipelineConfig {
                output_format,
                ..config
            },
            resolved_input_path: None,
            wav_path: None,
            subtitle_chunks: None,
            speaker_segments: None,
            translated_chunks: None,
            dubbed_audio_path: None,
            output_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    pub speaker_id: String,
    pub gender: Option<String>,
}

/// Минимальная длительность сегмента, пригодного как референс для клона голоса.
///
/// Единственный источник истины для двух потребителей:
/// - `merge_unclonable_speakers` (здесь) — спикер без сегмента такой длины
///   не считается различимым голосом, его сегменты присоединяются к соседу;
/// - `tts::pick_ref_segment` — не выбирает для клона сегмент короче этого порога.
///
/// Порог измеряет не «различимость голоса», а физическое требование движка:
/// CosyVoice3 строит референс из одного непрерывного отрезка исходного WAV, и
/// на отрезке меньше секунды эмбеддинг неустойчив — голос «плывёт».
pub const MIN_CLONE_REF_SEC: f64 = 1.0;

/// Убирает слова, в которых буква заменена символом цензуры: `бл*ть`, `п*здец`,
/// `f*ck`. Возвращает текст и число выброшенных слов.
///
/// Наблюдалось на Index-Translate-9B: чанк с «F you broke cheaters» пришёл как
/// «Бл*ть, обманщики…». Это не markdown и не ошибка формата — то же самое
/// получалось в chat- и в instTrans-промпте, тогда как 2B и Gemma в этом же
/// тесте не дали ни одной звёздочки. Модель узнаёт слово и цензурит его сама.
///
/// Почему слово выбрасывается, а не «достраивается»: восстановить заменённую
/// букву детерминированно нельзя, а вариантов ровно два — показать мусор или
/// выкинуть. Мусор хуже: `бл*ть` в SRT читается как опечатка, а в CosyVoice3
/// уходит небуквенный токен, на котором фонемизатор спотыкается и рвёт слово.
/// Правильное слово приходит с ретрая — `verification::assess` ловит дефект по
/// `censored_symbol`. Здесь просто страховка на случай, когда и ретрай вернул
/// цензуру.
///
/// Санитар, а не угадывание: словарь «`бл*ть` → `блять`» выглядел бы надёжнее,
/// но покрывает только известные маски, и первое же новое ругательство снова
/// поехало бы в озвучку.
///
/// Живёт в `comm`, а не в `translation`, потому что это чистая функция над
/// текстом без переводческого смысла: её зовут и перевод, и проверка, и тестовый
/// бинарник. Копия логики в тесте проверяла бы саму себя, а не production-код.
pub fn strip_masked_words(text: &str) -> (String, usize) {
    let mut kept: Vec<&str> = Vec::new();
    let mut dropped = 0usize;
    for token in text.split_whitespace() {
        // Знаки препинания вокруг слова («Бл*ть,») в проверку не входят: они
        // делали бы слово «не буквенным», и цензура проскакивала бы мимо.
        let core: String = token
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '*' || *c == '\'' || *c == '’')
            .collect();
        // Ровно одна `*` между буквами — наблюдавшийся вид маски. Две (`f**k`)
        // под `**` не попадают: `**` в этом проекте — markdown, и его режет
        // отдельная проверка `markdown_bold`.
        let masked = core.chars().filter(|c| *c == '*').count() == 1
            && core.chars().any(|c| c.is_alphabetic())
            && core.chars().all(|c| c.is_alphanumeric() || c == '*' || c == '\'' || c == '’');
        if masked {
            dropped += 1;
        } else {
            kept.push(token);
        }
    }
    if dropped > 0 {
        log::warn!(
            "Перевод: выброшено {dropped} замаскированных слов (модель цензурит мат). \
             Правильное написание обычно приходит с ретрая верификации."
        );
    }
    (kept.join(" "), dropped)
}

/// Финальная санитарная обработка перевода перед SRT и TTS.
///
/// Живёт в `comm`, а не в `translation::clean_output`, потому что это чистая
/// функция над текстом без переводческого смысла, и её зовут с двух сторон:
/// перевод и тест-бинарник. Пока последовательность была продублирована в тесте,
/// тест проверял бы свою копию, а не production-код, и разошёлся бы с ним при
/// первой же правке.
pub fn sanitize_for_output(text: &str) -> String {
    // Markdown по краям строки: `**Привет**` → `Привет`. Внутри фразы `**`
    // остаётся — его отдельно ловит проверка `markdown_bold`.
    let t = text
        .trim_start_matches(|c: char| c == '*' || c == '_')
        .trim_end_matches(|c: char| c == '*' || c == '_')
        .trim();
    let (cleaned, _) = strip_masked_words(t);
    // Схлопывание пробелов нужно потому, что удаление слова оставляет двойной
    // пробел перед знаком препинания («... ПК.  , обманщики»).
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Присоединяет спикеров, у которых нет сегмента длиной `MIN_CLONE_REF_SEC`.
///
/// Авто-кластеризация CrispASR (`--diarize-speakers auto`) недетерминирована
/// и на одном и том же аудио выдаёт то 4, то 5 спикеров: на ровном видео из
/// четырёх голосов лишний пятый кластер собирает из шума, смеха и кашля. Такие
/// сегменты проходят фильтр VAD-диаризации (>= 0.15 с), но для клона голоса
/// непригодны: `tts::pick_ref_segment` на них возвращает `None` и роняет весь
/// рендер на середине.
///
/// Такой кластер — не голос, а артефакт кластеризации, поэтому правится до
/// перевода и озвучки: речь спикера сохраняется в субтитрях, но получает голос
/// настоящего соседа. Менять номер у соседа нельзя — промпт перевода вставляет
/// имя текущего спикера буквально, и пропуск в нумерации читается как дыра в
/// данных.
///
/// Инвариант на выходе: у каждого оставшегося спикера есть сегмент
/// `>= MIN_CLONE_REF_SEC`. Если валидных спикеров нет (видео короче секунды
/// речи) слиянием это не лечится — референса в файле просто нет. Данные
/// оставляются как есть, а TTS падает с внятной причиной.
pub fn merge_unclonable_speakers(
    speaker_segments: &mut Vec<SpeakerSegment>,
    subtitle_chunks: &mut Vec<SubtitleChunk>,
) {
    let max_dur = |spk: &str| -> f64 {
        speaker_segments
            .iter()
            .filter(|s| s.speaker_id == spk)
            .map(|s| s.end_sec - s.start_sec)
            .fold(0.0f64, f64::max)
    };

    let valid: Vec<String> = speaker_segments
        .iter()
        .map(|s| s.speaker_id.as_str())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .filter(|spk| max_dur(spk) >= MIN_CLONE_REF_SEC)
        .map(str::to_string)
        .collect();
    // По первому появлению: нумерация идёт по хронологии.
    let valid: Vec<String> = {
        let mut v = valid;
        v.sort_by_key(|spk| {
            speaker_segments
                .iter()
                .position(|s| &s.speaker_id == spk)
                .unwrap_or(usize::MAX)
        });
        v
    };

    let dropped: Vec<String> = {
        let all: std::collections::HashSet<String> = speaker_segments
            .iter()
            .map(|s| s.speaker_id.clone())
            .collect();
        all.into_iter().filter(|spk| !valid.contains(spk)).collect()
    };
    if dropped.is_empty() {
        renumber_speakers(speaker_segments, subtitle_chunks);
        return;
    }

    if valid.is_empty() {
        let longest = speaker_segments
            .iter()
            .map(|s| s.end_sec - s.start_sec)
            .fold(0.0f64, f64::max);
        log::error!(
            "Diarization: ни одного сегмента спикера >= {}с (максимум {longest:.2}с) — \
             клонировать голос нечего, озвучка будет невозможна",
            MIN_CLONE_REF_SEC
        );
        return;
    }

    // Присоединять спикера к самому «долгоговорящему» нельзя: это отдало бы
    // его голос соседу даже там, где он говорит сам. Ближайший по времени
    // валидный спикер сохраняет интонационную связность диалога.
    let anchors: Vec<SpeakerSegment> = speaker_segments
        .iter()
        .filter(|s| valid.contains(&s.speaker_id))
        .cloned()
        .collect();

    let mut renames: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for spk in &dropped {
        let mine: Vec<(f64, f64)> = speaker_segments
            .iter()
            .filter(|s| &s.speaker_id == spk)
            .map(|s| (s.start_sec, s.end_sec))
            .collect();
        if mine.is_empty() {
            continue;
        }
        let total: f64 = mine.iter().map(|(a, b)| b - a).sum();
        let max: f64 = mine.iter().map(|(a, b)| b - a).fold(0.0f64, f64::max);

        // Медианный центр сегментов: один спикер может размазаться по трём
        // коротким вставкам в разных репликах, и медиана устойчивее первого.
        let mut mids: Vec<f64> = mine.iter().map(|(a, b)| (a + b) / 2.0).collect();
        mids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let my_center = mids[mids.len() / 2];
        let center = |an: &SpeakerSegment| (an.start_sec + an.end_sec) / 2.0;

        let target = anchors
            .iter()
            .filter(|an| &an.speaker_id != spk)
            .min_by(|a, b| {
                let da = (my_center - center(a)).abs();
                let db = (my_center - center(b)).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|an| an.speaker_id.clone())
            .unwrap_or_else(|| valid[0].clone());

        log::info!(
            "Diarization: {spk} ({} сегм., макс {:.2}с, всего {:.2}с) — нет референса >= {}с, \
             присоединяем к {target}",
            mine.len(),
            max,
            total,
            MIN_CLONE_REF_SEC
        );
        renames.insert(spk.clone(), target);
    }

    apply_renames(speaker_segments, subtitle_chunks, &renames);
    renumber_speakers(speaker_segments, subtitle_chunks);
}

/// Пересобирает `Speaker_N` в порядке первого появления по времени.
///
/// `speaker_segments` должен быть отсортирован по `start_sec`. Слияние
/// коротких спикеров оставляет пропуски в нумерации (`Speaker_2` исчезает,
/// остаются `Speaker_1` и `Speaker_3`), а промпт перевода вставляет в контекст
/// буквальное имя текущего спикера — «говорит Speaker_3» на первом же чанке
/// выглядит как дыра в данных. Поэтому номера перевыдаются после слияния.
fn renumber_speakers(
    speaker_segments: &mut Vec<SpeakerSegment>,
    subtitle_chunks: &mut Vec<SubtitleChunk>,
) {
    let mut order: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for seg in speaker_segments.iter() {
        let next = order.len() + 1;
        order.entry(seg.speaker_id.clone()).or_insert(next);
    }
    // Уже сплошная нумерация — ничего не трогаем (обычный случай без слияния).
    if order
        .iter()
        .all(|(id, num)| id == &format!("Speaker_{num}"))
    {
        return;
    }
    let renames: std::collections::HashMap<String, String> = order
        .into_iter()
        .map(|(old, num)| (old, format!("Speaker_{num}")))
        .collect();
    apply_renames(speaker_segments, subtitle_chunks, &renames);
}

/// Переименовывает спикера в обоих векторах разбора — сегменты и субтитры
/// строятся из одного прохода движка, поэтому их `speaker_id` обязаны
/// меняться синхронно, иначе перевод возьмёт голос не того спикера.
fn apply_renames(
    speaker_segments: &mut [SpeakerSegment],
    subtitle_chunks: &mut [SubtitleChunk],
    renames: &std::collections::HashMap<String, String>,
) {
    if renames.is_empty() {
        return;
    }
    for seg in speaker_segments.iter_mut() {
        if let Some(to) = renames.get(&seg.speaker_id) {
            seg.speaker_id = to.clone();
        }
    }
    for chunk in subtitle_chunks.iter_mut() {
        if let Some(spk) = chunk.speaker_id.as_ref() {
            if let Some(to) = renames.get(spk) {
                chunk.speaker_id = Some(to.clone());
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProgressUpdate {
    pub stage: String,
    pub percent: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// Находит спикера для чанка субтитров по максимальному перекрытию во времени.
/// Если перекрытия нет — берёт ближайшего по start_sec.
pub fn find_speaker_for_chunk<'a>(chunk: &SubtitleChunk, speakers: &'a [SpeakerSegment]) -> Option<&'a str> {
    if speakers.is_empty() {
        return None;
    }
    // Сначала ищем спикера с максимальным overlap
    let best = speakers
        .iter()
        .filter(|s| s.start_sec < chunk.end_sec && s.end_sec > chunk.start_sec)
        .max_by(|a, b| {
            let a_overlap = a.end_sec.min(chunk.end_sec) - a.start_sec.max(chunk.start_sec);
            let b_overlap = b.end_sec.min(chunk.end_sec) - b.start_sec.max(chunk.start_sec);
            a_overlap.partial_cmp(&b_overlap).unwrap_or(std::cmp::Ordering::Equal)
        });
    if best.is_some() {
        return best.map(|s| s.speaker_id.as_str());
    }
    // Нет перекрытия — берём ближайший по start_sec
    speakers
        .iter()
        .min_by(|a, b| {
            let da = (a.start_sec - chunk.start_sec).abs();
            let db = (b.start_sec - chunk.start_sec).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| s.speaker_id.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn speakers(segments: &[SpeakerSegment]) -> std::collections::HashSet<String> {
        segments
            .iter()
            .map(|s| s.speaker_id.clone())
            .collect()
    }

    /// Инвариант, ради которого всё и делается: `tts::pick_ref_segment`
    /// обязан находить референс для каждого спикера, иначе пайплайн падает
    /// на середине рендера.
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

    #[test]
    fn short_speaker_merges_into_nearest_by_time() {
        // Геометрия намеренно асимметричная: шумовой кластер (центр 5.4)
        // стоит в 2.9 с от Speaker_1 и в 8.1 с от Speaker_3 — при равных
        // расстояниях тест проверял бы случайность, а не правило выбора.
        let mut segs = vec![
            seg("Speaker_1", 0.0, 5.0),
            seg("Speaker_2", 5.2, 5.6), // 0.4 с — кластер из шума
            seg("Speaker_3", 12.0, 15.0),
        ];
        let mut chunks = vec![
            chunk("Speaker_1", 0.0, 5.0, "раз"),
            chunk("Speaker_2", 5.2, 5.6, "э-э-э"),
            chunk("Speaker_3", 12.0, 15.0, "два"),
        ];
        merge_unclonable_speakers(&mut segs, &mut chunks);

        assert_eq!(speakers(&segs).len(), 2, "после слияния должно быть 2 голоса");
        assert!(every_speaker_has_reference(&segs));
        // Сравниваем с МЕТКОЙ ЦЕЛИ, а не с её прежним именем: renumber_speakers
        // перевыдаёт номера после слияния, поэтому «Speaker_3» на выходе
        // вполне может стать «Speaker_2».
        let label_at = |start: f64| {
            segs.iter()
                .find(|s| (s.start_sec - start).abs() < 0.01)
                .map(|s| s.speaker_id.clone())
                .unwrap_or_default()
        };
        assert_eq!(label_at(5.2), label_at(0.0), "шум присоединён к соседу слева");
        assert_ne!(label_at(5.2), label_at(12.0));
    }

    #[test]
    fn merge_direction_follows_time_not_speaker_id() {
        // Тот же короткий кластер, но расположен ближе к третьему голосу:
        // выбор идёт по времени, а не по номеру спикера.
        let mut segs = vec![
            seg("Speaker_1", 0.0, 2.0),
            seg("Speaker_2", 3.0, 3.4),
            seg("Speaker_3", 3.6, 6.0),
        ];
        let mut chunks = vec![chunk("Speaker_1", 0.0, 2.0, "раз")];
        merge_unclonable_speakers(&mut segs, &mut chunks);

        let label_at = |start: f64| {
            segs.iter()
                .find(|s| (s.start_sec - start).abs() < 0.01)
                .map(|s| s.speaker_id.clone())
                .unwrap_or_default()
        };
        assert_eq!(label_at(3.0), label_at(3.6), "при сдвиге вправо — сосед справа");
        assert_ne!(label_at(3.0), label_at(0.0));
    }

    #[test]
    fn numbering_has_no_gaps_after_merge() {
        // Speaker_2 исчезает — нумерация обязана стать сплошной, иначе в
        // промпт перевода уедет «говорит Speaker_3» на первом же чанке.
        let mut segs = vec![
            seg("Speaker_1", 0.0, 4.0),
            seg("Speaker_2", 4.2, 4.5),
            seg("Speaker_3", 5.0, 8.0),
        ];
        let mut chunks = vec![chunk("Speaker_1", 0.0, 4.0, "раз")];
        merge_unclonable_speakers(&mut segs, &mut chunks);

        let mut nums: Vec<usize> = speakers(&segs)
            .iter()
            .map(|s| s.trim_start_matches("Speaker_").parse().unwrap())
            .collect();
        nums.sort();
        assert_eq!(nums, vec![1, 2], "пропусков в нумерации быть не должно");
    }

    #[test]
    fn subtitles_survive_merge_and_follow_voice() {
        let mut segs = vec![
            seg("Speaker_1", 0.0, 4.0),
            seg("Speaker_2", 4.2, 4.5),
            seg("Speaker_3", 12.0, 15.0),
        ];
        let mut chunks = vec![
            chunk("Speaker_1", 0.0, 4.0, "раз"),
            chunk("Speaker_2", 4.2, 4.5, "ага"),
            chunk("Speaker_3", 12.0, 15.0, "два"),
        ];
        merge_unclonable_speakers(&mut segs, &mut chunks);

        // Речь короткого кластера не теряется — она получает голос соседа.
        assert_eq!(chunks.len(), 3);
        let aga = chunks.iter().find(|c| c.text == "ага").unwrap();
        let raz = chunks.iter().find(|c| c.text == "раз").unwrap();
        assert_eq!(
            aga.speaker_id, raz.speaker_id,
            "субтитр и сегмент слитого спикера должны получить один голос"
        );
    }

    #[test]
    fn single_speaker_case_is_untouched() {
        let mut segs = vec![seg("Speaker_1", 0.0, 4.0), seg("Speaker_1", 5.0, 6.0)];
        let mut chunks = vec![chunk("Speaker_1", 0.0, 4.0, "раз")];
        merge_unclonable_speakers(&mut segs, &mut chunks);

        assert_eq!(segs.len(), 2);
        assert_eq!(speakers(&segs).len(), 1);
        assert_eq!(chunks[0].speaker_id.as_deref(), Some("Speaker_1"));
    }

    #[test]
    fn short_but_frequent_speaker_still_merges() {
        // Осознанное проектное решение: для клона важен САМЫЙ ДЛИННЫЙ
        // сегмент, а не суммарная длительность. Спикер из десяти вставок по
        // 0.6 с всё равно не даст движку пригодный референс.
        let mut segs = vec![seg("Speaker_1", 0.0, 4.0)];
        for i in 0..10 {
            segs.push(seg(
                "Speaker_2",
                5.0 + i as f64 * 2.0,
                5.6 + i as f64 * 2.0,
            ));
        }
        let mut chunks = vec![chunk("Speaker_1", 0.0, 4.0, "раз")];
        merge_unclonable_speakers(&mut segs, &mut chunks);

        assert_eq!(speakers(&segs).len(), 1);
        assert!(every_speaker_has_reference(&segs));
    }

    #[test]
    fn degenerate_all_short_is_left_alone() {
        // Ролик короче референса: сливать некуда и незачем. Функция обязана
        // оставить данные как есть — иначе создаст иллюзию пригодного голоса,
        // и падение уйдёт в TTS с ложной причиной.
        let mut segs = vec![seg("Speaker_1", 0.0, 0.5), seg("Speaker_2", 1.0, 1.7)];
        let mut chunks = vec![chunk("Speaker_1", 0.0, 0.5, "а")];
        merge_unclonable_speakers(&mut segs, &mut chunks);

        assert_eq!(speakers(&segs).len(), 2, "без референса данные не мутируются");
        assert!(segs.iter().all(|s| s.speaker_id == "Speaker_1" || s.speaker_id == "Speaker_2"));
    }
}
