//! Аудио-конвейер live-режима (шаг 2 PRP): сегменты без рекламы скармливаются
//! в потоковый ffmpeg-pipe (код пайпа — плагин tauri-plugin-ffmpeg, указание
//! юзера), выход — f32 16 kHz mono в кольцо, STT-поток тянет сэмплы в
//! realtime-WS crispasr и пересылает финалы фраз наружу событием `live-subs`.
//!
//! Протокол STT (проба на crispasr 0.8.43 реальным аудио bogur, 12.11.26):
//! корневой WS плагина (`SttWsClient`, бинарные f32) в 0.8.43 МЁРТВ — сервер
//! отвечает «GET / → 404 (no matching route)». Рабочий вход — vLLM Realtime
//! `ws://127.0.0.1:{ws_port+1}/v1/realtime` (JSON-кадры, аудио — base64 s16le
//! в `input_audio_buffer.append`; dedicated флага порта нет — движок
//! авто-стартует его на ws_port+1). События в порядке фразы:
//! `input_audio_buffer.speech_started` (пусто) → `speech_stopped` (пусто) →
//! `...transcription.delta` (промежуточный текст — игнорируем) →
//! `...transcription.completed` {transcript, audio_duration_ms, ...}.
//! Времени t0/t1 в событиях НЕТ — реконструируем: счётчик отправленных
//! сэмплов даёт аудио-время потока (приём ≈ аудио-время: бэклог движка ~0,
//! realtime_factor ≈ 30–100), t0 = abs_ms на приёме speech_started (чуть
//! позднее реального онсета — для оверлея/озвучки безопасно), t1 =
//! t0 + audio_duration_ms (audio_duration_ms квантован движком по ~секунде —
//! погрешность ±0.5 с, известная граница). На EOF блока дописываем 0.6 с
//! тишины: VAD закрывает открытую фразу → completed доезжает в drain-окно.
//!
//! Один ffmpeg-pipe на блок непрерывности (правило №2 специалиста): на
//! DISCONTINUITY / рекламе / дыре сегментов аудио-блок закрывается (stdin →
//! EOF → ffmpeg дописывает хвост), WS-сессия переоткрывается с новым блоком —
//! время фразы t0/t1 считается от начала аудио блока, и PDT фразы =
//! pdt0 блока + t (t0 == currentTime запрещено, правило №3 PRP: потребитель
//! фронта сверяет PDT-время фразы с hls.playingDate).
//!
//! Потоки на блок (самоочищаются, join из async не нужен):
//! - писец: сегменты из очереди → PipeStdin; разрыв очереди = закрытие stdin;
//! - читатель: PipeStdout → кольцо f32 (хвост + счётчик base); PipeGuard
//!   гасит ffmpeg, когда вывод дочитан;
//! - STT: ждёт готовности движка, шлёт сэмплы, читает финалы, эмитит событие.
//!
//! Каскад остановки блока: дроп Sender<SegMsg> → recv() Err у писца →
//! stdin закрыт → ffmpeg дописывает хвост → Ok(0) у читателя → set_eof →
//! STT дочитывает финалы FINAL_DRAIN_SECS → потоки сами завершаются.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine as _;
use tauri::Emitter;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

/// Частота аудио на выходе пайпа (гц).
pub const SAMPLE_RATE: u32 = 16000;
/// Сколько секунд аудио держать в кольце (антиутечка; STT вычитывает живьём).
const RING_CAP_SAMPLES: usize = 60 * SAMPLE_RATE as usize;
/// Порция отправки в WS (сэмплы): 0.5 с.
const SEND_CHUNK: usize = SAMPLE_RATE as usize / 2;
/// Буфер чтения stdout пайпа: 4096 сэмплов = 0.25 с.
const READ_BUF_BYTES: usize = 4096 * 4;
/// Сколько секунд ждать готовности движка в STT-потоке блока.
const ENGINE_WAIT_SECS: u64 = 120;
/// Сколько секунд после EOF пайпа дочитывать финалы (VAD-закрытие + redecode).
const FINAL_DRAIN_SECS: u64 = 3;
/// Бэкпрешер файлового источника: файл идёт в 30–100× реалтайма, и пока STT
/// не вычитал хвост кольца, читатель пайпа ждёт — иначе push сбрасывал бы
/// старейшие сэмплы (потеря начала фраз) вместо естественной паузы.
const BACKPRESSURE_LIMIT: usize = RING_CAP_SAMPLES - 5 * SAMPLE_RATE as usize;

/// Публичный счётчик распознанных фраз (для env-селфтеста live-режима).
pub static SUBS_N: AtomicU64 = AtomicU64::new(0);

/// Команды конвейера (FIFO — порядок фида = порядок сегментов).
enum Cmd {
    /// Явное открытие блока с t0-якорем (PDT первого МЕДИА-сегмента; init
    /// PDT не имеет и в фиде идёт следом — иначе якорь терялся).
    OpenBlock { pdt_secs: Option<f64> },
    Feed { bytes: Arc<Vec<u8>>, pdt_secs: Option<f64> },
    /// Файловый источник (шаг 5 PRP): ffmpeg сам читает файл с диска, один
    /// блок на весь файл, якорь t0 = 0 (шкала — секунды файла).
    OpenFile { path: PathBuf },
    /// YouTube-VOD (12.11.26): как файловый, но вход — удалённый URL с
    /// заголовками от yt-dlp.
    OpenUrl { url: String, headers: Vec<(String, String)> },
    Restart,
}

/// Сообщения писцу блока: сегмент на запись в ffmpeg.
enum SegMsg {
    Seg(Arc<Vec<u8>>),
}

/// Аудио-конвейер сессии: отправитель команд; таск живёт до drop Sender'а.
pub struct AudioPipeline {
    cmd_tx: tokio::sync::mpsc::Sender<Cmd>,
}

impl AudioPipeline {
    /// Запуск таска конвейера. `engine_port` — порт WS crispasr (0 = грузится);
    /// `mt_tx` — очередь переводчика (шаг 3): финалы фраз уходят в MT.
    pub fn start(
        app: tauri::AppHandle,
        engine_port: Arc<AtomicU64>,
        mt_tx: Option<std::sync::mpsc::Sender<super::mt::PhraseJob>>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::channel::<Cmd>(64);
        tauri::async_runtime::spawn(conveyor(app, engine_port, cmd_rx, mt_tx));
        Self { cmd_tx }
    }

    /// Явно открыть блок с t0-якорем `pdt_secs` (эпоха, с) — вызывается ПЕРЕД
    /// первым фидом блока (медиа-сегмент несёт PDT, init — нет).
    pub async fn open_block(&self, pdt_secs: Option<f64>) {
        let _ = self.cmd_tx.send(Cmd::OpenBlock { pdt_secs }).await;
    }

    /// Файловый источник: один непрерывный блок на весь файл (t0 = 0,
    /// шкала — секунды файла).
    pub async fn open_file(&self, path: PathBuf) {
        let _ = self.cmd_tx.send(Cmd::OpenFile { path }).await;
    }

    /// Удалённый медиа-блок (YouTube-VOD): ffmpeg читает URL с заголовками
    /// yt-dlp; остальная механика идентична файловому блоку.
    pub async fn open_url(&self, url: String, headers: Vec<(String, String)>) {
        let _ = self.cmd_tx.send(Cmd::OpenUrl { url, headers }).await;
    }

    /// Сегмент без рекламы скармливается в текущий блок (если блок ещё не
    /// открыт — откроется с pdt этого фида как фолбэк).
    /// `pdt_secs` — PDT сегмента (эпоха, с); нужен блоку как t0-якорь фраз.
    pub async fn feed(&self, bytes: Arc<Vec<u8>>, pdt_secs: Option<f64>) {
        let _ = self
            .cmd_tx
            .send(Cmd::Feed { bytes, pdt_secs })
            .await;
    }

    /// Закрыть текущий блок (разрыв/реклама/дыра): следующий Feed откроет новый.
    pub async fn restart(&self) {
        let _ = self.cmd_tx.send(Cmd::Restart).await;
    }
}

/// Цикл конвейера: владеет состоянием блока, раздаёт пайп потокам блока.
async fn conveyor(
    app: tauri::AppHandle,
    engine_port: Arc<AtomicU64>,
    mut cmd_rx: tokio::sync::mpsc::Receiver<Cmd>,
    mt_tx: Option<std::sync::mpsc::Sender<super::mt::PhraseJob>>,
) {
    // Блок = писец сегментов. Drop Sender'а → писец закрывает stdin → ffmpeg
    // дописывает хвост → читатель видит EOF → STT дочитывает и выходит.
    let mut block: Option<std::sync::mpsc::Sender<SegMsg>> = None;

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            Cmd::Restart => block = None,
            Cmd::OpenBlock { pdt_secs } => {
                if block.is_none() {
                    if let Err(e) = open_block(&app, &engine_port, pdt_secs, mt_tx.clone()) {
                        log::error!("Live-audio: блок не открыт: {e}");
                    }
                }
            }
            Cmd::OpenFile { path } => {
                // Файловый блок сам тянет аудио из файла; фидов не будет.
                if let Err(e) = open_file_block(&app, &engine_port, path, mt_tx.clone()) {
                    log::error!("Live-audio: файловый блок не открыт: {e}");
                }
            }
            Cmd::OpenUrl { url, headers } => {
                // URL-блок (YouTube-VOD): тот же pipeline, вход — сеть.
                if let Err(e) = open_url_block(&app, &engine_port, url, headers, mt_tx.clone()) {
                    log::error!("Live-audio: URL-блок не открыт: {e}");
                }
            }
            Cmd::Feed { bytes, pdt_secs } => {
                if block.is_none() {
                    match open_block(&app, &engine_port, pdt_secs, mt_tx.clone()) {
                        Ok(seg_tx) => block = Some(seg_tx),
                        Err(e) => {
                            log::error!("Live-audio: блок не открыт: {e}");
                            continue;
                        }
                    }
                }
                if let Some(b) = &block {
                    if b.send(SegMsg::Seg(bytes)).is_err() {
                        log::warn!("Live-audio: писец блока умер — сегмент потерян");
                        block = None;
                    }
                }
            }
        }
    }
    // Канал закрыт (сессия остановлена): блок закрывается дропом.
    drop(block);
}

/// Открыть блок непрерывности: ffmpeg-pipe + 3 потока (писец/читатель/STT).
/// Вход пайпа — конкатенация init-сегмента (CMAF) и медиа-сегментов; формат
/// пусть определяет проба ffmpeg по содержимому (fMP4 или TS-режим).
fn open_block(
    app: &tauri::AppHandle,
    engine_port: &Arc<AtomicU64>,
    pdt_secs: Option<f64>,
    mt_tx: Option<std::sync::mpsc::Sender<super::mt::PhraseJob>>,
) -> Result<std::sync::mpsc::Sender<SegMsg>, String> {
    let ffmpeg = crate::ffmpeg::resolve(&crate::config::load().ffmpeg_path);
    let args: Vec<String> = vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-i".into(),
        "pipe:0".into(),
        // Только звук под сабы/озвучку: mono 16 кГц, сырой f32.
        "-vn".into(),
        "-ac".into(),
        "1".into(),
        "-ar".into(),
        SAMPLE_RATE.to_string(),
        "-f".into(),
        "f32le".into(),
        "pipe:1".into(),
    ];
    log::info!("Live-audio: открываю аудио-блок (pdt0={pdt_secs:?})");
    let pipe = tauri_plugin_ffmpeg::engine::pipe::FfmpegPipe::spawn(Path::new(&ffmpeg), &args)
        .map_err(|e| format!("ffmpeg-pipe не стартовал: {e}"))?;
    let (mut pipe_in, mut pipe_out, guard) = pipe.split();

    let ring = Arc::new(Ring::default());
    let (seg_tx, seg_rx) = std::sync::mpsc::channel::<SegMsg>();

    // Писец: сегменты из очереди в stdin. Разрыв очереди = конец блока.
    let _ = std::thread::Builder::new().name("live-writer".into()).spawn(move || {
        while let Ok(msg) = seg_rx.recv() {
            let SegMsg::Seg(bytes) = msg;
            if let Err(e) = pipe_in.write(&bytes) {
                // ffmpeg умер (битые данные / убит сессией) — писать больше некуда.
                log::warn!("Live-audio: запись в ffmpeg прервана ({e}) — блок закрыт");
                break;
            }
        }
        pipe_in.close();
    });

    // Читатель: stdout → кольцо f32; EOF или смерть процесса = конец вывода.
    // Незавершённое слово f32 переносится на следующее чтение (read может
    // вернуться на середине float — выравнивание обязано сохраняться).
    let ring_reader = Arc::clone(&ring);
    let _ = std::thread::Builder::new().name("live-reader".into()).spawn(move || {
        let _guard = guard; // Drop гасит ffmpeg, когда вывод дочитан
        let mut buf = vec![0u8; READ_BUF_BYTES];
        let mut carry: Vec<u8> = Vec::new();
        loop {
            match pipe_out.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    carry.extend_from_slice(&buf[..n]);
                    let whole = carry.len() - carry.len() % 4;
                    if whole > 0 {
                        ring_reader.push(&le_bytes_to_f32(&carry[..whole]));
                        carry.drain(..whole);
                    }
                }
            }
        }
        ring_reader.set_eof();
    });

    // STT: сэмплы кольца → WS crispasr, финалы → событие live-subs + очередь MT.
    spawn_stt_thread(app.clone(), ring, engine_port.clone(), pdt_secs, mt_tx);

    Ok(seg_tx)
}

/// Открыть файловый блок: ffmpeg читает сам файл (`-i <path>`, не pipe:0),
/// читатель вычитывает stdout в кольцо с бэкпрешером. Якорь t0 = 0 — шкала
/// Общая механика файл/URL-блоков: ffmpeg читает `input` (файл или URL),
/// читатель с бэкпрешером → кольцо, STT с якорем 0 (шкала фраз — секунды).
fn open_media_block(
    app: &tauri::AppHandle,
    engine_port: &Arc<AtomicU64>,
    input: String,
    http_args: Vec<String>,
    label: &str,
    mt_tx: Option<std::sync::mpsc::Sender<super::mt::PhraseJob>>,
) -> Result<(), String> {
    let ffmpeg = crate::ffmpeg::resolve(&crate::config::load().ffmpeg_path);
    let mut args: Vec<String> = vec!["-hide_banner".into(), "-loglevel".into(), "error".into()];
    // http-опции ffmpeg должны идти ПЕРЕД -i (это входные опции).
    args.extend(http_args);
    args.extend([
        "-i".into(),
        input,
        // Только звук под сабы/озвучку: mono 16 кГц, сырой f32.
        "-vn".into(),
        "-ac".into(),
        "1".into(),
        "-ar".into(),
        SAMPLE_RATE.to_string(),
        "-f".into(),
        "f32le".into(),
        "pipe:1".into(),
    ]);
    log::info!("Live-audio: блок {} (якорь 0)", label);
    let pipe = tauri_plugin_ffmpeg::engine::pipe::FfmpegPipe::spawn(Path::new(&ffmpeg), &args)
        .map_err(|e| format!("ffmpeg-pipe не стартовал: {e}"))?;
    let (pipe_in, mut pipe_out, guard) = pipe.split();
    // ffmpeg с `-i <файл|url>` stdin не читает — конец пайпа закрываем сразу.
    drop(pipe_in);

    let ring = Arc::new(Ring::default());

    // Читатель: stdout → кольцо с бэкпрешером (источник идёт в 30–100×
    // реалтайма; пока STT не вычитал хвост, ждём — иначе push терял бы
    // старейшие сэмплы).
    let _ = std::thread::Builder::new().name("live-reader".into()).spawn({
        // Кольцо уходит и в STT-поток — читателю отдаём клон Arc.
        let ring = ring.clone();
        move || {
            let mut guard = guard; // Drop гасит ffmpeg, когда вывод дочитан
            let mut buf = vec![0u8; READ_BUF_BYTES];
            let mut carry: Vec<u8> = Vec::new();
            let mut total_read = 0u64;
            loop {
                while ring.pending() > BACKPRESSURE_LIMIT {
                    std::thread::sleep(Duration::from_millis(100));
                }
                match pipe_out.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        total_read += n as u64;
                        carry.extend_from_slice(&buf[..n]);
                        let whole = carry.len() - carry.len() % 4;
                        if whole > 0 {
                            ring.push(&le_bytes_to_f32(&carry[..whole]));
                            carry.drain(..whole);
                        }
                    }
                }
            }
            // Ранний EOF (меньше секунды аудио): ffmpeg умер — 403 googlevideo
            // при бот-проверке, битый файл, нет кодека. Раньше было МОЛЧА:
            // -loglevel error пишет только в stderr пайпа, и сабы просто
            // обрывались без единой строки в логе.
            if total_read < SAMPLE_RATE as u64 * 4 {
                let exit = guard.try_wait().and_then(|s| s.code());
                let tail = guard.stderr_tail();
                let last = tail.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("");
                log::error!(
                    "Live-audio: ffmpeg-подача умерла сразу (exit={exit:?}, {total_read} байт): {last}"
                );
            }
            ring.set_eof();
        }
    });

    // STT тот же, что у сегментного блока (якорь 0 → t0/t1 = секунды входа).
    spawn_stt_thread(app.clone(), ring, engine_port.clone(), Some(0.0), mt_tx);
    Ok(())
}

/// Файловый блок (шаг 5 PRP): ffmpeg сам читает файл с диска.
fn open_file_block(
    app: &tauri::AppHandle,
    engine_port: &Arc<AtomicU64>,
    path: PathBuf,
    mt_tx: Option<std::sync::mpsc::Sender<super::mt::PhraseJob>>,
) -> Result<(), String> {
    open_media_block(
        app,
        engine_port,
        path.to_string_lossy().to_string(),
        Vec::new(),
        &path.display().to_string(),
        mt_tx,
    )
}

/// URL-блок (YouTube-VOD, 12.11.26): вход — сеть. http-опции ffmpeg: UA
/// отдельным флагом, прочие заголовки через `-headers` (ожидает строки,
/// завершённые CRLF).
fn open_url_block(
    app: &tauri::AppHandle,
    engine_port: &Arc<AtomicU64>,
    url: String,
    headers: Vec<(String, String)>,
    mt_tx: Option<std::sync::mpsc::Sender<super::mt::PhraseJob>>,
) -> Result<(), String> {
    let mut ua: Option<String> = None;
    let mut extra: Vec<String> = Vec::new();
    for (k, v) in &headers {
        if k.eq_ignore_ascii_case("User-Agent") {
            ua = Some(v.clone());
        } else {
            extra.push(format!("{k}: {v}
"));
        }
    }
    let mut http_args: Vec<String> = Vec::new();
    if let Some(ua) = ua {
        http_args.extend(["-user_agent".into(), ua]);
    }
    if !extra.is_empty() {
        http_args.extend(["-headers".into(), extra.concat()]);
    }
    open_media_block(app, engine_port, url, http_args, "URL (удалённый медиа)", mt_tx)
}

/// STT-поток блока: ожидание движка, отправка сэмплов, пересылка финалов
/// (событие live-subs + мост в очередь переводчика, шаг 3).
fn spawn_stt_thread(
    app: tauri::AppHandle,
    ring: Arc<Ring>,
    engine_port: Arc<AtomicU64>,
    pdt_secs: Option<f64>,
    mt_tx: Option<std::sync::mpsc::Sender<super::mt::PhraseJob>>,
) {
    let _ = std::thread::Builder::new().name("live-stt".into()).spawn(move || {
        // Ждём готовности движка (модель грузится десятки секунд; ready
        // выставляется только после проверки realtime-порта — engine.rs).
        if !wait_for(ENGINE_WAIT_SECS, || engine_port.load(Ordering::SeqCst) != 0) {
            log::warn!("Live-audio: движок STT не поднялся за {ENGINE_WAIT_SECS} с — сабы блока пропущены");
            return;
        }
        let port = engine_port.load(Ordering::SeqCst) as u16;
        // Realtime-сервер движка авто-стартует на ws_port+1 (проба 0.8.43);
        // корневой WS плагина мёртв — см. док модуля.
        let rt_port = port + 1;

        let mut ws = {
            let mut connected = None;
            let mut last_err = String::new();
            for attempt in 0..10 {
                match RtClient::connect(rt_port) {
                    Ok(c) => {
                        connected = Some(c);
                        break;
                    }
                    Err(e) => {
                        last_err = e;
                        log::warn!("Live-audio: realtime-WS connect (попытка {attempt}): {last_err}");
                    }
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            match connected {
                Some(c) => c,
                None => {
                    log::error!("Live-audio: realtime-WS crispasr не открылся за 5 с ({last_err}) — сабы блока пропущены");
                    return;
                }
            }
        };
        ws.set_read_timeout(Duration::from_millis(25));

        let block_pdt0 = pdt_secs.unwrap_or(0.0);
        log::info!("Live-audio: STT-поток блока подключён (realtime :{rt_port})");
        let mut sent_abs: u64 = 0;
        let mut drain_deadline: Option<Instant> = None;
        let mut pending_t0_ms: Option<f64> = None;
        let mut eof_silence_sent = false;

        loop {
            let new = ring.take_new(&mut sent_abs, SEND_CHUNK * 4);
            if !new.is_empty() {
                let s16 = f32_to_s16_le_bytes(&new);
                for chunk in s16.chunks(SEND_CHUNK * 2) {
                    if let Err(e) = ws.append(chunk) {
                        log::warn!("Live-audio: отправка в realtime-WS оборвалась ({e}) — сабы блока остановлены");
                        return;
                    }
                }
            }
            // Неблокирующе вычитываем события; Err = соединение умерло.
            let events = match ws.poll_events() {
                Ok(evs) => evs,
                Err(e) => {
                    log::warn!("Live-audio: realtime-WS разорван ({e}) — сабы блока остановлены");
                    return;
                }
            };
            for ev in &events {
                match ev {
                    RtEvent::SpeechStarted => {
                        // Приёмное время ≈ аудио-время потока (бэклог движка ~0);
                        // t0 чуть позднее реального онсета — для оверлея/озвучки
                        // безопасно (поздний старт ждёт, ранний бы резал соседа).
                        // Лог: после пауз (реклама/музыка) движок может НЕ прислать
                        // started — тогда финал уходит в фолбэк t0 и таймкоды
                        // поплывут; последовательность событий видна здесь.
                        log::info!(
                            "Live-subs: speech_started @ {:.2}s",
                            sent_abs as f64 / SAMPLE_RATE as f64
                        );
                        pending_t0_ms = Some(sent_abs as f64 * 1000.0 / SAMPLE_RATE as f64);
                    }
                    RtEvent::SpeechStopped => {} // t1 считается от t0 + длительность
                    RtEvent::Completed { transcript, audio_duration_ms } => {
                        let t0_ms = pending_t0_ms.take().unwrap_or_else(|| {
                            // Фолбэк: финал без начатия — считаем от хвоста.
                            (sent_abs as f64 * 1000.0 / SAMPLE_RATE as f64) - audio_duration_ms
                        });
                        let t1_ms = t0_ms + audio_duration_ms;
                        if !transcript.trim().is_empty() {
                            // Мега-фраза движка (быстрая речь без пауз) дробится
                            // по предложениям — каждый кусок = свой суб, своя
                            // MT-джоба и свой юнит дубляжа.
                            for (s0, s1, txt) in split_phrase(t0_ms, t1_ms, &transcript) {
                                SUBS_N.fetch_add(1, Ordering::SeqCst);
                                log::info!(
                                    "Live-subs: фраза [{:.2}..{:.2}] {}",
                                    s0 / 1000.0,
                                    s1 / 1000.0,
                                    txt
                                );
                                if let Err(e) = app.emit(
                                    "live-subs",
                                    serde_json::json!({
                                        "t0": block_pdt0 + s0 / 1000.0,
                                        "t1": block_pdt0 + s1 / 1000.0,
                                        "en": txt,
                                    }),
                                ) {
                                    log::warn!("Live-subs: не отправить событие: {e}");
                                }
                                // Мост в переводчик (шаг 3). Полный буфер блокирует
                                // STT-поток — сознательный бэкпрешер файлового
                                // режима (MT ~0.3-1 с на фразу, это > темпа речи).
                                if let Some(tx) = &mt_tx {
                                    let job = super::mt::PhraseJob {
                                        t0: block_pdt0 + s0 / 1000.0,
                                        t1: block_pdt0 + s1 / 1000.0,
                                        en: txt,
                                    };
                                    let _ = tx.send(job);
                                }
                            }
                        }
                    }
                    RtEvent::Other => {}
                }
            }
            // Конец блока: после EOF дописываем 0.6 с тишины — VAD закрывает
            // открытую фразу (финал после 500 мс тишины), completed доезжает в drain.
            if ring.is_eof() && !eof_silence_sent {
                eof_silence_sent = true;
                let silence = vec![0u8; SAMPLE_RATE as usize / 1000 * 600 * 2];
                let _ = ws.append(&silence);
            }
            // После EOF дочитываем финалы FINAL_DRAIN_SECS.
            if ring.is_eof() {
                let dl = drain_deadline.get_or_insert_with(|| {
                    Instant::now() + Duration::from_secs(FINAL_DRAIN_SECS)
                });
                if Instant::now() >= *dl {
                    break;
                }
            }
            if new.is_empty() && events.is_empty() {
                std::thread::sleep(Duration::from_millis(25));
            }
        }
        log::info!("Live-audio: STT-поток блока завершён");
    });
}

/// Ожидание условия с таймаутом (проверка каждые 100 мс).
fn wait_for(timeout_secs: u64, cond: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        if cond() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Событие realtime-протокола, значимое для сабов.
#[derive(Debug, PartialEq)]
pub enum RtEvent {
    SpeechStarted,
    SpeechStopped,
    Completed {
        transcript: String,
        audio_duration_ms: f64,
    },
    /// session.created, delta (промежуточный текст), ping и пр.
    Other,
}

/// Разбор события realtime-протокола из JSON-кадра (чистая функция — тестируется).
pub fn parse_rt_event(v: &serde_json::Value) -> RtEvent {
    let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match typ {
        "input_audio_buffer.speech_started" => RtEvent::SpeechStarted,
        "input_audio_buffer.speech_stopped" => RtEvent::SpeechStopped,
        "conversation.item.input_audio_transcription.completed" => RtEvent::Completed {
            transcript: v
                .get("transcript")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string(),
            audio_duration_ms: v
                .get("audio_duration_ms")
                .and_then(|d| d.as_f64())
                .unwrap_or(0.0),
        },
        _ => RtEvent::Other,
    }
}

/// Жёсткий кап фразы для сабов/дубляжа. Движок склеивает быструю речь без
/// пауз (замер q1v4IuOoWiI: merge-gap 350 давал фразу 63 с / 1104 символа —
/// суб-«простыня» на весь экран и один гигантский юнит дубляжа; см. док
/// engine.rs). Всё длиннее капа дробим по предложениям (пунктуация в финалах
/// есть — FireRedPunc), таймкоды пропорциональны весу символов.
const MAX_PHRASE_SEC: f64 = 12.0;
const MAX_PHRASE_CHARS: usize = 220;
/// Хвост меньше этого доезжает до предыдущего куска (не плодим огрызки «Uh»).
const MIN_TAIL_CHARS: usize = 40;

/// Границы предложений: [.!?], затем необязательные закрывающие кавычки/скобки
/// и пробел/конец строки. Десятичные «10.5» не рвём: после точки цифра — не
/// граница. Возвращает спаны (байтовые индексы) по обрезанному тексту.
fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let b = text.as_bytes();
    let mut spans = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < b.len() {
        if matches!(b[i], b'.' | b'!' | b'?') {
            let mut j = i + 1;
            while j < b.len() && matches!(b[j], b'"' | b'\'' | b')' | b']') {
                j += 1;
            }
            if j >= b.len() || matches!(b[j], b' ' | b'\n') {
                spans.push((start, j));
                start = j;
                i = j;
            }
        }
        i += 1;
    }
    if start < b.len() {
        spans.push((start, b.len()));
    }
    spans
}

/// Слишком длинное предложение (без внутренней пунктуации) — рвём по словам.
fn word_split_span(text: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut piece = start;
    for i in start..end {
        if b[i] == b' ' && i - piece >= MAX_PHRASE_CHARS {
            out.push((piece, i));
            piece = i + 1;
        }
    }
    if piece < end {
        out.push((piece, end));
    }
    out
}

/// Нарезка финальной фразы движка на куски не длиннее капа. Короткая фраза
/// возвращается как есть (один элемент). Чистая функция — тестируется.
pub fn split_phrase(t0_ms: f64, t1_ms: f64, text: &str) -> Vec<(f64, f64, String)> {
    let trimmed = text.trim();
    let dur = (t1_ms - t0_ms).max(0.0);
    let chars = trimmed.chars().count();
    if trimmed.is_empty() || (dur <= MAX_PHRASE_SEC * 1000.0 && chars <= MAX_PHRASE_CHARS) {
        return vec![(t0_ms, t1_ms, trimmed.to_string())];
    }

    // Предложения; слишком длинные дополнительно рвём по словам.
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    for (s, e) in sentence_spans(trimmed) {
        if e - s > MAX_PHRASE_CHARS {
            pieces.extend(word_split_span(trimmed, s, e));
        } else {
            pieces.push((s, e));
        }
    }

    // Упаковка: накапливаем предложения, пока не выйдем за кап по символам
    // или по пропорциональной длительности.
    let total: usize = trimmed.chars().count().max(1);
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut cur: Option<(usize, usize)> = None;
    let mut cur_chars = 0usize;
    for (s, e) in pieces {
        let n = trimmed[s..e].chars().count();
        if let Some((cs, _)) = cur {
            let est_dur = dur * (cur_chars + n) as f64 / total as f64;
            if cur_chars + n > MAX_PHRASE_CHARS || est_dur > MAX_PHRASE_SEC * 1000.0 {
                chunks.push(cur.take().unwrap());
                cur = Some((s, e));
                cur_chars = n;
                continue;
            }
            cur = Some((cs, e));
            cur_chars += n;
        } else {
            cur = Some((s, e));
            cur_chars = n;
        }
    }
    if let Some(c) = cur {
        // Маленький хвост присоединяем к предыдущему (лёгкое превышение капа
        // лучше огрызка «Uh» отдельной строкой).
        if let Some(last) = chunks.last_mut() {
            if trimmed[c.0..c.1].chars().count() < MIN_TAIL_CHARS {
                last.1 = c.1;
            } else {
                chunks.push(c);
            }
        } else {
            chunks.push(c);
        }
    }

    // Таймкоды: пропорционально весу символов каждого куска.
    let mut out = Vec::with_capacity(chunks.len());
    let mut consumed = 0usize;
    for (s, e) in &chunks {
        let n = trimmed[*s..*e].chars().count();
        let a = t0_ms + dur * consumed as f64 / total as f64;
        consumed += n;
        let b = t0_ms + dur * consumed as f64 / total as f64;
        out.push((a, b, trimmed[*s..*e].trim().to_string()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_phrase_short_passes_through() {
        let out = split_phrase(1000.0, 5000.0, "Hello there. General Kenobi!");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], (1000.0, 5000.0, "Hello there. General Kenobi!".to_string()));
    }

    #[test]
    fn split_phrase_empty_text() {
        assert_eq!(split_phrase(0.0, 1000.0, "   ").len(), 1);
    }

    #[test]
    fn split_phrase_long_splits_at_sentences() {
        let text = "First sentence here. Second one follows it. Third sentence is a bit longer than the others, yes. Fourth wraps it up nicely!";
        // 127 символов — по символам влезает, но форсируем по времени:
        let out = split_phrase(0.0, 30_000.0, text);
        assert!(out.len() >= 2, "должна разбиться по капу 12 с, got {out:?}");
        // Таймкоды стыкуются и покрывают весь исходный диапазон.
        assert!((out[0].0 - 0.0).abs() < 1e-6);
        assert!((out.last().unwrap().1 - 30_000.0).abs() < 1e-6);
        for w in out.windows(2) {
            assert!((w[0].1 - w[1].0).abs() < 1e-6, "дыра в таймкодах: {w:?}");
        }
        // Текст без потерь (по конкатенации с учётом trim/пробелов).
        let joined: String = out.iter().map(|c| c.2.clone()).collect::<Vec<_>>().join(" ");
        assert_eq!(joined.split_whitespace().count(), text.split_whitespace().count());
    }

    #[test]
    fn split_phrase_keeps_decimals() {
        let text = "The rating is 9.5 out of ten and the season is season ten point five overall really.";
        let out = split_phrase(0.0, 40_000.0, text);
        for c in &out {
            assert!(!c.2.ends_with('.') || c.2.contains("really"), "порвано 9.5: {}", c.2);
        }
        let joined = out.iter().map(|c| c.2.clone()).collect::<Vec<_>>().join(" ");
        assert!(joined.contains("9.5"), "десятичное число разорвано: {joined}");
    }

    #[test]
    fn split_phrase_oversized_sentence_word_split() {
        // Одно предложение без точек, 300+ символов — рвётся по словам.
        let text = "a ".repeat(180).trim_end().to_string();
        let out = split_phrase(0.0, 20_000.0, &text);
        assert!(out.len() >= 2);
        for c in &out {
            assert!(c.2.chars().count() <= MAX_PHRASE_CHARS + 8, "кусок длиннее капа: {}", c.2.chars().count());
        }
    }

    #[test]
    fn split_phrase_tiny_tail_merged() {
        let text = "This is a reasonably long first sentence that keeps going. Hi.";
        let out = split_phrase(0.0, 30_000.0, text);
        // Хвост «Hi.» (3 симв.) не должен стать отдельной строкой.
        assert!(out.len() >= 1 && out.last().unwrap().2.ends_with("Hi."), "хвост оторван: {out:?}");
    }
}

/// WS-клиент vLLM Realtime протокола crispasr (см. док модуля: корневой WS
/// плагина в 0.8.43 мёртв, рабочий вход — /v1/realtime на ws_port+1).
struct RtClient {
    ws: WebSocket<MaybeTlsStream<std::net::TcpStream>>,
}

impl RtClient {
    /// Подключение к /v1/realtime (subprotocol серверу не нужен — проба).
    fn connect(port: u16) -> Result<Self, String> {
        let url = format!("ws://127.0.0.1:{port}/v1/realtime");
        let (ws, _resp) =
            tungstenite::connect(&url).map_err(|e| format!("{url}: {e}"))?;
        Ok(Self { ws })
    }

    /// Таймаут чтения для неблокирующего опроса: Err(WouldBlock/TimedOut) в
    /// tungstenite не фатален — кадр дорезюмируется на следующем read.
    fn set_read_timeout(&self, d: Duration) {
        if let MaybeTlsStream::Plain(s) = self.ws.get_ref() {
            let _ = s.set_read_timeout(Some(d));
        }
    }

    /// Порция s16le-PCM → input_audio_buffer.append (base64, JSON-кадр).
    fn append(&mut self, s16_bytes: &[u8]) -> Result<(), String> {
        let audio = base64::engine::general_purpose::STANDARD.encode(s16_bytes);
        let msg = serde_json::json!({ "type": "input_audio_buffer.append", "audio": audio }).to_string();
        self.ws
            .send(Message::Text(msg))
            .map_err(|e| format!("append: {e}"))
    }

    /// Неблокирующе вычитать накопившиеся события. Ok(пусто) = «пока тишина»;
    /// Err = соединение умерло (сабы блока останавливаются).
    fn poll_events(&mut self) -> Result<Vec<RtEvent>, String> {
        let mut out = Vec::new();
        loop {
            match self.ws.read() {
                Ok(Message::Text(t)) => match serde_json::from_str::<serde_json::Value>(&t) {
                    Ok(v) => out.push(parse_rt_event(&v)),
                    Err(e) => log::warn!("Live-audio: кривой JSON события realtime: {e}"),
                },
                Ok(_) => {} // Binary/Ping/Pong — протокол их не шлёт
                Err(tungstenite::Error::Io(e))
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    break;
                }
                Err(e) => return Err(format!("WS: {e}")),
            }
        }
        Ok(out)
    }
}

/// Кольцо f32-сэмплов одного блока: читатель пушит, STT вычитывает.
/// Абсолютные индексы переживают подрезку спереди (счётчик `base`): если STT
/// отстал больше ёмкости кольца (60 с), недочитанный хвост сбрасывается —
/// поток продолжается с актуального места, утечки нет.
#[derive(Default)]
pub struct Ring {
    inner: Mutex<RingIn>,
}

#[derive(Default)]
struct RingIn {
    /// Сэмплы, начиная с абсолютного индекса `base`.
    buf: Vec<f32>,
    /// Абсолютный индекс первого сэмпла в `buf`.
    base: u64,
    eof: bool,
}

impl Ring {
    /// Дописать сэмплы в хвост; при переполнении подрезать спереди.
    pub fn push(&self, samples: &[f32]) {
        let mut g = lock(&self.inner);
        g.buf.extend_from_slice(samples);
        if g.buf.len() > RING_CAP_SAMPLES {
            let drop = g.buf.len() - RING_CAP_SAMPLES;
            g.buf.drain(..drop);
            g.base += drop as u64;
        }
    }

    /// Пометить конец вывода пайпа (читатель дошёл до EOF).
    pub fn set_eof(&self) {
        lock(&self.inner).eof = true;
    }

    pub fn is_eof(&self) -> bool {
        lock(&self.inner).eof
    }

    /// Сколько несчитанных сэмплов в кольце (бэкпрешер файлового источника:
    /// файл идёт в 30–100× реалтайма — без паузы читатель терял бы хвост).
    pub fn pending(&self) -> usize {
        lock(&self.inner).buf.len()
    }

    /// Взять новые сэмплы, начиная с абсолютного индекса `*consumed`
    /// (не больше `max`); сдвинуть `*consumed` вперёд.
    pub fn take_new(&self, consumed: &mut u64, max: usize) -> Vec<f32> {
        let mut g = lock(&self.inner);
        let start = (*consumed).max(g.base);
        let end = g.base + g.buf.len() as u64;
        if start >= end {
            return Vec::new();
        }
        let from = (start - g.base) as usize;
        let to = g.buf.len().min(from + max);
        let out = g.buf[from..to].to_vec();
        *consumed = g.base + to as u64;
        out
    }
}

/// Лок с подстраховкой от паники в другом потоке (потоки живут дольше).
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Сырые little-endian f32 → вектор сэмплов. Хвост < 4 байт отбрасывается —
/// выравнивание обязан поддерживать вызывающий (carry в читателе).
pub fn le_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// f32 [-1..1] → байты little-endian s16le (PCM16 для realtime-протокола).
/// Зажим в диапазон: сэмплы ffmpeg f32le почти всегда внутри [-1..1],
/// но не гарантированы.
pub fn f32_to_s16_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// PDT-строка сегмента (RFC 3339) → секунды эпохи. None = не распарсилось
/// (блок получит t0-якорь 0.0 — фразы не совпадут с playingDate, без падения).
pub fn pdt_to_epoch_secs(pdt: &str) -> Option<f64> {
    chrono::DateTime::parse_from_rfc3339(pdt)
        .ok()
        .map(|dt| dt.timestamp_millis() as f64 / 1000.0)
}
