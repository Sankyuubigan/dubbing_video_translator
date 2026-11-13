import { useState, useEffect, useRef, useCallback } from "react";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import Hls from "hls.js";

/**
 * Live-вкладка (PRP «11.11.26 Live-режим Twitch.md» v2, шаги 1-5).
 *
 * Буферный time-shift: бэкенд качает поток в момент выхода сегментов, плеер
 * показывает через D секунд — за это время пайплайн готовит сабы и озвучку.
 * Шкала времени: эпоха (PDT) для twitch/url, секунды файла для file — фронт
 * берёт hls.playingDate, если есть, иначе video.currentTime.
 *
 * Очередь дубляжа — на ФРОНТЕ (правило №7 PRP): playhead дошёл до t0−LEAD →
 * live_synth; готовый WAV (live-dub) играет одиночный <audio> с даком видео
 * 0.15↔0.8; просрочка старта > SKIP_AFTER → скип со счётчиком.
 */

interface LiveStatus {
  running: boolean;
  state: string;
  port: number;
  channel: string;
  variant: string;
  delay_s: number;
  segments_downloaded: number;
  segments_failed: number;
  served_window_secs: number;
  target_duration: number;
  live_edge_pdt: string;
  stt_state: string;
  source_kind: string;
  mt_state: string;
  tts_state: string;
  dub_enabled: boolean;
  // YouTube-VOD (12.11.26): тип плеера и наличие отдельного аудио-прокси.
  player_kind?: string;
  remote_audio?: boolean;
  last_error: string;
}

interface LivePhrase {
  t0: number;
  t1: number;
  en: string;
}

type DubState = "wait" | "requested" | "ready" | "played" | "failed" | "skipped";

interface DubItem {
  id: number;
  t0: number;
  t1: number;
  ru: string;
  state: DubState;
  url?: string;
  dur?: number;
}

interface LiveCfg {
  source: "twitch" | "url" | "file";
  channel: string;
  url: string;
  filePath: string;
  voice: string;
  delay: number;
  quality: string;
  proxy: string;
  oauth: string;
  dub: boolean;
}

const CFG_KEY = "deedub_live_cfg";
const EN_SUBS_KEY = "deedub_live_en_subs";
const RU_SUBS_KEY = "deedub_live_ru_subs";
const VOL_KEY = "deedub_live_vol";
const QUALITIES = ["source", "1080p60", "1080p", "720p60", "720p", "480p", "360p", "160p"];
const DUB_LEAD_SECS = 3.0;
const DUB_SKIP_AFTER = 2.5;
// Саб держится на экране до t1 + GRACE: события STT/MT приходят ПОСЛЕ конца
// фразы (латентность конвейера), и строгое окно [t0..t1] к моменту прибытия
// уже прошло — саб никогда не показывался (баг вечернего прогона 12.11.26).
const SUBS_GRACE_SECS = 3.0;

function loadVol(): number {
  const n = Number(localStorage.getItem(VOL_KEY));
  return Number.isFinite(n) && n >= 0 && n <= 100 ? n : 80;
}

function loadCfg(): LiveCfg {  try {
    const raw = localStorage.getItem(CFG_KEY);
    if (raw) {
      const p = JSON.parse(raw) as Partial<LiveCfg>;
      const source: LiveCfg["source"] =
        p.source === "url" || p.source === "file" ? p.source : "twitch";
      return {
        source,
        channel: p.channel ?? "bogur",
        url: p.url ?? "",
        filePath: p.filePath ?? "",
        voice: p.voice ?? "",
        delay: typeof p.delay === "number" ? p.delay : 20,
        quality: p.quality ?? "source",
        proxy: p.proxy ?? "",
        oauth: p.oauth ?? "",
        dub: p.dub ?? true,
      };
    }
  } catch {
    // битый JSON — стартуем с дефолта, не падаем
  }
  return {
    source: "twitch",
    channel: "bogur",
    url: "",
    filePath: "",
    voice: "",
    delay: 20,
    quality: "source",
    proxy: "",
    oauth: "",
    dub: true,
  };
}

export default function Live() {
  const [cfg, setCfg] = useState<LiveCfg>(loadCfg);
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [showEn, setShowEn] = useState(() => localStorage.getItem(EN_SUBS_KEY) !== "0");
  const [showRu, setShowRu] = useState(() => localStorage.getItem(RU_SUBS_KEY) !== "0");
  const [currentEn, setCurrentEn] = useState("");
  const [currentRu, setCurrentRu] = useState("");
  const [attachedPort, setAttachedPort] = useState(0);
  const [skipN, setSkipN] = useState(0);
  // Громкость оригинала (0..100): у YouTube-VOD видеопоток БЕЗ звука (itag
  // video-only), оригинал играет отдельный <audio> без нативных контролов —
  // без слайдера громкость не настроить вообще. Меняет и <video> (мукс/
  // файл/twitch), и origAudio; дак дубляжа временно прижимает оба к 0.15.
  const [vol, setVol] = useState(loadVol);
  const volRef = useRef(0.8);

  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Оригинальный звук VOD (YouTube): отдельный <audio> на /audio, синк с видео.
  const origAudioRef = useRef<HTMLAudioElement | null>(null);
  const subsRef = useRef<LivePhrase[]>([]);
  const ruRef = useRef<DubItem[]>([]);
  const nextDubId = useRef(1);
  const playingDubRef = useRef<DubItem | null>(null);
  const prevVolumeRef = useRef(0.8);
  const dubOnRef = useRef(false);
  const showEnRef = useRef(true);
  const showRuRef = useRef(true);
  const attachedPortRef = useRef(0);
  const statusRef = useRef<LiveStatus | null>(null);

  const updateCfg = (patch: Partial<LiveCfg>) => setCfg((c) => ({ ...c, ...patch }));

  // Громкость: слайдер → оба медиа-элемента + база для возврата дака дубляжа.
  useEffect(() => {
    const v = Math.min(100, Math.max(0, vol)) / 100;
    volRef.current = v;
    prevVolumeRef.current = v;
    if (videoRef.current) videoRef.current.volume = v;
    if (origAudioRef.current) origAudioRef.current.volume = v;
    localStorage.setItem(VOL_KEY, String(vol));
  }, [vol]);

  useEffect(() => {
    dubOnRef.current = cfg.dub && cfg.voice !== "";
  }, [cfg.dub, cfg.voice]);

  useEffect(() => {
    showEnRef.current = showEn;
    localStorage.setItem(EN_SUBS_KEY, showEn ? "1" : "0");
  }, [showEn]);

  useEffect(() => {
    showRuRef.current = showRu;
    localStorage.setItem(RU_SUBS_KEY, showRu ? "1" : "0");
  }, [showRu]);

  useEffect(() => {
    localStorage.setItem(CFG_KEY, JSON.stringify(cfg));
  }, [cfg]);

  // Перетаскивание локального файла в окно (режим «file»): путь берём у
  // webview (нативный drop перехватывается Tauri), не браузерное событие.
  useEffect(() => {
    let un: (() => void) | undefined;
    let dead = false;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        const p = event.payload;
        if (p.type !== "drop" || !p.paths.length) return;
        const path = p.paths[0];
        setCfg((c) => (c.source === "file" ? { ...c, filePath: path } : c));
      })
      .then((f) => {
        if (dead) f();
        else un = f;
      });
    return () => {
      dead = true;
      un?.();
    };
  }, []);

  const destroyPlayer = useCallback(() => {
    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }
    const v = videoRef.current;
    if (v && v.src) {
      v.removeAttribute("src");
      v.load();
    }
    const oa = origAudioRef.current;
    if (oa) {
      oa.pause();
      oa.removeAttribute("src");
    }
    attachedPortRef.current = 0;
    setAttachedPort(0);
    subsRef.current = [];
    ruRef.current = [];
    playingDubRef.current = null;
    setCurrentEn("");
    setCurrentRu("");
  }, []);

  const attachPlayer = useCallback(
    (port: number, kind: "hls" | "file") => {
      if (attachedPortRef.current === port) return;
      const video = videoRef.current;
      if (!video) return;
      destroyPlayer();
      if (kind === "file") {
        // Локальный файл: бэкенд раздаёт его через /media с Range/206.
        video.src = `http://127.0.0.1:${port}/media`;
        video.play().catch(() => {});
      } else {
        const url = `http://127.0.0.1:${port}/live.m3u8`;
        const hls = new Hls({
          liveDurationInfinity: true,
          backBufferLength: 90,
          // Небольшой форвард-буфер: переживаем медленные сегменты, а
          // отставание от edge гасится лёгким ускорением ниже.
          maxBufferLength: 15,
          maxBufferSize: 60 * 1000 * 1000,
        });
        hlsRef.current = hls;
        hls.loadSource(url);
        hls.attachMedia(video);
        hls.on(Hls.Events.ERROR, (_evt, data) => {
          if (!data.fatal) return;
          if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
            // Сегмент мог уйти из скользящего окна — перезапрос с края.
            hls.startLoad(-2);
          } else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
            hls.recoverMediaError();
          }
        });
      }
      attachedPortRef.current = port;
      setAttachedPort(port);
    },
    [destroyPlayer],
  );

  useEffect(() => {
    let alive = true;

    const audio = new Audio();
    audioRef.current = audio;
    audio.onended = () => {
      const v = videoRef.current;
      if (v) v.volume = prevVolumeRef.current;
      const oa = origAudioRef.current;
      if (oa) oa.volume = volRef.current; // конец дака оригинального звука
      playingDubRef.current = null;
    };
    const orig = new Audio();
    orig.volume = volRef.current;
    origAudioRef.current = orig;

    const unSubs = listen<{ t0: number; t1: number; en: string }>("live-subs", (ev) => {
      if (!alive) return;
      subsRef.current.push(ev.payload);
    });

    const unRu = listen<{ t0: number; t1: number; ru: string }>("live-ru", (ev) => {
      if (!alive) return;
      ruRef.current.push({
        id: nextDubId.current++,
        t0: ev.payload.t0,
        t1: ev.payload.t1,
        ru: ev.payload.ru,
        state: "wait",
      });
    });

    const unDub = listen<{
      id: number;
      t0: number;
      t1: number;
      dur: number;
      url: string;
    }>("live-dub", (ev) => {
      if (!alive) return;
      const it = ruRef.current.find((x) => x.id === ev.payload.id);
      if (!it) return;
      if (ev.payload.url) {
        it.state = "ready";
        it.url = ev.payload.url;
        it.dur = ev.payload.dur;
      } else {
        // Дренаж бэкенда: фраза устарела до синтеза — RU-сабы остаются,
        // озвучки не будет.
        it.state = "failed";
      }
    });

    const poll = setInterval(async () => {
      try {
        const st = await invoke<LiveStatus>("live_status");
        if (!alive) return;
        statusRef.current = st;
        setStatus(st);
        if (st.running && st.port > 0 && attachedPortRef.current !== st.port) {
          attachPlayer(st.port, st.player_kind === "file" ? "file" : "hls");
          // Оригинальный звук VOD: запускается вместе с видео, синк в tick.
          const oa = origAudioRef.current;
          if (st.remote_audio && oa) {
            oa.src = `http://127.0.0.1:${st.port}/audio`;
            oa.play().catch(() => {});
          }
        }
        if (!st.running && attachedPortRef.current !== 0) {
          destroyPlayer();
        }
      } catch {
        // live_status может кинуть, пока бэкенд не готов — вкладка ждёт
      }
    }, 1000);

    const tick = setInterval(() => {
      if (!alive) return;
      const video = videoRef.current;
      const st = statusRef.current;
      if (!video || !st || !st.running || attachedPortRef.current === 0) return;
      const pd = hlsRef.current?.playingDate;
      const now = pd ? pd.getTime() / 1000 : video.currentTime;
      if (!Number.isFinite(now) || now <= 0) return;

      // Оригинальный звук VOD: держим на текущей позиции видео (сик по клику
      // пользователя тоже догоняется; порог 0.4 с — заметный рассинхрон).
      const oa0 = origAudioRef.current;
      if (oa0 && oa0.src) {
        if (Math.abs(oa0.currentTime - video.currentTime) > 0.4) {
          oa0.currentTime = video.currentTime;
        }
        if (video.paused && !oa0.paused) oa0.pause();
        else if (!video.paused && oa0.paused) oa0.play().catch(() => {});
      }

      // EN-сабы (шаг 2): последняя НАЧАВШАЯСЯ фраза (после late-прихода
      // find по строгому окну промахивается — фраза не показывалась вовсе),
      // держим до t1 + GRACE; новая начавшаяся фраза вытесняет старую.
      const subs = subsRef.current;
      while (subs.length > 0 && subs[0].t1 < now - 10) subs.shift();
      let cur: LivePhrase | undefined;
      for (let i = subs.length - 1; i >= 0; i--) {
        if (subs[i].t0 <= now) {
          cur = subs[i];
          break;
        }
      }
      if (cur && now > cur.t1 + SUBS_GRACE_SECS) cur = undefined;
      setCurrentEn(showEnRef.current && cur ? cur.en : "");

      // RU-сабы (шаг 3) + очередь дубляжа (шаг 4).
      const ru = ruRef.current;
      while (ru.length > 0 && ru[0].t1 < now - 30 && ru[0] !== playingDubRef.current) {
        ru.shift();
      }
      // RU-сабы показываются независимо от состояния озвучки: перевод уже
      // сделан, дак/скип касается только <audio>. Логика как у EN: последняя
      // начавшаяся фраза + держ до t1 + GRACE (перевод приходит позже речи).
      let curRu: DubItem | undefined;
      for (let i = ru.length - 1; i >= 0; i--) {
        if (ru[i].t0 <= now) {
          curRu = ru[i];
          break;
        }
      }
      if (curRu && now > curRu.t1 + SUBS_GRACE_SECS) curRu = undefined;
      setCurrentRu(showRuRef.current && curRu ? curRu.ru : "");

      if (dubOnRef.current) {
        for (const it of ru) {
          if (it.state !== "wait") continue;
          if (now > it.t0 + DUB_SKIP_AFTER) {
            it.state = "skipped";
            setSkipN((n) => n + 1);
          } else if (now >= it.t0 - DUB_LEAD_SECS) {
            it.state = "requested";
            invoke("live_synth", { id: it.id, t0: it.t0, t1: it.t1, ru: it.ru }).catch(() => {
              if (it.state === "requested") {
                it.state = "failed";
                setSkipN((n) => n + 1);
              }
            });
          }
        }
        // Проигрывание: последовательное, один <audio>, дак громкости видео.
        const a = audioRef.current;
        if (a && !playingDubRef.current) {
          const nxt = ru.find(
            (x) =>
              x.state === "ready" &&
              x.url &&
              now >= x.t0 - 0.15 &&
              now <= x.t0 + DUB_SKIP_AFTER,
          );
          if (nxt && nxt.url) {
            const url = nxt.url;
            playingDubRef.current = nxt;
            nxt.state = "played";
            prevVolumeRef.current = video.volume;
            video.volume = 0.15;
            const oaDuck = origAudioRef.current;
            if (oaDuck) oaDuck.volume = 0.15;
            a.src = url;
            a.play().catch(() => {
              // автоплей заблокирован / файл исчез — вернуть громкость
              video.volume = prevVolumeRef.current;
              if (nxt.state === "played") nxt.state = "failed";
              playingDubRef.current = null;
            });
          }
        }
      }
    }, 250);

    return () => {
      alive = false;
      clearInterval(poll);
      clearInterval(tick);
      audioRef.current = null;
      origAudioRef.current?.pause();
      origAudioRef.current = null;
      unSubs.then((f) => f());
      unRu.then((f) => f());
      unDub.then((f) => f());
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
    };
  }, [attachPlayer, destroyPlayer]);

  const running = !!status?.running;

  const handleStart = async () => {
    setBusy(true);
    setErr("");
    try {
      await invoke("live_start", {
        channel: cfg.channel,
        delayS: cfg.delay,
        quality: cfg.quality,
        proxy: cfg.proxy,
        oauthToken: cfg.oauth,
        sourceKind: cfg.source,
        url: cfg.url,
        filePath: cfg.filePath,
        voiceWav: cfg.voice,
        dubEnabled: cfg.dub && cfg.voice !== "",
      });
    } catch (e) {
      setErr(String(e));
    }
    setBusy(false);
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      await invoke("live_stop");
      destroyPlayer();
    } catch (e) {
      setErr(String(e));
    }
    setBusy(false);
  };

  const pickFile = async () => {
    const res = await open({
      multiple: false,
      filters: [{ name: "Видео", extensions: ["mp4", "mkv", "webm", "mov", "avi", "ts", "m4v"] }],
    });
    if (typeof res === "string") updateCfg({ filePath: res });
  };

  const pickVoice = async () => {
    const res = await open({
      multiple: false,
      filters: [{ name: "WAV", extensions: ["wav"] }],
    });
    if (typeof res === "string") updateCfg({ voice: res });
  };

  const startDisabled =
    busy ||
    (cfg.source === "twitch" && !cfg.channel.trim()) ||
    (cfg.source === "url" && !cfg.url.trim()) ||
    (cfg.source === "file" && !cfg.filePath);

  const isTwitch = cfg.source === "twitch";
  const isLiveLike = cfg.source === "twitch" || cfg.source === "url";

  return (
    <div className="live-tab">
      <div className="live-setup">
        <div className="live-source-row">
          <label>
            Источник:{" "}
            <select
              value={cfg.source}
              onChange={(e) => updateCfg({ source: e.target.value as LiveCfg["source"] })}
              disabled={running}
            >
              <option value="twitch">Twitch (канал)</option>
              <option value="url">Ссылка (YouTube и пр.)</option>
              <option value="file">Локальный файл</option>
            </select>
          </label>

          {isTwitch && (
            <label>
              Канал:{" "}
              <input
                value={cfg.channel}
                onChange={(e) => updateCfg({ channel: e.target.value })}
                placeholder="bogur"
                disabled={running}
              />
            </label>
          )}

          {cfg.source === "url" && (
            <label>
              Ссылка:{" "}
              <input
                value={cfg.url}
                onChange={(e) => updateCfg({ url: e.target.value })}
                placeholder="https://www.youtube.com/watch?v=…"
                disabled={running}
              />
            </label>
          )}

          {cfg.source === "file" && (
            <label className="live-file-row">
              Файл: <span className="live-file-path">{cfg.filePath || "не выбран"}</span>
              <button onClick={pickFile} disabled={running}>
                Выбрать…
              </button>
            </label>
          )}
        </div>

        <div className="live-source-row">
          {isLiveLike && (
            <label>
              Задержка: {cfg.delay}с{" "}
              <input
                type="range"
                min={10}
                max={60}
                step={1}
                value={cfg.delay}
                onChange={(e) => updateCfg({ delay: Number(e.target.value) })}
                disabled={running}
              />
            </label>
          )}

          {isTwitch && (
            <label>
              Качество:{" "}
              <select
                value={cfg.quality}
                onChange={(e) => updateCfg({ quality: e.target.value })}
                disabled={running}
              >
                {QUALITIES.map((q) => (
                  <option key={q} value={q}>
                    {q}
                  </option>
                ))}
              </select>
            </label>
          )}

          {isLiveLike && (
            <label>
              Прокси:{" "}
              <input
                value={cfg.proxy}
                onChange={(e) => updateCfg({ proxy: e.target.value })}
                placeholder="http://… (опционально)"
                disabled={running}
              />
            </label>
          )}

          {isTwitch && (
            <label>
              OAuth:{" "}
              <input
                type="password"
                value={cfg.oauth}
                onChange={(e) => updateCfg({ oauth: e.target.value })}
                placeholder="необязательно"
                disabled={running}
              />
            </label>
          )}
        </div>

        <div className="live-source-row">
          <label className="live-file-row">
            Голос (WAV-референс):{" "}
            <span className="live-file-path">{cfg.voice || "не выбран"}</span>
            <button onClick={pickVoice} disabled={running}>
              Выбрать…
            </button>
          </label>
          <label>
            <input
              type="checkbox"
              checked={cfg.dub}
              onChange={(e) => updateCfg({ dub: e.target.checked })}
            />{" "}
            Озвучка дубляжа
          </label>
          <label>
            Громкость оригинала: {vol}%{" "}
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={vol}
              onChange={(e) => setVol(Number(e.target.value))}
            />
          </label>
        </div>

        <div className="live-buttons">
          {!running ? (
            <button onClick={handleStart} disabled={startDisabled}>
              Запустить
            </button>
          ) : (
            <button onClick={handleStop} disabled={busy}>
              Остановить
            </button>
          )}
          <button onClick={() => setShowEn((v) => !v)}>
            {showEn ? "Скрыть EN-сабы" : "Показать EN-сабы"}
          </button>
          <button onClick={() => setShowRu((v) => !v)}>
            {showRu ? "Скрыть RU-сабы" : "Показать RU-сабы"}
          </button>
        </div>

        {err && <div className="live-error">{err}</div>}
      </div>

      <div className="live-player">
        <video ref={videoRef} controls autoPlay playsInline />
        {attachedPort === 0 && (
          <div className="live-placeholder">
            Выбери источник и запусти — плеер появится здесь (задержка = время
            буферизации и подготовки сабов/озвучки)
          </div>
        )}
        {currentEn && <div className="live-subs-overlay">{currentEn}</div>}
        {currentRu && <div className="live-subs-ru">{currentRu}</div>}
      </div>

      {status && (
        <div className="live-status">
          <div>
            {status.source_kind.toUpperCase()}: {status.channel || "—"} · состояние:{" "}
            {status.state}
            {status.state === "degraded" && ` (${status.last_error})`}
          </div>
          <div>
            STT: {status.stt_state} · MT: {status.mt_state} · TTS: {status.tts_state}
            {skipN > 0 && ` · озвучек пропущено: ${skipN}`}
          </div>
          <div>
            сегментов: {status.segments_downloaded} ok / {status.segments_failed} err ·
            задержка: {Math.round(status.delay_s)}с · окно:{" "}
            {Math.round(status.served_window_secs)}с
          </div>
        </div>
      )}
    </div>
  );
}
