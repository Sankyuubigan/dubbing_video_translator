import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

type Stage = "idle" | "select" | "processing" | "done" | "error";
type Tab = "main" | "logs";

interface LogEntry {
  level: string;
  message: string;
}

interface ProgressUpdate {
  stage: string;
  percent: number;
  result_path?: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("main");
  const [videoPath, setVideoPath] = useState<string>("");
  const [ggufPath, setGgufPath] = useState<string>("");
  const [sherpaOnnxDir, setSherpaOnnxDir] = useState<string>("");
  const [ffmpegPath, setFfmpegPath] = useState<string>("");
  const [vadThreshold, setVadThreshold] = useState<string>("-30");
  const [format, setFormat] = useState<string>("mp4");
  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState(0);
  const [progressStage, setProgressStage] = useState("");
  const [resultPath, setResultPath] = useState("");
  const [error, setError] = useState("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logPaths, setLogPaths] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    const unlisten = listen<ProgressUpdate>("pipeline-progress", (e) => {
      const { stage, percent, result_path } = e.payload;
      if (stage === "done") {
        setStage("done");
        setProgress(100);
        if (result_path) setResultPath(result_path);
      } else if (stage === "error") {
        setStage("error");
      } else {
        setProgressStage(stage);
        setProgress(percent);
        setStage((prev) => prev === "idle" || prev === "select" ? "processing" : prev);
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  useEffect(() => {
    const unlisten = listen<{ paths: string[] }>("tauri://drag-drop", (e) => {
      const path = e.payload.paths[0];
      if (path) {
        setVideoPath(path);
        const ext = path.split(".").pop()?.toLowerCase() || "mp4";
        setFormat(ext);
        setStage("select");
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  useEffect(() => {
    const unlisten = listen<LogEntry>("new-log", (e) => {
      setLogs((prev) => [...prev.slice(-4999), e.payload]);
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  useEffect(() => {
    invoke<LogEntry[]>("get_logs")
      .then(setLogs)
      .catch(() => {});
    invoke<string[]>("get_log_paths")
      .then(setLogPaths)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (autoScroll) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll]);

  useEffect(() => {
    invoke<{
      gguf_model_path: string | null;
      ffmpeg_path: string | null;
      output_format: string;
      vad_threshold_db: string | null;
      sherpa_onnx_dir: string | null;
    }>("get_config")
      .then((cfg) => {
        if (cfg.gguf_model_path) setGgufPath(cfg.gguf_model_path);
        if (cfg.ffmpeg_path) setFfmpegPath(cfg.ffmpeg_path);
        if (cfg.output_format) setFormat(cfg.output_format);
        if (cfg.vad_threshold_db) setVadThreshold(cfg.vad_threshold_db);
        if (cfg.sherpa_onnx_dir) setSherpaOnnxDir(cfg.sherpa_onnx_dir);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!ggufPath) return;
    const timer = setTimeout(() => {
      invoke("save_config", {
        cfg: {
          gguf_model_path: ggufPath || null,
          ffmpeg_path: ffmpegPath || null,
          output_format: format,
          vad_threshold_db: vadThreshold || null,
          sherpa_onnx_dir: sherpaOnnxDir || null,
        },
      }).catch(() => {});
    }, 500);
    return () => clearTimeout(timer);
  }, [ggufPath, ffmpegPath, format, vadThreshold, sherpaOnnxDir]);

  const handleSelectVideo = async () => {
    const file = await open({
      multiple: false,
      filters: [{ name: "Video", extensions: ["mp4", "mkv", "avi", "mov", "webm"] }],
    });
    if (file) {
      setVideoPath(file);
      const ext = file.split(".").pop()?.toLowerCase() || "mp4";
      setFormat(ext);
      setStage("select");
    }
  };

  const handleProcess = async () => {
    if (!videoPath) return;
    if (!ggufPath) {
      setError("Выберите GGUF-файл модели перевода");
      return;
    }
    if (!sherpaOnnxDir) {
      setError("Выберите папку с Sherpa-ONNX моделью ASR");
      return;
    }
    setStage("processing");
    setProgress(0);
    setProgressStage("start");
    setError("");
    setResultPath("");

    invoke("process_video", {
      inputPath: videoPath,
      ggufModelPath: ggufPath,
      outputFormat: format,
      vadThresholdDb: vadThreshold || null,
      sherpaOnnxDir: sherpaOnnxDir || null,
    }).catch((e) => {
      setError(String(e));
      setStage("error");
    });
  };

  const logLevelClass = (level: string) => {
    if (level === "ERROR") return "log-error";
    if (level === "WARN") return "log-warn";
    return "";
  };

  return (
    <div className="app">
      <div className="tabs">
        <button
          className={"tab" + (activeTab === "main" ? " tab-active" : "")}
          onClick={() => setActiveTab("main")}
        >
          Главная
        </button>
        <button
          className={"tab" + (activeTab === "logs" ? " tab-active" : "")}
          onClick={() => setActiveTab("logs")}
        >
          Логи
        </button>
      </div>

      {activeTab === "main" && (
        <>
          <h1>DubVidTra2</h1>
          <p className="subtitle">Локальный перевод видео с субтитрами</p>

          {stage === "idle" && (
            <div className="dropzone" onClick={handleSelectVideo}>
              <div className="drop-icon">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <p className="drop-text">Перетащите видео сюда или нажмите для выбора</p>
              <p className="drop-hint">MP4, MKV, AVI, MOV, WebM</p>
            </div>
          )}

          {stage !== "idle" && (
            <div className="card">
              <label>Видеофайл</label>
              <div className="file-row">
                <span className="file-name">
                  {videoPath.split("\\").pop()?.split("/").pop()}
                </span>
                <button className="btn-secondary" onClick={handleSelectVideo}>
                  Изменить
                </button>
              </div>
            </div>
          )}

          <div className="card">
            <label>Sherpa-ONNX модель (папка с conv_frontend.onnx и tokenizer/)</label>
            <div className="file-row">
              {sherpaOnnxDir ? (
                <span className="file-name">
                  {sherpaOnnxDir.split("\\").pop()?.split("/").pop()}
                </span>
              ) : (
                <span className="file-name dim">Не выбрана</span>
              )}
              <button className="btn-secondary" onClick={async () => {
                const file = await open({
                  multiple: false,
                  directory: true,
                });
                if (file) setSherpaOnnxDir(file);
              }}>
                Выбрать
              </button>
              {sherpaOnnxDir && (
                <button className="btn-secondary" onClick={() => setSherpaOnnxDir("")}>
                  Сбросить
                </button>
              )}
            </div>
          </div>

          <div className="card">
            <label>FFmpeg (оставьте пустым для auto-поиска)</label>
            <div className="file-row">
              {ffmpegPath ? (
                <span className="file-name">
                  {ffmpegPath.split("\\").pop()?.split("/").pop()}
                </span>
              ) : (
                <span className="file-name dim">Auto (PATH / рядом с exe)</span>
              )}
              <button className="btn-secondary" onClick={async () => {
                const file = await open({
                  multiple: false,
                  filters: [{ name: "FFmpeg", extensions: ["exe"] }],
                });
                if (file) setFfmpegPath(file);
              }}>
                Выбрать
              </button>
              {ffmpegPath && (
                <button className="btn-secondary" onClick={() => setFfmpegPath("")}>
                  Сбросить
                </button>
              )}
            </div>
          </div>

          <div className="card">
            <label>Модель перевода (GGUF)</label>
            <div className="file-row">
              {ggufPath ? (
                <span className="file-name">
                  {ggufPath.split("\\").pop()?.split("/").pop()}
                </span>
              ) : (
                <span className="file-name dim">Не выбрана</span>
              )}
              <button className="btn-secondary" onClick={async () => {
                const file = await open({
                  multiple: false,
                  filters: [{ name: "GGUF Model", extensions: ["gguf"] }],
                });
                if (file) setGgufPath(file);
              }}>
                Выбрать
              </button>
            </div>
          </div>

          <div className="card">
            <label>VAD порог (dB, тише = меньше ложных срабатываний)</label>
            <div className="file-row">
              <input
                className="input-text"
                type="text"
                value={vadThreshold}
                onChange={(e) => setVadThreshold(e.target.value)}
                placeholder="-30"
              />
            </div>
          </div>

          <div className="card">
            <label>Выходной формат</label>
            <select
              className="select"
              value={format}
              onChange={(e) => setFormat(e.target.value)}
            >
              <option value="mp4">MP4 (по умолчанию)</option>
              <option value="mkv">MKV</option>
              <option value="avi">AVI</option>
              <option value="mov">MOV</option>
              <option value="webm">WebM</option>
            </select>
          </div>

          {stage === "select" && !error && (
            <button className="btn-primary" onClick={handleProcess}>
              Обработать
            </button>
          )}

          {stage === "processing" && (
            <div className="card">
              <div className="progress-label">
                {progressStage === "start"
                  ? "Запуск..."
                  : `Этап: ${progressStage}`}
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {stage === "done" && (
            <div className="card success">
              <div className="result-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                  <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
              </div>
              <p className="result-text">Готово!</p>
              {resultPath && <p className="result-path">{resultPath}</p>}
              <button className="btn-primary" onClick={() => setStage("idle")}>
                Новое видео
              </button>
            </div>
          )}

          {stage === "error" && error && (
            <div className="card error-card">
              <p className="error-text">{error}</p>
              <button
                className="btn-secondary"
                onClick={() => { setStage("select"); setError(""); }}
              >
                Назад
              </button>
            </div>
          )}
        </>
      )}

      {activeTab === "logs" && (
        <div className="logs-panel">
          {logPaths.length > 0 && (
            <div className="log-file-path">
              Файлы логов:
              {logPaths.map((p, i) => (
                <div key={i} className="log-file-path-value">{p}</div>
              ))}
            </div>
          )}
          <div className="logs-toolbar">
            <span className="logs-count">Записей: {logs.length}</span>
            <button
              className="btn-secondary"
              onClick={() => setLogs([])}
            >
              Очистить
            </button>
            <button
              className="btn-secondary"
              onClick={async () => {
                const text = logs.map(e => `[${e.level}] ${e.message}`).join('\n');
                try {
                  await navigator.clipboard.writeText(text);
                } catch {}
              }}
            >
              Копировать все
            </button>
          </div>
          <div
            className="logs-container"
            onScroll={(e) => {
              const el = e.currentTarget;
              const atBottom =
                el.scrollHeight - el.scrollTop - el.clientHeight < 50;
              setAutoScroll(atBottom);
            }}
          >
            {logs.length === 0 && (
              <div className="logs-empty">Нет записей</div>
            )}
            {logs.map((entry, i) => (
              <div
                key={i}
                className={"log-entry " + logLevelClass(entry.level)}
              >
                <span className="log-level">[{entry.level}]</span>
                <span className="log-msg">{entry.message}</span>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}
