import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import {
  getEngineConfig,
  MODELS_CHANGED_EVENT,
  type EngineConfig,
} from "@my-tauri-plugins/plugin-llama-engine";
import "@my-tauri-plugins/plugin-logs";
import "./App.css";
import Settings from "./Settings";

type Stage = "idle" | "select" | "processing" | "done" | "error";
type Tab = "main" | "logs" | "settings";

/** Формат промпта перевода. `auto` определяет его по модели (см. PromptStyle в Rust). */
type PromptStyle = "auto" | "insttrans" | "chat";

interface ModelChoice {
  path: string;
  name: string;
  exists: boolean;
}

interface ProgressUpdate {
  stage: string;
  percent: number;
  result_path?: string;
  error_message?: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("main");
  const [videoPath, setVideoPath] = useState<string>("");
  const [engineCfg, setEngineCfg] = useState<EngineConfig | null>(null);
  const [ffmpegPath, setFfmpegPath] = useState<string>("");
  const [format, setFormat] = useState<string>("mp4");
  const [enableDubbing, setEnableDubbing] = useState<boolean>(false);
  const [mixVolume, setMixVolume] = useState<number>(15);
  const [promptStyle, setPromptStyle] = useState<PromptStyle>("auto");
  const [models, setModels] = useState<ModelChoice[]>([]);
  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState(0);
  const [progressStage, setProgressStage] = useState("");
  const [resultPath, setResultPath] = useState("");
  const [error, setError] = useState("");

  const modelPath = engineCfg?.last_model || "";

  const refreshEngineConfig = useCallback(async () => {
    try {
      setEngineCfg(await getEngineConfig());
      // Список для селекта берём своей командой: она отдаёт только пути и
      // признак «файл на месте», а `getEngineConfig` тащит за собой ещё и
      // параметры сэмплирования, которые селекту не нужны.
      setModels(await invoke<ModelChoice[]>("list_translation_models"));
    } catch (e) {
      console.error("Не удалось прочитать конфиг движка LLM", e);
    }
  }, []);

  useEffect(() => {
    const unlisten = listen<ProgressUpdate>("pipeline-progress", (e) => {
      const { stage, percent, result_path, error_message } = e.payload;
      if (stage === "done") {
        setStage("done");
        setProgress(100);
        if (result_path) setResultPath(result_path);
      } else if (stage === "error") {
        setStage("error");
        setError(error_message || "Неизвестная ошибка. Подробности в логах.");
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

  // Активная модель живёт в конфиге плагина движка: читаем его на старте и
  // обновляемся, когда модель меняется в Настройках (core §2.1, SSOT).
  useEffect(() => {
    void refreshEngineConfig();
    const onChanged = () => { void refreshEngineConfig(); };
    document.addEventListener(MODELS_CHANGED_EVENT, onChanged);
    return () => document.removeEventListener(MODELS_CHANGED_EVENT, onChanged);
  }, [refreshEngineConfig]);

  useEffect(() => {
    invoke<{
      ffmpeg_path: string | null;
      output_format: string;
      enable_dubbing: boolean;
      mix_volume: number;
      prompt_style: PromptStyle;
    }>("get_config")
      .then((cfg) => {
        if (cfg.ffmpeg_path) setFfmpegPath(cfg.ffmpeg_path);
        if (cfg.output_format) setFormat(cfg.output_format);
        setEnableDubbing(cfg.enable_dubbing);
        setMixVolume(Math.round(cfg.mix_volume * 100));
        // Значение из старой копии config.toml может быть любым мусором —
        // тогда оставляем auto, а не пишем мусор обратно в конфиг.
        if (["auto", "insttrans", "chat"].includes(cfg.prompt_style)) {
          setPromptStyle(cfg.prompt_style);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      // ВНИМАНИЕ: `save_config` принимает весь структур целиком, поэтому здесь
      // обязаны быть ВСЕ поля `AppConfig`. Забытое поле не «останется как
      // было» — оно пропадёт из config.toml, а при следующей загрузке
      // `config::load` вернёт дефолты для всего файла.
      invoke("save_config", {
        cfg: {
          ffmpeg_path: ffmpegPath || null,
          output_format: format,
          enable_dubbing: enableDubbing,
          mix_volume: mixVolume / 100,
          prompt_style: promptStyle,
        },
      }).catch(() => {});
    }, 500);
    return () => clearTimeout(timer);
  }, [ffmpegPath, format, enableDubbing, mixVolume, promptStyle]);

  const handleSelectModel = async (path: string) => {
    if (!path) return;
    try {
      await invoke("set_translation_model", { path });
      // Событие нужно, чтобы обновились и наш индикатор, и `●` у активной
      // модели в панели плагина — это публичный контракт плагина, а не костыль.
      document.dispatchEvent(new CustomEvent(MODELS_CHANGED_EVENT));
    } catch (e) {
      setError(`Не удалось выбрать модель: ${e}`);
    }
  };

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
    if (!modelPath) {
      setError("Модель перевода не выбрана — добавьте её в Настройках");
      return;
    }
    setStage("processing");
    setProgress(0);
    setProgressStage("start");
    setError("");
    setResultPath("");

    invoke("process_video", {
      inputPath: videoPath,
      outputFormat: format,
      enableDubbing: enableDubbing,
    }).catch((e) => {
      setError(String(e));
      setStage("error");
    });
  };

  return (
    <div className={"app" + (activeTab === "logs" ? " app-wide" : "")}>
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
        <button
          className={"tab" + (activeTab === "settings" ? " tab-active" : "")}
          onClick={() => setActiveTab("settings")}
        >
          Настройки
        </button>
      </div>

      {activeTab === "main" && (
        <>
          <h1>DeeDub</h1>
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
            <label>Модель перевода</label>
            <div className="file-row">
              <select
                className="select"
                value={modelPath}
                onChange={(e) => void handleSelectModel(e.target.value)}
              >
                <option value="">— не выбрана —</option>
                {models.map((m) => (
                  <option key={m.path} value={m.path}>
                    {m.name}
                    {m.exists ? "" : " — файл не найден"}
                  </option>
                ))}
              </select>
              <button
                className="btn-secondary"
                onClick={() => setActiveTab("settings")}
              >
                Добавить модель
              </button>
            </div>
            {models.length === 0 && (
              <p className="hint">
                Настройки → «Локальные модели перевода»: скачайте модель из
                каталога или добавьте свой GGUF-файл.
              </p>
            )}
          </div>

          <div className="card">
            <label>Формат промпта перевода</label>
            <select
              className="select"
              value={promptStyle}
              onChange={(e) => setPromptStyle(e.target.value as PromptStyle)}
            >
              <option value="auto">Авто — по модели (рекомендуется)</option>
              <option value="insttrans">instTrans — для Index-Translate</option>
              <option value="chat">Chat — для Gemma и прочих чат-моделей</option>
            </select>
            <p className="hint">
              {promptStyle === "auto"
                ? "Index-Translate получает свой родной формат instTrans, остальные модели — наш chat-промпт с контекстом."
                : promptStyle === "insttrans"
                  ? "Формат instTrans. С чужими моделями (Gemma) перевод станет хуже."
                  : "Наш chat-промпт. С Index-Translate он ломается: модель дублирует соседние реплики."}
            </p>
          </div>

          <div className="card checkbox-card">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={enableDubbing}
                onChange={(e) => setEnableDubbing(e.target.checked)}
              />
              <span>Добавить русскую озвучку (TTS)</span>
            </label>
          </div>

          {enableDubbing && (
            <div className="card">
              <label>Громкость оригинального звука: {mixVolume}%</label>
              <input
                type="range"
                min="15"
                max="80"
                value={mixVolume}
                onChange={(e) => setMixVolume(Number(e.target.value))}
                className="slider"
              />
              <div className="slider-labels">
                <span>15%</span>
                <span>80%</span>
              </div>
            </div>
          )}

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
              <button
                className="btn-cancel"
                onClick={() => invoke("cancel_pipeline")}
              >
                Отмена
              </button>
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

          {error && (
            <div className="card error-card">
              <p className="error-text">{error}</p>
              {stage === "error" && (
                <button
                  className="btn-secondary"
                  onClick={() => { setStage("select"); setError(""); }}
                >
                  Назад
                </button>
              )}
            </div>
          )}
        </>
      )}

      {activeTab === "settings" && (
        <Settings
          ffmpegPath={ffmpegPath}
          setFfmpegPath={setFfmpegPath}
        />
      )}

      {activeTab === "logs" && (
        <div className="logs-panel">
          <logs-panel />
        </div>
      )}
    </div>
  );
}
