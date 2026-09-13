import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";

interface EngineInfo {
  id: string;
  label: string;
  installed: boolean;
  installed_version: string | null;
  latest_version: string;
  update_available: boolean;
}

interface ModelInfo {
  id: string;
  label: string;
  installed: boolean;
  has_codec: boolean;
  has_voice: boolean;
  voice_type: string;
  size: string;
  supports_russian: boolean;
}

interface DownloadProgress {
  downloaded: number;
  total: number;
}

interface SettingsProps {
  sttModel: string;
  setSttModel: (v: string) => void;
  sherpaOnnxDir: string;
  setSherpaOnnxDir: (v: string) => void;
  ffmpegPath: string;
  setFfmpegPath: (v: string) => void;
}

const PRIMARY_MODELS = ["cosyvoice3-tts", "parakeet-tdt-0.6b-v3"];

export default function Settings({
  sttModel,
  setSttModel,
  sherpaOnnxDir,
  setSherpaOnnxDir,
  ffmpegPath,
  setFfmpegPath,
}: SettingsProps) {
  const [engineDir, setEngineDir] = useState("");
  const [modelsDir, setModelsDir] = useState("");
  const [dirsLoading, setDirsLoading] = useState(true);
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<Record<string, DownloadProgress>>({});
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [error, setError] = useState("");
  const progressRef = useRef<Record<string, DownloadProgress>>({});

  const loadDirs = async () => {
    try {
      const dirs = await invoke<{ engine_dir: string; models_dir: string }>(
        "plugin:speech|tts_default_dirs"
      );
      const saved = await invoke<{
        engine_dir?: string;
        models_dir?: string;
      }>("plugin:speech|tts_get_settings");
      setEngineDir(saved.engine_dir || dirs.engine_dir || "");
      setModelsDir(saved.models_dir || dirs.models_dir || "");
    } catch (e) {
      setError("Не удалось получить папки движка: " + String(e));
    } finally {
      setDirsLoading(false);
    }
  };

  const loadEngines = async () => {
    try {
      const res = await invoke<{
        ok: boolean;
        engines?: EngineInfo[];
        error?: string;
      }>("plugin:speech|tts_check_update");
      if (res.ok && res.engines) {
        setEngines(res.engines);
      } else {
        setError(res.error || "Не удалось получить список движка");
      }
    } catch (e) {
      setError("Ошибка списка движка: " + String(e));
    }
  };

  const loadModels = async (dir?: string) => {
    try {
      const list = await invoke<ModelInfo[]>("plugin:speech|tts_list_models", {
        modelsDir: dir ?? modelsDir,
      });
      const sorted = [...list].sort((a, b) => {
        const ia = PRIMARY_MODELS.indexOf(a.id);
        const ib = PRIMARY_MODELS.indexOf(b.id);
        return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
      });
      setModels(sorted);
    } catch (e) {
      setError("Ошибка списка моделей: " + String(e));
    }
  };

  useEffect(() => {
    loadDirs();
    loadEngines();
  }, []);

  useEffect(() => {
    if (!dirsLoading) {
      loadModels(modelsDir);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelsDir, dirsLoading]);

  useEffect(() => {
    const unlisten = listen<{
      kind: string;
      name: string;
      downloaded: number;
      total: number;
    }>("tts-download", (e) => {
      progressRef.current[e.payload.name] = {
        downloaded: e.payload.downloaded,
        total: e.payload.total,
      };
      setProgress({ ...progressRef.current });
      setActiveFile(e.payload.name);
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  const saveDirs = async (engine_dir: string, models_dir: string) => {
    const saved = await invoke<{
      engine_dir?: string;
      models_dir?: string;
      engine_backend?: string;
      preset?: string;
    }>("plugin:speech|tts_get_settings");
    await invoke("plugin:speech|tts_save_settings", {
      settings: {
        engine_dir,
        models_dir,
        engine_backend: saved.engine_backend || "cuda",
        preset: saved.preset || "cosyvoice3-tts",
      },
    });
  };

  const pickEngineDir = async () => {
    const file = await open({ multiple: false, directory: true });
    if (file && typeof file === "string") {
      setEngineDir(file);
      await saveDirs(file, modelsDir);
    }
  };

  const pickModelsDir = async () => {
    const file = await open({ multiple: false, directory: true });
    if (file && typeof file === "string") {
      setModelsDir(file);
      await saveDirs(engineDir, file);
    }
  };

  const downloadEngine = async (backendId: string) => {
    if (busy) return;
    setBusy(`engine:${backendId}`);
    setError("");
    try {
      await invoke("plugin:speech|tts_download_engine", {
        backendId,
        dest: engineDir,
      });
      await loadEngines();
    } catch (e) {
      setError("Ошибка скачивания движка: " + String(e));
    } finally {
      setBusy(null);
    }
  };

  const downloadModel = async (preset: string) => {
    if (busy) return;
    setBusy(`model:${preset}`);
    setError("");
    try {
      await invoke("plugin:speech|tts_download_model", {
        preset,
        dest: modelsDir,
      });
      const saved = await invoke<{
        engine_dir?: string;
        models_dir?: string;
        engine_backend?: string;
        preset?: string;
      }>("plugin:speech|tts_get_settings");
      await invoke("plugin:speech|tts_save_settings", {
        settings: {
          engine_dir: saved.engine_dir || engineDir,
          models_dir: saved.models_dir || modelsDir,
          engine_backend: saved.engine_backend || "cuda",
          preset: preset === "parakeet-tdt-0.6b-v3" ? "cosyvoice3-tts" : preset,
        },
      });
      await loadModels();
    } catch (e) {
      setError("Ошибка скачивания модели: " + String(e));
    } finally {
      setBusy(null);
    }
  };

  const percentLabel = (downloaded: number, total: number) => {
    if (!total) return "";
    const pct = Math.min(100, Math.round((downloaded / total) * 100));
    return pct < 100 ? ` — ${pct}%` : "";
  };

  const pickLegacy = async (kind: "asr" | "ffmpeg") => {
    const file =
      kind === "ffmpeg"
        ? await open({
            multiple: false,
            filters: [{ name: "FFmpeg", extensions: ["exe"] }],
          })
        : await open({ multiple: false, directory: true });
    if (file && typeof file === "string") {
      if (kind === "ffmpeg") setFfmpegPath(file);
      else setSherpaOnnxDir(file);
    }
  };

  const activeProg = activeFile ? progress[activeFile] : undefined;

  return (
    <div className="settings">
      <h1>Настройки</h1>

      {error && (
        <div className="card error-card">
          <p className="error-text">{error}</p>
          <button className="btn-secondary" onClick={() => setError("")}>
            Ок
          </button>
        </div>
      )}

      {activeProg && activeProg.total > 0 && (
        <div className="card">
          <div className="progress-label">
            Скачивание: {activeFile}
            {percentLabel(activeProg.downloaded, activeProg.total)}
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${Math.min(
                  100,
                  Math.round((activeProg.downloaded / activeProg.total) * 100)
                )}%`,
              }}
            />
          </div>
        </div>
      )}

      <div className="card">
        <label>Папка движка CrispASR</label>
        <div className="file-row">
          <span className="file-name dim">
            {dirsLoading
              ? "Загрузка..."
              : engineDir || "Не выбрана"}
          </span>
          <button className="btn-secondary" onClick={pickEngineDir}>
            Выбрать
          </button>
        </div>
      </div>

      <div className="card">
        <label>Папка моделей</label>
        <div className="file-row">
          <span className="file-name dim">
            {dirsLoading ? "Загрузка..." : modelsDir || "Не выбрана"}
          </span>
          <button className="btn-secondary" onClick={pickModelsDir}>
            Выбрать
          </button>
        </div>
      </div>

      <div className="card">
        <label>Движок (crispasr.exe)</label>
        <div className="tiles">
          {engines.map((e) => {
            const isBusy = busy === `engine:${e.id}`;
            return (
              <div key={e.id} className={"tile" + (e.installed ? " tile-installed" : "")}>
                <div className="tile-header">
                  <span className="tile-label">{e.label}</span>
                  {e.installed ? (
                    <span className="badge badge-ok">установлен</span>
                  ) : (
                    <span className="badge">не установлен</span>
                  )}
                </div>
                <div className="tile-meta">
                  {e.installed_version && (
                    <span>v{e.installed_version}</span>
                  )}
                  {e.update_available && (
                    <span className="update-flag">есть v{e.latest_version}</span>
                  )}
                </div>
                <div className="tile-actions">
                  <button
                    className="btn-secondary"
                    disabled={!!busy}
                    onClick={() => downloadEngine(e.id)}
                  >
                    {isBusy
                      ? "Скачивание... (см. прогресс)"
                      : e.installed && e.update_available
                      ? "Обновить"
                      : "Скачать"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="card">
        <label>Установленные модели</label>
        <div className="tiles">
          {models.map((m) => {
            const isBusy = busy === `model:${m.id}`;
            const isStt = m.id === "parakeet-tdt-0.6b-v3";
            return (
              <div key={m.id} className={"tile" + (m.installed ? " tile-installed" : "")}>
                <div className="tile-header">
                  <span className="tile-label">
                    {m.label}
                    {isStt && <span className="tile-tag">STT</span>}
                  </span>
                  {m.installed ? (
                    <span className="badge badge-ok">установлено</span>
                  ) : (
                    <span className="badge">не установлено</span>
                  )}
                </div>
                <div className="tile-meta">
                  <span>{m.size}</span>
                  {m.supports_russian && <span>RU</span>}
                </div>
                <div className="tile-actions">
                  <button
                    className="btn-secondary"
                    disabled={!!busy}
                    onClick={() => downloadModel(m.id)}
                  >
                    {isBusy ? "Скачивание... (см. прогресс)" : "Скачать"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
        <p className="hint">
          Parakeet TDT — модель распознавания речи (STT). CosyVoice3 — озвучка (TTS).
          Скачанные GGUF сохраняются в папке моделей.
        </p>
      </div>

      <div className="card">
        <label>Распознавание (ASR, текущий бэкенд sherpa-onnx)</label>
        <div className="file-row">
          <span className="file-name dim">
            {sherpaOnnxDir ? sherpaOnnxDir.split("\\").pop()?.split("/").pop() : "Не выбрана"}
          </span>
          <button className="btn-secondary" onClick={() => pickLegacy("asr")}>
            Выбрать
          </button>
          {sherpaOnnxDir && (
            <button className="btn-secondary" onClick={() => setSherpaOnnxDir("")}>
              Сбросить
            </button>
          )}
        </div>
        <select
          className="select"
          style={{ marginTop: 10 }}
          value={sttModel}
          onChange={(e) => setSttModel(e.target.value)}
        >
          <option value="qwen3-asr">Qwen3-ASR (мультиязычный)</option>
          <option value="parakeet-tdt">Parakeet TDT sherpa (английский)</option>
        </select>
      </div>

      <div className="card">
        <label>FFmpeg (оставьте пустым для auto-поиска)</label>
        <div className="file-row">
          <span className="file-name dim">
            {ffmpegPath ? ffmpegPath.split("\\").pop()?.split("/").pop() : "Auto (PATH / рядом с exe)"}
          </span>
          <button className="btn-secondary" onClick={() => pickLegacy("ffmpeg")}>
            Выбрать
          </button>
          {ffmpegPath && (
            <button className="btn-secondary" onClick={() => setFfmpegPath("")}>
              Сбросить
            </button>
          )}
        </div>
      </div>
    </div>
  );
}