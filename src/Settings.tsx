import "@my-tauri-plugins/plugin-speech";
import "@my-tauri-plugins/plugin-llama-engine";
import { open } from "@tauri-apps/plugin-dialog";

interface SettingsProps {
  ffmpegPath: string;
  setFfmpegPath: (v: string) => void;
}

export default function Settings({ ffmpegPath, setFfmpegPath }: SettingsProps) {
  const pickFfmpeg = async () => {
    const file = await open({
      multiple: false,
      filters: [{ name: "FFmpeg", extensions: ["exe"] }],
    });
    if (file && typeof file === "string") {
      setFfmpegPath(file);
    }
  };

  return (
    <div className="settings">
      <h1>Настройки</h1>

      <div className="card">
        <label>Движок перевода (LLM)</label>
        <llama-engine-panel />
      </div>

      <div className="card">
        <label>Локальные модели перевода</label>
        <llama-models-panel />
      </div>

      <div className="card">
        <label>Скачать модель перевода из каталога</label>
        <llama-download-panel />
      </div>

      <div className="card">
        <label>Движок CrispASR</label>
        <speech-engine-panel />
      </div>

      <div className="card">
        <label>Модели (TTS / STT)</label>
        <speech-models-panel />
      </div>

      <div className="card">
        <label>Хранилище голосов</label>
        <speech-voice-storage />
      </div>

      <div className="card">
        <label>FFmpeg (оставьте пустым для auto-поиска)</label>
        <div className="file-row">
          <span className="file-name dim">
            {ffmpegPath ? ffmpegPath.split("\\").pop()?.split("/").pop() : "Auto (PATH / рядом с exe)"}
          </span>
          <button className="btn-secondary" onClick={pickFfmpeg}>
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