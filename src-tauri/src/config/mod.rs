use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Конфиг приложения. Выбор LLM-модели здесь НЕ хранится: он живёт в
/// конфиге плагина `tauri-plugin-llama-engine` (`app_config.json`, ключ
/// `last_model`) — единый источник правды (core §2.1). Старый ключ
/// `gguf_model_path` в файле пользователя просто игнорируется serde.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub output_format: String,
    pub ffmpeg_path: Option<String>,
    pub enable_dubbing: bool,
    #[serde(default = "default_mix_volume")]
    pub mix_volume: f64,
    /// Формат промпта перевода: `auto` | `insttrans` | `chat`.
    ///
    /// Не в конфиге плагина, потому что это поведение нашего приложения, а не
    /// плагина: движку всё равно, в каком виде мы формулируем просьбу модели.
    /// `#[serde(default)]` обязателен — без него старая копия файла без этого
    /// ключа привела бы к откату всего конфига на дефолты (см. `load`).
    #[serde(default = "default_prompt_style")]
    pub prompt_style: String,
}

fn default_mix_volume() -> f64 { 0.15 }

/// `auto` — определять формат по модели. Единственный безопасный дефолт:
/// у Index-Translate канонический формат instTrans, у Gemma и прочих чат-
/// моделей наш собственный chat-промпт, и путать их нельзя (см.
/// `translation::PromptStyle`).
fn default_prompt_style() -> String { "auto".to_string() }

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            output_format: "mp4".to_string(),
            ffmpeg_path: None,
            enable_dubbing: false,
            mix_volume: 0.15,
            prompt_style: default_prompt_style(),
        }
    }
}

pub fn config_path() -> PathBuf {
    let mut path = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    path.push(".deedub");
    path
}

pub fn config_file() -> PathBuf {
    let mut path = config_path();
    std::fs::create_dir_all(&path).ok();
    path.push("config.toml");
    path
}

pub fn load() -> AppConfig {
    let path = config_file();
    if !path.exists() {
        let cfg = AppConfig::default();
        save(&cfg).ok();
        return cfg;
    }
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    match toml::from_str(&content) {
        Ok(cfg) => cfg,
        Err(e) => {
            // Молча отдать дефолт здесь означало бы «все настройки слетели»
            // без единого слова в логе: пользователь видит пустой ffmpeg-путь и
            // думает, что приложение забыло его настройки. Ошибка парсинга —
            // почти всегда правка файла руками, и путь к нему нужен в тексте
            // ошибки.
            log::error!(
                "Config: не удалось разобрать {}: {e}. Возвращаю значения по умолчанию — \
                 проверьте файл, иначе настройки будут перезаписаны при следующем сохранении.",
                path.display()
            );
            AppConfig::default()
        }
    }
}

pub fn save(cfg: &AppConfig) -> Result<()> {
    let path = config_file();
    let content = toml::to_string_pretty(cfg)?;
    std::fs::create_dir_all(config_path())?;
    std::fs::write(&path, content)?;
    Ok(())
}
