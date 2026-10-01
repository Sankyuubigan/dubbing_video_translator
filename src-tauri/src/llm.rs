//! Фасад инференса LLM поверх переиспользуемого плагина
//! `tauri-plugin-llama-engine` (SSOT движка, desktop §6.7/§7.1).
//!
//! Движок — ОТДЕЛЬНЫЙ процесс `llama-server.exe`, общение по HTTP localhost.
//! Приложение ничего не линкует нативно. Плагин отвечает за установку и
//! обновление бекенда, каталог моделей, выбор активной модели и VRAM.
//!
//! Модуль сознательно тонкий: вся специфика дубляжа (промпты, чистка вывода,
//! проверки изохронии) живёт в `translation` и `verification`. Здесь только
//! запуск сессии и один вызов генерации.

use std::path::Path;

use tauri::AppHandle;
use tauri_plugin_llama_engine::engine::llm_types::GenerationResult;
use tauri_plugin_llama_engine::engine::{llamacpp_installer, LlmMessage, LlamaEngine, ModelParams};

/// Контекстное окно движка (фиксируется при старте `llama-server.exe`).
pub const CONTEXT_SIZE: u32 = 8192;

/// Бюджет «размышлений» модели в токенах: 0 = выключен (global_ai_docs
/// `llama_cpp_engine.md` §3).
///
/// Думатель для дубляжа ВЫКЛЮЧЕН намеренно. Замер на ролике (238 чанков,
/// `test/last_logs.txt`): при бюджете 1500 модель писала 600-1500 токенов
/// рассуждений на КАЖДЫЙ чанк — даже на «Bam. Right now.» ушло 611 токенов
/// думателя. Итог: 25-30 с на реплику вместо ~1 с, перевод 238 чанков ≈ 100
/// минут вместо 3-5. Думатель при этом не давал выигрыша в качестве: ответ
/// и с ним, и без него — одна и та же короткая фраза, а ошибки распознавания
/// чинит source-side контекст (8 предыдущих пар EN/RU + lookahead), а не
/// длинный внутренний монолог.
///
/// Старый код (нативный llama-cpp-2) тоже думал, но коротко: стоп по
/// закрытию `<channel|>` ограничивал весь вывод 64-512 токенами НА ОТВЕТ.
/// Здесь бюджет 0 + `disable_reasoning` = тот же предел, но без CoT-мусора
/// в канале ответа.
const REASONING_BUDGET: u32 = 0;

// Параметры сэмплинга перевода. Раньше это была цепочка LlamaSampler
// (`penalties(512, 1.05) → top_k(64) → top_p(0.95) → temp(0.6)`); в плагине
// те же параметры передаются в llama-server как нативные параметры запроса.
const TEMPERATURE: f32 = 0.6;
const TOP_K: u32 = 64;
const TOP_P: f32 = 0.95;
const MIN_P: f32 = 0.0;
const REPETITION_PENALTY: f32 = 1.05;

/// Границы бюджета ОТВЕТА на один чанк (без «размышлений»).
const MIN_ANSWER_TOKENS: usize = 64;
const MAX_ANSWER_TOKENS: usize = 512;
const CONTEXT_RESERVE: usize = 50;

/// Живая сессия движка на время одного этапа пайплайна. При `drop` плагин
/// убивает процесс `llama-server.exe` и освобождает VRAM (desktop §6.5) —
/// поэтому следующий этап (TTS) стартует уже без модели в видеопамяти.
pub struct LlmSession {
    engine: LlamaEngine,
    params: ModelParams,
    model_path: String,
}

impl LlmSession {
    /// Поднимает движок на активной модели (её выбирает пользователь в
    /// Настройках → «Локальные модели перевода»; выбранная модель хранится
    /// в конфиге плагина, `last_model`).
    pub fn open(app: &AppHandle) -> Result<Self, String> {
        let engine_dir = tauri_plugin_llama_engine::get_engine_dir(app);
        if !llamacpp_installer::has_any_installed(&engine_dir) {
            return Err(format!(
                "Движок LLM не установлен (нет llama-server.exe в {}).\n\
                 Откройте Настройки → «Движок перевода (LLM)» и нажмите «Установить».",
                engine_dir.display()
            ));
        }

        let cfg = tauri_plugin_llama_engine::engine::load_config(app);
        let model_path = cfg
            .last_model
            .filter(|p| !p.is_empty())
            .ok_or_else(|| {
                "Модель перевода не выбрана.\nОткройте Настройки → «Локальные модели перевода» \
                 и добавьте модель (скачайте из каталога или выберите файл GGUF)."
                    .to_string()
            })?;
        if !Path::new(&model_path).is_file() {
            return Err(format!(
                "Файл выбранной модели не найден: {}\n\
                 Модель удалена или перемещена — выберите её заново в Настройках.",
                model_path
            ));
        }

        // Базовые значения берём из самого GGUF (плагин читает tokenizer.ggml.*),
        // сверху — параметры проекта для перевода.
        let mut params =
            tauri_plugin_llama_engine::commands::get_model_params(app.clone(), model_path.clone());
        params.temperature = TEMPERATURE;
        params.top_k = TOP_K;
        params.top_p = TOP_P;
        params.min_p = MIN_P;
        params.repetition_penalty = REPETITION_PENALTY;
        params.presence_penalty = 0.0;
        // DRY/XTC выключены: они не использовались в старой цепочке сэмплеров
        // и на дубляж только шумят.
        params.dry_multiplier = 0.0;
        params.dry_penalty_last_n = 0;
        params.xtc_probability = 0.0;

        log::info!(
            "LLM: запускаем движок на модели {} (ctx={}, temp={:.2}, top_k={}, top_p={:.2}, rep_pen={:.2})",
            model_path, CONTEXT_SIZE, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY
        );

        let engine = LlamaEngine::new(
            &engine_dir,
            &model_path,
            CONTEXT_SIZE,
            false,
            false,
            REASONING_BUDGET,
            |msg: String| log::info!("{}", msg),
            // Поток токенов в UI не показываем: DeeDub переводит пакетно, а
            // плагин сам пишет в лог метрики ответа (число токенов, стоп-причина,
            // скорость). Потокенный дамп в лог — это тысячи строк мусора.
            |_chunk: String| {},
        )?;

        Ok(Self { engine, params, model_path })
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// Один запрос к модели. `temperature` — переопределение для строгих
    /// ретраев верификации (детерминированный перевод), `None` — обычный
    /// перевод с параметрами проекта.
    pub fn generate(
        &self,
        messages: &[LlmMessage],
        temperature: Option<f32>,
    ) -> Result<GenerationResult, String> {
        if crate::is_cancelled() {
            return Err("Отменено пользователем".to_string());
        }

        let mut params = self.params.clone();
        if let Some(temp) = temperature {
            params.temperature = temp;
        }

        // Бюджет генерации: n_predict ограничивает только ОТВЕТ — думатель
        // выключен (REASONING_BUDGET = 0), поэтому резервировать токены «на
        // размышления» не нужно. Потолок не стоит генерации: llama-server
        // останавливается на EOS сам.
        let prompt_tokens = self.engine.get_tokens_count(messages, "Auto").unwrap_or(0);
        let answer_tokens = (prompt_tokens / 2).clamp(MIN_ANSWER_TOKENS, MAX_ANSWER_TOKENS);
        let context_left = CONTEXT_SIZE as usize - prompt_tokens - CONTEXT_RESERVE;
        if context_left <= answer_tokens {
            return Err(format!(
                "Промпт не помещается в контекст модели ({} токенов при CONTEXT_SIZE={})",
                prompt_tokens, CONTEXT_SIZE
            ));
        }
        let max_new = (answer_tokens + REASONING_BUDGET as usize).min(context_left);

        let result = self.engine.generate_chat(
            messages,
            max_new,
            &params,
            "Auto",
            true, // disable_reasoning: думатель выключен — ответ без CoT
            crate::cancel_flag(),
            "translate",
            None,
            |_, _| {},
            |msg: String| log::info!("{}", msg),
        )?;

        log::debug!(
            "LLM[{}]: {} токенов промпта, {} токенов ответа, stop_reason={}",
            self.model_path(),
            prompt_tokens,
            result.metrics.generated_tokens,
            result.stop_reason
        );
        if !result.reasoning.trim().is_empty() {
            log::debug!(
                "LLM: рассуждения модели ({} симв.): {}",
                result.reasoning.len(),
                result.reasoning.trim()
            );
        }

        Ok(result)
    }
}

/// Сообщение для перевода/ретрая: текст ответа модели без рассуждений.
pub fn message(role: &str, content: String) -> LlmMessage {
    LlmMessage {
        role: role.to_string(),
        content,
        tool_calls: None,
        tool_call_id: None,
    }
}