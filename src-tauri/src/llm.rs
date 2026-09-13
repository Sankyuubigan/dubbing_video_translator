use anyhow::{Context, Result};
use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;

pub const MAX_N_CTX: u32 = 8192;

/// Одноразовая генерация на уже загруженной модели (для ретраев верификации).
/// В отличие от translation-паспорта загружает модель сама: ретраи случаются
/// редко, поэтому перегруз (несколько секунд) допустима.
pub fn generate_once(
    model_path: &str,
    prompt: &str,
    temp: f32,
    seed: u32,
) -> Result<String> {
    let backend = crate::get_llama_backend().map_err(|e| anyhow::anyhow!(e))?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
    let model = LlamaModel::load_from_file(backend, model_path, &model_params)
        .with_context(|| format!("LLM: загрузка модели {}", model_path))?;

    log::info!("LLM: строгий ретрай на {:.60}...", model_path);

    let mut ctx = model.new_context(
        backend,
        LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(MAX_N_CTX))
            .with_n_batch(MAX_N_CTX),
    )?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(512, 1.05, 0.0, 0.0),
        LlamaSampler::top_k(64),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::temp(temp),
        LlamaSampler::dist(seed),
    ]);

    let eos = model.token_eos();
    let tok_len = model.str_to_token(prompt, AddBos::Always)?.len();
    let max_new = ((tok_len / 2).max(64)).min(512).min(MAX_N_CTX as usize - tok_len - 50);
    if max_new < 16 {
        anyhow::bail!("LLM: промпт слишком длинный ({} токенов)", tok_len);
    }

    let toks = generate_tokens(&model, &mut ctx, &mut sampler, prompt, eos, max_new, false)?;
    Ok(decode_tokens(&model, &toks))
}

/// Общая генерация с теми же стоп-условиями, что и в translation:
/// EOS-токен, открытие нового канала/тура, либо второй закрывающий `<channel|>`
/// (модель перешла в «режим редактора» — правильным считаем первый ответ).
///
/// `stop_on_channel_open`: в основном переводе промпт уже содержит открытый
/// thought-канал, и новое `<|channel>` означает уход в «редактор» — стоп.
/// В строгом ретрае модель сама открывает thought-канал (первый `<|channel>`),
/// поэтому стоп на открытии канала отключаем.
pub fn generate_tokens<'a>(
    model: &'a LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'a>,
    sampler: &mut LlamaSampler,
    prompt: &str,
    eos: LlamaToken,
    max_new: usize,
    stop_on_channel_open: bool,
) -> Result<Vec<LlamaToken>> {
    let tokens = model.str_to_token(prompt, AddBos::Always)?;

    let mut batch_llm = LlamaBatch::new(tokens.len(), 1);
    for (i, t) in tokens.iter().enumerate() {
        batch_llm.add(*t, i as i32, &[0], i == tokens.len() - 1)?;
    }
    ctx.decode(&mut batch_llm)?;

    let mut output_toks = Vec::new();
    let mut ctx_pos = batch_llm.n_tokens() as i32;

    for _ in 0..max_new {
        let mut cur_p = ctx.token_data_array();
        sampler.apply(&mut cur_p);

        let token = match cur_p.selected_token() {
            Some(t) => t,
            None => break,
        };

        if token == eos {
            break;
        }

        sampler.accept(token);
        output_toks.push(token);

        let current_text = decode_tokens(model, &output_toks);
        if current_text.contains("<turn|>") {
            break;
        }
        if stop_on_channel_open && current_text.contains("<|channel>") {
            break;
        }
        if current_text.matches("<channel|>").count() >= 2 {
            break;
        }

        let mut nb = LlamaBatch::new(1, 1);
        if let Err(e) = nb.add(token, ctx_pos, &[0], true) {
            log::warn!("LLM: ошибка batch.add: {:#}", e);
            break;
        }
        if let Err(e) = ctx.decode(&mut nb) {
            log::warn!("LLM: ошибка decode: {:#}", e);
            break;
        }
        ctx_pos += 1;
    }

    Ok(output_toks)
}

fn decode_tokens(model: &LlamaModel, tokens: &[LlamaToken]) -> String {
    let mut decoder = UTF_8.new_decoder();
    let mut result = String::new();
    for &token in tokens {
        if let Ok(piece) = model.token_to_piece(token, &mut decoder, false, None) {
            result.push_str(&piece);
        }
    }
    result
}