# DubVidTra2 Documentation

## Stack
- **Backend:** Rust + Tauri 2.0, `llama-cpp-2` v0.1.146 (CUDA), `sherpa-onnx` v1.13.2
- **Frontend:** React/TypeScript
- **Media:** FFmpeg (external, audio extraction + subtitle muxing)
- **Models:** Silero VAD v5, PyAnnote + TitaNet (diarization), Qwen3-ASR (STT), Tower-Plus-9B (translation)

## Architecture
Modules are isolated — they don't import each other directly. All communication goes through `comm.rs` (PipelineContext hub).

```
Pipeline order (DON'T CHANGE):
1. audio_extractor — extract WAV from video via FFmpeg
2. vad — Voice Activity Detection (Silero VAD v5, .onnx via sherpa-onnx)
3. diarization — Speaker Identification (PyAnnote + TitaNet, .onnx via sherpa-onnx)
4. stt — Speech-to-Text (Qwen3-ASR, .onnx via sherpa-onnx) with speaker-aware slicing
5. translation — LLM translation (Tower-Plus-9B GGUF via llama-cpp-2)
6. output — burn subtitles into video via FFmpeg
```

Models are loaded one at a time and dropped before the next (VRAM safety).

## Translation Module (`src-tauri/src/translation/mod.rs`)

### Model
- **File:** `D:\nn\models\llm\uncen\gemma-4-12B\gemma-4-12B-it-heretic-QAT-UD-Q4_K_XL.gguf`
- **Base:** `google/gemma-4` (Gemma4ForCausalLM)
- **Arch:** Gemma 4, 12B dense params (not MoE)
- **Quant:** Q4_K_XL, ~7.2 GiB
- **Vocab:** 256000, SPM (SentencePiece — no BPE Cyrillic bug)
- **BOS token:** 2 — **prepended** (`AddBos::Always`, BOS needed before `<|turn>` tokens)
- **EOS token:** 3 (model default)
- **Context window:** 8192 tokens

### Prompt format
**Gemma 4 turn format** (uses `<|turn>`/`<turn|>` control tokens):

```
<|turn>system
You are a professional translator. Translate English subtitles to Russian. Use previous dialogue segments to resolve pronouns and maintain topic consistency. Output only the translation, no explanations.<turn|>
<|turn>user

Previous dialogue:
   1. EN: "prev text"
      RU: "prev translation"
   2. EN: "older text"
      RU: "older translation"

Translate the following text into Russian. Fix any ASR errors. Output only the Russian translation.

source text<turn|>
<|turn>model
```

- BOS token is prepended automatically via `AddBos::Always`
- Context: up to 3 previous (EN, RU) pairs in reverse order (most recent first)
- Source-side context (EN) is critical — research shows it contributes more than target-side
- Output is plain Russian text (no JSON, no grammar)
- Generation stops at EOS token 3
- `clean_output` strips `<|...>` garbage tokens, known prefixes, and trailing artifacts

### Sampling chain (in order)
1. `LlamaSampler::penalties(512, 1.05, 0.0, 0.0)` — repetition penalty
2. `LlamaSampler::top_k(64)` — top-K filtering
3. `LlamaSampler::top_p(0.95, 1)` — nucleus sampling
4. `LlamaSampler::temp(1.0)` — temperature scaling (Gemma 4 standard)
5. `LlamaSampler::dist(42)` — random selection with seed

### Generation
- `max_new = (prompt_tokens / 2).max(64).min(512)`
- KV cache cleared per chunk: `ctx.clear_kv_cache()` + `sampler.reset()`
- Context window: 8192 tokens
- EOS check (token 3) in generation loop as safety stop
- `clean_output` pipeline: strip before first Cyrillic → strip from `<` → strip "thought" suffix → strip known prefixes → strip trailing non-Cyrillic after last Cyrillic char → keep only last line if multiline (multiple candidates)

### Key differences from Tower-Plus-9B
- **`AddBos::Always`** — BOS needed before `<|turn>` tokens (same as Tower-Plus-9B)
- **temp=1.0** instead of 0.15 — official Gemma 4 recommendation
- **top_k=64, top_p=0.95** — wider sampling for better translation diversity
- **Plain instruction prompt** — no turn tokens (Gemma 2-style format, Gemma 4 doesn't treat `<|turn>` as single SPM tokens)
- **12B params** — better STT error correction and context understanding

## Testing

### Environment quirks (Windows)
1. **MSVC required** — `llama-cpp-sys-2` needs `cl.exe`. Call `vcvarsall.bat` first.
2. **sccache broken** — `~/.cargo/config.toml` has `rustc-wrapper = "sccache"` which fails. Must override with `RUSTC_WRAPPER=` + `--config "rustc-wrapper = ''"` in cargo.
3. **MSVC env only in cmd.exe** — PowerShell loses env vars after batch files. Always test via `.bat`.
4. **`cargo test --lib` crashes** with `0xc000007b` (DLL path issue in debug). Use release binary instead.

### Commands

```batch
REM Full pipeline test (reliable)
call "D:\Programs\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set RUSTC_WRAPPER=
cd src-tauri
cargo run --bin test-pipeline --release --config "rustc-wrapper = ''" -- --video test/for_test.mp4
```

```batch
REM Compilation check only
call "D:\Programs\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set RUSTC_WRAPPER=
cd src-tauri
cargo test --lib --no-run --config "rustc-wrapper = ''"
```

### Batch files
| File | Purpose |
|------|---------|
| `build.bat` | Production build (npm install + cargo release + tauri bundle) |
| `run_pipeline.bat` | Full pipeline test |
| `test.bat` | Compilation check |

### Output files
- Video with subtitles: `test/for_test_subbed.mp4`
- English SRT: `%TEMP%\dubvidtra_subtitles_en.srt`
- Russian SRT: `%TEMP%\dubvidtra_subtitles_ru.srt`

## Config
File: `~/.dubvidtra2/config.toml`
Fields: `gguf_model_path`, `sherpa_onnx_dir`, `stt_model`, `ffmpeg_path`, `vad_threshold_db`, `diarization_threshold`, `diarization_num_speakers`, `output_format`

## Known Issues
- Occasional STT errors not corrected (e.g., "sea" → "море" instead of "sight" → "зрелище") — 9B model limitation
- "spотыкалась" instead of "соскальзывал" for "slipped" context — model doesn't infer the tube top slip meaning
- `cargo test --lib` crashes at runtime due to DLL search path in debug mode
- sccache wrapper in global cargo config breaks C++ builds

## Relevant Source Files
| File | Purpose |
|------|---------|
| `src-tauri/src/translation/mod.rs` | Translation pipeline |
| `src-tauri/src/lib.rs` | App init, PIPELINE_CANCEL, LlamaBackend |
| `src-tauri/src/pipeline.rs` | Pipeline orchestrator |
| `src-tauri/src/comm.rs` | PipelineContext, SubtitleChunk |
| `src-tauri/src/config/mod.rs` | Config load/save |
| `src-tauri/src/audio_extractor/mod.rs` | WAV extraction |
| `src-tauri/src/vad/mod.rs` | Voice Activity Detection |
| `src-tauri/src/diarization/mod.rs` | Speaker diarization |
| `src-tauri/src/stt/mod.rs` | Speech-to-Text |
| `src-tauri/src/output/mod.rs` | Subtitle muxing |
| `src-tauri/Cargo.toml` | Rust dependencies |

## Research
- `research_tower_plus_9b/` — full research findings on Tower-Plus-9B prompt format, translation prompting, and alternative approaches
- Key finding: source-side context (previous EN text) is more important than target-side context for translation quality
- Key finding: temperature 0.1-0.3 is optimal for factual translation tasks
- Key finding: GBNF grammar not needed for standard instruction-tuned models like Tower-Plus-9B
