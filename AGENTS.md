# DubVidTra2 Documentation

## Stack
- **Backend:** Rust + Tauri 2.0, `llama-cpp-2` v0.1.146 (CUDA), `sherpa-onnx` v1.13.2
- **Frontend:** React/TypeScript
- **Media:** FFmpeg (external, audio extraction + subtitle muxing)
- **Models:** Silero VAD v5, PyAnnote + TitaNet (diarization), Qwen3-ASR (STT), Hy-MT2-7B (translation)

## Architecture
Modules are isolated — they don't import each other directly. All communication goes through `comm.rs` (PipelineContext hub).

```
Pipeline order (DON'T CHANGE):
1. audio_extractor — extract WAV from video via FFmpeg
2. vad — Voice Activity Detection (Silero VAD v5, .onnx via sherpa-onnx)
3. diarization — Speaker Identification (PyAnnote + TitaNet, .onnx via sherpa-onnx)
4. stt — Speech-to-Text (Qwen3-ASR, .onnx via sherpa-onnx) with speaker-aware slicing
5. translation — LLM translation (Hy-MT2-7B GGUF via llama-cpp-2)
6. output — burn subtitles into video via FFmpeg
```

Models are loaded one at a time and dropped before the next (VRAM safety).

## Translation Module (`src-tauri/src/translation/mod.rs`)

### Model
- **File:** `D:\nn\models\translation\Hy-MT2-7B.i1-IQ4_NL.gguf`
- **Arch:** Hunyuan-dense, 7.5B params
- **Quant:** IQ4_NL (4.5 bpw), 4.07 GiB
- **Vocab:** 128167, BPE (GPT-2 style), BOS=127958, EOS=3
- **Special tokens:** `<|eos|>`=127960 (EOT), `<|extra_0|>`=127962 (SEP)
- **Context window:** 8192 tokens native (262144 trained, but effective ~8K)
- **Languages:** 36 (en, zh, fr, pt, es, ja, ru, ar, ko, etc.)

### Prompt format
Raw instruction, no special tokens:

```
Translate the following text into Russian. Note that you should only output the translated result without any additional explanation:{source_text}
```

BOS token (`<|startoftext|>`=127958) is prepended automatically via `AddBos::Always`.

Stop tokens: 3 (EOS), 127960 (`<|eos|>`), 127962 (`<|extra_0|>`), 127958 (`<|startoftext|>`).

### Sampling
Currently greedy (temp=0) for debugging. Official params: temp=0.7, top_p=0.6, top_k=20, repetition_penalty=1.05.

### Processing
One chunk at a time (no batching). KV cache cleared per chunk. Context window: 1024 tokens.

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
- Translation quality still broken — model outputs garbled text with current prompt format
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
