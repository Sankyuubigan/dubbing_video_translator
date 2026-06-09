# DubVidTra2

Локальный перевод видео с субтитрами. STT (Whisper/Parakeet) + Diarization (sherpa-onnx) + Перевод (Hy-MT2 GGUF).

## Тестирование

### Быстрый тест пайплайна (без Tauri окна)

```powershell
cd src-tauri
cargo run --bin test-pipeline --release -- --video "test/for_test.mp4"
```

- **Не открывает Tauri окно** — только консоль
- Автоматически завершается после теста (exit code 0/1)
- Логи пишутся в `last_logs.log` и в stderr
- Дефолтное видео: `test/for_test.mp4` из корня проекта

### Полный скрипт

```powershell
.\run_test.ps1                     # build + pipeline test
.\run_test.ps1 -Video "путь"       # свой файл
.\run_test.ps1 -All                # + audio extraction test
.\run_test.ps1 -AudioOnly          # только audio extraction
```

### Unit-тесты

```powershell
cd src-tauri
cargo test -- --nocapture
```

### GUI (полное приложение)

```powershell
cd src-tauri
cargo tauri dev
```

## Структура

- `src-tauri/src/` — Rust бэкенд (VAD, диаризация, STT, перевод)
- `src/` — React/Typescript фронтенд
- `models/` — ONNX и GGUF модели (VAD, диаризация, перевод)
- `test/` — тестовые видео
- `ffmpeg/` — ffmpeg для маскинга субтитров
