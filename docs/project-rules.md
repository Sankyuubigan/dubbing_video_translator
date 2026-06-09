# Правила проекта DubVidTra2

## Стек
- **Бэкенд:** Rust + Tauri 2.0
- **Фронтенд:** React (Tauri default)
- **Нейронки:** ort (ONNX), whisper-rs, llama-cpp-2
- **Медиа:** FFmpeg (внешняя зависимость)

## Архитектурные правила
1. Модули не импортят друг друга — только через comm.rs.
2. Каждый модуль — отдельная папка в src/.
3. Все модели выгружаются после использования (drop(model) перед загрузкой следующей).

## Соглашения по коду
- Комментарии на русском (личный проект).
- Публичные функции модуля — только те, что указаны в архитектуре.
- Ошибки пробрасывать через 	hiserror / nyhow.

## GUI (Tauri)
- Драг-н-дроп видео в основное окно.
- Кнопка "Выбрать GGUF-модель" — открывает файловый диалог для .gguf.
- Поле "Выходной формат" — по умолчанию совпадает с входным.
- Прогресс-бар: отображает стадию (VAD, STT, перевод...) и проценты.
- По окончании — кнопка "Открыть папку с результатом".

## Конфигурация
- Файл: ~/.dubvidtra2/config.toml
- Поля: gguf_model_path, ad_model_path, whisper_model_path, output_format

## Тестирование
- **Unit-тесты:** каждый модуль тестируется изолированно.
- **Интеграционный тест:** полный прогон пайплайна на or_test.mp4.
- cargo test — запуск всех тестов.
- cargo test -- --nocapture — с выводом логов.

## Зависимости (Cargo.toml)
`	oml
[dependencies]
tauri = { version = "2.0", features = ["dialog"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
thiserror = "1.0"
ort = "2.0"
whisper-rs = "0.11"
llama-cpp-2 = "0.3"
linfa = "0.7"
linfa-clustering = "0.7"
hound = "3.5"
toml = "0.8"
`

## Первый запуск
1. Установить FFmpeg (должен быть в PATH).
2. Скачать модели:
   - Silero VAD v5: https://github.com/snakers4/silero-vad
   - Whisper GGML: ggml-large-v3-turbo.bin
   - Qwen-2.5-7B GGUF: с HuggingFace
3. Выбрать GGUF-файл в GUI.
4. Кинуть видео → нажать "Обработать".
