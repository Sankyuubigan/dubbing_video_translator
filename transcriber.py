import whisperx
import torch
import os
from types import SimpleNamespace # Для хранения моделей

# Кэш для моделей
models_cache = SimpleNamespace(stt_model=None, diarization_model=None)

def load_stt_diarization_models(stt_model_name="large-v2", device="cpu", hf_token=None):
    """Загружает модели WhisperX STT и Diarization."""
    global models_cache
    # Определяем compute_type здесь, один раз, при загрузке модели STT
    stt_compute_type = "float16" if device == "cuda" else "int8"
    if device == "cpu" and stt_compute_type == "int8":
        try:
            import ctranslate2 # ctranslate2 нужен для int8 на CPU
            print("ctranslate2 found. Using compute_type='int8' for STT on CPU.")
        except ImportError:
            print("ctranslate2 not found, WhisperX STT on CPU will use 'float32'.")
            stt_compute_type = "float32" # Фолбэк, если нет ctranslate2 для int8

    if models_cache.stt_model is None:
        print(f"Loading WhisperX STT model: {stt_model_name} on device: {device} with compute_type: {stt_compute_type}")
        models_cache.stt_model = whisperx.load_model(
            stt_model_name,
            device,
            compute_type=stt_compute_type,
            # language="en" # Можно указать язык здесь, если он всегда один
            # download_root="path/to/your/model_cache" # Если хотите хранить модели не в дефолтном месте
        )
        print("STT model loaded.")
    else:
        print("Using cached STT model.")

    if models_cache.diarization_model is None:
        print("Loading Diarization model...")
        try:
            if hf_token:
                print("Using provided Hugging Face token for diarization model.")
                models_cache.diarization_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            else:
                print("No Hugging Face token provided, attempting to load diarization model without it (might fail for gated models).")
                models_cache.diarization_model = whisperx.DiarizationPipeline(device=device)
            print("Diarization model loaded.")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            print("You might need to accept user agreements on Hugging Face for pyannote/speaker-diarization and pyannote/segmentation and provide an HF auth token (use_auth_token='YOUR_TOKEN' or set HF_TOKEN environment variable).")
            models_cache.diarization_model = None
            raise
    else:
        print("Using cached Diarization model.")

    return models_cache.stt_model, models_cache.diarization_model


def transcribe_and_diarize(audio_path, language="en", batch_size=16):
    """
    Выполняет STT и Diarization с помощью WhisperX.
    Возвращает сегменты с информацией о спикерах.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_auth_token = os.environ.get("HF_TOKEN")

    # stt_model и diarization_model теперь загружаются с учетом compute_type
    stt_model, diarization_model = load_stt_diarization_models(device=device, hf_token=hf_auth_token)

    if diarization_model is None: # Проверяем, что модель диаризации загрузилась
        raise RuntimeError("Diarization model could not be loaded. Cannot proceed with speaker assignment.")

    print(f"Loading audio file: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    print("Transcribing audio...")
    # compute_type уже был учтен при загрузке stt_model через whisperx.load_model
    # Передавать его в stt_model.transcribe() не нужно и вызывает ошибку, если модель не faster-whisper
    result = stt_model.transcribe(audio, language=language, batch_size=batch_size)
    print("Transcription complete.")

    print("Aligning transcription...")
    try:
        # Убедимся, что язык из результата транскрипции корректный
        lang_code_for_align = result.get("language", language)
        if not lang_code_for_align: # Если язык не определен, пробуем английский
            print(f"Warning: Language code for alignment not found in transcription result. Defaulting to '{language}'.")
            lang_code_for_align = language

        align_model, metadata = whisperx.load_align_model(language_code=lang_code_for_align, device=device)
        # Передаем result["segments"], если они есть, иначе пустой список или сам result
        segments_to_align = result.get("segments")
        if segments_to_align is None:
            print("Warning: No 'segments' key in transcription result. Alignment might fail or be inaccurate.")
            # В этом случае, возможно, лучше пропустить выравнивание или использовать весь result, если API align это позволяет
            # Для простоты пока передадим пустой список, если сегментов нет, что приведет к пустому aligned_result
            segments_to_align = []

        aligned_result = whisperx.align(segments_to_align, align_model, metadata, audio, device, return_char_alignments=False)
        print("Alignment complete.")
        del align_model # Освобождаем память
        if device == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to align transcription - {e}. Speaker assignment might be less accurate.")
        aligned_result = result # Используем невыровненные сегменты для диаризации

    print("Performing diarization...")
    try:
        # Убедимся, что audio передается в правильном формате, если diarization_model ожидает путь к файлу
        # или numpy массив. whisperx.load_audio возвращает numpy массив, что должно быть ок.
        diarize_segments = diarization_model(audio) # или diarization_model(audio_path) если ожидает путь

        # Проверяем структуру aligned_result перед assign_word_speakers
        segments_for_speaker_assignment = aligned_result.get("segments")
        if segments_for_speaker_assignment is None:
            print("Warning: 'segments' key missing in aligned_result. Using original transcription segments for speaker assignment.")
            segments_for_speaker_assignment = result.get("segments", [])


        final_segments_with_speakers = whisperx.assign_word_speakers(diarize_segments, {"segments": segments_for_speaker_assignment})
        print("Diarization complete.")
        return final_segments_with_speakers.get("segments", []) # Возвращаем список сегментов или пустой список
    except Exception as e:
        print(f"Error during diarization or speaker assignment: {e}")
        print(f"Details: {traceback.format_exc()}")
        print("Falling back to transcription without speaker labels.")
        fallback_segments = aligned_result.get("segments", result.get("segments", []))
        for seg in fallback_segments:
            seg['speaker'] = 'SPEAKER_00'
        return fallback_segments