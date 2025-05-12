# Содержимое файла transcriber.py остается БЕЗ ИЗМЕНЕНИЙ
# по сравнению с предыдущим ответом.
# Включает: load_stt_diarization_models, transcribe_and_diarize
import whisperx
import torch
import os
from types import SimpleNamespace # Для хранения моделей

# Кэш для моделей
models_cache = SimpleNamespace(stt_model=None, diarization_model=None)

def load_stt_diarization_models(stt_model_name="large-v2", device="cpu", hf_token=None):
    """Загружает модели WhisperX STT и Diarization."""
    global models_cache
    if models_cache.stt_model is None:
        print(f"Loading WhisperX STT model: {stt_model_name} on device: {device}")
        # Указываем compute_type в зависимости от девайса для оптимизации
        compute_type = "float16" if device == "cuda" else "int8"
        # Убедитесь, что whisperx поддерживает int8 на CPU, если нет, используйте "float32"
        if device == "cpu" and compute_type == "int8":
             try:
                 # Проверка, что нужные библиотеки для int8 есть (может зависеть от версии whisperx/ctranslate2)
                 import ctranslate2
                 print("Using compute_type='int8' for STT on CPU.")
             except ImportError:
                 print("ctranslate2 not found, falling back to compute_type='float32' for STT on CPU.")
                 compute_type = "float32"

        models_cache.stt_model = whisperx.load_model(stt_model_name, device, compute_type=compute_type)
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

    stt_model, diarization_model = load_stt_diarization_models(device=device, hf_token=hf_auth_token)

    if diarization_model is None:
        raise RuntimeError("Diarization model could not be loaded. Cannot proceed with speaker assignment.")

    print(f"Loading audio file: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    print("Transcribing audio...")
    compute_type = "float16" if device == "cuda" else stt_model.model.compute_type # Используем compute_type загруженной модели
    result = stt_model.transcribe(audio, language=language, batch_size=batch_size) # compute_type передается в load_model
    print("Transcription complete.")

    print("Aligning transcription...")
    try:
        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
        print("Alignment complete.")
        del align_model # Освобождаем память
        if device == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to align transcription - {e}. Speaker assignment might be less accurate.")
        aligned_result = result # Используем невыровненные сегменты для диаризации

    print("Performing diarization...")
    try:
        diarize_segments = diarization_model(audio)
        # Убедимся, что aligned_result имеет нужную структуру
        if "segments" not in aligned_result:
             print("Warning: aligned_result missing 'segments' key. Using original result for speaker assignment.")
             final_segments_with_speakers = whisperx.assign_word_speakers(diarize_segments, result)
        else:
            final_segments_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        print("Diarization complete.")
        # print(final_segments_with_speakers["segments"]) # Debug
        return final_segments_with_speakers["segments"]
    except Exception as e:
        print(f"Error during diarization or speaker assignment: {e}")
        print("Falling back to transcription without speaker labels.")
        fallback_segments = aligned_result.get("segments", result.get("segments", []))
        for seg in fallback_segments:
            seg['speaker'] = 'SPEAKER_00'
        return fallback_segments