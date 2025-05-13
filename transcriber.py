# Содержимое файла transcriber.py остается БЕЗ ИЗМЕНЕНИЙ
# по сравнению с предыдущим ответом, где мы исправили AttributeError.
# Убедитесь, что DEFAULT_STT_MODEL_NAME установлен в "base.en", "small.en" или "medium.en"
# для более быстрой работы на CPU, если GPU не используется.
import whisperx
import torch
import os
from types import SimpleNamespace # Для хранения моделей
import traceback

# Кэш для моделей
models_cache = SimpleNamespace(stt_model=None, diarization_model=None)

DEFAULT_STT_MODEL_NAME = "medium.en" # Попробуйте "base.en" или "small.en" для CPU

def load_stt_diarization_models(stt_model_name=DEFAULT_STT_MODEL_NAME, device="cpu", hf_token=None):
    global models_cache
    stt_compute_type = "float16" if device == "cuda" else "int8"
    if device == "cpu" and stt_compute_type == "int8":
        try:
            import ctranslate2
            print("ctranslate2 found. Using compute_type='int8' for STT on CPU.")
        except ImportError:
            print("ctranslate2 not found, WhisperX STT on CPU will use 'float32'.")
            stt_compute_type = "float32"

    # Перезагружаем модель STT, если имя или compute_type изменились
    # или если она вообще не была загружена.
    reload_stt = models_cache.stt_model is None
    if models_cache.stt_model is not None:
        # Пытаемся получить предыдущие параметры загрузки. Это не очень надежно,
        # т.к. whisperx.WhisperModel не хранит их явно в простом виде.
        # Проще всегда перезагружать, если параметры могут меняться,
        # или жестко задать их и не менять во время сессии.
        # Для простоты, если модель уже есть, не будем ее перезагружать,
        # предполагая, что device и stt_model_name не меняются динамически.
        # Если вы хотите динамически менять модель или compute_type, нужна более сложная логика кэширования.
        pass # Не перезагружаем, если уже есть (для упрощения)

    if reload_stt: # Только если нужно грузить
        print(f"Loading WhisperX STT model: {stt_model_name} on device: {device} with compute_type: {stt_compute_type}")
        models_cache.stt_model = whisperx.load_model(
            stt_model_name,
            device,
            compute_type=stt_compute_type,
            language="en"
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
                print("No Hugging Face token provided, attempting to load diarization model without it.")
                models_cache.diarization_model = whisperx.DiarizationPipeline(device=device)
            print("Diarization model loaded.")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            print("Ensure you accepted user agreements on Hugging Face for pyannote/speaker-diarization and pyannote/segmentation, and using an HF auth token (HF_TOKEN env var or use_auth_token).")
            models_cache.diarization_model = None
            raise
    else:
        print("Using cached Diarization model.")
    return models_cache.stt_model, models_cache.diarization_model


def transcribe_and_diarize(audio_path, language="en", batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Transcribe & Diarize using device: {device}")
    hf_auth_token = os.environ.get("HF_TOKEN")

    stt_model, diarization_model = load_stt_diarization_models(stt_model_name=DEFAULT_STT_MODEL_NAME, device=device, hf_token=hf_auth_token)

    if diarization_model is None:
        raise RuntimeError("Diarization model is not loaded. Cannot proceed.")

    print(f"Loading audio file: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    print("Transcribing audio...")
    result = stt_model.transcribe(audio, language=language, batch_size=batch_size)
    print("Transcription complete.")

    print("Aligning transcription...")
    try:
        lang_code_for_align = result.get("language", language)
        if not lang_code_for_align: lang_code_for_align = language
        align_model, metadata = whisperx.load_align_model(language_code=lang_code_for_align, device=device)
        segments_to_align = result.get("segments")
        if segments_to_align is None: segments_to_align = []
        aligned_result = whisperx.align(segments_to_align, align_model, metadata, audio, device, return_char_alignments=False)
        print("Alignment complete.")
        del align_model
        if device == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to align transcription - {e}. Proceeding with unaligned segments.")
        aligned_result = result

    print("Performing diarization...")
    try:
        diarize_segments = diarization_model(audio)
        segments_for_assignment = aligned_result.get("segments")
        if segments_for_assignment is None:
            segments_for_assignment = result.get("segments", [])
        
        # assign_word_speakers ожидает словарь с ключом "segments"
        input_for_assignment = {"segments": segments_for_assignment}
        final_segments_with_speakers = whisperx.assign_word_speakers(diarize_segments, input_for_assignment)
        print("Diarization complete.")
        return final_segments_with_speakers.get("segments", [])
    except Exception as e:
        print(f"Error during diarization or speaker assignment: {e}")
        print(f"Details: {traceback.format_exc()}")
        print("Falling back to transcription without speaker labels.")
        fallback_segments = aligned_result.get("segments", result.get("segments", []))
        for seg in fallback_segments:
            seg['speaker'] = 'SPEAKER_00'
        return fallback_segments