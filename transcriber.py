import whisperx
import torch
import os
from types import SimpleNamespace # Для хранения моделей
import traceback

# Кэш для моделей
models_cache = SimpleNamespace(stt_model=None, diarization_model=None)

# --- ИЗМЕНЕНИЕ: Модель WhisperX по умолчанию ---
DEFAULT_STT_MODEL_NAME = "medium.en" # Используем "medium.en" для баланса

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

    # Проверяем, нужно ли перезагружать модель STT
    # (если еще не загружена, или если изменилось имя модели, или тип вычислений)
    # Для простоты, если модель уже есть, и параметры совпадают, не перезагружаем.
    # Это не идеально, если compute_type или device меняются динамически без смены stt_model_name.
    current_stt_params_match = False
    if models_cache.stt_model:
        # Пробуем получить параметры из загруженной модели, если возможно.
        # whisperx.WhisperModel может не хранить их явно и легкодоступно.
        # Для простоты будем считать, что если модель есть, параметры те же.
        # Если нужна смена параметров "на лету", логику кэширования нужно усложнить.
        if hasattr(models_cache.stt_model, 'model_name_or_path') and models_cache.stt_model.model_name_or_path == stt_model_name:
             # Здесь сложно проверить compute_type и device загруженной модели без хаков.
             # Поэтому, если имя совпадает, предполагаем, что все ок.
             current_stt_params_match = True


    if not current_stt_params_match:
        print(f"Loading WhisperX STT model: {stt_model_name} on device: {device} with compute_type: {stt_compute_type}")
        models_cache.stt_model = whisperx.load_model(
            stt_model_name,
            device,
            compute_type=stt_compute_type,
            language="en" # Для .en моделей это помогает
        )
        # Сохраняем параметры, с которыми модель была загружена, для будущих проверок
        if models_cache.stt_model: # Добавляем атрибуты после успешной загрузки
            models_cache.stt_model.loaded_model_name = stt_model_name
            models_cache.stt_model.loaded_compute_type = stt_compute_type
            models_cache.stt_model.loaded_device = device
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

    if diarization_model is None: # Проверяем еще раз на всякий случай
        raise RuntimeError("Diarization model is not loaded. Cannot proceed.")

    print(f"Loading audio file: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    print("Transcribing audio...")
    # compute_type уже был учтен при загрузке stt_model
    result = stt_model.transcribe(audio, language=language, batch_size=batch_size)
    print("Transcription complete.")

    print("Aligning transcription...")
    try:
        lang_code_for_align = result.get("language", language)
        if not lang_code_for_align: lang_code_for_align = language # Фолбэк
        align_model, metadata = whisperx.load_align_model(language_code=lang_code_for_align, device=device)
        segments_to_align = result.get("segments")
        if segments_to_align is None:
            print("Warning: No 'segments' in transcription result for alignment.")
            segments_to_align = [] # Пустой список, если сегментов нет
        
        # Проверяем, что segments_to_align - это список словарей
        if not isinstance(segments_to_align, list) or (segments_to_align and not isinstance(segments_to_align[0], dict)):
            print(f"Warning: segments_to_align is not a list of dicts. Type: {type(segments_to_align)}. Using empty list for alignment.")
            segments_to_align = []

        aligned_result = whisperx.align(segments_to_align, align_model, metadata, audio, device, return_char_alignments=False)
        print("Alignment complete.")
        del align_model
        if device == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to align transcription - {e}. Proceeding with unaligned segments for diarization.")
        aligned_result = result # Используем оригинальный результат транскрипции

    print("Performing diarization...")
    try:
        diarize_segments = diarization_model(audio, min_speakers=1, max_speakers=None) # Можно указать min/max speakers
        
        segments_for_assignment = aligned_result.get("segments")
        if segments_for_assignment is None:
            print("Warning: 'segments' key missing in aligned_result for speaker assignment. Using original transcription segments.")
            segments_for_assignment = result.get("segments", [])
        
        # Убедимся, что segments_for_assignment - это список словарей
        if not isinstance(segments_for_assignment, list) or (segments_for_assignment and not isinstance(segments_for_assignment[0], dict)):
            print(f"Warning: segments_for_assignment for speaker assignment is not a list of dicts. Type: {type(segments_for_assignment)}. Using empty list.")
            segments_for_assignment = []
            
        input_for_assignment = {"segments": segments_for_assignment}
        final_segments_with_speakers = whisperx.assign_word_speakers(diarize_segments, input_for_assignment)
        print("Diarization complete.")
        return final_segments_with_speakers.get("segments", [])
    except Exception as e:
        print(f"Error during diarization or speaker assignment: {e}")
        print(f"Details: {traceback.format_exc()}")
        print("Falling back to transcription without speaker labels (all SPEAKER_00).")
        fallback_segments = aligned_result.get("segments", result.get("segments", []))
        for seg in fallback_segments:
            seg['speaker'] = 'SPEAKER_00'
        return fallback_segments