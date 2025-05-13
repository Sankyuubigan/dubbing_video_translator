import os
import torch
import torchaudio
from TTS.api import TTS
import time
# import tempfile # Не используется напрямую здесь
import soundfile as sf # Используем soundfile для чтения длительности
import video_processor # Импортируем для вызова merge_audio_segments
from tqdm import tqdm # Для прогресс бара в консоли (опционально)

# --- Кэш для моделей TTS ---
class TTSCache:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = None

tts_cache = TTSCache()

def load_tts_model(model_name="tts_models/multilingual/multi-dataset/xtts_v2", device="cuda"):
    """Загружает модель TTS или возвращает из кэша."""
    # use_cuda = device == "cuda" # Не используется напрямую
    if tts_cache.model is None or tts_cache.model_name != model_name or tts_cache.device != device:
        print(f"Loading TTS model: {model_name} on device: {device}")
        try:
            tts_cache.model = TTS(model_name).to(device)
            tts_cache.model_name = model_name
            tts_cache.device = device
            print("TTS model loaded successfully.")
        except Exception as e:
            print(f"Error loading TTS model {model_name}: {e}")
            try:
                 print(f"Trying to load TTS model {model_name} without initial .to(device)...")
                 tts_cache.model = TTS(model_name)
                 tts_cache.model.to(device)
                 tts_cache.model_name = model_name
                 tts_cache.device = device
                 print("TTS model loaded successfully (moved to device after init).")
            except Exception as e2:
                 print(f"FATAL: Failed to load TTS model {model_name} even with fallback: {e2}")
                 tts_cache.model = None
                 raise

    # else: # Убираем этот print, т.к. он частый
    #     print("Using cached TTS model.")
    return tts_cache.model


def _get_or_create_speaker_wav(speaker_id, segment, base_audio_path, temp_dir, min_duration=2.0, max_duration=15.0):
    """
    Извлекает или находит подходящий аудиофрагмент для клонирования голоса спикера.
    Возвращает путь к WAV файлу.
    """
    speaker_wav_dir = os.path.join(temp_dir, "speaker_wavs")
    os.makedirs(speaker_wav_dir, exist_ok=True)
    speaker_ref_path = os.path.join(speaker_wav_dir, f"{speaker_id}_ref.wav")

    if os.path.exists(speaker_ref_path):
        return speaker_ref_path

    start_time = segment.get('start')
    end_time = segment.get('end')
    duration = end_time - start_time if start_time is not None and end_time is not None else 0

    # print(f"Attempting to create reference for {speaker_id} from segment [{start_time:.2f}s - {end_time:.2f}s] (duration: {duration:.2f}s)") # Слишком много логов

    if duration >= min_duration:
        actual_end_time = min(end_time, start_time + max_duration)
        # actual_duration = actual_end_time - start_time # Не используется

        # print(f"Extracting reference audio [{start_time:.2f}s - {actual_end_time:.2f}s] (target duration: {actual_duration:.2f}s) to {speaker_ref_path}") # Много логов
        try:
            if not os.path.exists(base_audio_path):
                 raise FileNotFoundError(f"Base audio path for speaker extraction not found: {base_audio_path}")

            waveform, sample_rate = torchaudio.load(base_audio_path)
            start_frame = int(start_time * sample_rate)
            end_frame = int(actual_end_time * sample_rate)

            if start_frame >= waveform.shape[1] or end_frame > waveform.shape[1] or start_frame >= end_frame:
                 print(f"Warning: Invalid frame indices for speaker {speaker_id}. Start: {start_frame}, End: {end_frame}, Total frames: {waveform.shape[1]}. Cannot extract reference from this segment.")
                 return None

            segment_waveform = waveform[:, start_frame:end_frame]
            torchaudio.save(speaker_ref_path, segment_waveform, sample_rate)

            try:
                info = sf.info(speaker_ref_path)
                if info.duration < min_duration / 2:
                     print(f"Warning: Extracted reference for {speaker_id} is too short ({info.duration:.2f}s). May lead to poor cloning.")
                elif info.duration == 0:
                     print(f"Warning: Extracted reference for {speaker_id} has zero duration. Deleting.")
                     os.remove(speaker_ref_path)
                     return None
            except Exception as sf_err:
                 print(f"Warning: Could not verify duration of {speaker_ref_path}: {sf_err}")

            print(f"Reference for {speaker_id} created: {os.path.basename(speaker_ref_path)}") # Сокращенный лог
            return speaker_ref_path

        except FileNotFoundError as fnf_err:
             print(f"Error extracting speaker wav: {fnf_err}")
             return None
        except Exception as e:
            print(f"Error extracting speaker wav for {speaker_id}: {e}")
            if os.path.exists(speaker_ref_path):
                try: os.remove(speaker_ref_path)
                except OSError: pass
            return None
    # else: # Этот лог тоже можно убрать, если он частый
    #     print(f"Segment duration ({duration:.2f}s) too short for reference (min: {min_duration:.2f}s). Skipping for {speaker_id}.")
    return None


def synthesize_speech_segments(segments, original_audio_path, temp_dir, language='ru', progress_callback=None):
    """
    Synthesizes speech for each segment using XTTS for voice cloning.
    Manages speaker references and merges final audio.
    """
    tts_model = load_tts_model()
    if tts_model is None:
        raise RuntimeError("TTS model could not be loaded.")

    output_segments_dir = os.path.join(temp_dir, "tts_segments")
    os.makedirs(output_segments_dir, exist_ok=True)

    segment_wav_files = []
    speaker_references = {}
    first_valid_speaker_ref = None

    total_segments = len(segments)
    print(f"Starting voice synthesis for {total_segments} segments...")

    # Используем tqdm только если не GUI, чтобы не дублировать прогресс
    # Для GUI progress_callback будет не None
    iterable_segments = segments if progress_callback else tqdm(segments, desc="Synthesizing Segments")

    for i, segment in enumerate(iterable_segments):
        # start_time_loop = time.time() # Не используется
        # --- ИЗМЕНЕНИЕ КЛЮЧА ДЛЯ ПЕРЕВЕДЕННОГО ТЕКСТА ---
        text = segment.get('translated_text', segment.get('text', '')) # Используем 'translated_text'
        speaker_id = segment.get('speaker', 'SPEAKER_UNKNOWN')
        # segment_start = segment.get('start', 0) # Не используется

        if not text.strip(): # Проверяем, что текст не пустой после strip()
            # print(f"Warning: Skipping segment {i+1}/{total_segments} due to empty text.") # Можно убрать этот лог
            continue

        segment_filename = os.path.join(output_segments_dir, f"segment_{i:04d}_{speaker_id}.wav")

        speaker_wav_path = speaker_references.get(speaker_id)
        if speaker_wav_path is None:
            speaker_wav_path = _get_or_create_speaker_wav(speaker_id, segment, original_audio_path, temp_dir)
            if speaker_wav_path:
                speaker_references[speaker_id] = speaker_wav_path
                if first_valid_speaker_ref is None:
                    first_valid_speaker_ref = speaker_wav_path
            else:
                if first_valid_speaker_ref:
                     print(f"Warning: Could not create reference for {speaker_id}. Using first available reference: {os.path.basename(first_valid_speaker_ref)}")
                     speaker_wav_path = first_valid_speaker_ref
                else:
                     print(f"FATAL ERROR: No speaker reference audio available. Cannot clone voice for segment {i+1} ({speaker_id}). Using default voice if possible, or skipping.")
                     # Вместо падения, попробуем синтезировать без speaker_wav (используется голос по умолчанию модели)
                     speaker_wav_path = None # Явно None, чтобы TTS использовал свой дефолтный голос
                     # Если и это не сработает, TTS сам вызовет ошибку, которую мы перехватим ниже.

        try:
            # Уменьшаем количество логов при синтезе
            if i % 20 == 0 or i == total_segments -1 : # Логируем каждый 20-й сегмент и последний
                 log_text = f"Synthesizing segment {i+1}/{total_segments} for {speaker_id} (text: '{text[:30].strip()}...')"
                 if speaker_wav_path:
                     log_text += f" Ref: {os.path.basename(speaker_wav_path)}"
                 else:
                     log_text += " (using default model voice)"
                 print(log_text)

            tts_model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav_path,
                language=language,
                file_path=segment_filename,
            )

            if os.path.exists(segment_filename) and os.path.getsize(segment_filename) > 0:
                 segment_wav_files.append(segment_filename)
                 # duration_loop = time.time() - start_time_loop # Не используется
                 # print(f"Segment {i+1} synthesized successfully in {duration_loop:.2f}s.") # Убираем этот частый лог
            else:
                 print(f"Warning: Synthesized file for segment {i+1} is missing or empty: {segment_filename}")

        except Exception as e:
            print(f"Error synthesizing segment {i+1} for speaker {speaker_id} (text: '{text[:30].strip()}...'): {e}")
            continue

        if progress_callback:
            progress_callback((i + 1) / total_segments)

    if not segment_wav_files:
        raise RuntimeError("No audio segments were successfully synthesized.")

    final_audio_path = os.path.join(temp_dir, "dubbed_full_audio.wav")
    # print(f"Merging {len(segment_wav_files)} synthesized segments into {final_audio_path}...") # Убираем, т.к. merge_audio_segments логирует
    video_processor.merge_audio_segments(segment_wav_files, final_audio_path)
    print("Final dubbed audio merged successfully.")
    return final_audio_path
