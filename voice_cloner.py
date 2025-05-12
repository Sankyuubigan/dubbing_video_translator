# Содержимое файла voice_cloner.py остается БЕЗ ИЗМЕНЕНИЙ
# по сравнению с предыдущим ответом.
# Включает: load_tts_model, find_best_speaker_clip, synthesize_speech_segments
from TTS.api import TTS
import torch
import os
import video_processor # Импортируем наш модуль для работы с аудио
from types import SimpleNamespace
import time

# Кэш для модели TTS
tts_cache = SimpleNamespace(model=None)
# Модель для клонирования голоса
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

def load_tts_model(model_name=XTTS_MODEL_NAME, device="cpu"):
    """Загружает модель Coqui TTS."""
    global tts_cache
    if tts_cache.model is None:
        print(f"Loading TTS model: {model_name} on device: {device}")
        try:
            tts_cache.model = TTS(model_name, gpu=(device == "cuda"))
            print("TTS model loaded.")
        except Exception as e:
            print(f"Error loading TTS model {model_name}: {e}")
            tts_cache.model = None
            raise
    else:
        print("Using cached TTS model.")
    return tts_cache.model

def find_best_speaker_clip(segments, speaker_id, original_audio_path, temp_dir, min_dur=2.5, max_dur=8.0):
    """Находит наиболее подходящий сегмент для клонирования голоса спикера."""
    best_clip_path = None
    print(f"Searching for suitable clip for {speaker_id}...")
    relevant_segments = [s for s in segments if s.get('speaker') == speaker_id]
    if not relevant_segments:
        print(f"Warning: No segments found for speaker {speaker_id}.")
        return None
    relevant_segments.sort(key=lambda s: s.get('end', 0) - s.get('start', 0), reverse=True)

    for segment in relevant_segments:
        start = segment.get('start')
        end = segment.get('end')
        if start is None or end is None: continue
        duration = end - start
        if duration >= min_dur:
            actual_end = min(end, start + max_dur)
            actual_duration = actual_end - start
            if actual_duration >= min_dur:
                print(f"Found suitable segment for {speaker_id}: duration {actual_duration:.2f}s")
                clip_path = os.path.join(temp_dir, f"speaker_{speaker_id}_ref.wav")
                try:
                    video_processor.extract_speaker_clip(original_audio_path, start, actual_end, clip_path)
                    best_clip_path = clip_path
                    break
                except Exception as e:
                    print(f"Warning: Failed to extract clip for segment {start}-{end} for speaker {speaker_id}: {e}")
                    continue
    if not best_clip_path:
        print(f"Warning: Could not find or extract a suitable reference clip (min_dur={min_dur}s) for speaker {speaker_id}.")
    return best_clip_path

def synthesize_speech_segments(translated_segments, original_audio_path, temp_dir, target_language="ru"):
    """
    Синтезирует речь для каждого сегмента, клонируя голос.
    Возвращает путь к объединенному аудиофайлу.
    """
    if not translated_segments: raise ValueError("No translated segments provided.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = load_tts_model(device=device)
    if tts_model is None: raise RuntimeError("TTS model could not be loaded.")

    print(f"Synthesizing speech for {len(translated_segments)} segments...")
    unique_speakers = sorted(list(set(seg.get('speaker', 'SPEAKER_00') for seg in translated_segments)))
    speaker_reference_clips = {}
    print(f"Found unique speakers: {unique_speakers}")
    for speaker_id in unique_speakers:
        clip_path = find_best_speaker_clip(translated_segments, speaker_id, original_audio_path, temp_dir)
        if clip_path and os.path.exists(clip_path):
            speaker_reference_clips[speaker_id] = clip_path
        else:
            print(f"Using default voice for speaker {speaker_id}.")
            speaker_reference_clips[speaker_id] = None

    segment_wav_files = []
    last_segment_end_time = 0.0
    total_segments = len(translated_segments) # Для примерного прогресса синтеза

    for i, segment in enumerate(translated_segments):
        start_time = segment.get('start')
        end_time = segment.get('end')
        text = segment.get('translated_text', '').strip()
        speaker_id = segment.get('speaker', 'SPEAKER_00')

        if start_time is None or end_time is None or not text:
            print(f"Skipping segment {i+1} due to missing data or empty text.")
            continue

        # Добавляем тишину
        silence_duration = start_time - last_segment_end_time
        if silence_duration > 0.05:
            print(f"Adding silence: {silence_duration:.2f}s before segment {i+1}")
            silence_wav = os.path.join(temp_dir, f"silence_{i}.wav")
            try:
                (ffmpeg.input('anullsrc', format='lavfi', r=24000)
                 .output(silence_wav, t=silence_duration).overwrite_output()
                 .run(capture_stdout=True, capture_stderr=True))
                segment_wav_files.append(silence_wav)
            except Exception as e: print(f"Warning: Failed to generate silence {silence_wav}: {e}")

        segment_wav_path = os.path.join(temp_dir, f"segment_{i}_{speaker_id}.wav")
        reference_wav = speaker_reference_clips.get(speaker_id)

        print(f"Synthesizing segment {i+1}/{total_segments} for speaker {speaker_id}...")
        if reference_wav: print(f"Using reference voice: {os.path.basename(reference_wav)}")
        else: print(f"Using default '{target_language}' voice.")

        try:
            # XTTS может быть медленным, особенно на CPU
            tts_model.tts_to_file(
                text=text,
                file_path=segment_wav_path,
                speaker_wav=reference_wav,
                language=target_language,
                split_sentences=True,
                speed=1.0
            )
            if os.path.exists(segment_wav_path): segment_wav_files.append(segment_wav_path)
            else: print(f"Warning: TTS failed to create file {segment_wav_path}")
        except Exception as e:
            print(f"Error synthesizing segment {i+1}: {e}")

        last_segment_end_time = end_time

    if not segment_wav_files: raise RuntimeError("No audio segments synthesized.")
    final_audio_path = os.path.join(temp_dir, "final_translated_audio.wav")
    video_processor.merge_audio_segments(segment_wav_files, final_audio_path)
    return final_audio_path
