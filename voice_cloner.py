# Содержимое файла voice_cloner.py остается БЕЗ ИЗМЕНЕНИЙ
# по сравнению с предыдущим ответом.
from TTS.api import TTS
import torch
import os
import video_processor
from types import SimpleNamespace
import time
import ffmpeg # Для генерации тишины

tts_cache = SimpleNamespace(model=None)
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

def load_tts_model(model_name=XTTS_MODEL_NAME, device="cpu"):
    global tts_cache
    if tts_cache.model is None:
        print(f"Loading TTS model: {model_name} on device: {device}")
        try:
            use_cuda_flag = True if device == "cuda" else False
            tts_cache.model = TTS(model_name, gpu=use_cuda_flag)
            print("TTS model loaded.")
        except Exception as e:
            print(f"Error loading TTS model {model_name}: {e}")
            tts_cache.model = None; raise
    else:
        print("Using cached TTS model.")
    return tts_cache.model

def find_best_speaker_clip(segments, speaker_id, original_audio_path, temp_dir, min_dur=2.5, max_dur=8.0):
    best_clip_path = None
    print(f"Searching for suitable clip for {speaker_id}...")
    relevant_segments = [
        s for s in segments 
        if s.get('speaker') == speaker_id and (s.get('end', 0) - s.get('start', 0)) >= min_dur
    ]
    if not relevant_segments:
        print(f"Warning: No segments long enough found for speaker {speaker_id} (min_dur: {min_dur}s).")
        return None
    relevant_segments.sort(key=lambda s: (s.get('end', 0) - s.get('start', 0)), reverse=True)

    for segment in relevant_segments:
        start = segment.get('start')
        end = segment.get('end')
        actual_end = min(end, start + max_dur)
        actual_duration = actual_end - start
        
        if actual_duration >= min_dur:
            print(f"Found suitable segment for {speaker_id}: duration {actual_duration:.2f}s (from {start:.2f} to {actual_end:.2f})")
            clip_path = os.path.join(temp_dir, f"speaker_{speaker_id}_ref.wav")
            try:
                video_processor.extract_speaker_clip(original_audio_path, start, actual_end, clip_path)
                best_clip_path = clip_path
                break
            except Exception as e:
                print(f"Warning: Failed to extract clip for segment {start}-{end} for speaker {speaker_id}: {e}")
                continue
    if not best_clip_path:
        print(f"Warning: Could not find or extract a suitable reference clip for speaker {speaker_id}.")
    return best_clip_path


def synthesize_speech_segments(translated_segments, original_audio_path, temp_dir, target_language="ru", progress_callback=None):
    if not translated_segments: raise ValueError("No translated segments provided.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Synthesizer using device: {device}")
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
            print(f"Using default XTTS voice for speaker {speaker_id} as reference clip was not found/extracted.")
            speaker_reference_clips[speaker_id] = None

    segment_wav_files = []
    last_segment_end_time = 0.0
    valid_segments_for_synth = [s for s in translated_segments if s.get('translated_text','').strip() and s.get('start') is not None and s.get('end') is not None]
    total_segments_to_synth = len(valid_segments_for_synth)
    synthesized_count = 0
    print(f"Number of valid segments to synthesize: {total_segments_to_synth}")

    for i, segment in enumerate(translated_segments):
        start_time = segment.get('start')
        end_time = segment.get('end')
        text = segment.get('translated_text', '').strip()
        speaker_id = segment.get('speaker', 'SPEAKER_00')
        current_segment_is_valid_for_synth = bool(text and start_time is not None and end_time is not None)

        if start_time is not None and last_segment_end_time is not None:
            silence_duration = start_time - last_segment_end_time
            if silence_duration > 0.05:
                print(f"Adding silence: {silence_duration:.2f}s before segment with text '{text[:20]}...' (start: {start_time:.2f})")
                silence_wav = os.path.join(temp_dir, f"silence_{i}.wav")
                try:
                    (ffmpeg.input('anullsrc', format='lavfi', r=24000, channel_layout='mono')
                     .output(silence_wav, t=silence_duration, acodec='pcm_s16le').overwrite_output()
                     .run(capture_stdout=True, capture_stderr=True))
                    if os.path.exists(silence_wav): segment_wav_files.append(silence_wav)
                except Exception as e: print(f"Warning: Failed to generate silence {silence_wav}: {e}")

        if not current_segment_is_valid_for_synth:
            if end_time is not None: last_segment_end_time = end_time
            continue

        segment_wav_path = os.path.join(temp_dir, f"segment_{i}_{speaker_id}.wav")
        reference_wav = speaker_reference_clips.get(speaker_id)

        print(f"Synthesizing segment {synthesized_count + 1}/{total_segments_to_synth} for speaker {speaker_id} (text: '{text[:30]}...')...")
        if reference_wav: print(f"Using reference voice: {os.path.basename(reference_wav)}")
        else: print(f"Using default '{target_language}' XTTS voice.")
        
        segment_synth_start_time = time.time()
        try:
            tts_model.tts_to_file(
                text=text,
                file_path=segment_wav_path,
                speaker_wav=reference_wav,
                language=target_language,
                split_sentences=True,
                speed=1.0
            )
            segment_synth_end_time = time.time()
            print(f"Segment synthesized in {segment_synth_end_time - segment_synth_start_time:.2f}s")

            if os.path.exists(segment_wav_path) and os.path.getsize(segment_wav_path) > 0:
                segment_wav_files.append(segment_wav_path)
                synthesized_count += 1
                if progress_callback:
                    progress_callback(synthesized_count / total_segments_to_synth if total_segments_to_synth > 0 else 1.0)
            else: print(f"Warning: TTS failed to create file or created empty file {segment_wav_path}")
        except Exception as e:
            print(f"Error synthesizing segment {i+1} for speaker {speaker_id} (text: '{text[:30]}...'): {e}")

        if end_time is not None: last_segment_end_time = end_time

    if not segment_wav_files: raise RuntimeError("No audio segments were successfully synthesized.")
    final_audio_path = os.path.join(temp_dir, "final_dubbed_audio.wav")
    video_processor.merge_audio_segments(segment_wav_files, final_audio_path)
    return final_audio_path