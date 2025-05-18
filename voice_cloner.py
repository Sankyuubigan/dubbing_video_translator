import os
import torch
import torchaudio
from TTS.api import TTS
import time
import soundfile as sf 
import video_processor 
from tqdm import tqdm
import ffmpeg
import re 
import pandas as pd 
import shutil 
import traceback 

# --- Кэш для моделей TTS ---
class TTSCache:
    def __init__(self):
        self.model = None; self.model_name = None; self.device = None
tts_cache = TTSCache()

XTTS_RU_MAX_CHARS = 180 

def load_tts_model(model_name="tts_models/multilingual/multi-dataset/xtts_v2", device="cuda"): # ... (без изменений) ...
    global tts_cache
    if tts_cache.model is None or tts_cache.model_name != model_name or tts_cache.device != device:
        print(f"Loading TTS model: {model_name} on device: {device}")
        try:
            tts_cache.model = TTS(model_name).to(device); tts_cache.model_name = model_name; tts_cache.device = device
            print("TTS model loaded successfully.")
        except Exception as e:
            print(f"Error loading TTS model {model_name}: {e}")
            try:
                 print(f"Trying to load TTS model {model_name} without initial .to(device)...")
                 tts_cache.model = TTS(model_name); tts_cache.model.to(device)
                 tts_cache.model_name = model_name; tts_cache.device = device
                 print("TTS model loaded successfully (moved to device after init).")
            except Exception as e2:
                 print(f"FATAL: Failed to load TTS model {model_name} even with fallback: {e2}")
                 tts_cache.model = None; raise
    return tts_cache.model

def _split_text_for_tts(text, max_length=XTTS_RU_MAX_CHARS): # ... (без изменений) ...
    if len(text) <= max_length: return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')); chunks = []; current_chunk = ""
    for sentence in sentences:
        if not sentence.strip(): continue
        if len(current_chunk) + len(sentence) + 1 <= max_length: 
            if current_chunk: current_chunk += " " + sentence
            else: current_chunk = sentence
        else:
            if current_chunk: chunks.append(current_chunk)
            if len(sentence) > max_length:
                temp_sub_chunk = ""
                words = sentence.split(' ')
                for word_idx, word in enumerate(words):
                    if len(temp_sub_chunk) + len(word) + 1 <= max_length:
                        if temp_sub_chunk: temp_sub_chunk += " " + word
                        else: temp_sub_chunk = word
                    else:
                        if temp_sub_chunk: chunks.append(temp_sub_chunk)
                        if len(word) > max_length: 
                            for i_w in range(0, len(word), max_length): chunks.append(word[i_w:i_w+max_length]) # Исправлена переменная цикла
                            temp_sub_chunk = ""
                        else: temp_sub_chunk = word
                if temp_sub_chunk: chunks.append(temp_sub_chunk)
                current_chunk = "" 
            else: current_chunk = sentence 
    if current_chunk: chunks.append(current_chunk)
    final_chunks = []
    for ch in chunks:
        if len(ch) > max_length:
            for i_ch in range(0, len(ch), max_length): final_chunks.append(ch[i_ch:i_ch+max_length]) # Исправлена переменная цикла
        elif ch.strip(): final_chunks.append(ch)
    if not final_chunks and text.strip(): 
        print(f"Warning: Text splitting failed to produce any chunks for text of length {len(text)}. Using original text split by max_length as fallback.")
        for i_f in range(0, len(text), max_length): final_chunks.append(text[i_f:i_f+max_length]) # Исправлена переменная цикла
    return [c for c in final_chunks if c.strip()] if final_chunks else ([text[:max_length]] if text.strip() else [])

def _get_or_create_speaker_wav(speaker_id, segment_for_ref_timing, base_audio_path, temp_dir, diarization_df=None, min_duration=2.5, max_duration=12.0): # ... (без изменений) ...
    speaker_wav_dir = os.path.join(temp_dir, "speaker_wavs"); os.makedirs(speaker_wav_dir, exist_ok=True)
    speaker_ref_path = os.path.join(speaker_wav_dir, f"{speaker_id}_ref.wav")
    if os.path.exists(speaker_ref_path) and os.path.getsize(speaker_ref_path) > 1000:
        try:
            info = sf.info(speaker_ref_path)
            if info.duration >= 0.5: return speaker_ref_path
        except Exception: pass
    ref_start, ref_end = None, None
    if diarization_df is not None and not diarization_df.empty and speaker_id in diarization_df['speaker'].unique():
        speaker_segments_df = diarization_df[diarization_df['speaker'] == speaker_id].copy()
        speaker_segments_df['duration'] = speaker_segments_df['end'] - speaker_segments_df['start']
        ideal_min_duration = 4.0; ideal_max_duration = 8.0 
        suitable_segments = speaker_segments_df[(speaker_segments_df['duration'] >= ideal_min_duration) & (speaker_segments_df['duration'] <= ideal_max_duration)]
        if not suitable_segments.empty:
            best_segment_row = suitable_segments.sample(1).iloc[0] if len(suitable_segments) > 1 else suitable_segments.iloc[0]
            ref_start, ref_end = best_segment_row['start'], best_segment_row['end']
        else:
            longer_segments = speaker_segments_df[speaker_segments_df['duration'] >= min_duration]
            if not longer_segments.empty:
                best_segment_row = longer_segments.sort_values(by='duration', ascending=False).iloc[0]
                ref_start = best_segment_row['start']
                ref_end = min(best_segment_row['end'], best_segment_row['start'] + max_duration)
    if ref_start is None and segment_for_ref_timing: 
        srt_start_time = segment_for_ref_timing.get('start'); srt_end_time = segment_for_ref_timing.get('end')
        if srt_start_time is not None and srt_end_time is not None:
            srt_duration = srt_end_time - srt_start_time
            if srt_duration >= min_duration: ref_start = srt_start_time; ref_end = min(srt_end_time, srt_start_time + max_duration)
    if ref_start is None or ref_end is None or (ref_end - ref_start) < 0.5: return None 
    try:
        if not os.path.exists(base_audio_path): raise FileNotFoundError(f"Base audio for speaker ref not found: {base_audio_path}")
        (ffmpeg.input(base_audio_path, ss=ref_start, to=ref_end).output(speaker_ref_path, ar=24000, ac=1, acodec='pcm_s16le').overwrite_output().run(capture_stdout=True, capture_stderr=True))
        if not os.path.exists(speaker_ref_path) or os.path.getsize(speaker_ref_path) == 0:
             if os.path.exists(speaker_ref_path): os.remove(speaker_ref_path); return None
        info = sf.info(speaker_ref_path)
        if info.duration < 0.5: os.remove(speaker_ref_path); return None
        return speaker_ref_path
    except Exception as e: 
        print(f"    Error extracting speaker reference for {speaker_id}: {e}")
        if os.path.exists(speaker_ref_path):
            try: os.remove(speaker_ref_path)
            except OSError: pass
        return None

def _apply_tempo_and_trim_pad(input_path, output_path, samplerate,
                              target_duration,
                              atempo_speed_limit_for_quality=1.8, 
                              log_prefix="    ",
                              radical_trim_pad_only=False): # ... (без изменений) ...
    base_input_name = os.path.basename(input_path)
    try:
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono').output(output_path, t=max(0.01, target_duration), acodec='pcm_s16le').overwrite_output().run(capture_stdout=True, capture_stderr=True))
            return True, max(0.01, target_duration)
        input_info = sf.info(input_path); current_duration = input_info.duration
        if current_duration <= 0.01 and target_duration <= 0.01 : 
             if input_path != output_path: shutil.copy(input_path, output_path)
             return True, current_duration
        elif target_duration <= 0.01: 
            (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono').output(output_path, t=0.01, acodec='pcm_s16le').overwrite_output().run(capture_stdout=True, capture_stderr=True))
            return True, 0.01
        duration_after_processing = current_duration; processed_audio_path = input_path
        if not radical_trim_pad_only: 
            desired_speed_factor = current_duration / target_duration if target_duration > 0.01 else float('inf')
            effective_atempo_speed = max(0.5, min(atempo_speed_limit_for_quality, desired_speed_factor))
            temp_atempo_output = os.path.join(os.path.dirname(output_path), f"temp_atempo_{base_input_name}")
            if abs(effective_atempo_speed - 1.0) > 0.01: 
                atempo_success, duration_after_atempo_val = _apply_tempo_adjustment_ffmpeg_internal(input_path, temp_atempo_output, effective_atempo_speed, samplerate, log_prefix=log_prefix + "  (atempo) ")
                if atempo_success: processed_audio_path = temp_atempo_output; duration_after_processing = duration_after_atempo_val
                else: 
                    if os.path.exists(temp_atempo_output): os.remove(temp_atempo_output)
            else: 
                if input_path != temp_atempo_output: shutil.copy(input_path, temp_atempo_output)
                else: pass 
                processed_audio_path = temp_atempo_output
        if not os.path.exists(processed_audio_path) or os.path.getsize(processed_audio_path) == 0:
            if output_path != input_path and os.path.exists(input_path): shutil.copy(input_path, output_path)
            elif output_path == input_path and not os.path.exists(output_path): 
                (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono').output(output_path, t=max(0.01, target_duration), acodec='pcm_s16le').overwrite_output().run(capture_stdout=True, capture_stderr=True))
                return True, max(0.01, target_duration)
            return False, current_duration
        final_stream_input_path_for_ffmpeg = processed_audio_path; temp_ffmpeg_input_if_needed = None
        if processed_audio_path == output_path:
             temp_ffmpeg_input_if_needed = os.path.join(os.path.dirname(output_path), f"temp_final_ff_in_{base_input_name}")
             shutil.copy(processed_audio_path, temp_ffmpeg_input_if_needed); final_stream_input_path_for_ffmpeg = temp_ffmpeg_input_if_needed
        final_stream = ffmpeg.input(final_stream_input_path_for_ffmpeg)
        duration_diff = target_duration - duration_after_processing 
        if abs(duration_diff) >= 0.005 and target_duration > 0.01:
            if duration_diff > 0: 
                pad_seconds = round(duration_diff, 3)
                final_stream = ffmpeg.filter(final_stream, 'apad', pad_dur=f'{pad_seconds}s')
                if log_prefix: print(f"{log_prefix}Padding '{base_input_name}' by {pad_seconds:.3f}s (dur_before_pad: {duration_after_processing:.3f}s, target: {target_duration:.3f}s)")
            elif duration_diff < 0: 
                final_stream = ffmpeg.filter(final_stream, 'atrim', start='0', end=str(round(target_duration,3)))
                if log_prefix: print(f"{log_prefix}Trimming '{base_input_name}' to {target_duration:.3f}s (dur_before_trim: {duration_after_processing:.3f}s)")
            ffmpeg.output(final_stream, output_path, acodec='pcm_s16le', ar=samplerate).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        else: 
            if final_stream_input_path_for_ffmpeg != output_path: shutil.copy(final_stream_input_path_for_ffmpeg, output_path)
        if processed_audio_path != input_path and os.path.exists(processed_audio_path): os.remove(processed_audio_path)
        if temp_ffmpeg_input_if_needed and os.path.exists(temp_ffmpeg_input_if_needed): os.remove(temp_ffmpeg_input_if_needed)
        final_duration = sf.info(output_path).duration if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else 0
        return True, final_duration
    except Exception as e:
        print(f"{log_prefix}Error in _apply_tempo_and_trim_pad for '{base_input_name}': {e} \n {traceback.format_exc()}")
        if input_path != output_path and os.path.exists(input_path):
            try: shutil.copy(input_path, output_path)
            except Exception as copy_err: print(f"{log_prefix}  Could not copy original on error: {copy_err}")
        return False, sf.info(input_path).duration if os.path.exists(input_path) and os.path.getsize(input_path) > 0 else 0


def synthesize_speech_segments(segments, reference_audio_path, temp_dir, diarization_result_df=None, language='ru', progress_callback=None, 
                               min_segment_duration_for_synth = 0.25, 
                               atempo_speed_cap = 2.0,
                               use_radical_trim_pad_only = False 
                               ):
    tts_model = load_tts_model()
    if tts_model is None: raise RuntimeError("TTS model could not be loaded.")

    output_segments_dir = os.path.join(temp_dir, "tts_adjusted_segments"); os.makedirs(output_segments_dir, exist_ok=True)
    raw_tts_dir = os.path.join(temp_dir, "tts_raw_segments"); os.makedirs(raw_tts_dir, exist_ok=True) 

    processed_segment_files = [] 
    speaker_references = {} 
    first_valid_speaker_ref = None 
    last_segment_original_end_time = 0.0
    
    total_raw_duration_sum = 0.0
    total_final_segment_duration_sum = 0.0

    total_segments = len(segments)
    print(f"Starting voice synthesis & tempo adjustment for {total_segments} segments...")
    print(f"  Min segment duration for synthesis: {min_segment_duration_for_synth}s")
    if use_radical_trim_pad_only:
        print(f"  RADICAL TEST MODE: Using ONLY TRIM/PAD for duration adjustment.")
    else:
        print(f"  Atempo speed cap for quality pass: {atempo_speed_cap:.1f}x (then trim/pad)")
    
    for i, segment in enumerate(segments):
        raw_text_from_segment = segment.get('translated_text', segment.get('text', ''))
        text_to_synth_cleaned = re.sub(r'<[^>]+>', '', raw_text_from_segment).strip()

        speaker_id = segment.get('speaker', 'SPEAKER_UNKNOWN')
        original_start = segment.get('start'); original_end = segment.get('end')

        if not text_to_synth_cleaned.strip() or original_start is None or original_end is None:
            if original_end is not None: last_segment_original_end_time = original_end 
            if progress_callback: progress_callback((i + 1) / total_segments)
            continue
        
        original_duration = original_end - original_start
        if original_duration <= 0.01: 
            if i % 50 == 0 : print(f"  Skipping very short original segment {i+1} (duration: {original_duration:.3f}s)") 
            last_segment_original_end_time = original_end
            if progress_callback: progress_callback((i + 1) / total_segments)
            continue
        
        log_this_segment_details = (i % 10 == 0) or (i == total_segments - 1) 

        if log_this_segment_details:
            print(f"\nProcessing segment {i+1}/{total_segments} (Speaker: {speaker_id}, Orig Dur: {original_duration:.3f}s)")
            print(f"  Text: '{text_to_synth_cleaned[:80]}...'")

        silence_before_duration = original_start - last_segment_original_end_time
        if silence_before_duration > 0.01: 
            silence_wav_path = os.path.join(output_segments_dir, f"silence_before_{i:04d}.wav")
            try:
                (ffmpeg.input('anullsrc', format='lavfi', r=24000, channel_layout='mono')
                 .output(silence_wav_path, t=silence_before_duration, acodec='pcm_s16le')
                 .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                if os.path.exists(silence_wav_path): 
                    processed_segment_files.append(silence_wav_path)
                    total_final_segment_duration_sum += silence_before_duration
            except Exception as e_sil: print(f"Warning: Failed to generate silence {silence_wav_path}: {e_sil}")

        if speaker_id not in speaker_references:
            current_speaker_wav_path = _get_or_create_speaker_wav(speaker_id, segment, reference_audio_path, temp_dir, diarization_df=diarization_result_df)
            if current_speaker_wav_path:
                speaker_references[speaker_id] = current_speaker_wav_path
                if first_valid_speaker_ref is None: # ИСПРАВЛЕННЫЙ ОТСТУП
                    first_valid_speaker_ref = current_speaker_wav_path
            else: 
                current_speaker_wav_path = first_valid_speaker_ref
        else: 
            current_speaker_wav_path = speaker_references[speaker_id]
        
        text_chunks = _split_text_for_tts(text_to_synth_cleaned, max_length=XTTS_RU_MAX_CHARS)
        synthesized_sub_segments_paths = [] # ... (TTS синтез без изменений) ...
        for chunk_idx, text_chunk in enumerate(text_chunks):
            if not text_chunk.strip(): continue
            raw_sub_segment_filename = os.path.join(raw_tts_dir, f"raw_sub_seg_{i:04d}_{chunk_idx:02d}_{speaker_id}.wav")
            try:
                tts_model.tts_to_file(text=text_chunk, speaker_wav=current_speaker_wav_path, language=language, file_path=raw_sub_segment_filename, split_sentences=True )
                if os.path.exists(raw_sub_segment_filename) and os.path.getsize(raw_sub_segment_filename) > 0: synthesized_sub_segments_paths.append(raw_sub_segment_filename)
            except Exception as e_tts_sub:
                if log_this_segment_details : print(f"    Error synthesizing sub-segment {i+1}-{chunk_idx+1}: {e_tts_sub}")
        
        raw_segment_filename_combined = os.path.join(raw_tts_dir, f"raw_segment_combined_{i:04d}_{speaker_id}.wav")
        synthesized_duration = 0.0; synthesized_samplerate = 24000  # ... (слияние и проверка сырого TTS без изменений) ...
        if not synthesized_sub_segments_paths: 
            if log_this_segment_details : print(f"  Warning: No sub-segments synthesized for segment {i+1}. Skipping.")
            last_segment_original_end_time = original_end; 
            if progress_callback: progress_callback((i + 1) / total_segments); continue
        elif len(synthesized_sub_segments_paths) == 1:
            try: shutil.copy(synthesized_sub_segments_paths[0], raw_segment_filename_combined)
            except Exception as e_copy_single: print(f"  Error copying single sub-segment {i+1}: {e_copy_single}"); continue
        else:
            try: video_processor.merge_audio_segments(synthesized_sub_segments_paths, raw_segment_filename_combined)
            except Exception as e_merge_sub:
                if log_this_segment_details : print(f"  Error merging sub-segments for segment {i+1}: {e_merge_sub}. Skipping.")
                last_segment_original_end_time = original_end; 
                if progress_callback: progress_callback((i + 1) / total_segments); continue
        if os.path.exists(raw_segment_filename_combined) and os.path.getsize(raw_segment_filename_combined) > 0:
            try:
                synthesized_info = sf.info(raw_segment_filename_combined)
                synthesized_duration = synthesized_info.duration; synthesized_samplerate = synthesized_info.samplerate
                total_raw_duration_sum += synthesized_duration
            except Exception as e_sfinfo: synthesized_duration = 0.0
        else:
            if log_this_segment_details : print(f"  Warning: Combined raw TTS for segment {i+1} is missing/empty. Skipping.")
            last_segment_original_end_time = original_end; 
            if progress_callback: progress_callback((i + 1) / total_segments); continue
            
        final_adjusted_segment_path = os.path.join(output_segments_dir, f"adj_segment_{i:04d}_{speaker_id}.wav")
        
        # ИСПРАВЛЕНИЕ ЗДЕСЬ: убедимся, что строка print полностью закрыта
        if original_duration <= min_segment_duration_for_synth or synthesized_duration <= 0.01 : 
            if log_this_segment_details : 
                print(f"  Segment {i+1} original_duration ({original_duration:.3f}s) <= min_synth_dur ({min_segment_duration_for_synth}s) OR synth_duration ({synthesized_duration:.3f}s) too short. Creating silence.")
            try:
                 (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                    .output(final_adjusted_segment_path, t=original_duration, acodec='pcm_s16le')
                    .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            except Exception as e_short_silence: print(f"  Error creating silence for short segment: {e_short_silence}")
        else:
            if log_this_segment_details:
                print(f"  Adjusting segment {i+1}: Synth Dur: {synthesized_duration:.3f}s, Target Orig Dur: {original_duration:.3f}s")
            
            _success, final_segment_duration = _apply_tempo_and_trim_pad(
                raw_segment_filename_combined,
                final_adjusted_segment_path,
                synthesized_samplerate,
                original_duration, 
                atempo_speed_limit_for_quality=atempo_speed_cap, 
                log_prefix="    (FinalAdj) ",
                radical_trim_pad_only=use_radical_trim_pad_only 
            )
            if log_this_segment_details:
                print(f"  Final duration for seg {i+1}: {final_segment_duration:.3f}s (Target: {original_duration:.3f}s)")
                if abs(final_segment_duration - original_duration) > 0.05: 
                     print(f"    ALERT: Final duration {final_segment_duration:.3f}s significantly differs from target {original_duration:.3f}s.")
        
        if os.path.exists(final_adjusted_segment_path) and os.path.getsize(final_adjusted_segment_path) > 0:
            processed_segment_files.append(final_adjusted_segment_path)
            try: total_final_segment_duration_sum += sf.info(final_adjusted_segment_path).duration
            except Exception: pass 
        else: 
            if log_this_segment_details : print(f"  Warning: Final adjusted segment {i+1} missing or empty. Not added to merge list.")
        
        last_segment_original_end_time = original_end
        if progress_callback: progress_callback((i + 1) / total_segments)
        
    final_audio_path = os.path.join(temp_dir, "dubbed_full_audio_final_tempo.wav")
    print(f"\nMerging {len(processed_segment_files)} processed audio segments into {os.path.basename(final_audio_path)}")
    video_processor.merge_audio_segments(processed_segment_files, final_audio_path)
    
    final_merged_audio_duration = 0.0 # ... (логирование и возврат суммарных длительностей без изменений) ...
    if os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 0:
        try:
            final_merged_audio_duration = sf.info(final_audio_path).duration
            print(f"Final dubbed audio (tempo adjusted and silences added) merged successfully. Duration: {final_merged_audio_duration:.2f}s")
        except Exception as e: print(f"Could not get duration of final merged audio: {e}")
    else: print("Warning: Final merged audio file is missing or empty.")
    return final_audio_path, total_raw_duration_sum, total_final_segment_duration_sum

def _apply_tempo_adjustment_ffmpeg_internal(input_path, output_path, speed_factor_for_atempo, samplerate,
                                   log_prefix="    "): # ... (без изменений) ...
    base_input_name = os.path.basename(input_path)
    if abs(speed_factor_for_atempo - 1.0) < 0.01:
        if input_path != output_path:
            try: shutil.copy(input_path, output_path)
            except Exception as e_copy_nochange: print(f"{log_prefix}  Error copying {base_input_name} (no tempo change): {e_copy_nochange}")
        return True, sf.info(output_path).duration if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else 0
    min_atempo_sf_abs = 0.5; max_atempo_sf_abs = 100.0 
    effective_max_single_pass = 4.0; effective_min_single_pass = 0.5
    atempo_filters_chain_values = []; current_sf_to_apply = speed_factor_for_atempo
    if not (min_atempo_sf_abs <= current_sf_to_apply <= max_atempo_sf_abs):
        current_sf_to_apply = max(min_atempo_sf_abs, min(max_atempo_sf_abs, current_sf_to_apply))
    if current_sf_to_apply > effective_max_single_pass:
        while current_sf_to_apply > effective_max_single_pass: 
            atempo_filters_chain_values.append(effective_max_single_pass); current_sf_to_apply /= effective_max_single_pass
        if current_sf_to_apply > 1.001 : atempo_filters_chain_values.append(current_sf_to_apply)
    elif current_sf_to_apply < effective_min_single_pass: 
        while current_sf_to_apply < effective_min_single_pass:
            atempo_filters_chain_values.append(effective_min_single_pass); current_sf_to_apply /= effective_min_single_pass
        if current_sf_to_apply < 0.999 and current_sf_to_apply > 0.01: atempo_filters_chain_values.append(current_sf_to_apply)
    else: atempo_filters_chain_values.append(current_sf_to_apply)
    if not atempo_filters_chain_values: 
        if input_path != output_path: shutil.copy(input_path, output_path)
        return True, sf.info(output_path).duration if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else 0
    stream = ffmpeg.input(input_path); filtered_stream = stream; filter_str_for_log = []
    for factor_val in atempo_filters_chain_values:
        rounded_factor = round(factor_val, 4); safe_factor = max(min_atempo_sf_abs, min(max_atempo_sf_abs, rounded_factor)) 
        if abs(safe_factor - rounded_factor)>0.0001 and log_prefix: print(f"{log_prefix}  Chain factor {rounded_factor} clamped to {safe_factor} for ffmpeg.")
        filtered_stream = ffmpeg.filter(filtered_stream, 'atempo', str(safe_factor)); filter_str_for_log.append(f"atempo={safe_factor}")
    output_node = None
    try:
        output_node = ffmpeg.output(filtered_stream, output_path, acodec='pcm_s16le', ar=samplerate).overwrite_output()
        _stdout, _stderr = output_node.run(capture_stdout=True, capture_stderr=True)
        stderr_str = _stderr.decode('utf8', 'ignore') if _stderr else ""
        actual_duration_after_tempo = 0
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0: actual_duration_after_tempo = sf.info(output_path).duration
        if "error" in stderr_str.lower() or not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
             print(f"{log_prefix}  ERROR applying atempo ({', '.join(filter_str_for_log)}) to {base_input_name}. FFmpeg stderr:\n{stderr_str[:500]}")
             if input_path != output_path: shutil.copy(input_path, output_path) 
             return False, sf.info(input_path).duration if os.path.exists(input_path) and os.path.getsize(input_path) > 0 else 0
        return True, actual_duration_after_tempo
    except ffmpeg.Error as e:
        error_message = e.stderr.decode('utf8', 'ignore') if e.stderr else str(e)
        print(f"{log_prefix}  EXCEPTION applying atempo ({', '.join(filter_str_for_log)}) to {base_input_name}: {error_message[:500]}")
        if input_path != output_path:
             try: shutil.copy(input_path, output_path)
             except Exception as copy_e: print(f"{log_prefix}    Fallback copy also failed for {base_input_name}: {copy_e}")
        return False, sf.info(input_path).duration if os.path.exists(input_path) and os.path.getsize(input_path) > 0 else 0
