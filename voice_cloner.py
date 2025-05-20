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
import io 
import sys 
from contextlib import redirect_stdout, redirect_stderr 

# --- Кэш для моделей TTS ---
class TTSCache:
    def __init__(self):
        self.model = None; self.model_name = None; self.device = None
tts_cache = TTSCache()

XTTS_RU_MAX_CHARS = 180
MIN_ORIGINAL_DURATION_FOR_TTS_ATTEMPT = 0.15 

ATEMPO_MIN_FACTOR = 0.5 
ATEMPO_MAX_FACTOR = 2.0  

MAX_SILENCE_BETWEEN_SEGMENTS = 2.5 # Увеличено значение по умолчанию


def load_tts_model(model_name="tts_models/multilingual/multi-dataset/xtts_v2", device="cuda"):
    global tts_cache
    if tts_cache.model is None or tts_cache.model_name != model_name or tts_cache.device != device:
        print(f"Loading TTS model: {model_name} on device: {device}")
        
        loaded_successfully = False
        original_stdin = sys.stdin  
        
        with io.StringIO() as captured_output, \
             redirect_stdout(captured_output), \
             redirect_stderr(captured_output):
            try:
                sys.stdin = io.StringIO('y\n') 
                
                tts_cache.model = TTS(model_name) 
                tts_cache.model.to(device)        
                
                tts_cache.model_name = model_name
                tts_cache.device = device
                loaded_successfully = True
            except Exception as e:
                try:
                    sys.stdin = io.StringIO('y\n') 
                    tts_cache.model = TTS(model_name)
                    tts_cache.model.to(device)
                    tts_cache.model_name = model_name
                    tts_cache.device = device
                    loaded_successfully = True
                except Exception as e2:
                    tts_cache.model = None 
            finally:
                sys.stdin = original_stdin 

        if loaded_successfully:
            print("TTS model loaded successfully (output suppressed).")
        else:
            print(f"FATAL: Failed to load TTS model {model_name}. Check logs if any were produced despite suppression.")
            
    return tts_cache.model

def _split_text_for_tts(text, max_length=XTTS_RU_MAX_CHARS):
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
                            for i_w in range(0, len(word), max_length): chunks.append(word[i_w:i_w+max_length])
                            temp_sub_chunk = ""
                        else: temp_sub_chunk = word
                if temp_sub_chunk: chunks.append(temp_sub_chunk)
                current_chunk = ""
            else: current_chunk = sentence
    if current_chunk: chunks.append(current_chunk)
    final_chunks = []
    for ch in chunks:
        if len(ch) > max_length:
            for i_ch in range(0, len(ch), max_length): final_chunks.append(ch[i_ch:i_ch+max_length])
        elif ch.strip(): final_chunks.append(ch)
    if not final_chunks and text.strip():
        for i_f in range(0, len(text), max_length): final_chunks.append(text[i_f:i_f+max_length])
    return [c for c in final_chunks if c.strip()] if final_chunks else ([text[:max_length]] if text.strip() else [])

def _get_or_create_speaker_wav(speaker_id, segment_for_ref_timing, base_audio_path, temp_dir, diarization_df=None, min_duration=2.5, max_duration=12.0):
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
        
        ideal_segments = speaker_segments_df[(speaker_segments_df['duration'] >= 3.5) & (speaker_segments_df['duration'] <= 10.0)]
        if not ideal_segments.empty:
            best_segment_row = ideal_segments.sample(1).iloc[0] if len(ideal_segments) > 1 else ideal_segments.iloc[0]
            ref_start, ref_end = best_segment_row['start'], best_segment_row['end']

        if ref_start is None:
            longer_segments = speaker_segments_df[speaker_segments_df['duration'] >= min_duration]
            if not longer_segments.empty:
                best_segment_row = longer_segments.sort_values(by='duration', ascending=False).iloc[0] 
                ref_start = best_segment_row['start']
                ref_end = min(best_segment_row['end'], best_segment_row['start'] + max_duration) 

    if ref_start is None and segment_for_ref_timing:
        if segment_for_ref_timing.get('speaker') == speaker_id:
            srt_start_time = segment_for_ref_timing.get('start'); srt_end_time = segment_for_ref_timing.get('end')
            if srt_start_time is not None and srt_end_time is not None:
                srt_duration = srt_end_time - srt_start_time
                if srt_duration >= min_duration: 
                    ref_start = srt_start_time
                    ref_end = min(srt_end_time, srt_start_time + max_duration) 
    
    if ref_start is None or ref_end is None or (ref_end - ref_start) < 0.5: 
        return None
        
    try:
        if not os.path.exists(base_audio_path): raise FileNotFoundError(f"Base audio for speaker ref not found: {base_audio_path}")
        (ffmpeg.input(base_audio_path, ss=ref_start, to=ref_end)
         .output(speaker_ref_path, ar=24000, ac=1, acodec='pcm_s16le')
         .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        if not os.path.exists(speaker_ref_path) or os.path.getsize(speaker_ref_path) < 1000:
             if os.path.exists(speaker_ref_path): os.remove(speaker_ref_path)
             return None
        info = sf.info(speaker_ref_path)
        if info.duration < 0.5:
            os.remove(speaker_ref_path)
            return None
        return speaker_ref_path
    except Exception as e:
        print(f"ERROR creating speaker ref for {speaker_id}: {e}")
        if os.path.exists(speaker_ref_path):
            try: os.remove(speaker_ref_path)
            except OSError: pass
        return None

def _apply_tempo_adjustment_ffmpeg_internal(input_path, output_path, speed_factor, samplerate, log_prefix="    "):
    base_input_name = os.path.basename(input_path)
    min_atempo_sf_abs = 0.5 
    max_atempo_sf_abs = 4.0 

    atempo_filters_chain_values = []
    current_sf_to_apply = speed_factor

    if current_sf_to_apply > max_atempo_sf_abs: 
        while current_sf_to_apply > max_atempo_sf_abs:
            atempo_filters_chain_values.append(max_atempo_sf_abs)
            current_sf_to_apply /= max_atempo_sf_abs
        if current_sf_to_apply > 1.001 : 
             atempo_filters_chain_values.append(current_sf_to_apply)
    elif current_sf_to_apply < min_atempo_sf_abs: 
        while current_sf_to_apply < min_atempo_sf_abs:
            atempo_filters_chain_values.append(min_atempo_sf_abs)
            current_sf_to_apply /= min_atempo_sf_abs
        if current_sf_to_apply < 0.999 and current_sf_to_apply > (1.0/max_atempo_sf_abs): 
            atempo_filters_chain_values.append(current_sf_to_apply)
    else: 
        atempo_filters_chain_values.append(current_sf_to_apply)

    if not atempo_filters_chain_values or (len(atempo_filters_chain_values) == 1 and abs(atempo_filters_chain_values[0] - 1.0) < 0.001) : 
        if input_path != output_path: shutil.copy(input_path, output_path)
        return True, sf.info(output_path).duration if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else 0

    stream = ffmpeg.input(input_path)
    filtered_stream = stream
    filter_str_for_log = []

    for factor_val in atempo_filters_chain_values:
        safe_factor_for_ffmpeg = max(0.5, min(100.0, round(factor_val, 4)))
        if abs(safe_factor_for_ffmpeg - factor_val) > 0.0001 and log_prefix:
            print(f"{log_prefix}  Atempo chain factor {factor_val:.4f} clamped to {safe_factor_for_ffmpeg:.4f} for ffmpeg.")
        filtered_stream = ffmpeg.filter(filtered_stream, 'atempo', str(safe_factor_for_ffmpeg))
        filter_str_for_log.append(f"atempo={safe_factor_for_ffmpeg}")
    
    if log_prefix: print(f"{log_prefix}Applying FFmpeg atempo chain: {', '.join(filter_str_for_log)} to '{base_input_name}'")

    output_node = None
    try:
        output_node = ffmpeg.output(filtered_stream, output_path, acodec='pcm_s16le', ar=samplerate).overwrite_output()
        _stdout, _stderr = output_node.run(capture_stdout=True, capture_stderr=True)
        stderr_str = _stderr.decode('utf8', 'ignore') if _stderr else ""
        
        actual_duration_after_tempo = 0
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            actual_duration_after_tempo = sf.info(output_path).duration
        
        if "error" in stderr_str.lower() or not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
             if log_prefix: print(f"{log_prefix}  ERROR applying atempo ({', '.join(filter_str_for_log)}) to {base_input_name}. FFmpeg stderr:\n{stderr_str[:500]}")
             if input_path != output_path: shutil.copy(input_path, output_path) 
             return False, sf.info(input_path).duration if os.path.exists(input_path) and os.path.getsize(input_path) > 0 else 0
        return True, actual_duration_after_tempo
    except ffmpeg.Error as e:
        error_message = e.stderr.decode('utf8', 'ignore') if e.stderr else str(e)
        if log_prefix: print(f"{log_prefix}  EXCEPTION applying atempo ({', '.join(filter_str_for_log)}) to {base_input_name}: {error_message[:500]}")
        if input_path != output_path:
             try: shutil.copy(input_path, output_path) 
             except Exception as copy_e: print(f"{log_prefix}    Fallback copy also failed for {base_input_name}: {copy_e}")
        return False, sf.info(input_path).duration if os.path.exists(input_path) and os.path.getsize(input_path) > 0 else 0


def _apply_final_adjustment(input_path, output_path, samplerate,
                            target_duration, current_duration, 
                            log_prefix="    "):
    base_input_name = os.path.basename(input_path)
    operation_performed = "None"
    log_this_adjustment = log_prefix is not None 

    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        if log_this_adjustment: print(f"{log_prefix}Input file '{base_input_name}' missing or empty. Creating silence of {target_duration:.3f}s.")
        (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
            .output(output_path, t=max(0.01, target_duration), acodec='pcm_s16le')
            .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return True, max(0.01, target_duration), "Created silence (input missing/empty)"

    if current_duration <= 0.01 and target_duration <= 0.01 : 
         if log_this_adjustment: print(f"{log_prefix}Both current ({current_duration:.3f}s) and target ({target_duration:.3f}s) for '{base_input_name}' are tiny. Copying.")
         if input_path != output_path: shutil.copy(input_path, output_path)
         return True, current_duration, "Copied (both durations tiny)"
    elif target_duration <= 0.01: 
        if log_this_adjustment: print(f"{log_prefix}Target duration for '{base_input_name}' is <= 0.01s ({target_duration:.3f}s). Creating minimal silence.")
        (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
            .output(output_path, t=0.01, acodec='pcm_s16le')
            .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return True, 0.01, "Created minimal silence (target tiny)"

    speech_tempo_factor = 1.0
    if current_duration > 0.01 and target_duration > 0.01:
        speech_tempo_factor = current_duration / target_duration 
    
    duration_after_atempo = current_duration
    path_after_atempo = input_path 
    temp_atempo_output = None
    
    adjusted_target_for_pad_trim = target_duration
    
    needs_atempo = abs(speech_tempo_factor - 1.0) > 0.05 
    
    if needs_atempo:
        if log_this_adjustment: print(f"{log_prefix}Target tempo factor for '{base_input_name}': {speech_tempo_factor:.3f} (Synth dur: {current_duration:.3f}s, Target SRT dur: {target_duration:.3f}s)")
        
        if (ATEMPO_MIN_FACTOR <= speech_tempo_factor <= ATEMPO_MAX_FACTOR):
            if log_this_adjustment: print(f"{log_prefix}Attempting FFmpeg atempo by factor {speech_tempo_factor:.3f}")
            
            temp_atempo_output = os.path.join(os.path.dirname(output_path), f"temp_atempo_{base_input_name}")
            success_atempo, dur_atempo = _apply_tempo_adjustment_ffmpeg_internal(
                input_path, temp_atempo_output, speech_tempo_factor, samplerate, log_prefix + "  (Atempo) " if log_this_adjustment else None
            )
            if success_atempo and os.path.exists(temp_atempo_output) and os.path.getsize(temp_atempo_output) > 0:
                duration_after_atempo = dur_atempo
                path_after_atempo = temp_atempo_output
                operation_performed = f"Atempo by {speech_tempo_factor:.3f}"
                if log_this_adjustment: print(f"{log_prefix}Atempo applied. New duration: {duration_after_atempo:.3f}s")
            else:
                if log_this_adjustment: print(f"{log_prefix}Atempo failed or produced empty file. Using original for trim/pad.")
                if temp_atempo_output and os.path.exists(temp_atempo_output): 
                    try: os.remove(temp_atempo_output)
                    except: pass
                temp_atempo_output = None 
                operation_performed = "Atempo_Failed"
        else:
             if log_this_adjustment: print(f"{log_prefix}Required atempo factor {speech_tempo_factor:.3f} is outside range [{ATEMPO_MIN_FACTOR}-{ATEMPO_MAX_FACTOR}]. Will use trim/pad only.")
             operation_performed = f"Atempo_Skipped (factor {speech_tempo_factor:.3f} out of range)"
             
             if speech_tempo_factor < ATEMPO_MIN_FACTOR and current_duration > 0.01: 
                 max_reasonable_duration_via_tempo_limit = current_duration / ATEMPO_MIN_FACTOR
                 if target_duration > max_reasonable_duration_via_tempo_limit:
                     adjusted_target_for_pad_trim = max_reasonable_duration_via_tempo_limit
                     if log_this_adjustment:
                         print(f"{log_prefix}  Atempo factor {speech_tempo_factor:.3f} too small. Capping padding/trimming target from {target_duration:.3f}s to {adjusted_target_for_pad_trim:.3f}s.")
    else:
        if log_this_adjustment: print(f"{log_prefix}Atempo not needed for '{base_input_name}' (factor {speech_tempo_factor:.3f} close to 1.0).")
        operation_performed = "Atempo_Not_Needed"


    final_stream_input_path_for_ffmpeg = path_after_atempo
    temp_ffmpeg_input_if_needed_for_trim_pad = None

    if os.path.abspath(path_after_atempo) == os.path.abspath(output_path):
         temp_ffmpeg_input_if_needed_for_trim_pad = os.path.join(os.path.dirname(output_path), f"temp_trimpad_ff_in_{base_input_name}")
         shutil.copy(path_after_atempo, temp_ffmpeg_input_if_needed_for_trim_pad)
         final_stream_input_path_for_ffmpeg = temp_ffmpeg_input_if_needed_for_trim_pad

    final_stream = ffmpeg.input(final_stream_input_path_for_ffmpeg)
    
    duration_diff = adjusted_target_for_pad_trim - duration_after_atempo 

    trim_pad_op_performed = False
    if abs(duration_diff) >= 0.005 and adjusted_target_for_pad_trim > 0.01: 
        if duration_diff > 0: 
            pad_seconds = round(duration_diff, 3)
            final_stream = ffmpeg.filter(final_stream, 'apad', pad_dur=f'{pad_seconds}s')
            current_op_detail = f"Padded by {pad_seconds:.3f}s"
            if log_this_adjustment: print(f"{log_prefix}Padding by {pad_seconds:.3f}s (current_dur: {duration_after_atempo:.3f}s, target: {adjusted_target_for_pad_trim:.3f}s)")
        elif duration_diff < 0: 
            final_stream = ffmpeg.filter(final_stream, 'atrim', start='0', end=str(round(adjusted_target_for_pad_trim,3)))
            current_op_detail = f"Trimmed to {adjusted_target_for_pad_trim:.3f}s"
            if log_this_adjustment: print(f"{log_prefix}Trimming to {adjusted_target_for_pad_trim:.3f}s (current_dur: {duration_after_atempo:.3f}s)")
        
        if operation_performed in ["None", "Atempo_Not_Needed"] or "Atempo_Skipped" in operation_performed or "Atempo_Failed" in operation_performed :
            operation_performed = current_op_detail
        else: 
            operation_performed += f"; {current_op_detail}"
        
        ffmpeg.output(final_stream, output_path, acodec='pcm_s16le', ar=samplerate).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        trim_pad_op_performed = True
    
    if not trim_pad_op_performed: 
        if final_stream_input_path_for_ffmpeg != output_path:
            shutil.copy(final_stream_input_path_for_ffmpeg, output_path)
        if operation_performed in ["None", "Atempo_Not_Needed"]: operation_performed = "Copied (duration diff small or target zero)"


    if temp_atempo_output and os.path.exists(temp_atempo_output) and \
       (temp_ffmpeg_input_if_needed_for_trim_pad is None or os.path.abspath(temp_atempo_output) != os.path.abspath(temp_ffmpeg_input_if_needed_for_trim_pad)):
        try: os.remove(temp_atempo_output)
        except: pass
    if temp_ffmpeg_input_if_needed_for_trim_pad and os.path.exists(temp_ffmpeg_input_if_needed_for_trim_pad):
        try: os.remove(temp_ffmpeg_input_if_needed_for_trim_pad)
        except: pass

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        if log_this_adjustment: print(f"{log_prefix}Output file '{os.path.basename(output_path)}' is missing or empty after {operation_performed}. Creating fallback silence.")
        (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
            .output(output_path, t=max(0.01, adjusted_target_for_pad_trim), acodec='pcm_s16le') 
            .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return True, max(0.01, adjusted_target_for_pad_trim), f"Fallback silence (output invalid after {operation_performed})"

    final_duration = sf.info(output_path).duration
    return True, final_duration, operation_performed


def synthesize_speech_segments(segments, reference_audio_path, temp_dir, diarization_result_df=None, language='ru', progress_callback=None,
                               min_segment_duration_for_synth_processing = 0.05
                               ): 
    tts_model = load_tts_model()
    if tts_model is None: raise RuntimeError("TTS model could not be loaded (remained None after load_tts_model).")

    output_segments_dir = os.path.join(temp_dir, "tts_adjusted_segments"); os.makedirs(output_segments_dir, exist_ok=True)
    raw_tts_dir = os.path.join(temp_dir, "tts_raw_segments"); os.makedirs(raw_tts_dir, exist_ok=True)

    processed_segment_files = []
    speaker_references = {}
    first_valid_speaker_ref = None
    last_segment_original_end_time = 0.0

    total_raw_tts_duration_s1_accumulator = 0.0 
    
    total_segments = len(segments)
    
    for i, segment in enumerate(tqdm(segments, desc="Synthesizing Segments", unit="segment", disable=bool(progress_callback)) if not progress_callback else segments):
        raw_text_from_segment = segment.get('translated_text', segment.get('text', ''))
        text_to_synth_cleaned = re.sub(r'<[^>]+>', '', raw_text_from_segment).strip()

        speaker_id = segment.get('speaker', 'SPEAKER_UNKNOWN')
        original_start = segment.get('start'); original_end = segment.get('end')
        
        segment_base_name = f"segment_{i:04d}_{speaker_id}"
        final_output_for_this_segment_path = os.path.join(output_segments_dir, f"{segment_base_name}_final.wav")

        log_this_segment_details = (i < 3) or (i > total_segments - 3) or (i % 50 == 0) 

        if original_start is None or original_end is None:
            if log_this_segment_details: print(f"Segment {i+1} missing start/end time. Skipping.")
            if progress_callback: progress_callback((i + 1) / total_segments)
            continue

        original_duration = original_end - original_start
        synthesized_samplerate = 24000 

        silence_duration_from_srt = original_start - last_segment_original_end_time
        actual_silence_to_insert = 0.0
        if silence_duration_from_srt > 0.001: # Было 0.01, уменьшаем порог для более точного следования SRT
            actual_silence_to_insert = min(silence_duration_from_srt, MAX_SILENCE_BETWEEN_SEGMENTS)
            # Важно: не урезаем паузы слишком сильно, если они действительно длинные в SRT
            # Если MAX_SILENCE_BETWEEN_SEGMENTS меньше, чем silence_duration_from_srt, то пауза будет урезана.
            # Это может быть одной из причин рассинхрона, если в оригинале много длинных пауз.
        
        if actual_silence_to_insert > 0.001: # Был 0.01
            silence_before_path = os.path.join(output_segments_dir, f"silence_before_seg_{i:04d}.wav")
            try:
                (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                 .output(silence_before_path, t=actual_silence_to_insert, acodec='pcm_s16le')
                 .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                if os.path.exists(silence_before_path) and os.path.getsize(silence_before_path) > 0:
                    processed_segment_files.append(silence_before_path)
            except Exception as e_sil: print(f"  Warning: Failed to generate silence before seg {i+1}: {e_sil}")

        segment_processed_successfully_with_sound = False
        current_segment_final_duration = 0.0
        
        try:
            if not text_to_synth_cleaned.strip() or original_duration < MIN_ORIGINAL_DURATION_FOR_TTS_ATTEMPT:
                reason = "empty text" if not text_to_synth_cleaned.strip() else f"orig_dur {original_duration:.3f}s < {MIN_ORIGINAL_DURATION_FOR_TTS_ATTEMPT}s"
                if log_this_segment_details:
                    print(f"\nSegment {i+1}/{total_segments} (Orig Dur: {original_duration:.3f}s, Target: {original_duration:.3f}s) - {reason}. Creating silence.")
                (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                    .output(final_output_for_this_segment_path, t=max(0.01, original_duration), acodec='pcm_s16le')
                    .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                if os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                     segment_processed_successfully_with_sound = True
                     current_segment_final_duration = sf.info(final_output_for_this_segment_path).duration
            else: 
                if log_this_segment_details:
                    print(f"\nProcessing TTS for segment {i+1}/{total_segments} (Speaker: {speaker_id}, Orig Dur: {original_duration:.3f}s, Target: {original_duration:.3f}s)")
                    print(f"  Text: '{text_to_synth_cleaned[:60]}...' -> {os.path.basename(final_output_for_this_segment_path)}")

                if speaker_id not in speaker_references:
                    current_speaker_wav_path = _get_or_create_speaker_wav(speaker_id, segment, reference_audio_path, temp_dir, diarization_df=diarization_result_df)
                    if current_speaker_wav_path:
                        speaker_references[speaker_id] = current_speaker_wav_path
                        if first_valid_speaker_ref is None: first_valid_speaker_ref = current_speaker_wav_path
                    else: 
                        current_speaker_wav_path = first_valid_speaker_ref 
                        if log_this_segment_details and current_speaker_wav_path:
                            print(f"  Warning: Could not create specific ref for {speaker_id}, using ref from {os.path.basename(current_speaker_wav_path)}")
                        elif log_this_segment_details:
                             print(f"  Warning: Could not create specific ref for {speaker_id} and no fallback ref available.")
                else: current_speaker_wav_path = speaker_references[speaker_id]
                
                active_speaker_ref = current_speaker_wav_path 
                if active_speaker_ref is None and log_this_segment_details:
                     print(f"  CRITICAL Warning: No speaker reference for seg {i+1} (speaker: {speaker_id}). TTS will use default voice or fail.")

                synthesized_at_speed_1_duration = 0.0
                
                synthesized_chunks_at_speed_1 = []
                text_chunks_for_tts = _split_text_for_tts(text_to_synth_cleaned, max_length=XTTS_RU_MAX_CHARS)

                with io.StringIO() as devnull_buffer, \
                     redirect_stdout(devnull_buffer), \
                     redirect_stderr(devnull_buffer):
                    try:
                        if text_chunks_for_tts:
                            for chunk_idx, text_chunk in enumerate(text_chunks_for_tts):
                                temp_chunk_file = os.path.join(raw_tts_dir, f"{segment_base_name}_chunk_{chunk_idx:02d}_s1.wav")
                                tts_model.tts_to_file(text=text_chunk, speaker_wav=active_speaker_ref, language=language, 
                                                        file_path=temp_chunk_file, split_sentences=True, speed=1.0) 
                                if os.path.exists(temp_chunk_file) and os.path.getsize(temp_chunk_file) > 0:
                                    synthesized_chunks_at_speed_1.append(temp_chunk_file)
                    except Exception as tts_e: 
                        if log_this_segment_details: print(f"    TTS error for seg {i+1}, chunk {chunk_idx}: {tts_e}")
                
                raw_segment_s1_combined_path = os.path.join(raw_tts_dir, f"{segment_base_name}_combined_s1.wav")
                if synthesized_chunks_at_speed_1:
                    if os.path.exists(raw_segment_s1_combined_path): os.remove(raw_segment_s1_combined_path)
                    video_processor.merge_audio_segments(synthesized_chunks_at_speed_1, raw_segment_s1_combined_path, log_prefix=None) 
                    
                    if os.path.exists(raw_segment_s1_combined_path) and os.path.getsize(raw_segment_s1_combined_path) > 0:
                        synthesized_at_speed_1_duration = sf.info(raw_segment_s1_combined_path).duration
                        if log_this_segment_details: print(f"    Synthesized TTS for seg {i+1} (speed=1.0) duration: {synthesized_at_speed_1_duration:.3f}s")
                        total_raw_tts_duration_s1_accumulator += synthesized_at_speed_1_duration
                else:
                    if log_this_segment_details : print(f"  Warning: No sub-segments synthesized for seg {i+1} at speed 1.0.")


                if synthesized_at_speed_1_duration <= 0.01 or original_duration <= min_segment_duration_for_synth_processing:
                    reason = "synth_dur_s1 too short" if synthesized_at_speed_1_duration <= 0.01 else f"orig_dur {original_duration:.3f}s <= min_process_dur"
                    if log_this_segment_details : print(f"  Seg {i+1} ({reason}). Creating silence instead of fine-tuning.")
                    (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                        .output(final_output_for_this_segment_path, t=max(0.01, original_duration), acodec='pcm_s16le')
                        .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                    if os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                         segment_processed_successfully_with_sound = True
                         current_segment_final_duration = sf.info(final_output_for_this_segment_path).duration
                else: 
                    if log_this_segment_details: print(f"  Fine-tuning seg {i+1}: SynthDur@Speed1.0: {synthesized_at_speed_1_duration:.3f}s, TargetSRT Dur: {original_duration:.3f}s")
                    
                    _success, final_dur_val, op_details = _apply_final_adjustment(
                        raw_segment_s1_combined_path, 
                        final_output_for_this_segment_path,
                        synthesized_samplerate, 
                        original_duration, 
                        synthesized_at_speed_1_duration, 
                        log_prefix="    (Atempo/Pad/Trim) " if log_this_segment_details else None 
                    )
                    if _success and os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                        segment_processed_successfully_with_sound = True
                        current_segment_final_duration = final_dur_val
                        if log_this_segment_details: print(f"  Seg {i+1} fine-tuned: Op='{op_details}', FinalDur={final_dur_val:.3f}s (Target={original_duration:.3f}s)")
                        if abs(final_dur_val - original_duration) > 0.05 and original_duration > 0.1: 
                             if log_this_segment_details: print(f"    ALERT: Final duration {final_dur_val:.3f}s differs significantly from target {original_duration:.3f}s for seg {i+1}.")
                    else:
                        if log_this_segment_details: print(f"  Error or empty output after _apply_final_adjustment for seg {i+1}. Op='{op_details}'. Fallback to silence.")
                        (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                            .output(final_output_for_this_segment_path, t=max(0.01, original_duration), acodec='pcm_s16le')
                            .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                        if os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                            segment_processed_successfully_with_sound = True
                            current_segment_final_duration = sf.info(final_output_for_this_segment_path).duration


        except Exception as e_segment_processing:
            print(f"  CRITICAL ERROR processing segment {i+1}: {e_segment_processing}\n{traceback.format_exc()}")
            print(f"  Creating fallback silence for segment {i+1} due to critical error.")
            try:
                (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                    .output(final_output_for_this_segment_path, t=max(0.01, original_duration), acodec='pcm_s16le')
                    .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                if os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                    segment_processed_successfully_with_sound = True
                    current_segment_final_duration = sf.info(final_output_for_this_segment_path).duration
            except Exception as e_crit_fallback:
                 print(f"    Failed to create fallback silence for seg {i+1} after critical error: {e_crit_fallback}")


        if segment_processed_successfully_with_sound and os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
            processed_segment_files.append(final_output_for_this_segment_path)
        else:
            if log_this_segment_details: print(f"  WARNING: Segment {i+1} processing failed to produce a valid sound file. A gap might appear or be handled by merge_audio_segments.")

        last_segment_original_end_time = original_end 
        if progress_callback: progress_callback((i + 1) / total_segments)
        if torch.cuda.is_available(): torch.cuda.empty_cache() 

    final_audio_path = os.path.join(temp_dir, "dubbed_full_audio_final_tempo.wav")
    print(f"\nAttempting to merge {len(processed_segment_files)} processed audio segments into {os.path.basename(final_audio_path)}")
    if not processed_segment_files:
        print("Warning: No processed segments to merge. Creating a short empty audio file for final output.")
        try:
            (ffmpeg.input('anullsrc', format='lavfi', r=24000, channel_layout='mono')
                .output(final_audio_path, t=0.01, acodec='pcm_s16le')
                .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        except Exception as e_empty_merge: print(f"Error creating empty final audio file: {e_empty_merge}")
    else:
        video_processor.merge_audio_segments(processed_segment_files, final_audio_path, log_prefix=None) 

    final_merged_audio_duration = 0.0
    if os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 0:
        try:
            final_merged_audio_duration = sf.info(final_audio_path).duration
        except Exception as e: print(f"Could not get duration of final merged audio: {e}")
    else: print(f"Warning: Final merged audio file '{os.path.basename(final_audio_path)}' is missing or empty after merge attempt.")
    
    return final_audio_path, total_raw_tts_duration_s1_accumulator, final_merged_audio_duration