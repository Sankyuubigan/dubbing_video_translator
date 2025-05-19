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
MIN_ORIGINAL_DURATION_FOR_TTS_ATTEMPT = 0.15 # 150 миллисекунд

# Диапазон для параметра скорости TTS.
TTS_SPEED_MIN = 0.7
TTS_SPEED_MAX = 1.5

# Диапазон для FFmpeg atempo. Более широкие значения могут привести к искажениям.
ATEMPO_MIN_FACTOR = 0.6  # Можно попробовать 0.5, если готовы к большим искажениям
ATEMPO_MAX_FACTOR = 1.8  # Можно попробовать 2.0


def load_tts_model(model_name="tts_models/multilingual/multi-dataset/xtts_v2", device="cuda"):
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
        ideal_segments = speaker_segments_df[(speaker_segments_df['duration'] >= 3.0) & (speaker_segments_df['duration'] <= 10.0)]
        if not ideal_segments.empty:
            best_segment_row = ideal_segments.sample(1).iloc[0] if len(ideal_segments) > 1 else ideal_segments.iloc[0]
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
        if os.path.exists(speaker_ref_path):
            try: os.remove(speaker_ref_path)
            except OSError: pass
        return None

def _apply_tempo_adjustment_ffmpeg_internal(input_path, output_path, speed_factor, samplerate, log_prefix="    "):
    base_input_name = os.path.basename(input_path)
    # Эта функция вызывается, только если speed_factor не близок к 1.0

    min_atempo_sf_abs = 0.5  # Минимальный фактор для одного прохода atempo
    max_atempo_sf_abs = 4.0  # Максимальный фактор для одного прохода atempo (ffmpeg atempo ограничен 0.5-100, но >4 уже сильно искажает)

    atempo_filters_chain_values = []
    current_sf_to_apply = speed_factor

    # Разборка фактора на несколько проходов, если он выходит за пределы одного прохода
    if current_sf_to_apply > max_atempo_sf_abs:
        while current_sf_to_apply > max_atempo_sf_abs:
            atempo_filters_chain_values.append(max_atempo_sf_abs)
            current_sf_to_apply /= max_atempo_sf_abs
        if current_sf_to_apply > 1.001 : # Добавляем остаток, если он значим
             atempo_filters_chain_values.append(current_sf_to_apply)
    elif current_sf_to_apply < min_atempo_sf_abs:
        while current_sf_to_apply < min_atempo_sf_abs:
            atempo_filters_chain_values.append(min_atempo_sf_abs)
            current_sf_to_apply /= min_atempo_sf_abs
        if current_sf_to_apply < 0.999 and current_sf_to_apply > (1.0/max_atempo_sf_abs): # Добавляем остаток
            atempo_filters_chain_values.append(current_sf_to_apply)
    else:
        atempo_filters_chain_values.append(current_sf_to_apply)

    if not atempo_filters_chain_values: # Если почему-то цепочка пуста
        if input_path != output_path: shutil.copy(input_path, output_path)
        return True, sf.info(output_path).duration if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else 0

    stream = ffmpeg.input(input_path)
    filtered_stream = stream
    filter_str_for_log = []

    for factor_val in atempo_filters_chain_values:
        # ffmpeg atempo имеет ограничения 0.5-100.0. Убедимся, что мы в них.
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
             if input_path != output_path: shutil.copy(input_path, output_path) # Fallback to copy
             return False, sf.info(input_path).duration if os.path.exists(input_path) and os.path.getsize(input_path) > 0 else 0
        return True, actual_duration_after_tempo
    except ffmpeg.Error as e:
        error_message = e.stderr.decode('utf8', 'ignore') if e.stderr else str(e)
        if log_prefix: print(f"{log_prefix}  EXCEPTION applying atempo ({', '.join(filter_str_for_log)}) to {base_input_name}: {error_message[:500]}")
        if input_path != output_path:
             try: shutil.copy(input_path, output_path) # Fallback to copy
             except Exception as copy_e: print(f"{log_prefix}    Fallback copy also failed for {base_input_name}: {copy_e}")
        return False, sf.info(input_path).duration if os.path.exists(input_path) and os.path.getsize(input_path) > 0 else 0


def _apply_final_adjustment(input_path, output_path, samplerate,
                            target_duration, current_duration,
                            log_prefix="    "):
    base_input_name = os.path.basename(input_path)
    operation_performed = "None"

    # 1. Обработка отсутствующего или пустого входного файла
    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        if log_prefix: print(f"{log_prefix}Input file '{base_input_name}' missing or empty for final adjustment. Creating silence of {target_duration:.3f}s.")
        (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
            .output(output_path, t=max(0.01, target_duration), acodec='pcm_s16le')
            .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return True, max(0.01, target_duration), "Created silence (input missing/empty)"

    # 2. Обработка очень коротких входных или целевых длительностей
    if current_duration <= 0.01 and target_duration <= 0.01 :
         if input_path != output_path: shutil.copy(input_path, output_path)
         return True, current_duration, "Copied (both durations tiny)"
    elif target_duration <= 0.01:
        if log_prefix: print(f"{log_prefix}Target duration for '{base_input_name}' is <= 0.01s. Creating minimal silence.")
        (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
            .output(output_path, t=0.01, acodec='pcm_s16le')
            .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return True, 0.01, "Created minimal silence (target tiny)"

    # 3. Основная логика: atempo если нужно, затем trim/pad
    speed_factor_for_atempo = 1.0
    if current_duration > 0.01 and target_duration > 0.01: # Осмысленно считать фактор
        speed_factor_for_atempo = current_duration / target_duration

    duration_after_atempo = current_duration
    path_after_atempo = input_path # По умолчанию, если atempo не применяется
    temp_atempo_output = None

    # Применять atempo, если фактор выходит за пределы "небольшого" отклонения
    # и находится в разумных пределах для atempo.
    needs_atempo = not (0.95 < speed_factor_for_atempo < 1.05) # Если отклонение > 5%
    
    if needs_atempo and (ATEMPO_MIN_FACTOR <= speed_factor_for_atempo <= ATEMPO_MAX_FACTOR):
        if log_prefix: print(f"{log_prefix}Attempting FFmpeg atempo for '{base_input_name}'. Current: {current_duration:.3f}s, Target: {target_duration:.3f}s, Factor: {speed_factor_for_atempo:.3f}")
        
        temp_atempo_output = os.path.join(os.path.dirname(output_path), f"temp_atempo_{base_input_name}")
        success_atempo, dur_atempo = _apply_tempo_adjustment_ffmpeg_internal(
            input_path, temp_atempo_output, speed_factor_for_atempo, samplerate, log_prefix + "  (Atempo) "
        )
        if success_atempo and os.path.exists(temp_atempo_output) and os.path.getsize(temp_atempo_output) > 0:
            duration_after_atempo = dur_atempo
            path_after_atempo = temp_atempo_output
            operation_performed = f"Atempo by {speed_factor_for_atempo:.3f}"
            if log_prefix: print(f"{log_prefix}Atempo applied to '{base_input_name}'. New duration: {duration_after_atempo:.3f}s")
        else:
            if log_prefix: print(f"{log_prefix}Atempo failed or produced empty file for '{base_input_name}'. Using original for trim/pad.")
            # path_after_atempo остается input_path, duration_after_atempo остается current_duration
            if temp_atempo_output and os.path.exists(temp_atempo_output): # Удаляем неудавшийся временный файл
                try: os.remove(temp_atempo_output)
                except: pass
            temp_atempo_output = None # Сбрасываем, чтобы не пытаться удалить его позже
            operation_performed = "Atempo_Failed"

    elif needs_atempo: # Требуется изменение скорости, но фактор выходит за пределы ATEMPO_MIN/MAX_FACTOR
         if log_prefix: print(f"{log_prefix}Required speed factor {speed_factor_for_atempo:.3f} for '{base_input_name}' is outside atempo range [{ATEMPO_MIN_FACTOR}-{ATEMPO_MAX_FACTOR}]. Will use trim/pad only.")
         operation_performed = f"Atempo_Skipped (factor {speed_factor_for_atempo:.3f} out of range)"


    # 4. Применение trim или pad к результату (либо исходному, либо после atempo)
    final_stream_input_path_for_ffmpeg = path_after_atempo
    temp_ffmpeg_input_if_needed_for_trim_pad = None

    if os.path.abspath(path_after_atempo) == os.path.abspath(output_path):
         temp_ffmpeg_input_if_needed_for_trim_pad = os.path.join(os.path.dirname(output_path), f"temp_trimpad_ff_in_{base_input_name}")
         shutil.copy(path_after_atempo, temp_ffmpeg_input_if_needed_for_trim_pad)
         final_stream_input_path_for_ffmpeg = temp_ffmpeg_input_if_needed_for_trim_pad

    final_stream = ffmpeg.input(final_stream_input_path_for_ffmpeg)
    duration_diff = target_duration - duration_after_atempo

    trim_pad_op_performed = False
    if abs(duration_diff) >= 0.005 and target_duration > 0.01: # Порог для операции
        if duration_diff > 0: # Нужно добавить тишину (Padding)
            pad_seconds = round(duration_diff, 3)
            final_stream = ffmpeg.filter(final_stream, 'apad', pad_dur=f'{pad_seconds}s')
            current_op_detail = f"Padded by {pad_seconds:.3f}s"
            if log_prefix: print(f"{log_prefix}Padding '{base_input_name}' by {pad_seconds:.3f}s (current_dur: {duration_after_atempo:.3f}s, target: {target_duration:.3f}s)")
        elif duration_diff < 0: # Нужно обрезать (Trimming)
            final_stream = ffmpeg.filter(final_stream, 'atrim', start='0', end=str(round(target_duration,3)))
            current_op_detail = f"Trimmed to {target_duration:.3f}s"
            if log_prefix: print(f"{log_prefix}Trimming '{base_input_name}' to {target_duration:.3f}s (current_dur: {duration_after_atempo:.3f}s)")
        
        if operation_performed == "None" or "Atempo_Skipped" in operation_performed or "Atempo_Failed" in operation_performed :
            operation_performed = current_op_detail
        else:
            operation_performed += f"; {current_op_detail}"
        
        ffmpeg.output(final_stream, output_path, acodec='pcm_s16le', ar=samplerate).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        trim_pad_op_performed = True
    
    if not trim_pad_op_performed: # Если trim/pad не делался (разница мала или целевая 0)
        if final_stream_input_path_for_ffmpeg != output_path:
            shutil.copy(final_stream_input_path_for_ffmpeg, output_path)
        if operation_performed == "None": operation_performed = "Copied (duration diff small or target zero)"


    # Очистка временных файлов
    if temp_atempo_output and os.path.exists(temp_atempo_output) and \
       (temp_ffmpeg_input_if_needed_for_trim_pad is None or os.path.abspath(temp_atempo_output) != os.path.abspath(temp_ffmpeg_input_if_needed_for_trim_pad)):
        try: os.remove(temp_atempo_output)
        except: pass
    if temp_ffmpeg_input_if_needed_for_trim_pad and os.path.exists(temp_ffmpeg_input_if_needed_for_trim_pad):
        try: os.remove(temp_ffmpeg_input_if_needed_for_trim_pad)
        except: pass

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        if log_prefix: print(f"{log_prefix}Output file '{os.path.basename(output_path)}' is missing or empty after {operation_performed}. Creating fallback silence.")
        (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
            .output(output_path, t=max(0.01, target_duration), acodec='pcm_s16le')
            .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return True, max(0.01, target_duration), f"Fallback silence (output invalid after {operation_performed})"

    final_duration = sf.info(output_path).duration
    return True, final_duration, operation_performed

    # Исключения обрабатываются в вызывающей функции synthesize_speech_segments


def synthesize_speech_segments(segments, reference_audio_path, temp_dir, diarization_result_df=None, language='ru', progress_callback=None,
                               min_segment_duration_for_synth_processing = 0.05
                               ): # Убрали флаг use_radical_trim_pad_only_test_flag
    tts_model = load_tts_model()
    if tts_model is None: raise RuntimeError("TTS model could not be loaded.")

    output_segments_dir = os.path.join(temp_dir, "tts_adjusted_segments"); os.makedirs(output_segments_dir, exist_ok=True)
    raw_tts_dir = os.path.join(temp_dir, "tts_raw_segments"); os.makedirs(raw_tts_dir, exist_ok=True)

    processed_segment_files = []
    speaker_references = {}
    first_valid_speaker_ref = None
    last_segment_original_end_time = 0.0

    total_raw_tts_duration_s1_accumulator = 0.0 
    total_final_segment_duration_sum = 0.0 

    total_segments = len(segments)
    print(f"Starting voice synthesis & tempo adjustment for {total_segments} segments...")
    print(f"  Min original duration for TTS attempt: {MIN_ORIGINAL_DURATION_FOR_TTS_ATTEMPT}s")
    print(f"  Min segment duration for synth processing (after TTS): {min_segment_duration_for_synth_processing}s")
    print(f"  TTS Speed adjustment range: [{TTS_SPEED_MIN} - {TTS_SPEED_MAX}]")
    print(f"  FFmpeg Atempo factor range for fine-tuning: [{ATEMPO_MIN_FACTOR} - {ATEMPO_MAX_FACTOR}]")


    for i, segment in enumerate(tqdm(segments, desc="Synthesizing Segments", unit="segment") if not progress_callback else segments):
        raw_text_from_segment = segment.get('translated_text', segment.get('text', ''))
        text_to_synth_cleaned = re.sub(r'<[^>]+>', '', raw_text_from_segment).strip()

        speaker_id = segment.get('speaker', 'SPEAKER_UNKNOWN')
        original_start = segment.get('start'); original_end = segment.get('end')
        
        segment_base_name = f"segment_{i:04d}_{speaker_id}"
        final_output_for_this_segment_path = os.path.join(output_segments_dir, f"{segment_base_name}_final.wav")

        log_this_segment_details = (i < 5) or (i > total_segments - 5) or (i % 20 == 0) # Логируем чаще

        if original_start is None or original_end is None:
            if log_this_segment_details: print(f"Segment {i+1} missing start/end time. Skipping.")
            if progress_callback: progress_callback((i + 1) / total_segments)
            continue

        original_duration = original_end - original_start
        synthesized_samplerate = 24000 

        silence_before_duration = original_start - last_segment_original_end_time
        if silence_before_duration > 0.01: 
            silence_before_path = os.path.join(output_segments_dir, f"silence_before_seg_{i:04d}.wav")
            try:
                (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                 .output(silence_before_path, t=silence_before_duration, acodec='pcm_s16le')
                 .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                if os.path.exists(silence_before_path) and os.path.getsize(silence_before_path) > 0:
                    processed_segment_files.append(silence_before_path)
                    if log_this_segment_details: print(f"  Added silence before seg {i+1}: {silence_before_duration:.3f}s")
            except Exception as e_sil: print(f"  Warning: Failed to generate silence before seg {i+1}: {e_sil}")

        segment_processed_successfully_with_sound = False
        current_segment_final_duration = 0.0
        
        try:
            if not text_to_synth_cleaned.strip() or original_duration < MIN_ORIGINAL_DURATION_FOR_TTS_ATTEMPT:
                reason = "empty text" if not text_to_synth_cleaned.strip() else f"orig_dur {original_duration:.3f}s < {MIN_ORIGINAL_DURATION_FOR_TTS_ATTEMPT}s"
                if log_this_segment_details:
                    print(f"\nSegment {i+1}/{total_segments} (Orig Dur: {original_duration:.3f}s) - {reason}. Creating silence.")
                (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                    .output(final_output_for_this_segment_path, t=max(0.01, original_duration), acodec='pcm_s16le')
                    .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                if os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                     segment_processed_successfully_with_sound = True
                     current_segment_final_duration = sf.info(final_output_for_this_segment_path).duration
            else: 
                if log_this_segment_details:
                    print(f"\nProcessing TTS for segment {i+1}/{total_segments} (Speaker: {speaker_id}, Orig Dur: {original_duration:.3f}s)")
                    print(f"  Text: '{text_to_synth_cleaned[:60]}...' -> {os.path.basename(final_output_for_this_segment_path)}")

                if speaker_id not in speaker_references:
                    current_speaker_wav_path = _get_or_create_speaker_wav(speaker_id, segment, reference_audio_path, temp_dir, diarization_df=diarization_result_df)
                    if current_speaker_wav_path:
                        speaker_references[speaker_id] = current_speaker_wav_path
                        if first_valid_speaker_ref is None: first_valid_speaker_ref = current_speaker_wav_path
                    else: current_speaker_wav_path = first_valid_speaker_ref
                else: current_speaker_wav_path = speaker_references[speaker_id]
                active_speaker_ref = current_speaker_wav_path if current_speaker_wav_path else first_valid_speaker_ref
                if active_speaker_ref is None and log_this_segment_details:
                     print(f"  Warning: No speaker reference for seg {i+1}. TTS may use default voice or fail.")

                calculated_speed = 1.0
                raw_tts_duration_for_speed_calc = 0.0
                
                probe_chunks_dir = os.path.join(raw_tts_dir, f"probe_{segment_base_name}")
                os.makedirs(probe_chunks_dir, exist_ok=True)
                
                text_chunks_for_probe_and_final = _split_text_for_tts(text_to_synth_cleaned, max_length=XTTS_RU_MAX_CHARS)
                probe_sub_segment_paths = []

                if text_chunks_for_probe_and_final:
                    for chunk_idx, text_chunk_probe in enumerate(text_chunks_for_probe_and_final):
                        probe_sub_file = os.path.join(probe_chunks_dir, f"probe_sub_{chunk_idx:02d}.wav")
                        tts_model.tts_to_file(text=text_chunk_probe, speaker_wav=active_speaker_ref, language=language, 
                                                file_path=probe_sub_file, split_sentences=True, speed=1.0)
                        if os.path.exists(probe_sub_file) and os.path.getsize(probe_sub_file) > 0:
                            probe_sub_segment_paths.append(probe_sub_file)
                    
                    if probe_sub_segment_paths:
                        temp_combined_probe_file = os.path.join(probe_chunks_dir, "combined_probe.wav")
                        if os.path.exists(temp_combined_probe_file): os.remove(temp_combined_probe_file)
                        video_processor.merge_audio_segments(probe_sub_segment_paths, temp_combined_probe_file)
                        
                        if os.path.exists(temp_combined_probe_file) and os.path.getsize(temp_combined_probe_file) > 0:
                            raw_tts_duration_for_speed_calc = sf.info(temp_combined_probe_file).duration
                            if log_this_segment_details: print(f"    Probe TTS total duration for seg {i+1} (speed=1.0): {raw_tts_duration_for_speed_calc:.3f}s")
                
                if raw_tts_duration_for_speed_calc > 0:
                     total_raw_tts_duration_s1_accumulator += raw_tts_duration_for_speed_calc
                
                if os.path.exists(probe_chunks_dir): shutil.rmtree(probe_chunks_dir)

                if raw_tts_duration_for_speed_calc > 0.05 and original_duration > 0.05:
                    calculated_speed = raw_tts_duration_for_speed_calc / original_duration
                    calculated_speed = max(TTS_SPEED_MIN, min(TTS_SPEED_MAX, calculated_speed))
                    if log_this_segment_details: print(f"    Calculated TTS speed for seg {i+1}: {calculated_speed:.3f}")
                elif log_this_segment_details: print(f"    Using default TTS speed 1.0 for seg {i+1}")

                synthesized_sub_segments_paths = []
                if text_chunks_for_probe_and_final:
                    for chunk_idx, text_chunk_final in enumerate(text_chunks_for_probe_and_final):
                        raw_sub_segment_filename = os.path.join(raw_tts_dir, f"{segment_base_name}_sub_{chunk_idx:02d}.wav")
                        tts_model.tts_to_file(text=text_chunk_final, speaker_wav=active_speaker_ref, language=language, 
                                                file_path=raw_sub_segment_filename, split_sentences=True, speed=calculated_speed)
                        if os.path.exists(raw_sub_segment_filename) and os.path.getsize(raw_sub_segment_filename) > 0:
                            synthesized_sub_segments_paths.append(raw_sub_segment_filename)
                
                raw_segment_filename_combined = os.path.join(raw_tts_dir, f"{segment_base_name}_combined.wav")
                synthesized_duration_after_speed_adj = 0.0

                if synthesized_sub_segments_paths:
                    if os.path.exists(raw_segment_filename_combined): os.remove(raw_segment_filename_combined)
                    video_processor.merge_audio_segments(synthesized_sub_segments_paths, raw_segment_filename_combined)
                    if os.path.exists(raw_segment_filename_combined) and os.path.getsize(raw_segment_filename_combined) > 0:
                        synthesized_duration_after_speed_adj = sf.info(raw_segment_filename_combined).duration
                        if log_this_segment_details: print(f"    Combined final TTS for seg {i+1} (TTS speed: {calculated_speed:.2f}) duration: {synthesized_duration_after_speed_adj:.3f}s")
                else:
                    if log_this_segment_details : print(f"  Warning: No sub-segments synthesized for seg {i+1} with TTS speed {calculated_speed:.2f}.")


                if synthesized_duration_after_speed_adj <= 0.01 or original_duration <= min_segment_duration_for_synth_processing:
                    reason = "synth_dur_adj too short" if synthesized_duration_after_speed_adj <= 0.01 else f"orig_dur {original_duration:.3f}s <= min_process_dur"
                    if log_this_segment_details : print(f"  Seg {i+1} ({reason}). Creating silence instead of fine-tuning.")
                    (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                        .output(final_output_for_this_segment_path, t=max(0.01, original_duration), acodec='pcm_s16le')
                        .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                    if os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                         segment_processed_successfully_with_sound = True
                         current_segment_final_duration = sf.info(final_output_for_this_segment_path).duration
                else: 
                    if log_this_segment_details: print(f"  Fine-tuning seg {i+1}: SynthDur@SpeedAdj: {synthesized_duration_after_speed_adj:.3f}s, TargetOrigDur: {original_duration:.3f}s")
                    
                    _success, final_dur_val, op_details = _apply_final_adjustment(
                        raw_segment_filename_combined, final_output_for_this_segment_path,
                        synthesized_samplerate, original_duration, synthesized_duration_after_speed_adj,
                        log_prefix="    (FineTune) "
                    )
                    if _success and os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                        segment_processed_successfully_with_sound = True
                        current_segment_final_duration = final_dur_val
                        if log_this_segment_details: print(f"  Seg {i+1} fine-tuned: Op='{op_details}', FinalDur={final_dur_val:.3f}s (Target={original_duration:.3f}s)")
                        if abs(final_dur_val - original_duration) > 0.05 and original_duration > 0.1: 
                             if log_this_segment_details: print(f"    ALERT: Final duration {final_dur_val:.3f}s differs significantly from target {original_duration:.3f}s for seg {i+1}.")
                    else:
                        if log_this_segment_details: print(f"  Error or empty output after _apply_final_adjustment for seg {i+1}. Op='{op_details}'. Fallback to silence.")
                        # Fallback to silence if fine-tuning failed
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
            if log_this_segment_details: print(f"  Appended to merge list: {os.path.basename(final_output_for_this_segment_path)} (ActualDur: {current_segment_final_duration:.3f}s)")
        else:
            # This case should ideally be handled by fallbacks within the try-except block above
            if log_this_segment_details: print(f"  WARNING: Segment {i+1} processing failed to produce a valid sound file. A gap might appear or be handled by merge_audio_segments.")
            # No explicit fallback here, as merge_audio_segments has its own logic for missing/empty files.

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
        video_processor.merge_audio_segments(processed_segment_files, final_audio_path)

    final_merged_audio_duration = 0.0
    if os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 0:
        try:
            final_merged_audio_duration = sf.info(final_audio_path).duration
            print(f"Final dubbed audio merged. Actual duration: {final_merged_audio_duration:.2f}s")
        except Exception as e: print(f"Could not get duration of final merged audio: {e}")
    else: print(f"Warning: Final merged audio file '{os.path.basename(final_audio_path)}' is missing or empty after merge attempt.")
    
    calculated_sum_from_processed_files = 0
    for f_path in processed_segment_files:
        if os.path.exists(f_path) and os.path.getsize(f_path) > 0:
            try: calculated_sum_from_processed_files += sf.info(f_path).duration
            except: pass 
    print(f"Sum of durations of individual files sent to merge: {calculated_sum_from_processed_files:.2f}s")

    return final_audio_path, total_raw_tts_duration_s1_accumulator, final_merged_audio_duration