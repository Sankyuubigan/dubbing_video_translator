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
        # print(f"Warning: Text splitting failed to produce any chunks for text of length {len(text)}. Using original text split by max_length as fallback.")
        for i_f in range(0, len(text), max_length): final_chunks.append(text[i_f:i_f+max_length])
    return [c for c in final_chunks if c.strip()] if final_chunks else ([text[:max_length]] if text.strip() else [])

def _get_or_create_speaker_wav(speaker_id, segment_for_ref_timing, base_audio_path, temp_dir, diarization_df=None, min_duration=2.5, max_duration=12.0):
    speaker_wav_dir = os.path.join(temp_dir, "speaker_wavs"); os.makedirs(speaker_wav_dir, exist_ok=True)
    speaker_ref_path = os.path.join(speaker_wav_dir, f"{speaker_id}_ref.wav")
    if os.path.exists(speaker_ref_path) and os.path.getsize(speaker_ref_path) > 1000: # Проверка на размер > 1KB
        try:
            info = sf.info(speaker_ref_path)
            if info.duration >= 0.5: return speaker_ref_path # Минимальная длительность референса 0.5с
        except Exception: pass # Если sf.info падает, файл битый, пересоздаем
    ref_start, ref_end = None, None
    if diarization_df is not None and not diarization_df.empty and speaker_id in diarization_df['speaker'].unique():
        speaker_segments_df = diarization_df[diarization_df['speaker'] == speaker_id].copy()
        speaker_segments_df['duration'] = speaker_segments_df['end'] - speaker_segments_df['start']
        # Ищем сегмент идеальной длины
        ideal_segments = speaker_segments_df[(speaker_segments_df['duration'] >= 3.0) & (speaker_segments_df['duration'] <= 10.0)]
        if not ideal_segments.empty:
            best_segment_row = ideal_segments.sample(1).iloc[0] if len(ideal_segments) > 1 else ideal_segments.iloc[0]
            ref_start, ref_end = best_segment_row['start'], best_segment_row['end']
        else: # Если нет идеальных, берем самый длинный подходящий
            longer_segments = speaker_segments_df[speaker_segments_df['duration'] >= min_duration]
            if not longer_segments.empty:
                best_segment_row = longer_segments.sort_values(by='duration', ascending=False).iloc[0]
                ref_start = best_segment_row['start']
                ref_end = min(best_segment_row['end'], best_segment_row['start'] + max_duration) # Ограничиваем максимальную длину
    # Если по диаризации не нашли, пробуем из текущего SRT сегмента (если он достаточно длинный)
    if ref_start is None and segment_for_ref_timing:
        srt_start_time = segment_for_ref_timing.get('start'); srt_end_time = segment_for_ref_timing.get('end')
        if srt_start_time is not None and srt_end_time is not None:
            srt_duration = srt_end_time - srt_start_time
            if srt_duration >= min_duration:
                 ref_start = srt_start_time
                 ref_end = min(srt_end_time, srt_start_time + max_duration)
    if ref_start is None or ref_end is None or (ref_end - ref_start) < 0.5: # Минимальная длительность референса 0.5с
        # print(f"    Could not find suitable audio part for speaker {speaker_id} reference.")
        return None
    try:
        if not os.path.exists(base_audio_path): raise FileNotFoundError(f"Base audio for speaker ref not found: {base_audio_path}")
        # print(f"    Extracting speaker ref for {speaker_id} from {ref_start:.2f}s to {ref_end:.2f}s")
        (ffmpeg.input(base_audio_path, ss=ref_start, to=ref_end)
         .output(speaker_ref_path, ar=24000, ac=1, acodec='pcm_s16le') # XTTS v2 требует 24kHz
         .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        if not os.path.exists(speaker_ref_path) or os.path.getsize(speaker_ref_path) < 1000: # Проверка на размер
             if os.path.exists(speaker_ref_path): os.remove(speaker_ref_path)
             return None
        info = sf.info(speaker_ref_path) # Проверка, что файл читается
        if info.duration < 0.5: # Минимальная итоговая длительность референса
            os.remove(speaker_ref_path)
            return None
        return speaker_ref_path
    except Exception as e:
        # print(f"    Error extracting speaker reference for {speaker_id}: {e}")
        if os.path.exists(speaker_ref_path):
            try: os.remove(speaker_ref_path)
            except OSError: pass
        return None

def _apply_tempo_and_trim_pad(input_path, output_path, samplerate,
                              target_duration,
                              log_prefix="    ",
                              radical_trim_pad_only=True):
    base_input_name = os.path.basename(input_path)
    operation_performed = "None" # Для логирования

    try:
        # 1. Обработка отсутствующего или пустого входного файла
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            # print(f"{log_prefix}Input file '{base_input_name}' missing or empty. Creating silence of {target_duration:.3f}s.")
            (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
                .output(output_path, t=max(0.01, target_duration), acodec='pcm_s16le')
                .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            return True, max(0.01, target_duration), "Created silence (input missing/empty)"

        input_info = sf.info(input_path)
        current_duration = input_info.duration

        # 2. Обработка очень коротких входных или целевых длительностей
        if current_duration <= 0.01 and target_duration <= 0.01 :
             if input_path != output_path: shutil.copy(input_path, output_path)
             return True, current_duration, "Copied (both durations tiny)"
        elif target_duration <= 0.01:
            # print(f"{log_prefix}Target duration for '{base_input_name}' is <= 0.01s. Creating minimal silence.")
            (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
                .output(output_path, t=0.01, acodec='pcm_s16le')
                .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            return True, 0.01, "Created minimal silence (target tiny)"

        # 3. Основная логика (только trim/pad, если radical_trim_pad_only=True)
        duration_after_processing = current_duration
        processed_audio_path = input_path # По умолчанию, если atempo не применяется

        if radical_trim_pad_only:
            if log_prefix: print(f"{log_prefix}Adjusting '{base_input_name}' from {current_duration:.3f}s to {target_duration:.3f}s using ONLY TRIM/PAD.")
        else: # Эта ветка сейчас не должна выполняться
            # ... (старый код atempo, который сейчас не используется) ...
            pass

        # Проверка, что processed_audio_path все еще валиден
        if not os.path.exists(processed_audio_path) or os.path.getsize(processed_audio_path) == 0:
            # print(f"{log_prefix}File '{os.path.basename(processed_audio_path)}' became invalid. Copying original to output.")
            if output_path != input_path and os.path.exists(input_path): shutil.copy(input_path, output_path)
            # Если и исходный пропал, или это тот же файл, то что-то совсем не так
            elif not os.path.exists(output_path) or os.path.getsize(output_path) == 0 :
                 (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
                    .output(output_path, t=max(0.01, target_duration), acodec='pcm_s16le')
                    .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                 return True, max(0.01, target_duration), "Created silence (processed invalid, input too)"
            return False, current_duration, "Error (processed invalid)"


        # 4. Применение trim или pad
        final_stream_input_path_for_ffmpeg = processed_audio_path
        temp_ffmpeg_input_if_needed = None

        # Если processed_audio_path это и есть output_path, нужно создать временную копию для ffmpeg
        if os.path.abspath(processed_audio_path) == os.path.abspath(output_path):
             temp_ffmpeg_input_if_needed = os.path.join(os.path.dirname(output_path), f"temp_final_ff_in_{base_input_name}")
             shutil.copy(processed_audio_path, temp_ffmpeg_input_if_needed)
             final_stream_input_path_for_ffmpeg = temp_ffmpeg_input_if_needed

        final_stream = ffmpeg.input(final_stream_input_path_for_ffmpeg)
        duration_diff = target_duration - duration_after_processing # duration_after_processing это current_duration в radical_trim_pad_only режиме

        if abs(duration_diff) >= 0.005 and target_duration > 0.01: # Порог для операции
            if duration_diff > 0: # Нужно добавить тишину (Padding)
                pad_seconds = round(duration_diff, 3)
                final_stream = ffmpeg.filter(final_stream, 'apad', pad_dur=f'{pad_seconds}s')
                operation_performed = f"Padded by {pad_seconds:.3f}s"
                if log_prefix: print(f"{log_prefix}Padding '{base_input_name}' by {pad_seconds:.3f}s (current_dur: {duration_after_processing:.3f}s, target: {target_duration:.3f}s)")
            elif duration_diff < 0: # Нужно обрезать (Trimming)
                final_stream = ffmpeg.filter(final_stream, 'atrim', start='0', end=str(round(target_duration,3)))
                operation_performed = f"Trimmed to {target_duration:.3f}s"
                if log_prefix: print(f"{log_prefix}Trimming '{base_input_name}' to {target_duration:.3f}s (current_dur: {duration_after_processing:.3f}s)")
            ffmpeg.output(final_stream, output_path, acodec='pcm_s16le', ar=samplerate).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        else: # Если разница очень мала или целевая длительность 0, просто копируем
            if final_stream_input_path_for_ffmpeg != output_path:
                shutil.copy(final_stream_input_path_for_ffmpeg, output_path)
            operation_performed = "Copied (duration diff small or target zero)"

        # Очистка временных файлов
        if processed_audio_path != input_path and os.path.exists(processed_audio_path) and \
           (temp_ffmpeg_input_if_needed is None or os.path.abspath(processed_audio_path) != os.path.abspath(temp_ffmpeg_input_if_needed)):
            os.remove(processed_audio_path)
        if temp_ffmpeg_input_if_needed and os.path.exists(temp_ffmpeg_input_if_needed):
            os.remove(temp_ffmpeg_input_if_needed)

        # Проверка выходного файла
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            # print(f"{log_prefix}Output file '{os.path.basename(output_path)}' is missing or empty after {operation_performed}. Creating fallback silence.")
            (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
                .output(output_path, t=max(0.01, target_duration), acodec='pcm_s16le')
                .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            return True, max(0.01, target_duration), f"Fallback silence (output invalid after {operation_performed})"

        final_duration = sf.info(output_path).duration
        return True, final_duration, operation_performed

    except Exception as e:
        print(f"{log_prefix}CRITICAL Error in _apply_tempo_and_trim_pad for '{base_input_name}': {e} \n {traceback.format_exc()}")
        # В случае критической ошибки, создаем тишину нужной длины в выходном файле
        try:
            (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
                .output(output_path, t=max(0.01, target_duration), acodec='pcm_s16le')
                .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            return True, max(0.01, target_duration), "Fallback silence (exception in function)"
        except Exception as e_fallback:
            print(f"{log_prefix}  Failed to create fallback silence for '{base_input_name}': {e_fallback}")
            return False, 0, "Error (exception and fallback failed)"


def synthesize_speech_segments(segments, reference_audio_path, temp_dir, diarization_result_df=None, language='ru', progress_callback=None,
                               min_segment_duration_for_synth_processing = 0.05, # Уменьшил, чтобы больше сегментов проходило обработку
                               use_radical_trim_pad_only_test_flag = True
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
    total_final_segment_duration_sum = 0.0 # Будет суммой длительностей файлов в processed_segment_files

    total_segments = len(segments)
    print(f"Starting voice synthesis & tempo adjustment for {total_segments} segments...")
    print(f"  Min original duration for TTS attempt: {MIN_ORIGINAL_DURATION_FOR_TTS_ATTEMPT}s")
    print(f"  Min segment duration for synth processing (after TTS): {min_segment_duration_for_synth_processing}s")
    if use_radical_trim_pad_only_test_flag: print(f"  Using ONLY TRIM/PAD for duration adjustment.")


    for i, segment in enumerate(tqdm(segments, desc="Synthesizing Segments", unit="segment") if not progress_callback else segments):
        raw_text_from_segment = segment.get('translated_text', segment.get('text', ''))
        text_to_synth_cleaned = re.sub(r'<[^>]+>', '', raw_text_from_segment).strip()

        speaker_id = segment.get('speaker', 'SPEAKER_UNKNOWN')
        original_start = segment.get('start'); original_end = segment.get('end')
        
        # Определяем путь к файлу, который будет добавлен в `processed_segment_files`
        # Это может быть синтезированный и обработанный сегмент, или тишина.
        # Имя файла теперь включает и "adj" и "silence" для ясности, если это тишина.
        segment_base_name = f"segment_{i:04d}_{speaker_id}"
        final_output_for_this_segment_path = os.path.join(output_segments_dir, f"{segment_base_name}_final.wav")

        log_this_segment_details = (i < 5) or (i > total_segments - 5) or (i % 50 == 0) # Логируем начало, конец и каждые 50

        if original_start is None or original_end is None:
            if log_this_segment_details: print(f"Segment {i+1} missing start/end time. Skipping (no silence will be added for this gap).")
            if progress_callback: progress_callback((i + 1) / total_segments)
            # Не меняем last_segment_original_end_time, чтобы следующий сегмент корректно рассчитал тишину до себя
            continue

        original_duration = original_end - original_start
        synthesized_samplerate = 24000 # Стандартная для XTTS

        # 1. Добавляем тишину ПЕРЕД текущим сегментом, если есть разрыв
        silence_before_duration = original_start - last_segment_original_end_time
        if silence_before_duration > 0.01: # Минимальная тишина 10мс
            silence_before_path = os.path.join(output_segments_dir, f"silence_before_seg_{i:04d}.wav")
            try:
                (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                 .output(silence_before_path, t=silence_before_duration, acodec='pcm_s16le')
                 .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                if os.path.exists(silence_before_path) and os.path.getsize(silence_before_path) > 0:
                    processed_segment_files.append(silence_before_path)
                    # total_final_segment_duration_sum += silence_before_duration # Суммируем в конце по файлам
                    if log_this_segment_details: print(f"  Added silence before seg {i+1}: {silence_before_duration:.3f}s")
                else:
                     if log_this_segment_details: print(f"  Warning: Generated silence_before_seg_{i:04d}.wav is missing or empty.")
            except Exception as e_sil: print(f"  Warning: Failed to generate silence before seg {i+1}: {e_sil}")

        # 2. Логика обработки самого сегмента (TTS или тишина)
        segment_processed_successfully_with_sound = False
        if not text_to_synth_cleaned.strip() or original_duration < MIN_ORIGINAL_DURATION_FOR_TTS_ATTEMPT:
            reason = "empty text" if not text_to_synth_cleaned.strip() else f"orig_dur {original_duration:.3f}s < {MIN_ORIGINAL_DURATION_FOR_TTS_ATTEMPT}s"
            if log_this_segment_details:
                print(f"\nSegment {i+1}/{total_segments} (Orig Dur: {original_duration:.3f}s) - {reason}. Creating silence directly for path: {final_output_for_this_segment_path}")
            # final_output_for_this_segment_path уже определен выше
            try:
                 (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                    .output(final_output_for_this_segment_path, t=max(0.01, original_duration), acodec='pcm_s16le')
                    .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                 if os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                     segment_processed_successfully_with_sound = True # Технически это тишина, но файл создан
                 else:
                     if log_this_segment_details: print(f"  Error: Direct silence file for seg {i+1} is missing or empty.")
            except Exception as e_direct_silence:
                if log_this_segment_details: print(f"  Error creating direct silence for seg {i+1}: {e_direct_silence}")
        else: # Попытка синтеза речи
            if log_this_segment_details:
                print(f"\nProcessing TTS for segment {i+1}/{total_segments} (Speaker: {speaker_id}, Orig Dur: {original_duration:.3f}s)")
                print(f"  Text: '{text_to_synth_cleaned[:60]}...' -> {final_output_for_this_segment_path}")

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

            text_chunks = _split_text_for_tts(text_to_synth_cleaned, max_length=XTTS_RU_MAX_CHARS)
            synthesized_sub_segments_paths = []
            for chunk_idx, text_chunk in enumerate(text_chunks):
                if not text_chunk.strip(): continue
                raw_sub_segment_filename = os.path.join(raw_tts_dir, f"{segment_base_name}_sub_{chunk_idx:02d}.wav")
                try:
                    tts_model.tts_to_file(text=text_chunk, speaker_wav=active_speaker_ref, language=language, file_path=raw_sub_segment_filename, split_sentences=True) # XTTS сама делит на предложения
                    if os.path.exists(raw_sub_segment_filename) and os.path.getsize(raw_sub_segment_filename) > 0:
                        synthesized_sub_segments_paths.append(raw_sub_segment_filename)
                except Exception as e_tts_sub:
                    if log_this_segment_details : print(f"    Error synthesizing sub-segment {i+1}-{chunk_idx+1}: {e_tts_sub}")

            raw_segment_filename_combined = os.path.join(raw_tts_dir, f"{segment_base_name}_combined.wav")
            synthesized_duration = 0.0
            if not synthesized_sub_segments_paths:
                if log_this_segment_details : print(f"  Warning: No sub-segments synthesized for seg {i+1}.")
            elif len(synthesized_sub_segments_paths) == 1:
                try: shutil.copy(synthesized_sub_segments_paths[0], raw_segment_filename_combined)
                except Exception as e_copy_single:
                    if log_this_segment_details : print(f"  Error copying single sub-segment for seg {i+1}: {e_copy_single}"); raw_segment_filename_combined = None
            else:
                try: video_processor.merge_audio_segments(synthesized_sub_segments_paths, raw_segment_filename_combined)
                except Exception as e_merge_sub:
                    if log_this_segment_details : print(f"  Error merging sub-segments for seg {i+1}: {e_merge_sub}."); raw_segment_filename_combined = None

            if raw_segment_filename_combined and os.path.exists(raw_segment_filename_combined) and os.path.getsize(raw_segment_filename_combined) > 0:
                try:
                    synthesized_info = sf.info(raw_segment_filename_combined)
                    synthesized_duration = synthesized_info.duration; # synthesized_samplerate уже определена
                    total_raw_duration_sum += synthesized_duration # Суммируем сырую длительность TTS
                except Exception as e_sfinfo:
                    synthesized_duration = 0.0
                    if log_this_segment_details : print(f"  Warning: Could not get info for combined raw TTS for seg {i+1}: {e_sfinfo}")
            else:
                if log_this_segment_details : print(f"  Warning: Combined raw TTS for seg {i+1} is missing/empty.")
                synthesized_duration = 0.0

            # Решаем, обрабатывать ли синтезированный звук или создавать тишину для final_output_for_this_segment_path
            if synthesized_duration <= 0.01 or original_duration <= min_segment_duration_for_synth_processing:
                reason = "synth_dur too short" if synthesized_duration <= 0.01 else f"orig_dur {original_duration:.3f}s <= min_process_dur {min_segment_duration_for_synth_processing}s"
                if log_this_segment_details : print(f"  Seg {i+1} ({reason}). Creating silence for path: {final_output_for_this_segment_path}")
                try:
                     (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                        .output(final_output_for_this_segment_path, t=max(0.01, original_duration), acodec='pcm_s16le')
                        .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                     if os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                         segment_processed_successfully_with_sound = True
                except Exception as e_short_silence:
                    if log_this_segment_details: print(f"  Error creating silence for short/failed synth seg {i+1}: {e_short_silence}")
            else: # Есть что обрабатывать (trim/pad)
                if log_this_segment_details: print(f"  Adjusting seg {i+1}: Synth Dur: {synthesized_duration:.3f}s, Target Orig Dur: {original_duration:.3f}s")
                _success, final_dur_val, op_details = _apply_tempo_and_trim_pad(
                    raw_segment_filename_combined, final_output_for_this_segment_path,
                    synthesized_samplerate, original_duration,
                    log_prefix="    (FinalAdj) ", radical_trim_pad_only=use_radical_trim_pad_only_test_flag
                )
                if _success and os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                    segment_processed_successfully_with_sound = True
                    if log_this_segment_details: print(f"  Seg {i+1} processed: Op='{op_details}', FinalDur={final_dur_val:.3f}s (Target={original_duration:.3f}s)")
                    if abs(final_dur_val - original_duration) > 0.05 and original_duration > 0.1: # Порог для "значительного" расхождения
                         if log_this_segment_details: print(f"    ALERT: Final duration {final_dur_val:.3f}s differs from target {original_duration:.3f}s for seg {i+1}.")
                else:
                    if log_this_segment_details: print(f"  Error or empty output after _apply_tempo_and_trim_pad for seg {i+1}. Op='{op_details}'.")


        # 3. Гарантированное добавление файла в processed_segment_files
        # Либо final_output_for_this_segment_path (если он успешно создан), либо принудительная тишина.
        if segment_processed_successfully_with_sound and os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
            processed_segment_files.append(final_output_for_this_segment_path)
            if log_this_segment_details: print(f"  Appended to merge list: {os.path.basename(final_output_for_this_segment_path)}")
        else:
            # Если final_output_for_this_segment_path не был успешно создан (неважно, TTS это был или тишина),
            # создаем здесь ГАРАНТИРОВАННУЮ тишину, чтобы заполнить пробел.
            # Используем то же имя файла, т.к. предыдущая попытка его создать провалилась.
            if log_this_segment_details: print(f"  Fallback: Seg {i+1} processing failed or resulted in empty file. Creating GUARANTEED silence of {original_duration:.3f}s for {os.path.basename(final_output_for_this_segment_path)}.")
            try:
                (ffmpeg.input('anullsrc', format='lavfi', r=synthesized_samplerate, channel_layout='mono')
                 .output(final_output_for_this_segment_path, t=max(0.01, original_duration), acodec='pcm_s16le')
                 .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                if os.path.exists(final_output_for_this_segment_path) and os.path.getsize(final_output_for_this_segment_path) > 0:
                    processed_segment_files.append(final_output_for_this_segment_path)
                    if log_this_segment_details: print(f"    Guaranteed fallback silence created and appended: {os.path.basename(final_output_for_this_segment_path)}")
                else: # Это очень плохая ситуация
                    if log_this_segment_details: print(f"    CRITICAL ERROR: Failed to create GUARANTEED fallback silence for seg {i+1}.")
            except Exception as e_guaranteed_sil:
                if log_this_segment_details: print(f"    CRITICAL ERROR creating GUARANTEED fallback silence for seg {i+1}: {e_guaranteed_sil}")

        last_segment_original_end_time = original_end
        if progress_callback: progress_callback((i + 1) / total_segments)
        if torch.cuda.is_available(): torch.cuda.empty_cache() # Освобождаем память GPU после каждого сегмента

    # 4. Слияние всех собранных сегментов
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
        # Логируем первые и последние несколько файлов, которые идут на слияние
        if len(processed_segment_files) > 10:
            print("  Files for merge (first 5):")
            for f_idx, f_path in enumerate(processed_segment_files[:5]): print(f"    {f_idx}: {os.path.basename(f_path)}")
            print("  Files for merge (last 5):")
            for f_idx, f_path in enumerate(processed_segment_files[-5:]): print(f"    {len(processed_segment_files)-5+f_idx}: {os.path.basename(f_path)}")
        else:
            print("  Files for merge:")
            for f_idx, f_path in enumerate(processed_segment_files): print(f"    {f_idx}: {os.path.basename(f_path)}")

        video_processor.merge_audio_segments(processed_segment_files, final_audio_path)

    # 5. Подсчет итоговой длительности по факту
    final_merged_audio_duration = 0.0
    if os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 0:
        try:
            final_merged_audio_duration = sf.info(final_audio_path).duration
            print(f"Final dubbed audio merged. Actual duration: {final_merged_audio_duration:.2f}s")
        except Exception as e: print(f"Could not get duration of final merged audio: {e}")
    else: print(f"Warning: Final merged audio file '{os.path.basename(final_audio_path)}' is missing or empty after merge attempt.")
    
    # Пересчитываем total_final_segment_duration_sum на основе фактически добавленных файлов
    # (хотя final_merged_audio_duration должно быть точнее, если слияние прошло успешно)
    calculated_sum_from_processed_files = 0
    for f_path in processed_segment_files:
        if os.path.exists(f_path) and os.path.getsize(f_path) > 0:
            try: calculated_sum_from_processed_files += sf.info(f_path).duration
            except: pass # Игнорируем ошибки чтения отдельных файлов при этом подсчете
    print(f"Sum of durations of individual files sent to merge: {calculated_sum_from_processed_files:.2f}s")

    return final_audio_path, total_raw_duration_sum, final_merged_audio_duration # Возвращаем фактическую длительность слитого файла

def _apply_tempo_adjustment_ffmpeg_internal(input_path, output_path, speed_factor_for_atempo, samplerate,
                                   log_prefix="    "): # Эта функция не используется, если radical_trim_pad_only=True
    base_input_name = os.path.basename(input_path)
    if abs(speed_factor_for_atempo - 1.0) < 0.01: # No change needed
        if input_path != output_path:
            try: shutil.copy(input_path, output_path)
            except Exception as e_copy_nochange: print(f"{log_prefix}Error copying {base_input_name} (no tempo change): {e_copy_nochange}")
        return True, sf.info(output_path).duration if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else 0
    # ... (остальная часть функции _apply_tempo_adjustment_ffmpeg_internal без изменений, т.к. она сейчас не вызывается)
    min_atempo_sf_abs = 0.5; max_atempo_sf_abs = 100.0
    effective_max_single_pass = 4.0; effective_min_single_pass = 0.5
    atempo_filters_chain_values = []; current_sf_to_apply = speed_factor_for_atempo
    if not (min_atempo_sf_abs <= current_sf_to_apply <= max_atempo_sf_abs):
        # print(f"{log_prefix}  Atempo speed factor {current_sf_to_apply:.3f} for '{base_input_name}' is outside safe range [{min_atempo_sf_abs}-{max_atempo_sf_abs}]. Clamping.")
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
             # print(f"{log_prefix}  ERROR applying atempo ({', '.join(filter_str_for_log)}) to {base_input_name}. FFmpeg stderr:\n{stderr_str[:500]}")
             if input_path != output_path: shutil.copy(input_path, output_path)
             return False, sf.info(input_path).duration if os.path.exists(input_path) and os.path.getsize(input_path) > 0 else 0
        return True, actual_duration_after_tempo
    except ffmpeg.Error as e:
        error_message = e.stderr.decode('utf8', 'ignore') if e.stderr else str(e)
        # print(f"{log_prefix}  EXCEPTION applying atempo ({', '.join(filter_str_for_log)}) to {base_input_name}: {error_message[:500]}")
        if input_path != output_path:
             try: shutil.copy(input_path, output_path)
             except Exception as copy_e: print(f"{log_prefix}    Fallback copy also failed for {base_input_name}: {copy_e}")
        return False, sf.info(input_path).duration if os.path.exists(input_path) and os.path.getsize(input_path) > 0 else 0