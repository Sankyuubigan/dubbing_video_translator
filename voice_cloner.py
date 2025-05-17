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

# --- Кэш для моделей TTS ---
class TTSCache:
    def __init__(self):
        self.model = None; self.model_name = None; self.device = None
tts_cache = TTSCache()

XTTS_RU_MAX_CHARS = 180 

def load_tts_model(model_name="tts_models/multilingual/multi-dataset/xtts_v2", device="cuda"):
    # ... (без изменений)
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
    # ... (без изменений)
    if len(text) <= max_length:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')) 
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk: 
                chunks.append(current_chunk)
            if len(sentence) > max_length:
                for i in range(0, len(sentence), max_length):
                    chunks.append(sentence[i:i+max_length])
                current_chunk = "" 
            else:
                current_chunk = sentence 
    if current_chunk: 
        chunks.append(current_chunk)
    final_chunks = []
    for ch in chunks:
        if len(ch) > max_length:
            for i in range(0, len(ch), max_length):
                final_chunks.append(ch[i:i+max_length])
        elif ch.strip(): 
            final_chunks.append(ch)
    if not final_chunks and text.strip(): 
        print(f"Warning: Text splitting failed to produce chunks for text of length {len(text)}. Using hard split by max_length.")
        for i in range(0, len(text), max_length):
            final_chunks.append(text[i:i+max_length])
    return [c for c in final_chunks if c.strip()] if final_chunks else [text[:max_length]] 


def _get_or_create_speaker_wav(speaker_id, segment_for_ref_timing, base_audio_path, temp_dir, diarization_df=None, min_duration=2.5, max_duration=12.0):
    # ... (без изменений)
    speaker_wav_dir = os.path.join(temp_dir, "speaker_wavs"); os.makedirs(speaker_wav_dir, exist_ok=True)
    speaker_ref_path = os.path.join(speaker_wav_dir, f"{speaker_id}_ref.wav")
    ref_start, ref_end = None, None
    if diarization_df is not None and not diarization_df.empty and speaker_id in diarization_df['speaker'].unique():
        speaker_segments_df = diarization_df[diarization_df['speaker'] == speaker_id].copy()
        speaker_segments_df['duration'] = speaker_segments_df['end'] - speaker_segments_df['start']
        ideal_min_duration = 4.0
        ideal_max_duration = 8.0
        suitable_segments = speaker_segments_df[
            (speaker_segments_df['duration'] >= ideal_min_duration) & 
            (speaker_segments_df['duration'] <= ideal_max_duration)
        ]
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
        srt_start = segment_for_ref_timing.get('start')
        srt_end = segment_for_ref_timing.get('end')
        if srt_start is not None and srt_end is not None:
            srt_duration = srt_end - srt_start
            if srt_duration >= min_duration:
                ref_start, ref_end = srt_start, min(srt_end, srt_start + max_duration)
    if ref_start is None or ref_end is None or (ref_end - ref_start) < 0.5: 
        print(f"Warning: Could not find suitable reference segment >= 0.5s for {speaker_id}. Cloning quality may suffer. Path: {base_audio_path}")
        return None 
    # print(f"Extracting reference for {speaker_id} from main audio [{ref_start:.2f}s - {ref_end:.2f}s]") # Убрал этот лог, т.к. он частый
    try:
        if not os.path.exists(base_audio_path): raise FileNotFoundError(f"Base audio for speaker ref not found: {base_audio_path}")
        (ffmpeg.input(base_audio_path, ss=ref_start, to=ref_end)
         .output(speaker_ref_path, ar=24000, ac=1, acodec='pcm_s16le')
         .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        if not os.path.exists(speaker_ref_path) or os.path.getsize(speaker_ref_path) == 0:
             print(f"Warning: Extracted reference for {speaker_id} is missing or empty after ffmpeg. Deleting if exists."); 
             if os.path.exists(speaker_ref_path): os.remove(speaker_ref_path); 
             return None
        info = sf.info(speaker_ref_path)
        if info.duration < 0.5: 
             print(f"Warning: Extracted reference for {speaker_id} is too short ({info.duration:.2f}s) after extraction. Deleting."); os.remove(speaker_ref_path); return None
        # print(f"Reference for {speaker_id} created: {os.path.basename(speaker_ref_path)} (Duration: {info.duration:.2f}s)") # Убрал этот лог
        return speaker_ref_path
    except Exception as e:
        print(f"Error extracting speaker wav for {speaker_id}: {e}")
        if os.path.exists(speaker_ref_path):
            try: os.remove(speaker_ref_path)
            except OSError: pass
        return None

def _apply_atempo_iteratively(input_path, output_path, target_tempo, samplerate, 
                             min_step_tempo=0.5, max_step_tempo=2.0, max_steps=5):
    """
    Применяет atempo итеративно, чтобы достичь target_tempo.
    target_tempo: <1 для ускорения, >1 для замедления.
    """
    if abs(target_tempo - 1.0) < 0.01: # Если изменение темпа очень мало
        if input_path != output_path: shutil.copy(input_path, output_path)
        return

    # print(f"    Target atempo: {target_tempo:.3f} for {os.path.basename(input_path)}")
    current_input = input_path
    temp_files = []
    
    # Определяем, сколько раз нужно применить максимальное/минимальное изменение за шаг
    # чтобы приблизиться к target_tempo.
    # Например, если target_tempo = 0.25 (ускорить в 4 раза), и max_step_tempo = 0.5 (ускорить в 2 раза за шаг),
    # то нужно 2 шага: 0.5 * 0.5 = 0.25
    # Если target_tempo = 4.0 (замедлить в 4 раза), и min_step_tempo = 2.0 (замедлить в 2 раза за шаг),
    # то нужно 2 шага: 2.0 * 2.0 = 4.0

    effective_tempo = 1.0 # Начинаем с оригинального темпа
    
    for step_num in range(max_steps):
        if abs(effective_tempo - target_tempo) < 0.01: # Если мы уже достаточно близки
            break

        step_atempo_val = 1.0
        if effective_tempo > target_tempo: # Нужно ускорить (effective_tempo сейчас больше, чем нужно)
            # Хотим effective_tempo * step_atempo_val = target_tempo (или ближе)
            # step_atempo_val = target_tempo / effective_tempo
            # Ограничиваем step_atempo_val значением min_step_tempo (например, 0.5)
            required_change_ratio = target_tempo / effective_tempo
            step_atempo_val = max(min_step_tempo, required_change_ratio)
            # print(f"      Step {step_num+1} (Accelerate): current_eff_tempo={effective_tempo:.3f}, target_tempo={target_tempo:.3f}, req_ratio={required_change_ratio:.3f}, step_val={step_atempo_val:.3f}")

        elif effective_tempo < target_tempo: # Нужно замедлить
            required_change_ratio = target_tempo / effective_tempo
            step_atempo_val = min(max_step_tempo, required_change_ratio)
            # print(f"      Step {step_num+1} (Decelerate): current_eff_tempo={effective_tempo:.3f}, target_tempo={target_tempo:.3f}, req_ratio={required_change_ratio:.3f}, step_val={step_atempo_val:.3f}")

        if abs(step_atempo_val - 1.0) < 0.01: # Если шаг очень маленький, не делаем его
            break

        temp_output_path = os.path.join(os.path.dirname(output_path), f"temp_atempo_{step_num}_{os.path.basename(input_path)}")
        
        try:
            (ffmpeg.input(current_input)
             .filter('atempo', f"{step_atempo_val:.4f}")
             .output(temp_output_path, acodec='pcm_s16le', ar=samplerate)
             .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            
            effective_tempo *= step_atempo_val
            
            if current_input != input_path: # Если current_input был временным файлом
                temp_files.append(current_input)
            current_input = temp_output_path
            
        except ffmpeg.Error as e_atempo:
            print(f"    Error during atempo step {step_num+1} (val: {step_atempo_val:.3f}): {e_atempo.stderr.decode('utf8', 'ignore') if e_atempo.stderr else e_atempo}")
            # Если ошибка, используем предыдущий результат
            break 
    else: # Если цикл завершился без break (т.е. все max_steps выполнены)
        if abs(effective_tempo - target_tempo) > 0.1 : # Если мы все еще далеки от цели
             print(f"    Warning: Max atempo steps ({max_steps}) reached. Effective tempo: {effective_tempo:.3f}, Target: {target_tempo:.3f}")

    if current_input != output_path:
        shutil.move(current_input, output_path)
        if current_input in temp_files: # current_input может быть последним temp_output_path
             temp_files.remove(current_input)
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError as e_clean:
                print(f"    Warning: Could not clean temp atempo file {temp_file}: {e_clean}")


def synthesize_speech_segments(segments, reference_audio_path, temp_dir, diarization_result_df=None, language='ru', progress_callback=None, 
                               min_overall_tempo_factor=0.6, max_overall_tempo_factor=1.8): 
    tts_model = load_tts_model()
    if tts_model is None: raise RuntimeError("TTS model could not be loaded.")

    output_segments_dir = os.path.join(temp_dir, "tts_adjusted_segments"); os.makedirs(output_segments_dir, exist_ok=True)
    raw_tts_dir = os.path.join(temp_dir, "tts_raw_segments"); os.makedirs(raw_tts_dir, exist_ok=True) 

    processed_segment_files = [] 
    speaker_references = {}
    first_valid_speaker_ref = None 
    
    last_segment_original_end_time = 0.0 

    total_segments = len(segments)
    print(f"Starting voice synthesis & tempo adjustment for {total_segments} segments...")
    
    for i, segment in enumerate(segments):
        raw_text_from_segment = segment.get('translated_text', segment.get('text', ''))
        
        text_to_synth_cleaned = re.sub(r'<(\d{2}:\d{2}:\d{2}\.\d{3})><c>', '', raw_text_from_segment)
        text_to_synth_cleaned = re.sub(r'</c>', '', text_to_synth_cleaned)
        text_to_synth_cleaned = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', text_to_synth_cleaned)
        text_to_synth_cleaned = re.sub(r'<[^>]+>', '', text_to_synth_cleaned).strip()


        speaker_id = segment.get('speaker', 'SPEAKER_UNKNOWN')
        original_start = segment.get('start')
        original_end = segment.get('end')

        # Используем уже очищенный текст для логирования
        # print(f"Segment {i+1}/{total_segments} (Speaker: {speaker_id}, Orig Dur: {original_start:.2f}-{original_end:.2f} = {(original_end or 0) - (original_start or 0):.2f}s): '{text_to_synth_cleaned[:60]}...'")

        if not text_to_synth_cleaned.strip() or original_start is None or original_end is None:
            if original_end is not None: last_segment_original_end_time = original_end 
            if progress_callback: progress_callback((i + 1) / total_segments)
            # print(f"  Skipping segment {i+1} due to empty text or missing timing.")
            continue
        
        original_duration = original_end - original_start
        if original_duration <= 0.05: 
            if i % 20 == 0 : print(f"  Skipping very short original segment {i+1} (duration: {original_duration:.2f}s)") # Логируем реже
            last_segment_original_end_time = original_end
            if progress_callback: progress_callback((i + 1) / total_segments)
            continue
        
        # Логируем только для каждого 10-го сегмента или если есть проблема, чтобы не засорять вывод
        if i % 10 == 0 or i == total_segments -1:
            print(f"Processing segment {i+1}/{total_segments} (Speaker: {speaker_id}, Orig Dur: {original_duration:.2f}s): '{text_to_synth_cleaned[:60]}...'")


        silence_before_duration = original_start - last_segment_original_end_time
        if silence_before_duration > 0.05: 
            silence_wav_path = os.path.join(output_segments_dir, f"silence_before_{i:04d}.wav")
            try:
                (ffmpeg.input('anullsrc', format='lavfi', r=24000, channel_layout='mono')
                 .output(silence_wav_path, t=silence_before_duration, acodec='pcm_s16le')
                 .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                if os.path.exists(silence_wav_path): processed_segment_files.append(silence_wav_path)
            except Exception as e_sil: print(f"Warning: Failed to generate silence {silence_wav_path}: {e_sil}")

        current_speaker_wav_path = _get_or_create_speaker_wav(
            speaker_id, segment, reference_audio_path, temp_dir, diarization_df=diarization_result_df
        )
        if current_speaker_wav_path:
            speaker_references[speaker_id] = current_speaker_wav_path 
            if first_valid_speaker_ref is None:
                first_valid_speaker_ref = current_speaker_wav_path
        else: 
            current_speaker_wav_path = speaker_references.get(speaker_id) 
            if not current_speaker_wav_path:
                 current_speaker_wav_path = first_valid_speaker_ref 
                 if current_speaker_wav_path:
                     if i % 10 == 0 : print(f"  Warning: No specific reference for {speaker_id} for segment {i+1}. Using fallback speaker ref: {os.path.basename(current_speaker_wav_path)}")
                 else:
                     if i % 10 == 0 : print(f"  CRITICAL WARNING: No reference audio for {speaker_id} (segment {i+1}) and no fallback. Using default model voice.")
        
        text_chunks = _split_text_for_tts(text_to_synth_cleaned, max_length=XTTS_RU_MAX_CHARS)
        synthesized_sub_segments_paths = []
        
        # if len(text_chunks)>1 and (i % 10 == 0 or i == total_segments -1) : print(f"  Text Chunks for synthesis: {len(text_chunks)}")

        for chunk_idx, text_chunk in enumerate(text_chunks):
            if not text_chunk.strip(): continue
            raw_sub_segment_filename = os.path.join(raw_tts_dir, f"raw_sub_seg_{i:04d}_{chunk_idx:02d}_{speaker_id}.wav")
            try:
                tts_model.tts_to_file(
                    text=text_chunk, 
                    speaker_wav=current_speaker_wav_path, 
                    language=language, 
                    file_path=raw_sub_segment_filename,
                    split_sentences=True 
                )
                if os.path.exists(raw_sub_segment_filename) and os.path.getsize(raw_sub_segment_filename) > 0:
                    synthesized_sub_segments_paths.append(raw_sub_segment_filename)
                else:
                    if i % 10 == 0 : print(f"    Warning: Raw TTS for sub-segment {i+1}-{chunk_idx+1} is missing or empty.")
            except Exception as e_tts_sub:
                if i % 10 == 0 : print(f"    Error synthesizing sub-segment {i+1}-{chunk_idx+1} for speaker {speaker_id}: {e_tts_sub}")
        
        raw_segment_filename_combined = os.path.join(raw_tts_dir, f"raw_segment_combined_{i:04d}_{speaker_id}.wav")
        synthesized_duration = 0.0
        synthesized_samplerate = 24000 

        if not synthesized_sub_segments_paths:
            if i % 10 == 0 : print(f"  Warning: No sub-segments synthesized for segment {i+1}. Skipping tempo adjustment.")
            last_segment_original_end_time = original_end
            if progress_callback: progress_callback((i + 1) / total_segments)
            continue
        elif len(synthesized_sub_segments_paths) == 1:
            shutil.copy(synthesized_sub_segments_paths[0], raw_segment_filename_combined)
        else:
            try:
                video_processor.merge_audio_segments(synthesized_sub_segments_paths, raw_segment_filename_combined)
            except Exception as e_merge_sub:
                if i % 10 == 0 : print(f"  Error merging sub-segments for segment {i+1}: {e_merge_sub}. Skipping this segment.")
                last_segment_original_end_time = original_end
                if progress_callback: progress_callback((i + 1) / total_segments)
                continue
        
        if os.path.exists(raw_segment_filename_combined) and os.path.getsize(raw_segment_filename_combined) > 0:
            try:
                synthesized_info = sf.info(raw_segment_filename_combined)
                synthesized_duration = synthesized_info.duration
                synthesized_samplerate = synthesized_info.samplerate
            except Exception as e_sfinfo:
                if i % 10 == 0 : print(f"  Error getting info for combined raw segment {i+1}: {e_sfinfo}. Assuming zero duration.")
                synthesized_duration = 0.0
        else:
            if i % 10 == 0 : print(f"  Warning: Combined raw TTS for segment {i+1} is missing or empty. Skipping.")
            last_segment_original_end_time = original_end
            if progress_callback: progress_callback((i + 1) / total_segments)
            continue
            
        adjusted_segment_filename = os.path.join(output_segments_dir, f"adj_segment_{i:04d}_{speaker_id}.wav")
        
        if synthesized_duration <= 0.05: 
            if i % 10 == 0 : print(f"  Synthesized audio for segment {i+1} is too short ({synthesized_duration:.2f}s). Using raw.")
            shutil.copy(raw_segment_filename_combined, adjusted_segment_filename)
        else:
            target_atempo_value = original_duration / synthesized_duration
            if i % 10 == 0 or i == total_segments - 1 or target_atempo_value < min_overall_tempo_factor or target_atempo_value > max_overall_tempo_factor : # Логируем если за пределами или выборочно
                print(f"  Segment {i+1}: Orig Dur: {original_duration:.3f}s, Synth Dur: {synthesized_duration:.3f}s, Calculated atempo: {target_atempo_value:.3f}")

            clamped_atempo_value = target_atempo_value
            if target_atempo_value < min_overall_tempo_factor:
                clamped_atempo_value = min_overall_tempo_factor
                print(f"    Atempo value ({target_atempo_value:.3f}) below min_overall ({min_overall_tempo_factor}). Clamping to {clamped_atempo_value:.3f}.")
            elif target_atempo_value > max_overall_tempo_factor:
                clamped_atempo_value = max_overall_tempo_factor
                print(f"    Atempo value ({target_atempo_value:.3f}) above max_overall ({max_overall_tempo_factor}). Clamping to {clamped_atempo_value:.3f}.")
            
            try:
                _apply_atempo_iteratively(
                    raw_segment_filename_combined, 
                    adjusted_segment_filename, 
                    clamped_atempo_value, 
                    synthesized_samplerate
                )
                if os.path.exists(adjusted_segment_filename):
                    adjusted_info = sf.info(adjusted_segment_filename)
                    if i % 10 == 0 or i == total_segments -1 :print(f"    Adjusted segment duration: {adjusted_info.duration:.3f}s (SR: {adjusted_info.samplerate} Hz)")
                else:
                    if i % 10 == 0 : print(f"    Warning: Adjusted segment file {adjusted_segment_filename} not found after _apply_atempo_iteratively.")

            except Exception as e_tempo_adj:
                 if i % 10 == 0 : print(f"    Error during _apply_atempo_iteratively for segment {i+1}: {e_tempo_adj}. Using raw combined segment.")
                 if os.path.exists(raw_segment_filename_combined):
                    shutil.copy(raw_segment_filename_combined, adjusted_segment_filename)
                 else:
                    if i % 10 == 0 : print(f"    Critical: Raw combined segment {raw_segment_filename_combined} also missing for segment {i+1}.")

        if os.path.exists(adjusted_segment_filename):
            processed_segment_files.append(adjusted_segment_filename)
        else:
            if i % 10 == 0 : print(f"  Warning: Adjusted segment {i+1} not created. Trying to use raw combined (if exists).")
            if os.path.exists(raw_segment_filename_combined): processed_segment_files.append(raw_segment_filename_combined)
        
        last_segment_original_end_time = original_end
        if progress_callback: progress_callback((i + 1) / total_segments)

    if not processed_segment_files:
        print("CRITICAL: No audio segments were successfully processed for the entire video.")
        final_audio_path = os.path.join(temp_dir, "dubbed_full_audio_final_tempo.wav")
        (ffmpeg.input('anullsrc', format='lavfi', r=24000)
             .output(final_audio_path, t=0.01, acodec='pcm_s16le')
             .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return final_audio_path
        
    final_audio_path = os.path.join(temp_dir, "dubbed_full_audio_final_tempo.wav")
    video_processor.merge_audio_segments(processed_segment_files, final_audio_path)
    print("Final dubbed audio (tempo adjusted and silences added) merged successfully.")
    return final_audio_path