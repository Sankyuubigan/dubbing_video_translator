import whisperx
import torch
import os
from types import SimpleNamespace
import traceback
import srt
import datetime
import re
import pandas as pd

models_cache = SimpleNamespace(stt_model=None, diarization_model=None, align_model_cache=None)
DEFAULT_STT_MODEL_NAME = "medium.en" # Используем medium.en для лучшего распознавания, если SRT не предоставлен

def load_stt_diarization_models(stt_model_name=DEFAULT_STT_MODEL_NAME, device="cpu", hf_token=None, load_stt=True, load_diarization=True):
    global models_cache; stt_model_to_return = models_cache.stt_model; diarization_model_to_return = models_cache.diarization_model
    if load_stt:
        stt_compute_type = "float16" if device == "cuda" else "int8"
        if device == "cpu" and stt_compute_type == "int8":
            try: import ctranslate2
            except ImportError: print("ctranslate2 not found, WhisperX STT on CPU will use 'float32'."); stt_compute_type = "float32"
        needs_stt_reload = True
        if models_cache.stt_model:
            if hasattr(models_cache.stt_model, 'loaded_model_name') and \
               models_cache.stt_model.loaded_model_name == stt_model_name and \
               hasattr(models_cache.stt_model, 'loaded_compute_type') and \
               models_cache.stt_model.loaded_compute_type == stt_compute_type and \
               hasattr(models_cache.stt_model, 'loaded_device') and \
               models_cache.stt_model.loaded_device == device:
                 needs_stt_reload = False
        if needs_stt_reload:
            print(f"Loading WhisperX STT model: {stt_model_name} on device: {device} with compute_type: {stt_compute_type}")
            # Указываем language="en" при загрузке, если модель многоязычная, но мы ожидаем английский для лучшего выравнивания
            models_cache.stt_model = whisperx.load_model(stt_model_name, device, compute_type=stt_compute_type, language="en" if "large" in stt_model_name else None)
            if models_cache.stt_model:
                models_cache.stt_model.loaded_model_name = stt_model_name
                models_cache.stt_model.loaded_compute_type = stt_compute_type
                models_cache.stt_model.loaded_device = device
            print("STT model loaded.")
        stt_model_to_return = models_cache.stt_model
    if load_diarization:
        if models_cache.diarization_model is None:
            print("Loading Diarization model...")
            try:
                if hf_token: models_cache.diarization_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
                else: models_cache.diarization_model = whisperx.DiarizationPipeline(device=device)
                print("Diarization model loaded.")
            except Exception as e:
                print(f"Error loading diarization model: {e}\nEnsure HF agreements for pyannote models are accepted and HF_TOKEN is set if needed.")
                models_cache.diarization_model = None;
        diarization_model_to_return = models_cache.diarization_model
    return stt_model_to_return, diarization_model_to_return

def load_align_model_cached(language_code, device):
    global models_cache
    if models_cache.align_model_cache is None: models_cache.align_model_cache = {}
    align_lang_code_to_use = language_code if language_code else "en"
    if align_lang_code_to_use not in models_cache.align_model_cache or \
       models_cache.align_model_cache[align_lang_code_to_use].get('device') != device:
        print(f"Loading align model for language '{align_lang_code_to_use}' on device '{device}'...")
        model, metadata = whisperx.load_align_model(language_code=align_lang_code_to_use, device=device)
        models_cache.align_model_cache[align_lang_code_to_use] = {'model': model, 'metadata': metadata, 'device': device}
        print("Align model loaded.")
    return models_cache.align_model_cache[align_lang_code_to_use]['model'], models_cache.align_model_cache[align_lang_code_to_use]['metadata']

def segments_to_srt(segments, use_translated_text_field=None):
    if not srt: print("ERROR: srt library is not available. Cannot generate SRT content."); return ""
    srt_subs = []
    for i, segment_data in enumerate(segments):
        start_time = segment_data.get('start'); end_time = segment_data.get('end')
        text_to_use = segment_data.get('text', '')
        if use_translated_text_field and use_translated_text_field in segment_data:
            text_to_use = segment_data.get(use_translated_text_field, '')
        if start_time is None or end_time is None or not str(text_to_use).strip(): continue
        start_td = datetime.timedelta(seconds=start_time); end_td = datetime.timedelta(seconds=end_time)
        speaker = segment_data.get('speaker')
        srt_text_final = f"[{speaker}]: {text_to_use}" if speaker and speaker != "SPEAKER_00" else str(text_to_use)
        srt_subs.append(srt.Subtitle(index=i + 1, start=start_td, end=end_td, content=srt_text_final))
    return srt.compose(srt_subs)

def transcribe_and_diarize(audio_path, language="en", batch_size=16, output_srt_path=None, return_diarization_df=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Transcribe & Diarize using device: {device}")
    hf_auth_token = os.environ.get("HF_TOKEN")
    stt_model, diarization_model_instance = load_stt_diarization_models(stt_model_name=DEFAULT_STT_MODEL_NAME, device=device, hf_token=hf_auth_token, load_stt=True, load_diarization=True)
    if stt_model is None: raise RuntimeError("STT model is not loaded.")
    audio = whisperx.load_audio(audio_path); print(f"Transcribing audio '{os.path.basename(audio_path)}' (language hint: {language})...")
    # Явно указываем язык "en" для транскрипции, если используем англоязычную модель
    # Если модель многоязычная (large), то язык можно не указывать или указывать "en" для фокусировки.
    transcribe_language_param = "en" # DEFAULT_STT_MODEL_NAME = "medium.en"
    if "large" in DEFAULT_STT_MODEL_NAME.lower() and language: # Если используется large модель и указан язык
        transcribe_language_param = language
    elif "large" in DEFAULT_STT_MODEL_NAME.lower():
        transcribe_language_param = None # Позволяем large-модели самой определять язык

    result = stt_model.transcribe(audio, language=transcribe_language_param, batch_size=batch_size); print("Transcription complete.")

    segments_after_align = result.get("segments", []); print("Aligning transcription...")
    try:
        lang_code_for_align = result.get("language", "en")
        if not lang_code_for_align: print(f"Warning: WhisperX did not detect language for alignment. Defaulting to 'en'."); lang_code_for_align = "en"
        align_model, metadata = load_align_model_cached(language_code=lang_code_for_align, device=device)
        segments_to_align = result.get("segments", []);
        if not isinstance(segments_to_align, list) or (segments_to_align and not isinstance(segments_to_align[0], dict)): segments_to_align = []
        if segments_to_align:
            aligned_result = whisperx.align(segments_to_align, align_model, metadata, audio, device, return_char_alignments=False)
            segments_after_align = aligned_result.get("segments", segments_to_align)
            print(f"Alignment complete (used lang_code: {lang_code_for_align}).")
        else: print("No segments to align. Using original transcription segments."); segments_after_align = result.get("segments", [])
        if device == "cuda": torch.cuda.empty_cache()
    except Exception as e: print(f"Warning: Failed to align transcription - {e}. Using unaligned segments."); segments_after_align = result.get("segments", [])
    final_segments = segments_after_align; diarize_segments_df_result = None
    if diarization_model_instance:
        print("Performing diarization and assigning speakers to aligned transcription...")
        try:
            diarize_segments_df_result = diarization_model_instance(audio_path)
            if not isinstance(segments_after_align, list) or \
               (segments_after_align and not isinstance(segments_after_align[0], dict)):
                print("Warning: Segments for speaker assignment are not in the expected list-of-dicts format. Skipping speaker assignment.")
                final_segments = segments_after_align
                for seg in final_segments:
                    if 'speaker' not in seg: seg['speaker'] = 'SPEAKER_00'
            elif not segments_after_align: print("Warning: No segments available for speaker assignment."); final_segments = []
            else:
                input_for_assignment = {"segments": segments_after_align}
                if diarize_segments_df_result is not None and not diarize_segments_df_result.empty:
                    final_segments_with_speakers_data = whisperx.assign_word_speakers(diarize_segments_df_result, input_for_assignment)
                    final_segments = final_segments_with_speakers_data.get("segments", segments_after_align)
                    print("Diarization and speaker assignment complete.")
                else:
                    print("Diarization result was empty. Assigning default speaker to all segments.")
                    for seg_idx in range(len(final_segments)): final_segments[seg_idx]['speaker'] = 'SPEAKER_00'
        except Exception as e:
            print(f"Error during diarization/speaker assignment: {e}. Falling back to transcription without speaker labels.");
            final_segments = segments_after_align
            for seg_idx in range(len(final_segments)):
                if 'speaker' not in final_segments[seg_idx]: final_segments[seg_idx]['speaker'] = 'SPEAKER_00'
            diarize_segments_df_result = None
    else:
        print("Diarization model not loaded. Skipping speaker assignment. Assigning default speaker.")
        for seg_idx in range(len(final_segments)):
            if 'speaker' not in final_segments[seg_idx]: final_segments[seg_idx]['speaker'] = 'SPEAKER_00'
    if output_srt_path and final_segments and srt:
        try:
            srt_data = segments_to_srt(final_segments)
            with open(output_srt_path, 'w', encoding='utf-8') as f: f.write(srt_data)
            print(f"WhisperX SRT (source transcription) saved to: {output_srt_path}")
        except Exception as e_srt_save: print(f"Warning: Could not save WhisperX SRT to {output_srt_path}: {e_srt_save}")
    if return_diarization_df: return final_segments, diarize_segments_df_result
    return final_segments

def perform_diarization_only(audio_path, device="cpu", hf_token=None):
    print(f"Performing diarization ONLY on: {os.path.basename(audio_path)} using device: {device}")
    _stt_model, diarization_model = load_stt_diarization_models(device=device, hf_token=hf_token, load_stt=False, load_diarization=True )
    if diarization_model is None: print("Diarization model is not loaded. Cannot perform diarization."); return pd.DataFrame()
    try:
        diarization_result_df = diarization_model(audio_path)
        if diarization_result_df is not None and not diarization_result_df.empty: print(f"Diarization found {len(diarization_result_df['speaker'].unique())} unique speakers.")
        else: print("Diarization did not return any speaker segments."); return pd.DataFrame()
        return diarization_result_df
    except Exception as e: print(f"Error during standalone diarization: {e}\nDetails: {traceback.format_exc()}"); return pd.DataFrame()

def assign_srt_segments_to_speakers(srt_segments, diarization_df):
    if diarization_df is None or diarization_df.empty:
        print("Warning: Diarization result is empty or None. Assigning default speaker 'SPEAKER_00' to SRT segments.")
        output_segments = []
        for srt_seg in srt_segments: new_seg = srt_seg.copy(); new_seg['speaker'] = 'SPEAKER_00'; output_segments.append(new_seg)
        return output_segments
    print(f"Assigning speakers to {len(srt_segments)} SRT segments based on diarization...")
    output_segments = []
    for srt_seg in srt_segments:
        srt_start_time = srt_seg['start']; srt_end_time = srt_seg['end']; overlapping_speakers = {}
        for _idx, dia_row in diarization_df.iterrows():
            dia_start = dia_row['start']; dia_end = dia_row['end']; speaker_label = dia_row['speaker']
            overlap_start = max(srt_start_time, dia_start); overlap_end = min(srt_end_time, dia_end)
            overlap_duration = overlap_end - overlap_start
            if overlap_duration > 0.01: overlapping_speakers[speaker_label] = overlapping_speakers.get(speaker_label, 0) + overlap_duration
        assigned_speaker = 'SPEAKER_00'
        if overlapping_speakers: assigned_speaker = max(overlapping_speakers, key=overlapping_speakers.get)
        new_seg = srt_seg.copy(); new_seg['speaker'] = assigned_speaker; output_segments.append(new_seg)
    print("Speaker assignment for SRT segments complete.")
    return output_segments

def _clean_srt_text_line(line: str) -> str:
    line = re.sub(r'^\[SPEAKER_\d+\]\s*:\s*', '', line, flags=re.IGNORECASE)
    line = re.sub(r'^SPEAKER_\d+\s*:\s*', '', line, flags=re.IGNORECASE)
    line = re.sub(r'<[^>]+>', '', line)
    line = re.sub(r'<v\s+[^>]+>', '', line, flags=re.IGNORECASE)
    line = re.sub(r'<v\.[^>]+>', '', line, flags=re.IGNORECASE)
    line = re.sub(r'</v>', '', line, flags=re.IGNORECASE)
    line = line.replace('♪', '').replace('♫', '')
    cleaned_line_for_check = line.lower().replace('хм', '').replace('мм', '').replace('гм', '').replace('ага', '').replace('ого', '').replace('угу', '')
    cleaned_line_for_check = re.sub(r'[хмг]', '', cleaned_line_for_check, flags=re.IGNORECASE)
    if re.fullmatch(r'^[ \t\n\r\f\v\x00-\x1F\x7F-\x9F.,!?"\'«»…()*#%;:/@<>=\[\]\\^`{}|~+-]*$', cleaned_line_for_check):
        return ""
    line = line.replace('\xa0', ' ').replace('\t', ' ')
    line = re.sub(r'\s+', ' ', line).strip()
    line = re.sub(r'\\c[Hh][0-9a-fA-F]+', '', line)
    line = re.sub(r'^[^\w\s]+$', '', line)
    return line.strip()

def _preprocess_vtt_content(content: str) -> str:
    if content.startswith("WEBVTT"):
        header_end_match = re.search(r"WEBVTT.*?\n\n", content, re.DOTALL | re.IGNORECASE)
        if header_end_match: content = content[header_end_match.end():]
        else:
            lines = content.splitlines()
            if lines and "WEBVTT" in lines[0].upper(): content = "\n".join(lines[1:])
    content = re.sub(r"(\d{2}:\d{2}:\d{2})[.](\d{3})", r"\1,\2", content) # VTT to SRT time format
    content = re.sub(r'\n\s*\n', '\n\n', content).strip() # Normalize blank lines
    return content

def _postprocess_srt_segments(segments, min_duration_threshold=0.25, merge_gap_threshold=0.1):
    if not segments: return []
    print(f"Post-processing {len(segments)} SRT segments...")
    processed_segments = []
    idx = 0
    while idx < len(segments):
        current_seg = segments[idx]
        current_text = current_seg.get('text', '').strip()
        current_start = current_seg.get('start')
        current_end = current_seg.get('end')
        current_speaker = current_seg.get('speaker', 'SPEAKER_00')
        current_duration = current_end - current_start if current_start is not None and current_end is not None else 0

        # 1. Удаление дубликатов или слишком коротких сегментов с тем же текстом, что и предыдущий
        if processed_segments:
            prev_seg = processed_segments[-1]
            if prev_seg['text'] == current_text and prev_seg['speaker'] == current_speaker and current_duration < 0.1:
                print(f"  Skipping duplicate/tiny segment {idx+1} with same text as previous: '{current_text[:30]}...'")
                idx += 1
                continue

        # 2. Попытка объединить очень короткие сегменты со следующим, если спикер тот же
        if current_duration < min_duration_threshold and (idx + 1) < len(segments):
            next_seg = segments[idx+1]
            next_start = next_seg.get('start')
            next_end = next_seg.get('end')
            next_text = next_seg.get('text', '').strip()
            next_speaker = next_seg.get('speaker', 'SPEAKER_00')
            gap = next_start - current_end if next_start is not None and current_end is not None else float('inf')

            if current_speaker == next_speaker and gap < merge_gap_threshold:
                # Если текст короткого сегмента является началом текста следующего, или наоборот (редко, но бывает)
                # или если тексты просто разные, но сегмент очень короткий и лучше объединить.
                merged_text = current_text
                if next_text and not next_text.startswith(current_text): # Избегаем дублирования если один текст содержит другой
                     merged_text = (current_text + " " + next_text).strip()
                elif not current_text and next_text:
                    merged_text = next_text
                
                # Если текст короткого сегмента полностью содержится в начале следующего, берем текст следующего
                if next_text.startswith(current_text) and len(current_text) > 0:
                    merged_text = next_text
                elif current_text.endswith(next_text) and len(next_text) > 0: # Менее вероятно, но возможно
                    merged_text = current_text
                else: # Просто конкатенируем, если нет явного включения
                    merged_text = (current_text + " " + next_text).strip()
                
                merged_text = re.sub(r'\s+', ' ', merged_text).strip() # Очистка двойных пробелов

                if merged_text: # Только если есть текст после слияния
                    new_merged_segment = {
                        'text': merged_text,
                        'start': current_start,
                        'end': next_end,
                        'speaker': current_speaker
                    }
                    print(f"  Merged short segment {idx+1} (dur: {current_duration:.3f}s, text: '{current_text[:30]}...') "
                          f"with next {idx+2} (text: '{next_text[:30]}...'). New dur: {next_end - current_start:.3f}s")
                    
                    # Вместо добавления сразу, заменяем текущий сегмент и пропускаем следующий
                    # Это позволит на следующей итерации снова проверить объединенный сегмент
                    segments[idx] = new_merged_segment
                    segments.pop(idx+1) # Удаляем следующий, так как он объединен
                    # Не инкрементируем idx, чтобы перепроверить объединенный сегмент
                    continue # Начинаем цикл while заново с текущим (теперь объединенным) idx

        # 3. Если после попыток объединения сегмент все еще слишком короткий или пустой, пропускаем его
        current_duration_after_potential_merge = segments[idx]['end'] - segments[idx]['start'] # Пересчитываем, если было слияние
        current_text_after_potential_merge = segments[idx]['text']
        if not current_text_after_potential_merge or current_duration_after_potential_merge < 0.05: # 50ms - очень маленький порог
            print(f"  Skipping very short/empty segment {idx+1} after processing (dur: {current_duration_after_potential_merge:.3f}s, text: '{current_text_after_potential_merge[:30]}...')")
            idx += 1
            continue

        processed_segments.append(segments[idx])
        idx += 1

    print(f"SRT post-processing finished. {len(processed_segments)} segments remaining.")
    if processed_segments and len(processed_segments) > 3:
        print("  DEBUG: First 3 segments after post-processing:")
        for i_debug, seg_debug in enumerate(processed_segments[:3]):
             print(f"    {i_debug+1}: S={seg_debug['start']:.3f}, E={seg_debug['end']:.3f}, Spk={seg_debug['speaker']}, Txt='{seg_debug['text'][:60]}...'")
    return processed_segments


def parse_srt_file(srt_file_path, post_process=False): # Добавлен параметр post_process
    if not srt: print("ERROR: srt library is not available. Cannot parse SRT/VTT file."); return []
    print(f"Parsing SRT/VTT file: {os.path.basename(srt_file_path)}")
    segments = []
    try:
        with open(srt_file_path, 'r', encoding='utf-8-sig') as f: raw_content = f.read()
        processed_content = _preprocess_vtt_content(raw_content)
        if not processed_content.strip():
            print(f"Warning: SRT/VTT file {srt_file_path} is empty or became empty after VTT header removal.")
            return []
        parsed_subs = list(srt.parse(processed_content))
        for sub_idx, sub in enumerate(parsed_subs):
            lines = sub.content.splitlines() if sub.content else []
            
            # Извлечение спикера из текста, если он есть (например, "[SPEAKER_01]: текст")
            speaker_in_text = 'SPEAKER_00' # По умолчанию
            full_raw_text = " ".join(lines).strip()
            speaker_match = re.match(r'^\[(SPEAKER_\d+)\]\s*:\s*(.*)', full_raw_text, re.IGNORECASE)
            if speaker_match:
                speaker_in_text = speaker_match.group(1).upper()
                text_content_for_cleaning = speaker_match.group(2)
            else: # Пробуем второй формат "SPEAKER_01 : текст"
                speaker_match_alt = re.match(r'^(SPEAKER_\d+)\s*:\s*(.*)', full_raw_text, re.IGNORECASE)
                if speaker_match_alt:
                    speaker_in_text = speaker_match_alt.group(1).upper()
                    text_content_for_cleaning = speaker_match_alt.group(2)
                else:
                    text_content_for_cleaning = full_raw_text
            
            # Очищаем только текстовое содержимое
            cleaned_lines = [_clean_srt_text_line(line) for line in text_content_for_cleaning.splitlines()]
            full_cleaned_text = " ".join(filter(None, cleaned_lines)).strip()
            full_cleaned_text = re.sub(r'\s+', ' ', full_cleaned_text).strip()

            if not full_cleaned_text: continue # Пропускаем сегменты без текста ПОСЛЕ очистки

            segment = {
                'text': full_cleaned_text,
                'start': sub.start.total_seconds(),
                'end': sub.end.total_seconds(),
                'speaker': speaker_in_text # Используем извлеченного спикера или дефолт
            }
            segments.append(segment)

        print(f"Successfully parsed {len(segments)} segments from {os.path.basename(srt_file_path)} before post-processing.")
        if segments and len(segments) > 5:
            print("  DEBUG: First 5 parsed SRT segments (before post-processing):")
            for idx_debug, seg_debug in enumerate(segments[:5]):
                print(f"    {idx_debug+1}: S={seg_debug['start']:.3f}, E={seg_debug['end']:.3f}, Spk={seg_debug['speaker']}, Txt='{seg_debug['text'][:60]}...'")
        elif segments:
             print(f"  DEBUG: Parsed {len(segments)} SRT segment(s) (before post-processing): {segments}")

        if post_process:
            segments = _postprocess_srt_segments(segments)

        return segments
    except FileNotFoundError: print(f"Error: SRT/VTT file not found at {srt_file_path}"); return []
    except srt.SRTParseError as e_srt_parse:
        print(f"Error parsing SRT/VTT file {srt_file_path} with `srt` library: {e_srt_parse}")
        print(f"Problematic content snippet (first 500 chars after preprocessing):\n{processed_content[:500]}")
        return []
    except Exception as e_gen:
        print(f"General error parsing SRT/VTT file {srt_file_path}: {e_gen}\nDetails: {traceback.format_exc()}")
        return []