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
DEFAULT_STT_MODEL_NAME = "medium.en"

def load_stt_diarization_models(stt_model_name=DEFAULT_STT_MODEL_NAME, device="cpu", hf_token=None, load_stt=True, load_diarization=True): # ... (без изменений) ...
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
            models_cache.stt_model = whisperx.load_model(stt_model_name, device, compute_type=stt_compute_type, language="en")
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

def load_align_model_cached(language_code, device): # ... (без изменений) ...
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

def segments_to_srt(segments, use_translated_text_field=None): # ... (без изменений) ...
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

def transcribe_and_diarize(audio_path, language="en", batch_size=16, output_srt_path=None, return_diarization_df=False): # ... (без изменений) ...
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Transcribe & Diarize using device: {device}")
    hf_auth_token = os.environ.get("HF_TOKEN")
    stt_model, diarization_model_instance = load_stt_diarization_models(stt_model_name=DEFAULT_STT_MODEL_NAME, device=device, hf_token=hf_auth_token, load_stt=True, load_diarization=True)
    if stt_model is None: raise RuntimeError("STT model is not loaded.")
    audio = whisperx.load_audio(audio_path); print(f"Transcribing audio '{os.path.basename(audio_path)}' (language hint: {language})...")
    result = stt_model.transcribe(audio, language="en", batch_size=batch_size); print("Transcription complete.")
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

def perform_diarization_only(audio_path, device="cpu", hf_token=None): # ... (без изменений) ...
    print(f"Performing diarization ONLY on: {os.path.basename(audio_path)} using device: {device}")
    _stt_model, diarization_model = load_stt_diarization_models(device=device, hf_token=hf_token, load_stt=False, load_diarization=True )
    if diarization_model is None: print("Diarization model is not loaded. Cannot perform diarization."); return pd.DataFrame() 
    try:
        diarization_result_df = diarization_model(audio_path) 
        if diarization_result_df is not None and not diarization_result_df.empty: print(f"Diarization found {len(diarization_result_df['speaker'].unique())} unique speakers.")
        else: print("Diarization did not return any speaker segments."); return pd.DataFrame() 
        return diarization_result_df 
    except Exception as e: print(f"Error during standalone diarization: {e}\nDetails: {traceback.format_exc()}"); return pd.DataFrame()

def assign_srt_segments_to_speakers(srt_segments, diarization_df): # ... (без изменений) ...
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
    # 1. Удаляем маркеры спикера типа "[SPEAKER_01]: " или "SPEAKER_01 : " (с возможными пробелами)
    line = re.sub(r'^\[SPEAKER_\d+\]\s*:\s*', '', line, flags=re.IGNORECASE)
    line = re.sub(r'^SPEAKER_\d+\s*:\s*', '', line, flags=re.IGNORECASE)
    # 2. Удаляем HTML-подобные теги (например, <i>, <b>, <font>, <c.color>)
    line = re.sub(r'<[^>]+>', '', line)
    # 3. Удаляем VTT-теги спикеров типа <v Speaker 1> или <v.SpeakerName> (если они остались после п.2)
    line = re.sub(r'<v\s+[^>]+>', '', line, flags=re.IGNORECASE) # <v Speaker 1>
    line = re.sub(r'<v\.[^>]+>', '', line, flags=re.IGNORECASE)  # <v.SpeakerName>
    line = re.sub(r'</v>', '', line, flags=re.IGNORECASE)       # </v>
    # 4. Удаляем символы музыкальных нот и подобные "мусорные" символы
    line = line.replace('♪', '').replace('♫', '')
    # 5. Удаляем строки, состоящие только из "хм", "мм", "ага", "гм", "хм-хм" и т.п. или только из знаков препинания.
    #    Также удаляем строки, которые после удаления всех вышеперечисленных элементов становятся пустыми.
    #    Добавляем проверки на строки типа "х х х х" или "мм мм мм"
    cleaned_line_for_check = line.lower().replace('хм', '').replace('мм', '').replace('гм', '').replace('ага', '').replace('ого', '').replace('угу', '')
    cleaned_line_for_check = re.sub(r'[хмг]', '', cleaned_line_for_check, flags=re.IGNORECASE) # Удаляем отдельные х, м, г
    if re.fullmatch(r'^[ \t\n\r\f\v\x00-\x1F\x7F-\x9F.,!?"\'«»…()*#%;:/@<>=\[\]\\^`{}|~+-]*$', cleaned_line_for_check):
        return ""
    # 6. Заменяем неразрывные пробелы и табы на обычные, затем множественные пробелы на один
    line = line.replace('\xa0', ' ').replace('\t', ' ')
    line = re.sub(r'\s+', ' ', line).strip()
    # 7. Дополнительная очистка от странных символов или последовательностей, если они появляются в логах
    # Например, если есть что-то вроде "\cHFFFF..." - это может быть остаток цветовых кодов или чего-то подобного
    line = re.sub(r'\\c[Hh][0-9a-fA-F]+', '', line) # Удаление \cH FFFFFF...
    line = re.sub(r'^[^\w\s]+$', '', line) # Удаление строки, если она состоит ТОЛЬКО из спецсимволов (кроме пробелов)
    
    return line.strip()

def _preprocess_vtt_content(content: str) -> str: # ... (без изменений) ...
    if content.startswith("WEBVTT"):
        header_end_match = re.search(r"WEBVTT.*?\n\n", content, re.DOTALL | re.IGNORECASE)
        if header_end_match: content = content[header_end_match.end():]
        else:
            lines = content.splitlines()
            if lines and "WEBVTT" in lines[0].upper(): content = "\n".join(lines[1:])
    content = re.sub(r"(\d{2}:\d{2}:\d{2})[.](\d{3})", r"\1,\2", content)
    content = re.sub(r'\n\s*\n', '\n\n', content).strip()
    return content

def parse_srt_file(srt_file_path): # ... (без изменений, кроме использования новой _clean_srt_text_line) ...
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
            cleaned_lines = [_clean_srt_text_line(line) for line in lines]
            full_cleaned_text = " ".join(filter(None, cleaned_lines)).strip()
            full_cleaned_text = re.sub(r'\s+', ' ', full_cleaned_text).strip()
            if not full_cleaned_text: continue
            segment = {
                'text': full_cleaned_text,
                'start': sub.start.total_seconds(),
                'end': sub.end.total_seconds(),
                'speaker': 'SPEAKER_00' 
            }
            segments.append(segment)
        print(f"Successfully parsed {len(segments)} segments from {os.path.basename(srt_file_path)}.")
        # Отладочный вывод нескольких первых сегментов после парсинга
        if segments and len(segments) > 5:
            print("  DEBUG: First 5 parsed SRT segments:")
            for idx_debug, seg_debug in enumerate(segments[:5]):
                print(f"    {idx_debug+1}: Start={seg_debug['start']:.3f}, End={seg_debug['end']:.3f}, Text='{seg_debug['text'][:80]}...'")
        elif segments:
             print(f"  DEBUG: Parsed {len(segments)} SRT segment(s): {segments}")

        return segments
    except FileNotFoundError: print(f"Error: SRT/VTT file not found at {srt_file_path}"); return []
    except srt.SRTParseError as e_srt_parse: 
        print(f"Error parsing SRT/VTT file {srt_file_path} with `srt` library: {e_srt_parse}")
        print(f"Problematic content snippet (first 500 chars after preprocessing):\n{processed_content[:500]}")
        return []
    except Exception as e_gen: 
        print(f"General error parsing SRT/VTT file {srt_file_path}: {e_gen}\nDetails: {traceback.format_exc()}")
        return []