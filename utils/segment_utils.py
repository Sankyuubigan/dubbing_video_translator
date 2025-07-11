import srt
import datetime
import re
import pandas as pd

# Константы, перенесенные из transcriber.py
MAX_SEGMENT_DURATION_FROM_WORDS = 10.0 
MAX_PAUSE_BETWEEN_WORDS = 0.7 
CHARS_PER_SECOND_ESTIMATE = 13 
MAX_TEXT_TO_DURATION_RATIO_TOLERANCE = 3.5 
MIN_TEXT_DURATION_AFTER_ESTIMATE_CAP = 0.2 
MAX_SILENCE_AFTER_TEXT_IN_LONG_SEGMENT = 1.5 
MIN_SEGMENT_DURATION_FOR_POSTPROCESS = 0.05 
MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS = 0.01 
DEFAULT_TARGET_CHUNK_DURATION_SEC = 45.0
DEFAULT_MAX_CHUNK_DURATION_SEC = 60.0
MIN_CHUNK_DURATION_THRESHOLD_SEC = 0.2

def segments_to_srt_string(segments, use_translated_text_field=None):
    """
    Преобразует список сегментов в строку формата SRT.
    Перенесено из transcriber.py.
    """
    if not srt: 
        print("ERROR: srt library is not available. Cannot generate SRT content.")
        return ""
        
    srt_subs = []
    if not segments: 
        return srt.compose(srt_subs) if srt else ""

    for i, segment_data in enumerate(segments):
        start_time = segment_data.get('start')
        end_time = segment_data.get('end')
        
        if not (isinstance(start_time, (int, float)) and isinstance(end_time, (int, float))):
            continue

        text_to_use = segment_data.get('text', '')
        if use_translated_text_field and use_translated_text_field in segment_data:
            text_to_use = segment_data.get(use_translated_text_field, '')
        
        if not str(text_to_use).strip() and not segment_data.get('is_pause_segment'): 
            continue
        
        if end_time <= start_time:
            end_time = start_time + 0.050 

        start_td = datetime.timedelta(seconds=start_time)
        end_td = datetime.timedelta(seconds=end_time)
        
        speaker_tag = segment_data.get('speaker', 'SPEAKER_00') 
        
        srt_text_final = ""
        if str(text_to_use).strip(): 
             srt_text_final = f"[{speaker_tag}]: {text_to_use}" if speaker_tag and speaker_tag not in ['SPEAKER_00', 'SPEAKER_UNKNOWN'] else str(text_to_use)
        
        if srt_text_final or segment_data.get('is_pause_segment'):
            try:
                subtitle_obj = srt.Subtitle(index=len(srt_subs) + 1, start=start_td, end=end_td, content=srt_text_final)
                srt_subs.append(subtitle_obj)
            except Exception as e_srt_create: 
                print(f"Error creating srt.Subtitle object for segment {i}: {e_srt_create}")

    return srt.compose(srt_subs) if srt else ""


def create_phrase_segments_from_words(word_segments, 
                                      max_duration=MAX_SEGMENT_DURATION_FROM_WORDS, 
                                      max_pause=MAX_PAUSE_BETWEEN_WORDS):
    """
    Создает сегменты на уровне фраз из сегментов на уровне слов.
    Перенесено из transcriber.py.
    """
    if not word_segments: return []
    phrase_segments = []
    current_phrase_text_parts = [] 
    current_phrase_start_time = -1
    current_phrase_end_time = -1 
    
    if not isinstance(word_segments, list) or not all(isinstance(w, dict) for w in word_segments):
        print("Warning (create_phrase_segments_from_words): word_segments is not a list of dicts.")
        return []

    for i, word_data in enumerate(word_segments):
        word_text = word_data.get("word", "").strip() 
        start_time = word_data.get("start")
        end_time = word_data.get("end")

        if not word_text or not isinstance(start_time, (int, float)) : continue
        if not isinstance(end_time, (int, float)) or end_time <= start_time: end_time = start_time + 0.01 

        if not current_phrase_text_parts: 
            current_phrase_text_parts.append(word_text)
            current_phrase_start_time = start_time
            current_phrase_end_time = end_time
        else:
            pause_since_last_word = start_time - current_phrase_end_time
            duration_if_added = end_time - current_phrase_start_time
            if pause_since_last_word > max_pause or duration_if_added > max_duration:
                phrase_segments.append({
                    "text": " ".join(current_phrase_text_parts),
                    "start": current_phrase_start_time,
                    "end": current_phrase_end_time, 
                })
                current_phrase_text_parts = [word_text]
                current_phrase_start_time = start_time
                current_phrase_end_time = end_time
            else: 
                current_phrase_text_parts.append(word_text)
                current_phrase_end_time = end_time 
        
        if i == len(word_segments) - 1 and current_phrase_text_parts:
            phrase_segments.append({
                "text": " ".join(current_phrase_text_parts),
                "start": current_phrase_start_time,
                "end": current_phrase_end_time,
            })
    return phrase_segments


def clean_srt_text_line(line: str) -> str:
    """Очищает одну строку текста субтитра. Перенесено из transcriber.py."""
    if not isinstance(line, str): return "" 
    line = re.sub(r'^\[SPEAKER_\d+\]\s*:\s*', '', line, flags=re.IGNORECASE)
    line = re.sub(r'^SPEAKER_\d+\s*:\s*', '', line, flags=re.IGNORECASE)
    line = re.sub(r'<[^>]+>', '', line) 
    line = line.replace('♪', '').replace('♫', '') 
    cleaned_line_for_check = line.lower()
    interjections_to_remove_for_check = ['хм', 'мм', 'гм', 'ага', 'ого', 'угу', 'эм', 'ээ', 'ой', 'ай', 'ну', 'а-а', 'о-о']
    for interjection in interjections_to_remove_for_check:
        cleaned_line_for_check = re.sub(r'\b' + re.escape(interjection) + r'\b', '', cleaned_line_for_check)
    if not cleaned_line_for_check.replace(' ', '').strip(): 
        if re.fullmatch(r'^[ \t\n\r\f\v\x00-\x1F\x7F-\x9F.,!?"\'«»…()*#%;:/@<>=\[\]\\^`{}|~+\-–—]*$', line):
            return "" 
    line = line.replace('\xa0', ' ').replace('\r', '')
    line = re.sub(r'\s+', ' ', line).strip() 
    return line.strip() 


def preprocess_vtt_content(content: str) -> str:
    """Предобрабатывает содержимое VTT файла для парсинга. Перенесено из transcriber.py."""
    if not isinstance(content, str): return ""
    if content.startswith("WEBVTT"):
        header_end_match = re.search(r"WEBVTT.*?\n\n", content, re.DOTALL | re.IGNORECASE)
        if header_end_match: content = content[header_end_match.end():]
        else: 
            lines = content.splitlines()
            if lines and "WEBVTT" in lines[0].upper(): content = "\n".join(lines[1:]) 
    content = re.sub(r"(\d{2}:\d{2}:\d{2})[.](\d{3})", r"\1,\2", content) 
    content = re.sub(r'\n\s*\n', '\n\n', content).strip() 
    return content


def postprocess_srt_segments(segments, is_external_srt=False):
    """
    Постобработка списка сегментов SRT (коррекция наложений, длительностей).
    Перенесено из transcriber.py.
    """
    if not segments or not isinstance(segments, list): return []
    valid_initial_segments = []
    for s_idx, s_data in enumerate(segments):
        if isinstance(s_data, dict) and \
           isinstance(s_data.get('start'), (int, float)) and \
           isinstance(s_data.get('end'), (int, float)) and \
           s_data['start'] < s_data['end']:
            valid_initial_segments.append(s_data)
    segments = valid_initial_segments
    if not segments: return []
    segments.sort(key=lambda x: (x['start'], x['end']))
    processed_segments_list = []
    if is_external_srt:
        temp_processed_external_segments = []
        if not segments: return []
        segments_copy_for_external = [s.copy() for s in segments]
        i_ext = 0
        while i_ext < len(segments_copy_for_external):
            current_seg_ext = segments_copy_for_external[i_ext]
            if i_ext + 1 < len(segments_copy_for_external):
                next_seg_ext = segments_copy_for_external[i_ext+1]
                curr_start_ext = current_seg_ext['start']; curr_end_ext = current_seg_ext['end']
                next_start_ext = next_seg_ext['start']
                if curr_end_ext > next_start_ext: 
                    overlap_amount_ext = curr_end_ext - next_start_ext
                    meeting_point_ext = next_start_ext + overlap_amount_ext / 2 
                    new_curr_end_ext = meeting_point_ext - (MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS / 2)
                    new_next_start_ext = meeting_point_ext + (MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS / 2)
                    if new_curr_end_ext > curr_start_ext + MIN_SEGMENT_DURATION_FOR_POSTPROCESS / 2 :
                        current_seg_ext['end'] = new_curr_end_ext
                    else: 
                        current_seg_ext['end'] = next_start_ext - MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS
                        if current_seg_ext['end'] < current_seg_ext['start']: current_seg_ext['end'] = current_seg_ext['start'] + MIN_SEGMENT_DURATION_FOR_POSTPROCESS
                    if new_next_start_ext < next_seg_ext.get('end', float('inf')) - MIN_SEGMENT_DURATION_FOR_POSTPROCESS / 2 :
                         segments_copy_for_external[i_ext+1]['start'] = new_next_start_ext 
                    else: 
                        original_next_duration_ext = next_seg_ext.get('end', new_next_start_ext + MIN_SEGMENT_DURATION_FOR_POSTPROCESS) - next_start_ext
                        segments_copy_for_external[i_ext+1]['start'] = current_seg_ext['end'] + MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS
                        segments_copy_for_external[i_ext+1]['end'] = segments_copy_for_external[i_ext+1]['start'] + max(MIN_SEGMENT_DURATION_FOR_POSTPROCESS, original_next_duration_ext)
            temp_processed_external_segments.append(current_seg_ext)
            i_ext += 1
        segments = temp_processed_external_segments 
    last_valid_segment_end_time = 0.0
    for i_proc, seg_original_data_proc in enumerate(segments):
        current_segment_data_proc = seg_original_data_proc.copy() 
        text_content_proc = current_segment_data_proc.get('text', '').strip()
        start_time_proc = current_segment_data_proc['start'] 
        end_time_proc = current_segment_data_proc['end']
        if start_time_proc < last_valid_segment_end_time:
            if not is_external_srt: 
                 new_start_time_proc = last_valid_segment_end_time + MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS
                 original_duration_proc = end_time_proc - start_time_proc
                 end_time_proc = new_start_time_proc + max(MIN_SEGMENT_DURATION_FOR_POSTPROCESS, original_duration_proc if original_duration_proc > 0 else MIN_SEGMENT_DURATION_FOR_POSTPROCESS)
                 start_time_proc = new_start_time_proc
        current_segment_data_proc['start'] = start_time_proc
        current_segment_data_proc['end'] = end_time_proc
        current_srt_duration_proc = end_time_proc - start_time_proc
        if current_srt_duration_proc <= 0: 
             current_srt_duration_proc = MIN_SEGMENT_DURATION_FOR_POSTPROCESS
             current_segment_data_proc['end'] = current_segment_data_proc['start'] + current_srt_duration_proc
        if text_content_proc: 
            estimated_speech_duration_val = len(text_content_proc) / CHARS_PER_SECOND_ESTIMATE
            if current_srt_duration_proc > (estimated_speech_duration_val * MAX_TEXT_TO_DURATION_RATIO_TOLERANCE) and current_srt_duration_proc > 0.2:
                capped_duration_val = estimated_speech_duration_val + MAX_SILENCE_AFTER_TEXT_IN_LONG_SEGMENT
                current_segment_data_proc['end'] = current_segment_data_proc['start'] + capped_duration_val
            current_segment_data_proc['end'] = max(current_segment_data_proc['end'], current_segment_data_proc['start'] + MIN_TEXT_DURATION_AFTER_ESTIMATE_CAP)
        elif not text_content_proc: 
            pause_duration_val = current_segment_data_proc['end'] - current_segment_data_proc['start']
            if pause_duration_val > MAX_SILENCE_AFTER_TEXT_IN_LONG_SEGMENT: 
                current_segment_data_proc['end'] = current_segment_data_proc['start'] + MAX_SILENCE_AFTER_TEXT_IN_LONG_SEGMENT
            current_segment_data_proc['end'] = max(current_segment_data_proc['end'], current_segment_data_proc['start'] + MIN_SEGMENT_DURATION_FOR_POSTPROCESS)
        final_duration_segment = current_segment_data_proc['end'] - current_segment_data_proc['start']
        if final_duration_segment >= MIN_SEGMENT_DURATION_FOR_POSTPROCESS:
            processed_segments_list.append(current_segment_data_proc)
            last_valid_segment_end_time = current_segment_data_proc['end']
    return processed_segments_list

def assign_speakers_to_phrases(phrase_segments, word_segments_with_speakers):
    """Присваивает спикеров фразам на основе слов с уже присвоенными спикерами."""
    if not isinstance(phrase_segments, list) or not isinstance(word_segments_with_speakers, list):
        for phrase in phrase_segments if isinstance(phrase_segments, list) else []:
            if isinstance(phrase, dict): phrase['speaker'] = 'SPEAKER_00'
        return phrase_segments
    output_phrase_segments_list = []
    word_idx_counter = 0 
    for phrase_dict in phrase_segments:
        if not isinstance(phrase_dict, dict): continue 
        phrase_start_time = phrase_dict.get('start'); phrase_end_time = phrase_dict.get('end')
        if not (isinstance(phrase_start_time, (int,float)) and isinstance(phrase_end_time, (int,float))):
            new_phrase_copy = phrase_dict.copy(); new_phrase_copy['speaker'] = 'SPEAKER_00'
            output_phrase_segments_list.append(new_phrase_copy); continue
        speakers_overlap_in_phrase = {} 
        temp_word_idx = word_idx_counter; current_word_processed_for_this_phrase_flag = False
        while temp_word_idx < len(word_segments_with_speakers):
            word_dict = word_segments_with_speakers[temp_word_idx]
            if not isinstance(word_dict, dict): temp_word_idx += 1; continue
            word_start_time = word_dict.get('start'); word_end_time = word_dict.get('end')
            word_speaker_id = word_dict.get('speaker') 
            if not (isinstance(word_start_time, (int,float)) and isinstance(word_end_time, (int,float))):
                temp_word_idx += 1; continue 
            if word_start_time > phrase_end_time + 0.5: break 
            if word_end_time < phrase_start_time - 0.01: 
                if word_idx_counter == temp_word_idx: word_idx_counter = temp_word_idx + 1 
                temp_word_idx += 1; continue
            current_word_processed_for_this_phrase_flag = True 
            overlap_start_calc = max(phrase_start_time, word_start_time)
            overlap_end_calc = min(phrase_end_time, word_end_time)
            overlap_duration_calc = overlap_end_calc - overlap_start_calc
            if overlap_duration_calc > 0.001: 
                speaker_to_assign = word_speaker_id if word_speaker_id else 'SPEAKER_00' 
                speakers_overlap_in_phrase[speaker_to_assign] = speakers_overlap_in_phrase.get(speaker_to_assign, 0.0) + overlap_duration_calc
            if word_end_time <= phrase_end_time:
                 if word_idx_counter == temp_word_idx: word_idx_counter = temp_word_idx + 1 
                 temp_word_idx +=1
            else: break 
        assigned_speaker_final = 'SPEAKER_00' 
        if speakers_overlap_in_phrase: assigned_speaker_final = max(speakers_overlap_in_phrase, key=speakers_overlap_in_phrase.get)
        new_phrase_copy_out = phrase_dict.copy(); new_phrase_copy_out['speaker'] = assigned_speaker_final
        output_phrase_segments_list.append(new_phrase_copy_out)
    return output_phrase_segments_list

def assign_srt_segments_to_speakers(srt_segments, diarization_df, trust_srt_speaker_field=True):
    """Присваивает спикеров сегментам SRT на основе DataFrame диаризации."""
    output_segments_list = []
    if not isinstance(srt_segments, list): srt_segments = []
    diarization_data_available = isinstance(diarization_df, pd.DataFrame) and not diarization_df.empty
    if not diarization_data_available:
        for srt_seg_item in srt_segments:
            new_seg_copy = srt_seg_item.copy() if isinstance(srt_seg_item, dict) else {'text': str(srt_seg_item), 'start':0, 'end':0}
            if trust_srt_speaker_field and 'speaker' in new_seg_copy and new_seg_copy['speaker'] not in ['SPEAKER_00', 'SPEAKER_UNKNOWN', None, '']: pass
            else: new_seg_copy['speaker'] = 'SPEAKER_00' 
            output_segments_list.append(new_seg_copy)
        return output_segments_list
    for srt_seg_item_orig in srt_segments:
        new_seg_copy_assign = srt_seg_item_orig.copy() if isinstance(srt_seg_item_orig, dict) else {'text': str(srt_seg_item_orig), 'start':0, 'end':0, 'speaker':'SPEAKER_00'}
        srt_speaker_original_val = new_seg_copy_assign.get('speaker')
        if trust_srt_speaker_field and srt_speaker_original_val and srt_speaker_original_val not in ['SPEAKER_00', 'SPEAKER_UNKNOWN', None, '']: pass
        else: 
            srt_start_time_val = new_seg_copy_assign.get('start'); srt_end_time_val = new_seg_copy_assign.get('end')     
            if not (isinstance(srt_start_time_val, (int,float)) and isinstance(srt_end_time_val, (int,float))): 
                new_seg_copy_assign['speaker'] = 'SPEAKER_00'; output_segments_list.append(new_seg_copy_assign); continue
            overlapping_speakers_map = {} 
            for _idx, dia_row_data in diarization_df.iterrows():
                dia_start_time = dia_row_data.get('start'); dia_end_time = dia_row_data.get('end')
                speaker_label_from_diar_val = dia_row_data.get('speaker')
                if not (isinstance(dia_start_time, (int,float)) and isinstance(dia_end_time, (int,float)) and speaker_label_from_diar_val): continue 
                overlap_start_val = max(srt_start_time_val, dia_start_time); overlap_end_val = min(srt_end_time_val, dia_end_time)
                overlap_duration_val = overlap_end_val - overlap_start_val
                if overlap_duration_val > 0.01: 
                    overlapping_speakers_map[speaker_label_from_diar_val] = overlapping_speakers_map.get(speaker_label_from_diar_val, 0) + overlap_duration_val
            assigned_speaker_from_diar_val = 'SPEAKER_00' 
            if overlapping_speakers_map: assigned_speaker_from_diar_val = max(overlapping_speakers_map, key=overlapping_speakers_map.get)
            new_seg_copy_assign['speaker'] = assigned_speaker_from_diar_val
        output_segments_list.append(new_seg_copy_assign)
    return output_segments_list

def get_chunk_definitions_from_phrases(all_phrases, 
                                       target_duration_sec=DEFAULT_TARGET_CHUNK_DURATION_SEC, 
                                       max_duration_sec=DEFAULT_MAX_CHUNK_DURATION_SEC,
                                       processing_start_offset_sec=0.0):
    """
    Формирует определения чанков на основе списка фраз и их таймингов.
    Перенесено из transcriber.py.
    """
    chunk_definitions_list = []
    if not all_phrases: return chunk_definitions_list
    current_chunk_phrases_buffer = []
    relevant_phrases_sorted = sorted(
        [p for p in all_phrases if p.get('end', 0) > processing_start_offset_sec and p.get('start', float('inf')) < p.get('end', 0)],
        key=lambda p: p.get('start', float('inf')))
    if not relevant_phrases_sorted: return chunk_definitions_list
    current_chunk_abs_start_time = -1 
    for i_phrase, phrase_data in enumerate(relevant_phrases_sorted):
        phrase_abs_start = phrase_data['start']; phrase_abs_end = phrase_data['end']
        if not current_chunk_phrases_buffer:
            current_chunk_abs_start_time = max(processing_start_offset_sec, phrase_abs_start)
            current_chunk_phrases_buffer.append(phrase_data); continue
        potential_chunk_abs_end_time = phrase_abs_end
        potential_chunk_duration = potential_chunk_abs_end_time - current_chunk_abs_start_time
        should_close_chunk_now = False
        if potential_chunk_duration > max_duration_sec: should_close_chunk_now = True
        elif (current_chunk_phrases_buffer[-1]['end'] - current_chunk_abs_start_time >= target_duration_sec):
            if i_phrase < len(relevant_phrases_sorted) -1: # Если есть следующая фраза
                next_phrase_data = relevant_phrases_sorted[i_phrase] # Текущая станет следующей для чанка без нее
                if (next_phrase_data['end'] - current_chunk_abs_start_time) > max_duration_sec: should_close_chunk_now = True
            # Если это последняя фраза, should_close_chunk_now останется False, и фраза добавится к текущему чанку.
        if should_close_chunk_now:
            if current_chunk_phrases_buffer: 
                chunk_actual_start = current_chunk_abs_start_time
                chunk_actual_end = current_chunk_phrases_buffer[-1]['end']
                if chunk_definitions_list and chunk_actual_start < chunk_definitions_list[-1]['end']:
                    chunk_actual_start = chunk_definitions_list[-1]['end'] + MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS
                if chunk_actual_end > chunk_actual_start + MIN_CHUNK_DURATION_THRESHOLD_SEC: 
                    chunk_definitions_list.append({'id': len(chunk_definitions_list), 'start': chunk_actual_start, 'end': chunk_actual_end, 'phrases_abs': list(current_chunk_phrases_buffer)})
            current_chunk_abs_start_time = max(processing_start_offset_sec, phrase_abs_start)
            if chunk_definitions_list and current_chunk_abs_start_time < chunk_definitions_list[-1]['end']:
                 current_chunk_abs_start_time = chunk_definitions_list[-1]['end'] + MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS
            current_chunk_phrases_buffer = [phrase_data]
        else:
            current_chunk_phrases_buffer.append(phrase_data)
    if current_chunk_phrases_buffer:
        chunk_actual_start = current_chunk_abs_start_time
        chunk_actual_end = current_chunk_phrases_buffer[-1]['end']
        if chunk_definitions_list and chunk_actual_start < chunk_definitions_list[-1]['end']:
            chunk_actual_start = chunk_definitions_list[-1]['end'] + MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS
        if chunk_actual_end > chunk_actual_start + MIN_CHUNK_DURATION_THRESHOLD_SEC:
            chunk_definitions_list.append({'id': len(chunk_definitions_list), 'start': chunk_actual_start, 'end': chunk_actual_end, 'phrases_abs': list(current_chunk_phrases_buffer)})
    for chunk_def_item in chunk_definitions_list:
        chunk_start_time_val = chunk_def_item['start']
        chunk_def_item['phrases_relative'] = []
        for p_abs in chunk_def_item.get('phrases_abs', []): # .get для безопасности
            p_rel = p_abs.copy()
            p_rel['start'] = p_abs['start'] - chunk_start_time_val
            p_rel['end'] = p_abs['end'] - chunk_start_time_val
            p_rel['original_abs_start'] = p_abs['start']
            p_rel['original_abs_end'] = p_abs['end']
            chunk_def_item['phrases_relative'].append(p_rel)
        if 'phrases_abs' in chunk_def_item: del chunk_def_item['phrases_abs'] 
    return chunk_definitions_list
