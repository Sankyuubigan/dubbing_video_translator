import whisperx
import torch
import os
from types import SimpleNamespace
import traceback
import srt
import datetime
import re
import pandas as pd
import shutil 
from difflib import SequenceMatcher 

try:
    from deepmultilingualpunctuation import PunctuationModel
    punctuation_model_cache = SimpleNamespace(model=None)
except ImportError:
    PunctuationModel = None
    punctuation_model_cache = None
    print("WARNING: deepmultilingualpunctuation library not found. Punctuation restoration will be skipped. Install with 'pip install deepmultilingualpunctuation'")


models_cache = SimpleNamespace(stt_model=None, diarization_model=None, align_model_cache=None)
DEFAULT_STT_MODEL_NAME = "medium.en" 

MAX_SEGMENT_DURATION_FROM_WORDS = 10.0 
MAX_PAUSE_BETWEEN_WORDS = 0.7 

CHARS_PER_SECOND_ESTIMATE = 13 
MAX_TEXT_TO_DURATION_RATIO_TOLERANCE = 3.5 
MIN_TEXT_DURATION_AFTER_ESTIMATE_CAP = 0.2 # Минимальная длительность сегмента с текстом
MAX_SILENCE_AFTER_TEXT_IN_LONG_SEGMENT = 1.5 
MIN_SEGMENT_DURATION_FOR_POSTPROCESS = 0.05 # Сегменты короче этого будут удалены (если они не пустые специально)
MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS = 0.01 

def ähnlich(a, b): 
    return SequenceMatcher(None, a, b).ratio()

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
            lang_for_load = "en" if ".en" in stt_model_name.lower() else None
            
            try:
                models_cache.stt_model = whisperx.load_model(stt_model_name, device, compute_type=stt_compute_type, language=lang_for_load)
            finally:
                pass

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
    if not language_code: 
        print(f"Warning: Language code for alignment is not specified, defaulting to 'en'.")
        align_lang_code_to_use = "en"

    if align_lang_code_to_use not in models_cache.align_model_cache or \
       models_cache.align_model_cache[align_lang_code_to_use].get('device') != device:
        print(f"Loading align model for language '{align_lang_code_to_use}' on device '{device}'...")
        try:
            model, metadata = whisperx.load_align_model(language_code=align_lang_code_to_use, device=device)
            models_cache.align_model_cache[align_lang_code_to_use] = {'model': model, 'metadata': metadata, 'device': device}
            print("Align model loaded.")
        except Exception as e_align_load:
            print(f"ERROR loading align model for language '{align_lang_code_to_use}': {e_align_load}")
            print("Falling back to 'en' align model if available, or skipping alignment.")
            if align_lang_code_to_use != "en" and "en" in models_cache.align_model_cache:
                print("Using cached 'en' align model as fallback.")
                return models_cache.align_model_cache["en"]['model'], models_cache.align_model_cache["en"]['metadata']
            return None, None 
            
    return models_cache.align_model_cache[align_lang_code_to_use]['model'], models_cache.align_model_cache[align_lang_code_to_use]['metadata']


def _load_punctuation_model():
    global punctuation_model_cache
    if PunctuationModel is None: 
        return None
    if punctuation_model_cache.model is None:
        print("Loading punctuation model (deepmultilingualpunctuation)...")
        try:
            punctuation_model_cache.model = PunctuationModel() 
            print("Punctuation model loaded.")
        except Exception as e:
            print(f"Error loading punctuation model: {e}")
            punctuation_model_cache.model = None
    return punctuation_model_cache.model

def _restore_punctuation(text_list):
    if not text_list: return [] 
    p_model = _load_punctuation_model()
    if p_model is None: return text_list 
    
    restored_texts_all = []
    try:
        if hasattr(p_model, 'restore_punctuation') and callable(p_model.restore_punctuation):
            for text_item in text_list:
                if not text_item or not text_item.strip(): restored_texts_all.append(text_item) 
                else: restored_texts_all.append(p_model.restore_punctuation(text_item))
        elif callable(p_model): 
            for text_item in text_list:
                if not text_item or not text_item.strip(): restored_texts_all.append(text_item)
                else: restored_texts_all.append(p_model(text_item))
        else:
            print("Punctuation model does not have a recognized method for restoration. Texts will be used as is.")
            return text_list
        return restored_texts_all
    except Exception as e:
        print(f"Error during punctuation restoration: {e}. Texts will be used as is.")
        return text_list


def segments_to_srt(segments, use_translated_text_field=None):
    if not srt: print("ERROR: srt library is not available. Cannot generate SRT content."); return ""
    srt_subs = []
    for i, segment_data in enumerate(segments):
        start_time = segment_data.get('start'); end_time = segment_data.get('end')
        text_to_use = segment_data.get('text', '')
        if use_translated_text_field and use_translated_text_field in segment_data:
            text_to_use = segment_data.get(use_translated_text_field, '')
        
        if start_time is None or end_time is None or (not str(text_to_use).strip() and not segment_data.get('is_pause_segment')): 
            continue
        
        if end_time <= start_time:
            end_time = start_time + 0.050 

        start_td = datetime.timedelta(seconds=start_time); end_td = datetime.timedelta(seconds=end_time)
        speaker = segment_data.get('speaker', 'SPEAKER_00') 
        
        srt_text_final = ""
        if str(text_to_use).strip(): 
             srt_text_final = f"[{speaker}]: {text_to_use}" if speaker and speaker not in ['SPEAKER_00', 'SPEAKER_UNKNOWN'] else str(text_to_use)
        
        if srt_text_final or segment_data.get('is_pause_segment'):
            srt_subs.append(srt.Subtitle(index=len(srt_subs) + 1, start=start_td, end=end_td, content=srt_text_final))
            
    return srt.compose(srt_subs)

def _create_phrase_segments_from_words(word_segments, max_duration=MAX_SEGMENT_DURATION_FROM_WORDS, max_pause=MAX_PAUSE_BETWEEN_WORDS):
    if not word_segments:
        return []

    phrase_segments = []
    current_phrase_text = []
    current_phrase_start = -1
    current_phrase_end = -1
    last_word_end_time = -1

    for i, word_data in enumerate(word_segments):
        word_text = word_data.get("word", "").strip() 
        if not word_text or 'start' not in word_data or word_data.get('start') is None: 
            if word_text: print(f"Warning: Word '{word_text}' missing timing info from align, skipping.")
            continue

        start_time = word_data.get("start")
        end_time = word_data.get("end") 

        if end_time is None : 
            end_time = start_time + 0.01 
        elif end_time <= start_time: 
            end_time = start_time + 0.01

        if not current_phrase_text: 
            current_phrase_text.append(word_text)
            current_phrase_start = start_time
            current_phrase_end = end_time
            last_word_end_time = end_time
        else:
            pause_duration = start_time - last_word_end_time
            current_segment_would_be_duration = end_time - current_phrase_start

            if pause_duration > max_pause or \
               current_segment_would_be_duration > max_duration :
                
                phrase_segments.append({
                    "text": " ".join(current_phrase_text),
                    "start": current_phrase_start,
                    "end": current_phrase_end,
                })
                current_phrase_text = [word_text]
                current_phrase_start = start_time
                current_phrase_end = end_time
                last_word_end_time = end_time
            else: 
                current_phrase_text.append(word_text)
                current_phrase_end = end_time 
                last_word_end_time = end_time
        
        if i == len(word_segments) - 1 and current_phrase_text:
            phrase_segments.append({
                "text": " ".join(current_phrase_text),
                "start": current_phrase_start,
                "end": current_phrase_end,
            })
    return phrase_segments


def transcribe_and_diarize_audio( 
                           audio_path, 
                           language_for_stt="en", 
                           batch_size=16, 
                           return_diarization_df=False,
                           stt_model_name_override=None
                           ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Transcribing and Aligning audio ({language_for_stt}) using device: {device}")
    hf_auth_token = os.environ.get("HF_TOKEN")
    
    stt_model_to_use = stt_model_name_override if stt_model_name_override else DEFAULT_STT_MODEL_NAME
    stt_model, diarization_model_instance = load_stt_diarization_models(stt_model_name=stt_model_to_use, device=device, hf_token=hf_auth_token, load_stt=True, load_diarization=True)
    if stt_model is None: raise RuntimeError(f"STT model '{stt_model_to_use}' is not loaded.")
    
    audio_for_stt = whisperx.load_audio(audio_path)
    
    print(f"Performing STT with Whisper model '{stt_model_to_use}' (language: {language_for_stt})...")
    raw_transcription_result = stt_model.transcribe(audio_for_stt, language=language_for_stt, batch_size=batch_size)
    
    whisper_stt_segments = raw_transcription_result.get("segments", [])
    detected_lang_code_for_align = raw_transcription_result.get("language", language_for_stt) 
    if not detected_lang_code_for_align: detected_lang_code_for_align = language_for_stt 
    print(f"STT complete. Whisper detected/using language: {detected_lang_code_for_align}. Found {len(whisper_stt_segments)} base segments.")

    if whisper_stt_segments:
        texts_for_punct = [s.get('text','') for s in whisper_stt_segments]
        restored_texts = _restore_punctuation(texts_for_punct)
        if len(restored_texts) == len(whisper_stt_segments):
            for idx, s_data in enumerate(whisper_stt_segments):
                s_data['text'] = restored_texts[idx]
    
    print(f"Aligning {len(whisper_stt_segments)} STT segments (language for align model: {detected_lang_code_for_align})...")
    segments_after_align = [] 
    word_segments_for_diarization = [] 
    
    align_model, metadata = load_align_model_cached(language_code=detected_lang_code_for_align, device=device)

    if align_model and metadata:
        valid_segments_for_align_input = [s for s in whisper_stt_segments if s.get('text','').strip() and 'start' in s and 'end' in s]
            
        if valid_segments_for_align_input:
            temp_aligned_word_segments_list = []
            temp_aligned_phrase_segments_list = [] 

            for s_idx, single_seg_to_align in enumerate(valid_segments_for_align_input):
                try:
                    current_aligned_result = whisperx.align([single_seg_to_align], align_model, metadata, audio_for_stt, device, return_char_alignments=False)
                    
                    if current_aligned_result.get("word_segments"):
                        for w_seg_idx, w_seg in enumerate(current_aligned_result["word_segments"]):
                            if 'start' not in w_seg: 
                                print(f"    WARNING: Word segment {w_seg_idx} ('{w_seg.get('word','NO_WORD')}') missing 'start' after aligning main segment {s_idx}. Skipping this word.")
                                continue 
                            temp_aligned_word_segments_list.append(w_seg)
                    
                    if current_aligned_result.get("segments"):
                        temp_aligned_phrase_segments_list.extend(current_aligned_result["segments"])
                except Exception as e_align_single:
                    print(f"    EXCEPTION during whisperx.align for segment {s_idx} (Text: {single_seg_to_align.get('text','')[:30]}...): {e_align_single}")
                    continue 
            
            aligned_result = { "word_segments": temp_aligned_word_segments_list, "segments": temp_aligned_phrase_segments_list }

            if aligned_result.get("word_segments"):
                word_segments_for_diarization = aligned_result["word_segments"] 
                if aligned_result.get("segments"): segments_after_align = aligned_result["segments"]
                else: segments_after_align = _create_phrase_segments_from_words(word_segments_for_diarization)
            else: segments_after_align = []
            print(f"Alignment complete. Produced {len(segments_after_align)} phrase segments and {len(word_segments_for_diarization)} valid word segments.")
        else: print("No valid STT segments to align.")
    else:
        print("Warning: Align model not loaded or metadata missing. Using STT segments without fine-grained alignment.")
        segments_after_align = whisper_stt_segments 
        if segments_after_align and all("words" in s for s in segments_after_align):
            for s in segments_after_align: word_segments_for_diarization.extend(s["words"])
        elif segments_after_align: 
             word_segments_for_diarization = [{"word": s["text"], "start":s["start"], "end":s["end"]} for s in segments_after_align if "text" in s and "start" in s and "end" in s]

    if device == "cuda": torch.cuda.empty_cache()
    
    final_segments_with_speakers = segments_after_align 
    diarize_segments_df_result = None

    if diarization_model_instance:
        print("Performing diarization...")
        try:
            diarize_segments_df_result = diarization_model_instance(audio_path) 
            if diarize_segments_df_result is not None and not diarize_segments_df_result.empty:
                if word_segments_for_diarization and segments_after_align: 
                    final_segments_with_speakers = assign_speakers_to_phrases(segments_after_align, word_segments_for_diarization)
                elif segments_after_align: 
                    final_segments_with_speakers = assign_srt_segments_to_speakers(segments_after_align, diarize_segments_df_result, trust_srt_speaker_field=False)
            else: 
                for seg in final_segments_with_speakers: seg['speaker'] = 'SPEAKER_00'
        except Exception as e:
            print(f"Error during diarization/speaker assignment: {e}. Falling back to default speaker.");
            for seg in final_segments_with_speakers: seg['speaker'] = 'SPEAKER_00'
            diarize_segments_df_result = None
    else: 
        for seg in final_segments_with_speakers: seg['speaker'] = 'SPEAKER_00'

    valid_timed_segments = []
    for seg_idx, seg_val in enumerate(final_segments_with_speakers):
        if seg_val.get('start') is not None and seg_val.get('end') is not None and seg_val.get('text','').strip() : 
            valid_timed_segments.append(seg_val)
    
    final_segments_with_speakers = valid_timed_segments
    print(f"Returning {len(final_segments_with_speakers)} final, timed, non-empty segments from transcribe_and_diarize_audio.")

    if return_diarization_df: return final_segments_with_speakers, diarize_segments_df_result
    return final_segments_with_speakers


def assign_speakers_to_phrases(phrase_segments, word_segments_with_speakers):
    if not word_segments_with_speakers:
        for phrase in phrase_segments:
            phrase['speaker'] = 'SPEAKER_00'
        return phrase_segments

    output_phrase_segments = []
    word_idx = 0
    for phrase in phrase_segments:
        phrase_start = phrase.get('start')
        phrase_end = phrase.get('end')
        if phrase_start is None or phrase_end is None:
            new_phrase = phrase.copy()
            new_phrase['speaker'] = 'SPEAKER_00'
            output_phrase_segments.append(new_phrase)
            continue

        speakers_in_phrase = {}
        
        temp_word_idx = word_idx
        current_word_processed_for_this_phrase = False

        while temp_word_idx < len(word_segments_with_speakers):
            word = word_segments_with_speakers[temp_word_idx]
            
            word_start = word.get('start')
            word_end = word.get('end')
            word_speaker = word.get('speaker')

            if word_start is None or word_end is None or word_speaker is None:
                temp_word_idx += 1
                continue

            if word_start > phrase_end + 0.5: 
                break 
            
            if word_end < phrase_start - 0.01:
                if word_idx == temp_word_idx: 
                    word_idx = temp_word_idx + 1
                temp_word_idx += 1
                continue
            
            current_word_processed_for_this_phrase = True
            overlap_start = max(phrase_start, word_start)
            overlap_end = min(phrase_end, word_end)
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > 0.001: 
                speakers_in_phrase[word_speaker] = speakers_in_phrase.get(word_speaker, 0.0) + overlap_duration
            
            if word_end <= phrase_end:
                 if word_idx == temp_word_idx: 
                     word_idx = temp_word_idx + 1 
                 temp_word_idx +=1
            else: 
                break 
        
        if not current_word_processed_for_this_phrase and temp_word_idx >= len(word_segments_with_speakers):
             pass


        assigned_speaker = 'SPEAKER_00' 
        if speakers_in_phrase:
            assigned_speaker = max(speakers_in_phrase, key=speakers_in_phrase.get)
        
        new_phrase = phrase.copy()
        new_phrase['speaker'] = assigned_speaker
        output_phrase_segments.append(new_phrase)
        
    return output_phrase_segments


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

def assign_srt_segments_to_speakers(srt_segments, diarization_df, trust_srt_speaker_field=True):
    output_segments = []
    
    if (diarization_df is None or diarization_df.empty):
        for srt_seg in srt_segments:
            new_seg = srt_seg.copy()
            if trust_srt_speaker_field and 'speaker' in new_seg and new_seg['speaker'] not in ['SPEAKER_00', 'SPEAKER_UNKNOWN', None]:
                pass 
            else:
                new_seg['speaker'] = 'SPEAKER_00' 
            output_segments.append(new_seg)
        if diarization_df is None or diarization_df.empty:
             print("Diarization data is empty. Speaker assignment based on trust_srt_speaker_field or default.")
        return output_segments

    for srt_seg in srt_segments:
        new_seg = srt_seg.copy()
        srt_speaker_original = new_seg.get('speaker')

        if trust_srt_speaker_field and srt_speaker_original and srt_speaker_original not in ['SPEAKER_00', 'SPEAKER_UNKNOWN', None]:
            pass 
        else: 
            srt_start_time = new_seg.get('start') 
            srt_end_time = new_seg.get('end')     
            
            if srt_start_time is None or srt_end_time is None: 
                new_seg['speaker'] = 'SPEAKER_00'
                output_segments.append(new_seg)
                continue

            overlapping_speakers = {}

            for _idx, dia_row in diarization_df.iterrows():
                dia_start = dia_row['start']
                dia_end = dia_row['end']
                speaker_label_from_diar = dia_row['speaker']
                
                overlap_start = max(srt_start_time, dia_start)
                overlap_end = min(srt_end_time, dia_end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 0.01: 
                    overlapping_speakers[speaker_label_from_diar] = overlapping_speakers.get(speaker_label_from_diar, 0) + overlap_duration
            
            assigned_speaker_from_diar = 'SPEAKER_00' 
            if overlapping_speakers:
                assigned_speaker_from_diar = max(overlapping_speakers, key=overlapping_speakers.get)
            new_seg['speaker'] = assigned_speaker_from_diar
        
        output_segments.append(new_seg)
        
    return output_segments


def _clean_srt_text_line(line: str) -> str:
    line = re.sub(r'^\[SPEAKER_\d+\]\s*:\s*', '', line, flags=re.IGNORECASE)
    line = re.sub(r'^SPEAKER_\d+\s*:\s*', '', line, flags=re.IGNORECASE)
    line = re.sub(r'<[^>]+>', '', line) 
    line = line.replace('♪', '').replace('♫', '')
    
    cleaned_line_for_check = line.lower()
    interjections_to_remove = ['хм', 'мм', 'гм', 'ага', 'ого', 'угу', 'эм', 'ээ', 'ой', 'ай', 'ну']
    for interjection in interjections_to_remove:
        cleaned_line_for_check = cleaned_line_for_check.replace(interjection, '')
    
    if not cleaned_line_for_check.replace(' ', '').strip():
        if re.fullmatch(r'^[ \t\n\r\f\v\x00-\x1F\x7F-\x9F.,!?"\'«»…()*#%;:/@<>=\[\]\\^`{}|~+-]*$', line):
            return ""
            
    line = line.replace('\xa0', ' ') 
    line = re.sub(r'\s+', ' ', line).strip() 
    return line.strip()

def _preprocess_vtt_content(content: str) -> str:
    if content.startswith("WEBVTT"):
        header_end_match = re.search(r"WEBVTT.*?\n\n", content, re.DOTALL | re.IGNORECASE)
        if header_end_match: content = content[header_end_match.end():]
        else:
            lines = content.splitlines()
            if lines and "WEBVTT" in lines[0].upper(): content = "\n".join(lines[1:])
    content = re.sub(r"(\d{2}:\d{2}:\d{2})[.](\d{3})", r"\1,\2", content) 
    content = re.sub(r'\n\s*\n', '\n\n', content).strip() 
    return content

def _postprocess_srt_segments(segments, is_external_srt=False): # Добавлен флаг
    if not segments: return []
    
    try:
        segments.sort(key=lambda x: (x.get('start', float('inf')), x.get('end', float('inf'))))
    except TypeError: 
        segments = [s for s in segments if s.get('start') is not None and s.get('end') is not None]
        segments.sort(key=lambda x: (x['start'], x['end']))

    processed_segments = []
    
    # --- Логика для is_external_srt ---
    if is_external_srt:
        # print(f"Post-processing {len(segments)} EXTERNAL SRT segments with overlap 'meeting point' logic...")
        # Сначала пройдемся и скорректируем наложения "встречей посередине"
        # Этот цикл может изменить end_time предыдущих сегментов, поэтому нужен второй проход для других коррекций
        
        # Создаем копию для итерации, чтобы изменять оригинальный список `segments` безопасно
        # Однако, мы будем строить новый список `temp_processed_segments`
        temp_processed_segments = []
        if not segments: return []

        # Первым делом, убедимся, что у всех сегментов валидная начальная длительность
        for i in range(len(segments)):
            start_time = segments[i].get('start')
            end_time = segments[i].get('end')
            if start_time is None or end_time is None or end_time <= start_time:
                # print(f"  Segment {i} (text: '{segments[i].get('text','')[:20]}') has invalid initial timing. Setting minimal duration.")
                segments[i]['start'] = start_time if start_time is not None else (segments[i-1]['end'] + MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS if i > 0 and segments[i-1].get('end') is not None else 0)
                segments[i]['end'] = segments[i]['start'] + MIN_SEGMENT_DURATION_FOR_POSTPROCESS
        
        # Теперь обрабатываем наложения
        i = 0
        while i < len(segments):
            current_seg = segments[i].copy() # Работаем с копией для добавления в новый список
            
            if i + 1 < len(segments):
                next_seg = segments[i+1]
                
                # Проверка на валидность времен
                curr_start = current_seg.get('start')
                curr_end = current_seg.get('end')
                next_start = next_seg.get('start')
                
                if curr_start is not None and curr_end is not None and next_start is not None:
                    if curr_end > next_start: # Есть наложение
                        overlap_amount = curr_end - next_start
                        # print(f"  Overlap detected between seg {i} (end: {curr_end:.3f}) and seg {i+1} (start: {next_start:.3f}). Overlap: {overlap_amount:.3f}s")
                        
                        # Логика "встречи посередине"
                        meeting_point = next_start + overlap_amount / 2 
                        # meeting_point = (curr_end + next_start) / 2 # Альтернатива
                        
                        new_curr_end = meeting_point - (MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS / 2)
                        new_next_start = meeting_point + (MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS / 2)

                        # Убедимся, что сегменты не станут слишком короткими или с инвертированными временами
                        if new_curr_end > curr_start + MIN_SEGMENT_DURATION_FOR_POSTPROCESS / 2 : # Делим на 2, т.к. оба сегмента могут укоротиться
                            current_seg['end'] = new_curr_end
                        else: # Если текущий становится слишком коротким, просто обрезаем его до начала следующего
                            current_seg['end'] = next_start - MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS
                            if current_seg['end'] < current_seg['start']: current_seg['end'] = current_seg['start'] + MIN_SEGMENT_DURATION_FOR_POSTPROCESS
                            # print(f"    Adjusted seg {i} end to {current_seg['end']:.3f} (cut before next due to meeting point making it too short)")
                        
                        if new_next_start < next_seg.get('end', float('inf')) - MIN_SEGMENT_DURATION_FOR_POSTPROCESS / 2 :
                             segments[i+1]['start'] = new_next_start # Изменяем начало следующего сегмента в исходном списке для следующей итерации
                        else: # Если следующий становится слишком коротким
                            original_next_duration = next_seg.get('end', new_next_start + MIN_SEGMENT_DURATION_FOR_POSTPROCESS) - next_start
                            segments[i+1]['start'] = current_seg['end'] + MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS
                            segments[i+1]['end'] = segments[i+1]['start'] + max(MIN_SEGMENT_DURATION_FOR_POSTPROCESS, original_next_duration)
                            # print(f"    Adjusted seg {i+1} start to {segments[i+1]['start']:.3f} (pushed after current due to meeting point making it too short)")
                        # print(f"    Resolved to: seg {i} end: {current_seg['end']:.3f}, seg {i+1} start: {segments[i+1]['start']:.3f}")

            temp_processed_segments.append(current_seg)
            i += 1
        
        segments = temp_processed_segments # Перезаписываем исходный список уже без грубых наложений

    # --- Общая постобработка (коррекция длин, удаление слишком коротких) ---
    # print(f"Performing general post-processing for {'external' if is_external_srt else 'internal'} {len(segments)} segments...")
    last_valid_end_time = 0.0
    for i, seg_original_data in enumerate(segments):
        current_segment_data = seg_original_data.copy()
        text = current_segment_data.get('text', '').strip()
        start_time = current_segment_data.get('start')
        end_time = current_segment_data.get('end')

        if start_time is None or end_time is None: continue

        # Если это внешний SRT, мы уже пытались исправить наложения выше.
        # Здесь можно сделать более мягкую проверку или доверять результату "встречи посередине".
        # Но для сегментов от Whisper (is_external_srt=False), наложения маловероятны,
        # но все же проверим и сдвинем, если нужно.
        if start_time < last_valid_end_time:
            if not is_external_srt: # Для Whisper-сегментов просто сдвигаем
                 new_start_time = last_valid_end_time + MIN_GAP_BETWEEN_ADJUSTED_SEGMENTS
                 original_duration = end_time - start_time
                 end_time = new_start_time + max(MIN_SEGMENT_DURATION_FOR_POSTPROCESS, original_duration if original_duration > 0 else MIN_SEGMENT_DURATION_FOR_POSTPROCESS)
                 start_time = new_start_time
            # Если is_external_srt, то наложения уже должны были быть исправлены "встречей",
            # но если что-то осталось, эта логика может еще немного поправить.
            # Однако, если "встреча посередине" отработала, то start_time > last_valid_end_time.
            # Эта ветка здесь больше для подстраховки или для не-внешних SRT.
        
        current_segment_data['start'] = start_time
        current_segment_data['end'] = end_time
        current_srt_duration = end_time - start_time

        if current_srt_duration <= 0: # После всех манипуляций
             current_srt_duration = MIN_SEGMENT_DURATION_FOR_POSTPROCESS
             current_segment_data['end'] = current_segment_data['start'] + current_srt_duration
        
        if text:
            estimated_speech_duration = len(text) / CHARS_PER_SECOND_ESTIMATE
            if current_srt_duration > (estimated_speech_duration * MAX_TEXT_TO_DURATION_RATIO_TOLERANCE) and current_srt_duration > 0.2:
                capped_duration = estimated_speech_duration + MAX_SILENCE_AFTER_TEXT_IN_LONG_SEGMENT
                current_segment_data['end'] = current_segment_data['start'] + capped_duration
            current_segment_data['end'] = max(current_segment_data['end'], current_segment_data['start'] + MIN_TEXT_DURATION_AFTER_ESTIMATE_CAP)
        elif not text: # Пустой сегмент (пауза)
            pause_duration = current_segment_data['end'] - current_segment_data['start']
            if pause_duration > MAX_SILENCE_AFTER_TEXT_IN_LONG_SEGMENT:
                current_segment_data['end'] = current_segment_data['start'] + MAX_SILENCE_AFTER_TEXT_IN_LONG_SEGMENT
            current_segment_data['end'] = max(current_segment_data['end'], current_segment_data['start'] + MIN_SEGMENT_DURATION_FOR_POSTPROCESS)

        final_duration = current_segment_data['end'] - current_segment_data['start']

        if final_duration >= MIN_SEGMENT_DURATION_FOR_POSTPROCESS:
            processed_segments.append(current_segment_data)
            last_valid_end_time = current_segment_data['end']
            
    # print(f"Post-processing finished. Returning {len(processed_segments)} segments.")
    return processed_segments


def parse_srt_file(srt_file_path, post_process=False): # post_process флаг здесь не используется
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
            speaker_in_text = 'SPEAKER_00' 
            full_raw_text = " ".join(lines).strip()
            
            speaker_match = re.match(r'^\[(SPEAKER_\d+)\]\s*:\s*(.*)', full_raw_text, re.IGNORECASE)
            text_content_for_cleaning = full_raw_text
            if speaker_match:
                speaker_in_text = speaker_match.group(1).upper()
                text_content_for_cleaning = speaker_match.group(2)
            else: 
                speaker_match_alt = re.match(r'^(SPEAKER_\d+)\s*:\s*(.*)', full_raw_text, re.IGNORECASE)
                if speaker_match_alt:
                    speaker_in_text = speaker_match_alt.group(1).upper()
                    text_content_for_cleaning = speaker_match_alt.group(2)
            
            cleaned_lines = [_clean_srt_text_line(line) for line in text_content_for_cleaning.splitlines()]
            full_cleaned_text = " ".join(filter(None, cleaned_lines)).strip()
            full_cleaned_text = re.sub(r'\s+', ' ', full_cleaned_text).strip()

            current_start_time = sub.start.total_seconds()
            current_end_time = sub.end.total_seconds()

            if current_end_time <= current_start_time: # Обеспечиваем минимальную положительную длительность
                current_end_time = current_start_time + MIN_SEGMENT_DURATION_FOR_POSTPROCESS 
 
            segments.append({
                'text': full_cleaned_text, 
                'start': current_start_time,
                'end': current_end_time,
                'speaker': speaker_in_text 
            })
            
        return segments
    except FileNotFoundError: print(f"Error: SRT/VTT file not found at {srt_file_path}"); return []
    except srt.SRTParseError as e_srt_parse:
        print(f"Error parsing SRT/VTT file {srt_file_path} with `srt` library: {e_srt_parse}")
        # print(f"Problematic content snippet (first 500 chars after preprocessing):\n{processed_content[:500]}")
        return []
    except Exception as e_gen:
        print(f"General error parsing SRT/VTT file {srt_file_path}: {e_gen}\nDetails: {traceback.format_exc()}")
        return []
