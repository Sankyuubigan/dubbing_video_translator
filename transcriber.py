import whisperx
import torch
import os
from types import SimpleNamespace
import traceback
import pandas as pd # srt больше не нужен здесь напрямую
import shutil 
from difflib import SequenceMatcher 

# Импортируем утилиты из нового файла
from utils import segment_utils 

try:
    from deepmultilingualpunctuation import PunctuationModel
    punctuation_model_cache = SimpleNamespace(model=None)
except ImportError:
    PunctuationModel = None
    punctuation_model_cache = None
    print("WARNING: deepmultilingualpunctuation library not found. Punctuation restoration will be skipped.")


models_cache = SimpleNamespace(stt_model=None, diarization_model=None, align_model_cache=None)
DEFAULT_STT_MODEL_NAME = "medium.en" 
# Константы, специфичные для транскрипции, остаются здесь
# Например, DEFAULT_STT_MODEL_NAME

# Константы для обработки чанков (перенесены в segment_utils)
# DEFAULT_TARGET_CHUNK_DURATION_SEC
# DEFAULT_MAX_CHUNK_DURATION_SEC
# MIN_CHUNK_DURATION_THRESHOLD_SEC

# Константа для максимальной длины аудио для одного прохода WhisperX при transcribe_full_audio_for_phrases
# Это поможет избежать CUDA out of memory при транскрипции всего файла сразу.
# Значение в секундах, например, 10 минут = 600 секунд. Подбирается экспериментально.
MAX_AUDIO_LENGTH_FOR_FULL_TRANSCRIPTION_PASS_SEC = 600 # 10 минут


def ähnlich(a, b): 
    return SequenceMatcher(None, a, b).ratio()

def load_stt_diarization_models(stt_model_name=DEFAULT_STT_MODEL_NAME, device="cpu", hf_token=None, load_stt=True, load_diarization=True):
    global models_cache
    stt_model_to_return = models_cache.stt_model
    diarization_model_to_return = models_cache.diarization_model
    stt_load_successful = not load_stt 
    diar_load_successful = not load_diarization
    if load_stt:
        stt_compute_type = "float16" if device == "cuda" else "int8"
        if device == "cpu" and stt_compute_type == "int8":
            ctranslate2_available = False
            try: import ctranslate2; ctranslate2_available = True
            except ImportError: pass 
            if not ctranslate2_available: print("ctranslate2 not found, WhisperX STT on CPU will use 'float32'."); stt_compute_type = "float32"
        needs_stt_reload = True
        if models_cache.stt_model:
            if hasattr(models_cache.stt_model, 'loaded_model_name') and models_cache.stt_model.loaded_model_name == stt_model_name and \
               hasattr(models_cache.stt_model, 'loaded_compute_type') and models_cache.stt_model.loaded_compute_type == stt_compute_type and \
               hasattr(models_cache.stt_model, 'loaded_device') and models_cache.stt_model.loaded_device == device:
                 needs_stt_reload = False; stt_load_successful = True 
        if needs_stt_reload:
            print(f"Loading WhisperX STT model: {stt_model_name} on device: {device} with compute_type: {stt_compute_type}")
            lang_for_load = "en" if ".en" in stt_model_name.lower() else None
            temp_stt_model = None
            try: temp_stt_model = whisperx.load_model(stt_model_name, device, compute_type=stt_compute_type, language=lang_for_load)
            except Exception as e_load_stt: print(f"CRITICAL ERROR during whisperx.load_model for STT: {e_load_stt}"); temp_stt_model = None
            if temp_stt_model is not None:
                models_cache.stt_model = temp_stt_model
                models_cache.stt_model.loaded_model_name = stt_model_name; models_cache.stt_model.loaded_compute_type = stt_compute_type
                models_cache.stt_model.loaded_device = device; print("STT model loaded."); stt_load_successful = True
            else: print(f"ERROR: Failed to load STT model {stt_model_name}."); models_cache.stt_model = None; stt_load_successful = False
        stt_model_to_return = models_cache.stt_model
    if load_diarization:
        if models_cache.diarization_model is None:
            print("Loading Diarization model...")
            temp_diar_model = None
            try:
                if hf_token: temp_diar_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
                else: temp_diar_model = whisperx.DiarizationPipeline(device=device)
            except Exception as e_load_diar: print(f"CRITICAL ERROR during whisperx.DiarizationPipeline init: {e_load_diar}"); temp_diar_model = None
            if temp_diar_model is not None: models_cache.diarization_model = temp_diar_model; print("Diarization model loaded."); diar_load_successful = True
            else: print(f"ERROR: Failed to load Diarization model."); models_cache.diarization_model = None; diar_load_successful = False
        else: diar_load_successful = True
        diarization_model_to_return = models_cache.diarization_model
    return stt_model_to_return, diarization_model_to_return, stt_load_successful, diar_load_successful

def load_align_model_cached(language_code, device):
    global models_cache
    if models_cache.align_model_cache is None: models_cache.align_model_cache = {}
    align_lang_code_to_use = language_code if language_code else "en" 
    if not language_code: print(f"Warning: Language code for alignment is not specified, defaulting to 'en'."); align_lang_code_to_use = "en"
    model_in_cache = models_cache.align_model_cache.get(align_lang_code_to_use)
    if model_in_cache is None or model_in_cache.get('device') != device:
        print(f"Loading align model for language '{align_lang_code_to_use}' on device '{device}'...")
        align_model_loaded = None; metadata_loaded = None
        try: align_model_loaded, metadata_loaded = whisperx.load_align_model(language_code=align_lang_code_to_use, device=device)
        except Exception as e_align_load: print(f"ERROR loading align model for language '{align_lang_code_to_use}': {e_align_load}"); return None, None, False
        if align_model_loaded is not None and metadata_loaded is not None:
            models_cache.align_model_cache[align_lang_code_to_use] = {'model': align_model_loaded, 'metadata': metadata_loaded, 'device': device}
            print("Align model loaded."); return align_model_loaded, metadata_loaded, True
        else: print(f"Failed to load align model or metadata for '{align_lang_code_to_use}'."); return None, None, False
    return model_in_cache['model'], model_in_cache['metadata'], True

def _load_punctuation_model():
    global punctuation_model_cache
    if PunctuationModel is None: return None, False
    if punctuation_model_cache.model is None:
        print("Loading punctuation model (deepmultilingualpunctuation)...")
        loaded_model = None
        try: loaded_model = PunctuationModel() 
        except Exception as e: print(f"Error loading punctuation model: {e}")
        if loaded_model is not None: punctuation_model_cache.model = loaded_model; print("Punctuation model loaded."); return punctuation_model_cache.model, True
        else: return None, False
    return punctuation_model_cache.model, True

def _restore_punctuation(text_list):
    if not text_list: return [] 
    p_model, load_success = _load_punctuation_model()
    if not load_success or p_model is None: return text_list 
    restored_texts_all = []; all_restored_successfully = True
    can_restore_punc = hasattr(p_model, 'restore_punctuation') and callable(p_model.restore_punctuation)
    can_call_punc_model = callable(p_model)
    if not (can_restore_punc or can_call_punc_model): print("Punctuation model does not have a recognized method for restoration."); return text_list
    for text_item in text_list:
        if not text_item or not text_item.strip(): restored_texts_all.append(text_item) 
        else:
            restored_text_item = text_item 
            try:
                if can_restore_punc: restored_text_item = p_model.restore_punctuation(text_item)
                elif can_call_punc_model: restored_text_item = p_model(text_item)
                restored_texts_all.append(restored_text_item)
            except Exception as e_punc:
                print(f"Error during punctuation restoration for text: '{text_item[:50]}...': {e_punc}. Using original.")
                restored_texts_all.append(text_item); all_restored_successfully = False
    return restored_texts_all

def transcribe_full_audio_for_phrases(audio_path, language_for_stt="en", batch_size=16, stt_model_name_override=None, 
                                     max_pass_duration_sec=MAX_AUDIO_LENGTH_FOR_FULL_TRANSCRIPTION_PASS_SEC):
    """
    Транскрибирует весь аудиофайл (возможно, по частям, если он слишком длинный) 
    и возвращает сегменты на уровне фраз с таймингами.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Performing full audio transcription for phrase timings ({language_for_stt}) using device: {device}")
    
    stt_model_name_to_use = stt_model_name_override if stt_model_name_override else DEFAULT_STT_MODEL_NAME
    stt_model, _diar_model_unused, stt_loaded, _diar_loaded_unused = load_stt_diarization_models(
        stt_model_name=stt_model_name_to_use, device=device, load_stt=True, load_diarization=False)
    if not stt_loaded or stt_model is None:
        print(f"ERROR: STT model '{stt_model_name_to_use}' could not be loaded for full transcription.")
        return [], False 

    # Получаем общую длительность аудио
    audio_duration_total_sec = 0
    # Проверка существования файла перед использованием sf.info
    if os.path.exists(audio_path):
        try:
            audio_info = sf.info(audio_path) # soundfile.info
            audio_duration_total_sec = audio_info.duration
        except Exception as e_sf_info:
             print(f"Error getting audio info for {audio_path} with soundfile: {e_sf_info}")
             # Попробуем через ffmpeg, если soundfile не справился
             from video_processor import get_video_duration # Локальный импорт для избежания циклов
             audio_duration_total_sec = get_video_duration(audio_path) # get_video_duration работает и для аудио

    if audio_duration_total_sec <= 0:
        print(f"Error: Could not determine duration of audio file {audio_path} or it is empty.")
        return [], False

    all_final_phrases_from_audio = []
    current_offset_sec = 0.0
    pass_num = 0

    while current_offset_sec < audio_duration_total_sec:
        pass_num += 1
        segment_duration_for_pass = min(max_pass_duration_sec, audio_duration_total_sec - current_offset_sec)
        if segment_duration_for_pass < 1.0: # Слишком короткий остаток для обработки
            break 
        
        print(f"  Full transcription pass {pass_num}: Processing audio from {current_offset_sec:.2f}s for {segment_duration_for_pass:.2f}s")
        
        audio_input_for_stt_pass = None
        try:
            # Загружаем только необходимый сегмент аудио
            audio_input_for_stt_pass, sr = torchaudio.load(audio_path, frame_offset=int(current_offset_sec * 16000), num_frames=int(segment_duration_for_pass * 16000))
            if sr != 16000: # WhisperX ожидает 16kHz
                 audio_input_for_stt_pass = torchaudio.transforms.Resample(sr, 16000)(audio_input_for_stt_pass)
            if audio_input_for_stt_pass.ndim > 1: # Преобразуем в моно, если нужно
                 audio_input_for_stt_pass = torch.mean(audio_input_for_stt_pass, dim=0, keepdim=True)
        except Exception as e_load_audio_segment:
            print(f"ERROR: Could not load audio segment for pass {pass_num} from {audio_path}: {e_load_audio_segment}")
            if device == "cuda": torch.cuda.empty_cache()
            # Если одна часть не загрузилась, это проблема для всей транскрипции
            return [], False 

        raw_transcription_result_pass = None
        try:
            raw_transcription_result_pass = stt_model.transcribe(audio_input_for_stt_pass.squeeze(0).numpy(), language=language_for_stt, batch_size=batch_size)
        except Exception as e_transcribe_pass:
            print(f"ERROR: Transcription failed for pass {pass_num}: {e_transcribe_pass}")
            if device == "cuda": torch.cuda.empty_cache()
            # Если одна часть не транскрибировалась, это проблема
            if pass_num == 1 and not all_final_phrases_from_audio : return [], False # Если первая же часть не удалась
            else: break # Прерываем цикл, если последующие части не удались

        if raw_transcription_result_pass is None or "segments" not in raw_transcription_result_pass:
            print(f"ERROR: Transcription result for pass {pass_num} is invalid or does not contain segments.")
            if device == "cuda": torch.cuda.empty_cache()
            if pass_num == 1 and not all_final_phrases_from_audio : return [], False
            else: break

        whisper_stt_segments_raw_pass = raw_transcription_result_pass.get("segments", [])
        detected_lang_code_for_align_pass = raw_transcription_result_pass.get("language", language_for_stt) 
        if not detected_lang_code_for_align_pass: detected_lang_code_for_align_pass = language_for_stt 
        
        if whisper_stt_segments_raw_pass:
            texts_for_punct_restore_pass = [s.get('text','') for s in whisper_stt_segments_raw_pass]
            restored_texts_list_pass = _restore_punctuation(texts_for_punct_restore_pass)
            if len(restored_texts_list_pass) == len(whisper_stt_segments_raw_pass):
                for idx, s_data_item_pass in enumerate(whisper_stt_segments_raw_pass):
                    s_data_item_pass['text'] = restored_texts_list_pass[idx]
        
        aligned_phrase_segments_pass = [] 
        align_model_instance, align_metadata_instance, align_model_loaded = load_align_model_cached(language_code=detected_lang_code_for_align_pass, device=device)
        if align_model_loaded and align_model_instance and align_metadata_instance:
            valid_segments_for_align_pass = [s for s in whisper_stt_segments_raw_pass if s.get('text','').strip() and 'start' in s and 'end' in s and s['start'] < s['end']]
            if valid_segments_for_align_pass:
                try:
                    aligned_result_dict_pass = whisperx.align(valid_segments_for_align_pass, align_model_instance, align_metadata_instance, audio_input_for_stt_pass.squeeze(0).numpy(), device, return_char_alignments=False)
                    aligned_phrase_segments_pass = aligned_result_dict_pass.get("segments", [])
                    if not aligned_phrase_segments_pass and aligned_result_dict_pass.get("word_segments"):
                        aligned_phrase_segments_pass = segment_utils.create_phrase_segments_from_words(aligned_result_dict_pass["word_segments"])
                except Exception as e_align_pass:
                    print(f"ERROR during whisperx.align for pass {pass_num}: {e_align_pass}. Using raw STT segments for this pass.")
                    aligned_phrase_segments_pass = whisper_stt_segments_raw_pass 
            else: aligned_phrase_segments_pass = []
        else: aligned_phrase_segments_pass = whisper_stt_segments_raw_pass 

        # Корректируем тайминги сегментов относительно начала всего аудио и добавляем в общий список
        for seg_data_val_pass in aligned_phrase_segments_pass:
            if seg_data_val_pass.get('text','').strip() and \
               isinstance(seg_data_val_pass.get('start'), (int, float)) and \
               isinstance(seg_data_val_pass.get('end'), (int, float)) and \
               seg_data_val_pass['start'] < seg_data_val_pass['end']:
                corrected_seg = seg_data_val_pass.copy()
                corrected_seg['start'] += current_offset_sec
                corrected_seg['end'] += current_offset_sec
                all_final_phrases_from_audio.append(corrected_seg)
        
        current_offset_sec += segment_duration_for_pass
        if device == "cuda": torch.cuda.empty_cache() # Очищаем память после каждой части
    
    if not all_final_phrases_from_audio and audio_duration_total_sec > 0 : # Если ничего не получили, но аудио было
        print("Warning: Full audio transcription resulted in no valid phrases after all passes.")
        return [], False # Считаем это неудачей, если аудио не пустое

    print(f"Full transcription finished. Returning {len(all_final_phrases_from_audio)} final phrase segments.")
    return all_final_phrases_from_audio, True


def transcribe_and_diarize_audio(audio_path, language_for_stt="en", batch_size=16, return_diarization_df=False, stt_model_name_override=None):
    """Транскрибирует и диаризует ОДИН АУДИО ЧАНК."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Transcribing and Aligning audio chunk ({language_for_stt}) using device: {device}")
    hf_auth_token = os.environ.get("HF_TOKEN")
    stt_model_to_use = stt_model_name_override if stt_model_name_override else DEFAULT_STT_MODEL_NAME
    stt_model, diarization_model_instance, stt_loaded, diar_loaded = load_stt_diarization_models(
        stt_model_name=stt_model_to_use, device=device, hf_token=hf_auth_token, load_stt=True, load_diarization=True)
    if not stt_loaded or stt_model is None:
        print(f"ERROR: STT model '{stt_model_to_use}' could not be loaded for chunk."); return ([], None) if return_diarization_df else [] 
    audio_for_stt = None
    try: audio_for_stt = whisperx.load_audio(audio_path)
    except Exception as e_load_audio_chunk: print(f"ERROR: Could not load audio {audio_path} for chunk: {e_load_audio_chunk}"); return ([], None) if return_diarization_df else []
    raw_transcription_result = None
    try: raw_transcription_result = stt_model.transcribe(audio_for_stt, language=language_for_stt, batch_size=batch_size)
    except Exception as e_transcribe_chunk: print(f"ERROR: Transcription failed for chunk: {e_transcribe_chunk}"); if device == "cuda": torch.cuda.empty_cache(); return ([], None) if return_diarization_df else []
    if raw_transcription_result is None or "segments" not in raw_transcription_result:
        print("ERROR: Transcription result for chunk invalid."); if device == "cuda": torch.cuda.empty_cache(); return ([], None) if return_diarization_df else []
    whisper_stt_segments = raw_transcription_result.get("segments", [])
    detected_lang_code_for_align = raw_transcription_result.get("language", language_for_stt) 
    if not detected_lang_code_for_align: detected_lang_code_for_align = language_for_stt 
    if whisper_stt_segments:
        texts_for_punct = [s.get('text','') for s in whisper_stt_segments]
        restored_texts = _restore_punctuation(texts_for_punct)
        if len(restored_texts) == len(whisper_stt_segments):
            for idx, s_data in enumerate(whisper_stt_segments): s_data['text'] = restored_texts[idx]
    segments_after_align = []; word_segments_for_diarization_assignment = []
    align_model, metadata, align_model_loaded_flag = load_align_model_cached(language_code=detected_lang_code_for_align, device=device)
    if align_model_loaded_flag and align_model and metadata:
        valid_segments_for_align_input = [s for s in whisper_stt_segments if s.get('text','').strip() and 'start' in s and 'end' in s and s['start'] < s['end']]
        if valid_segments_for_align_input:
            aligned_result_data = None
            try: aligned_result_data = whisperx.align(valid_segments_for_align_input, align_model, metadata, audio_for_stt, device, return_char_alignments=False)
            except Exception as e_align_c: print(f"ERROR during whisperx.align for chunk: {e_align_c}. Using raw STT for this chunk."); aligned_result_data = {"segments": whisper_stt_segments}
            if aligned_result_data:
                segments_after_align = aligned_result_data.get("segments", [])
                word_segments_for_diarization_assignment = aligned_result_data.get("word_segments", [])
                if not segments_after_align and word_segments_for_diarization_assignment: segments_after_align = segment_utils.create_phrase_segments_from_words(word_segments_for_diarization_assignment)
            else: segments_after_align = whisper_stt_segments
        else: segments_after_align = []
    else:
        segments_after_align = whisper_stt_segments 
        if segments_after_align: 
            for s_raw in segments_after_align:
                if "words" in s_raw and isinstance(s_raw["words"], list): word_segments_for_diarization_assignment.extend(s_raw["words"])
                elif s_raw.get("text") and isinstance(s_raw.get("start"), float) and isinstance(s_raw.get("end"), float):
                    word_segments_for_diarization_assignment.append({"word": s_raw["text"], "start": s_raw["start"], "end": s_raw["end"], "score": 1.0})
    if device == "cuda": torch.cuda.empty_cache()
    final_segments_with_speakers_assigned = segments_after_align; diarize_segments_df_for_return = pd.DataFrame()
    if diar_loaded and diarization_model_instance: 
        temp_diarize_df = None
        try: temp_diarize_df = diarization_model_instance(audio_path) 
        except Exception as e_diar_chunk: print(f"Error during diarization of chunk: {e_diar_chunk}."); temp_diarize_df = None
        if temp_diarize_df is not None and not temp_diarize_df.empty:
            diarize_segments_df_for_return = temp_diarize_df 
            if word_segments_for_diarization_assignment and segments_after_align: 
                words_with_speakers_assigned = whisperx.assign_word_speakers(diarize_segments_df_for_return, word_segments_for_diarization_assignment)
                final_segments_with_speakers_assigned = segment_utils.assign_speakers_to_phrases(segments_after_align, words_with_speakers_assigned)
            elif segments_after_align: 
                final_segments_with_speakers_assigned = segment_utils.assign_srt_segments_to_speakers(segments_after_align, diarize_segments_df_for_return, trust_srt_speaker_field=False)
        else: 
            for seg_item in final_segments_with_speakers_assigned: seg_item['speaker'] = 'SPEAKER_00'
    else: 
        for seg_item in final_segments_with_speakers_assigned: seg_item['speaker'] = 'SPEAKER_00'
    valid_timed_segments_final = []
    for seg_val_item in final_segments_with_speakers_assigned:
        if seg_val_item.get('text','').strip() and \
           isinstance(seg_val_item.get('start'), (int, float)) and \
           isinstance(seg_val_item.get('end'), (int, float)) and \
           seg_val_item['start'] < seg_val_item['end']: 
            valid_timed_segments_final.append(seg_val_item)
    final_segments_with_speakers_assigned = valid_timed_segments_final
    if return_diarization_df: return final_segments_with_speakers_assigned, diarize_segments_df_for_return
    return final_segments_with_speakers_assigned

def perform_diarization_only(audio_path, device="cpu", hf_token=None):
    """Выполняет только диаризацию аудиофайла."""
    _stt_model_unused, diarization_model_inst, _stt_load_ok, diar_load_ok = load_stt_diarization_models(
        device=device, hf_token=hf_token, load_stt=False, load_diarization=True)
    if not diar_load_ok or diarization_model_inst is None:
        print("Diarization model not loaded. Cannot perform diarization."); return pd.DataFrame()
    diarization_result_df_out = pd.DataFrame() 
    try:
        temp_result_df = diarization_model_inst(audio_path)
        if temp_result_df is not None and isinstance(temp_result_df, pd.DataFrame) and not temp_result_df.empty:
            print(f"Diarization found {len(temp_result_df['speaker'].unique())} unique speakers.")
            diarization_result_df_out = temp_result_df
        else: print("Diarization did not return any speaker segments or result was not a DataFrame.")
    except Exception as e_diar_only: print(f"Error during standalone diarization: {e_diar_only}\nDetails: {traceback.format_exc()}")
    return diarization_result_df_out

def parse_srt_file(srt_file_path): # post_process_parsed больше не нужен как параметр
    """Парсит SRT/VTT файл. Возвращает (list_of_segments, success_flag)."""
    # srt импортируется в segment_utils, здесь он не нужен
    print(f"Parsing SRT/VTT file: {os.path.basename(srt_file_path)}")
    segments_list = []
    if not os.path.exists(srt_file_path):
        print(f"Error: SRT/VTT file not found at {srt_file_path}"); return [], False
    raw_content_str = ""; f_srt = None
    try:
        f_srt = open(srt_file_path, 'r', encoding='utf-8-sig'); raw_content_str = f_srt.read()
    except IOError as e_io_srt: print(f"Error reading SRT/VTT file {srt_file_path}: {e_io_srt}"); return [], False
    finally:
        if f_srt: f_srt.close()
    processed_content_for_srt_lib = segment_utils.preprocess_vtt_content(raw_content_str)
    if not processed_content_for_srt_lib.strip():
        print(f"Warning: SRT/VTT file {srt_file_path} is empty or became empty after VTT header removal."); return [], True 
    parsed_subs_list = []
    try:
        # srt.parse должен быть доступен, если srt_lib_available в main.py True
        # Это означает, что импорт srt в segment_utils тоже должен быть доступен.
        global srt # Делаем srt глобальным, если он был импортирован в main и передан сюда
        if 'srt' not in globals() or globals()['srt'] is None: # Если srt не импортирован глобально (маловероятно)
             # Попробуем импортировать srt здесь, если он не доступен глобально
             # Это нарушение принципа единого импорта, но как фоллбэк
             try:
                 import srt as srt_local
                 parsed_subs_list = list(srt_local.parse(processed_content_for_srt_lib))
             except ImportError:
                  print("FATAL: srt library is missing for parse_srt_file and was not passed/imported.")
                  return [], False
             except srt_local.SRTParseError as e_srt_p_lib: print(f"Error parsing SRT/VTT {srt_file_path} with `srt`: {e_srt_p_lib}"); return [], False
        else: # srt должен быть доступен из глобальной области видимости (импортирован в segment_utils)
            parsed_subs_list = list(globals()['srt'].parse(processed_content_for_srt_lib))

    except Exception as e_gen_srt_parse: # Ловим другие ошибки от srt.parse
        print(f"General error during srt.parse for {srt_file_path}: {e_gen_srt_parse}"); return [], False

    for sub_idx_val, sub_obj in enumerate(parsed_subs_list):
        lines_from_sub = sub_obj.content.splitlines() if sub_obj.content else []
        speaker_in_text_val = 'SPEAKER_00'; full_raw_text_from_sub = " ".join(lines_from_sub).strip()
        speaker_match_result = re.match(r'^\[(SPEAKER_\d+)\]\s*:\s*(.*)', full_raw_text_from_sub, re.IGNORECASE)
        text_content_for_cleaning_val = full_raw_text_from_sub
        if speaker_match_result: speaker_in_text_val = speaker_match_result.group(1).upper(); text_content_for_cleaning_val = speaker_match_result.group(2)
        else: 
            speaker_match_alt_result = re.match(r'^(SPEAKER_\d+)\s*:\s*(.*)', full_raw_text_from_sub, re.IGNORECASE)
            if speaker_match_alt_result: speaker_in_text_val = speaker_match_alt_result.group(1).upper(); text_content_for_cleaning_val = speaker_match_alt_result.group(2)
        cleaned_lines_list = [segment_utils.clean_srt_text_line(line_item) for line_item in text_content_for_cleaning_val.splitlines()]
        full_cleaned_text_val = " ".join(filter(None, cleaned_lines_list)).strip()
        full_cleaned_text_val = re.sub(r'\s+', ' ', full_cleaned_text_val).strip()
        current_start_time_sec = sub_obj.start.total_seconds(); current_end_time_sec = sub_obj.end.total_seconds()
        if current_end_time_sec <= current_start_time_sec: current_end_time_sec = current_start_time_sec + segment_utils.MIN_SEGMENT_DURATION_FOR_POSTPROCESS 
        segments_list.append({'text': full_cleaned_text_val, 'start': current_start_time_sec, 'end': current_end_time_sec, 'speaker': speaker_in_text_val})
    return segments_list, True 