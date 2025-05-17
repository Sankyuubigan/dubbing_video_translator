import whisperx
import torch
import os
from types import SimpleNamespace
import traceback
import srt 
import datetime 
import re 
import pandas as pd # Добавлено для DataFrame диаризации

models_cache = SimpleNamespace(stt_model=None, diarization_model=None, align_model_cache=None) 
DEFAULT_STT_MODEL_NAME = "medium.en"

def load_stt_diarization_models(stt_model_name=DEFAULT_STT_MODEL_NAME, device="cpu", hf_token=None, load_stt=True, load_diarization=True):
    global models_cache
    stt_model_to_return = models_cache.stt_model
    diarization_model_to_return = models_cache.diarization_model

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
                models_cache.diarization_model = None; # Не поднимаем ошибку, чтобы можно было продолжить без диаризации, если она не критична
        diarization_model_to_return = models_cache.diarization_model
        
    return stt_model_to_return, diarization_model_to_return

def load_align_model_cached(language_code, device):
    global models_cache
    if models_cache.align_model_cache is None:
        models_cache.align_model_cache = {}
    
    if language_code not in models_cache.align_model_cache or \
       models_cache.align_model_cache[language_code].get('device') != device:
        print(f"Loading align model for language '{language_code}' on device '{device}'...")
        model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        models_cache.align_model_cache[language_code] = {'model': model, 'metadata': metadata, 'device': device}
        print("Align model loaded.")
    return models_cache.align_model_cache[language_code]['model'], models_cache.align_model_cache[language_code]['metadata']


def segments_to_srt(segments):
    srt_subs = []
    for i, segment_data in enumerate(segments):
        start_time = segment_data.get('start'); end_time = segment_data.get('end'); text = segment_data.get('text', '')
        if start_time is None or end_time is None or not text.strip(): continue
        start_td = datetime.timedelta(seconds=start_time); end_td = datetime.timedelta(seconds=end_time)
        speaker = segment_data.get('speaker'); srt_text = f"[{speaker}]: {text}" if speaker and speaker != "SPEAKER_00" else text 
        srt_subs.append(srt.Subtitle(index=i + 1, start=start_td, end=end_td, content=srt_text))
    return srt.compose(srt_subs)

def transcribe_and_diarize(audio_path, language="en", batch_size=16, output_srt_path=None, return_diarization_df=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Transcribe & Diarize using device: {device}")
    hf_auth_token = os.environ.get("HF_TOKEN")
    
    stt_model, diarization_model_instance = load_stt_diarization_models(
        stt_model_name=DEFAULT_STT_MODEL_NAME, device=device, hf_token=hf_auth_token, load_stt=True, load_diarization=True
    ) # Переименовал в diarization_model_instance для ясности
    
    if stt_model is None: raise RuntimeError("STT model is not loaded.")
    # Diarization model может быть None, если не загрузилась, обработаем это ниже
    
    audio = whisperx.load_audio(audio_path)
    print("Transcribing audio...")
    result = stt_model.transcribe(audio, language=language, batch_size=batch_size)
    print("Transcription complete.")
    
    segments_after_align = result.get("segments", []) # Начальное значение

    print("Aligning transcription...")
    try:
        lang_code_for_align = result.get("language", language)
        align_model, metadata = load_align_model_cached(language_code=lang_code_for_align, device=device)
        segments_to_align = result.get("segments", []); 
        if not isinstance(segments_to_align, list) or (segments_to_align and not isinstance(segments_to_align[0], dict)): segments_to_align = []
        
        # whisperx.align может вернуть пустой результат или ошибку, если segments_to_align пуст или некорректен
        if segments_to_align:
            aligned_result = whisperx.align(segments_to_align, align_model, metadata, audio, device, return_char_alignments=False)
            segments_after_align = aligned_result.get("segments", segments_to_align) # Используем выровненные, если есть, иначе исходные
            print("Alignment complete.")
        else:
            print("No segments to align. Using original transcription segments.")
            segments_after_align = result.get("segments", [])

        if device == "cuda": torch.cuda.empty_cache() 
    except Exception as e: 
        print(f"Warning: Failed to align transcription - {e}. Using unaligned segments."); 
        segments_after_align = result.get("segments", []) # Фолбэк на невыровненные
    
    final_segments = segments_after_align
    diarize_segments_df_result = None # Для возврата

    if diarization_model_instance: # Проверяем, что модель диаризации загружена
        print("Performing diarization and assigning speakers to aligned transcription...")
        try:
            # Передаем путь к файлу, а не загруженный audio объект, т.к. pyannote этого ожидает
            diarize_segments_df_result = diarization_model_instance(audio_path) 
            
            # Убедимся, что segments_after_align - это список словарей
            if not isinstance(segments_after_align, list) or \
               (segments_after_align and not isinstance(segments_after_align[0], dict)):
                print("Warning: Segments for speaker assignment are not in the expected list-of-dicts format. Skipping speaker assignment.")
                final_segments = segments_after_align # Оставляем как есть
                for seg in final_segments: 
                    if 'speaker' not in seg: seg['speaker'] = 'SPEAKER_00'
            elif not segments_after_align: # Если список пуст
                 print("Warning: No segments available for speaker assignment.")
                 final_segments = []
            else:
                input_for_assignment = {"segments": segments_after_align}
                final_segments_with_speakers_data = whisperx.assign_word_speakers(diarize_segments_df_result, input_for_assignment)
                final_segments = final_segments_with_speakers_data.get("segments", segments_after_align) # Фолбэк на сегменты до присвоения
                print("Diarization and speaker assignment complete.")

        except Exception as e:
            print(f"Error during diarization/speaker assignment: {e}. Falling back to transcription without speaker labels."); 
            final_segments = segments_after_align # Используем то, что было после выравнивания
            for seg in final_segments: 
                if 'speaker' not in seg: seg['speaker'] = 'SPEAKER_00' # Убедимся, что поле спикера есть
            diarize_segments_df_result = None # Сбрасываем, если была ошибка
    else:
        print("Diarization model not loaded. Skipping speaker assignment. Assigning default speaker.")
        for seg in final_segments: seg['speaker'] = 'SPEAKER_00'
            
    if output_srt_path and final_segments:
        try:
            srt_data = segments_to_srt(final_segments)
            with open(output_srt_path, 'w', encoding='utf-8') as f: f.write(srt_data)
            print(f"WhisperX SRT saved to: {output_srt_path}")
        except Exception as e_srt_save: print(f"Warning: Could not save WhisperX SRT to {output_srt_path}: {e_srt_save}")
    
    if return_diarization_df:
        return final_segments, diarize_segments_df_result
    return final_segments


def perform_diarization_only(audio_path, device="cpu", hf_token=None):
    print(f"Performing diarization ONLY on: {os.path.basename(audio_path)} using device: {device}")
    _stt_model, diarization_model = load_stt_diarization_models(
        device=device, hf_token=hf_token, load_stt=False, load_diarization=True 
    )
    if diarization_model is None:
        print("Diarization model is not loaded. Cannot perform diarization.")
        return pd.DataFrame() # Возвращаем пустой DataFrame
    try:
        diarization_result_df = diarization_model(audio_path) 
        if diarization_result_df is not None and not diarization_result_df.empty:
            print(f"Diarization found {len(diarization_result_df['speaker'].unique())} unique speakers.")
        else:
            print("Diarization did not return any speaker segments.")
            return pd.DataFrame() # Возвращаем пустой DataFrame
        return diarization_result_df 
    except Exception as e:
        print(f"Error during standalone diarization: {e}")
        print(f"Details: {traceback.format_exc()}")
        return pd.DataFrame() # Возвращаем пустой DataFrame

def assign_srt_segments_to_speakers(srt_segments, diarization_df):
    if diarization_df is None or diarization_df.empty:
        print("Warning: Diarization result is empty or None. Assigning default speaker to SRT segments.")
        for seg in srt_segments:
            seg['speaker'] = 'SPEAKER_00'
        return srt_segments

    print(f"Assigning speakers to {len(srt_segments)} SRT segments based on diarization...")
    
    output_segments = []
    for srt_seg in srt_segments:
        srt_start_time = srt_seg['start']
        srt_end_time = srt_seg['end']
        
        overlapping_speakers = {} 
        
        for _idx, dia_row in diarization_df.iterrows():
            dia_start = dia_row['start']
            dia_end = dia_row['end']
            speaker_label = dia_row['speaker']
            
            overlap_start = max(srt_start_time, dia_start)
            overlap_end = min(srt_end_time, dia_end)
            overlap_duration = overlap_end - overlap_start
            
            if overlap_duration > 0.01: # Небольшой порог для учета пересечения
                overlapping_speakers[speaker_label] = overlapping_speakers.get(speaker_label, 0) + overlap_duration
        
        assigned_speaker = 'SPEAKER_00' 
        if overlapping_speakers:
            assigned_speaker = max(overlapping_speakers, key=overlapping_speakers.get)

        new_seg = srt_seg.copy()
        new_seg['speaker'] = assigned_speaker
        output_segments.append(new_seg)
        
    print("Speaker assignment for SRT segments complete.")
    return output_segments

def _preprocess_vtt_content(content: str) -> str:
    if content.startswith("WEBVTT"):
        header_end_match = re.search(r"WEBVTT.*?\n\n", content, re.DOTALL | re.IGNORECASE) # Добавил IGNORECASE
        if header_end_match:
            content = content[header_end_match.end():]
        else:
            lines = content.splitlines()
            if lines and "WEBVTT" in lines[0].upper(): # Сравнение без учета регистра
                content = "\n".join(lines[1:])
    content = re.sub(r"(\d{2}:\d{2}:\d{2})[.](\d{3})", r"\1,\2", content)
    # Удаление пустых строк между блоками субтитров, если они есть после обработки WEBVTT
    content = re.sub(r'\n\s*\n', '\n\n', content).strip()
    return content

def parse_srt_file(srt_file_path):
    print(f"Parsing SRT/VTT file: {os.path.basename(srt_file_path)}")
    segments = []
    try:
        with open(srt_file_path, 'r', encoding='utf-8-sig') as f:
            srt_content = f.read()

        processed_content = _preprocess_vtt_content(srt_content)
        
        if not processed_content.strip():
            print(f"Warning: SRT/VTT file {srt_file_path} is empty or became empty after VTT header removal.")
            return []

        parsed_subs = list(srt.parse(processed_content))
        for sub_idx, sub in enumerate(parsed_subs):
            text_content = sub.content if sub.content is not None else ""
            # Удаляем тэги спикеров из VTT, если они есть, типа <v Speaker 1>
            text_content = re.sub(r'<v [^>]+>', '', text_content).strip()
            text_content = re.sub(r'</v>', '', text_content).strip()
            
            # Иногда в VTT многострочные субтитры разделены \n, заменяем их на пробел
            text_content = text_content.replace('\n', ' ').strip()

            if not text_content: # Пропускаем пустые субтитры после очистки
                # print(f"Skipping empty subtitle content at index {sub_idx} from {os.path.basename(srt_file_path)}")
                continue

            segment = {
                'text': text_content,
                'start': sub.start.total_seconds(),
                'end': sub.end.total_seconds(),
                'speaker': 'SPEAKER_00' 
            }
            segments.append(segment)
        print(f"Successfully parsed {len(segments)} segments from {os.path.basename(srt_file_path)}.")
        return segments
    except FileNotFoundError:
        print(f"Error: SRT/VTT file not found at {srt_file_path}")
        return []
    except srt.SRTParseError as e_srt:
        print(f"Error parsing SRT/VTT file {srt_file_path} with `srt` library: {e_srt}")
        print(f"Problematic content (first 500 chars after preprocessing):\n{processed_content[:500]}")
        print(f"Details: {traceback.format_exc()}")
        return []
    except Exception as e: 
        print(f"General error parsing SRT/VTT file {srt_file_path}: {e}")
        print(f"Details: {traceback.format_exc()}")
        return []