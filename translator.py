# Содержимое файла translator.py остается БЕЗ ИЗМЕНЕНИЙ
# по сравнению с предыдущим ответом.
from transformers import pipeline
import torch
from types import SimpleNamespace
import time

translator_cache = SimpleNamespace(pipeline=None)
DEFAULT_TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-en-ru"

def load_translator_model(model_name=DEFAULT_TRANSLATOR_MODEL, device="cpu"):
    global translator_cache
    if translator_cache.pipeline is None:
        print(f"Loading translation model: {model_name} on device: {device}")
        device_id = 0 if device == "cuda" else -1
        try:
            translator_cache.pipeline = pipeline("translation", model=model_name, device=device_id)
            print("Translation model loaded.")
        except Exception as e:
            print(f"Error loading translation model {model_name}: {e}")
            translator_cache.pipeline = None; raise
    else:
        print("Using cached translation model.")
    return translator_cache.pipeline

def translate_segments(segments, target_language="ru"):
    if not segments: return []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Translator using device: {device}")
    translator_pipeline = load_translator_model(device=device)
    if translator_pipeline is None: raise RuntimeError("Translation model could not be loaded.")

    print(f"Translating {len(segments)} segments to {target_language}...")
    texts_to_translate = [segment['text'] for segment in segments if segment.get('text','').strip()]
    if not texts_to_translate:
        print("No text found in segments to translate.")
        for segment in segments: segment['translated_text'] = segment.get('text', "")
        return segments

    start_time_trans = time.time()
    try:
        # Увеличим max_length, если тексты могут быть длинными, но это может потребовать больше памяти
        translations = translator_pipeline(texts_to_translate, max_length=512, truncation=True)
    except Exception as e:
        print(f"Error during batch translation: {e}. Falling back to one-by-one.")
        translations = []
        for i, text_segment in enumerate(texts_to_translate):
            print(f"Translating segment {i+1}/{len(texts_to_translate)} individually...")
            try:
                translation_result = translator_pipeline(text_segment, max_length=512, truncation=True)
                translations.append(translation_result[0])
            except Exception as e_single:
                 print(f"Error translating segment: {text_segment[:50]}... Error: {e_single}")
                 translations.append({'translation_text': f"[Translation Error: {text_segment[:30]}...]"})
    
    end_time_trans = time.time()
    print(f"Translation of {len(texts_to_translate)} texts took {end_time_trans - start_time_trans:.2f} seconds.")

    # Сопоставляем переводы с оригинальными текстами для правильного присвоения
    # Это важно, т.к. texts_to_translate мог быть отфильтрован от пустых
    translated_text_iter = iter(translations)
    translated_segments_list = []
    for segment in segments:
        new_segment = segment.copy()
        original_text = segment.get('text', '').strip()
        if original_text:
            try:
                # Берем следующий перевод из итератора
                new_segment['translated_text'] = next(translated_text_iter)['translation_text']
            except StopIteration:
                 # Этого не должно случиться, если texts_to_translate и translations имеют одинаковую длину
                 print(f"Warning: Ran out of translations for segment with original text: {original_text[:30]}...")
                 new_segment['translated_text'] = "[Translation Missing]"
            except KeyError:
                 print(f"Warning: 'translation_text' key missing for segment with original text: {original_text[:30]}...")
                 new_segment['translated_text'] = "[Translation Error]"
        else:
            new_segment['translated_text'] = ""
        translated_segments_list.append(new_segment)
    return translated_segments_list