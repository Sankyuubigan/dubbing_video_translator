# Содержимое файла translator.py остается БЕЗ ИЗМЕНЕНИЙ
# по сравнению с предыдущим ответом.
# Включает: load_translator_model, translate_segments
from transformers import pipeline
import torch
from types import SimpleNamespace

# Кэш для модели переводчика
translator_cache = SimpleNamespace(pipeline=None)
# Используем модель Helsinki-NLP для перевода EN->RU
DEFAULT_TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-en-ru"

def load_translator_model(model_name=DEFAULT_TRANSLATOR_MODEL, device="cpu"):
    """Загружает модель перевода Hugging Face."""
    global translator_cache
    if translator_cache.pipeline is None:
        print(f"Loading translation model: {model_name} on device: {device}")
        device_id = 0 if device == "cuda" else -1
        try:
            translator_cache.pipeline = pipeline("translation", model=model_name, device=device_id)
            print("Translation model loaded.")
        except Exception as e:
            print(f"Error loading translation model {model_name}: {e}")
            translator_cache.pipeline = None
            raise
    else:
        print("Using cached translation model.")
    return translator_cache.pipeline

def translate_segments(segments, target_language="ru"):
    """
    Переводит текст в сегментах.
    """
    if not segments: return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = load_translator_model(device=device)
    if translator is None: raise RuntimeError("Translation model could not be loaded.")

    print(f"Translating {len(segments)} segments to {target_language}...")
    texts_to_translate = [segment['text'] for segment in segments if segment.get('text')]
    if not texts_to_translate:
        print("No text found in segments to translate.")
        for segment in segments: segment['translated_text'] = ""
        return segments

    try:
        translations = translator(texts_to_translate, max_length=512)
    except Exception as e:
        print(f"Error during batch translation: {e}. Falling back to one-by-one.")
        translations = []
        for i, text in enumerate(texts_to_translate):
            print(f"Translating segment {i+1}/{len(texts_to_translate)} individually...")
            try:
                translation_result = translator(text, max_length=512)
                translations.append(translation_result[0])
            except Exception as e_single:
                 print(f"Error translating segment: {text[:50]}... Error: {e_single}")
                 translations.append({'translation_text': f"[Translation Error: {text[:30]}...]"})

    translation_map = {orig_text: trans['translation_text'] for orig_text, trans in zip(texts_to_translate, translations)}
    translated_segments_list = []
    for segment in segments:
        new_segment = segment.copy()
        original_text = segment.get('text')
        if original_text:
            translated_text = translation_map.get(original_text, f"[Translation Missing: {original_text[:30]}...]")
            new_segment['translated_text'] = translated_text
        else:
            new_segment['translated_text'] = ""
        translated_segments_list.append(new_segment)

    print("Translation complete.")
    return translated_segments_list