import os
import re
from llama_cpp import Llama
from utils.segment_utils import segments_to_srt_string

_llm_instance = None

def get_llm(models_dir):
    global _llm_instance
    if _llm_instance: return _llm_instance
    
    # Новое имя файла Qwen 3.5 4B
    gguf_path = os.path.join(models_dir, "llm_translator", "Qwen3.5-4B-Q4_K_M.gguf")
    
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"LLM model not found at {gguf_path}")
        
    print("Loading Llama.cpp (Qwen 3.5 4B 4-bit)...")
    _llm_instance = Llama(
        model_path=gguf_path,
        n_ctx=8192,
        n_gpu_layers=-1, # -1 означает выгрузку всех слоев на видеокарту для макс. скорости
        verbose=False
    )
    return _llm_instance

def translate_segments(segments, models_dir):
    """
    Переводит сегменты с учетом контекста, конвертируя их в SRT и пропуская через LLM.
    """
    if not segments: return segments, True
    
    llm = get_llm(models_dir)
    
    english_srt = segments_to_srt_string(segments)
    
    print("Translating with context via Llama.cpp...")
    
    system_prompt = (
        "Ты профессиональный переводчик фильмов. Твоя задача перевести английские SRT субтитры на русский язык. "
        "СОХРАНЯЙ тайм-коды, пустые строки, нумерацию и теги спикеров (например[SPEAKER_00]). "
        "Переводи литературно, учитывая контекст диалога. Не пиши ничего кроме валидного SRT кода."
    )
    
    # Формат промпта ChatML, который использует Qwen
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{english_srt}<|im_end|>\n<|im_start|>assistant\n"
    
    # temperature=0.3 делает перевод более стабильным и точным, без лишней "фантазии" нейросети
    response = llm(prompt, max_tokens=8192, stop=["<|im_end|>"], temperature=0.3)
    russian_srt = response['choices'][0]['text'].strip()
    
    return parse_translated_srt_to_segments(russian_srt, segments), True

def parse_translated_srt_to_segments(translated_srt, original_segments):
    """
    Читает переведенный SRT и обновляет поле 'translated_text' в оригинальных сегментах.
    """
    blocks = translated_srt.strip().split('\n\n')
    
    for i, block in enumerate(blocks):
        lines = block.split('\n')
        # Ищем текст (обычно он идет после таймкодов, т.е. с 3-й строки)
        if len(lines) >= 3 and i < len(original_segments):
            text_lines = lines[2:]
            full_text = " ".join(text_lines)
            
            # Очищаем от тега спикера, если LLM оставила его внутри текста
            full_text = re.sub(r'^\[SPEAKER_\d+\]:\s*', '', full_text)
            
            original_segments[i]['translated_text'] = full_text.strip()
            
    # Защита от потери строк: если LLM пропустила сегмент, оставляем оригинал
    for seg in original_segments:
        if 'translated_text' not in seg:
            seg['translated_text'] = seg.get('text', '')
            
    return original_segments