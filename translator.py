import ctranslate2
import sentencepiece as spm
import os
from types import SimpleNamespace

# Кэш
translator_cache = SimpleNamespace(translator=None, tokenizer=None)

def load_translator(models_dir, device="cpu"):
    global translator_cache
    if translator_cache.translator: return translator_cache.translator, translator_cache.tokenizer

    mt_dir = os.path.join(models_dir, "mt_nllb_ct2")
    
    # NLLB в CT2 обычно ищет model.bin
    # Иногда при распаковке создается подпапка. Найдем её.
    model_path = mt_dir
    for root, dirs, files in os.walk(mt_dir):
        if "model.bin" in files:
            model_path = root
            break
            
    print(f"Loading CTranslate2 NLLB from {model_path} on {device}...")
    
    # Определяем device для CT2
    ct_device = "cuda" if device == "cuda" else "cpu"
    
    translator = ctranslate2.Translator(model_path, device=ct_device)
    
    # Загружаем токенизатор (SentencePiece)
    # В папке модели должен быть файл sentencepiece.model или shared_vocabulary.txt
    # NLLB обычно использует sentencepiece.
    sp_model = os.path.join(model_path, "sentencepiece.bpe.model") # Имя может отличаться
    if not os.path.exists(sp_model):
        # Попробуем поискать любой .model файл
        for f in os.listdir(model_path):
            if f.endswith(".model"):
                sp_model = os.path.join(model_path, f)
                break
                
    if not os.path.exists(sp_model):
        # Если нет SP модели, возможно это HF токенизатор, но CT2 нужен SP для NLLB обычно.
        # Для NLLB-200-distilled-600M часто идет `flores200_sacrebleu_tokenizer_spm.model`
        raise FileNotFoundError(f"SentencePiece model not found in {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model)
    
    translator_cache.translator = translator
    translator_cache.tokenizer = sp
    return translator, sp

def translate_segments(segments, models_dir, target_lang="rus_Cyrl"):
    """
    Переводит сегменты используя CTranslate2 + NLLB.
    """
    if not segments: return segments, True
    
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    translator, sp = load_translator(models_dir, device)
    
    texts = [s.get('text', '').strip() for s in segments]
    non_empty_indices = [i for i, t in enumerate(texts) if t]
    non_empty_texts = [texts[i] for i in non_empty_indices]
    
    if not non_empty_texts: return segments, True

    print(f"Translating {len(non_empty_texts)} segments to {target_lang}...")
    
    # 1. Tokenize
    # NLLB требует исходный язык, например eng_Latn
    source_tokens = [sp.encode_as_pieces(t) for t in non_empty_texts]
    
    # Добавляем тег языка источника в начало (для NLLB это важно, если модель не делает это сама)
    # CT2 NLLB обычно ожидает target_prefix. Source prefix подается как первый токен?
    # Обычно: </s> eng_Latn ... 
    # Но проще использовать API translate_batch с target_prefix
    
    # 2. Translate
    # target_prefix=[['rus_Cyrl']] * len
    results = translator.translate_batch(
        source_tokens,
        target_prefix=[[target_lang]] * len(source_tokens),
        beam_size=4
    )
    
    # 3. Detokenize
    translated_texts = []
    for res in results:
        # res.hypotheses[0] - лучший вариант
        # Убираем тег языка если он есть в начале
        tokens = res.hypotheses[0]
        if tokens and tokens[0] == target_lang: tokens = tokens[1:]
        decoded = sp.decode(tokens)
        translated_texts.append(decoded)
        
    # 4. Assign back
    for i, idx in enumerate(non_empty_indices):
        segments[idx]['translated_text'] = translated_texts[i]
        
    for i, s in enumerate(segments):
        if 'translated_text' not in s: s['translated_text'] = s.get('text', '')

    return segments, True