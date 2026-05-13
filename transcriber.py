import os
import torch
import whisperx
import gc
import nltk
from pyannote.audio import Pipeline

def _ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK 'punkt_tab' for WhisperX alignment...")
        try:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"CRITICAL WARNING: NLTK download failed ({e}).")

def transcribe_and_diarize(audio_path, models_dir):
    _ensure_nltk_data()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    # Защита от зависания на CPU
    if device == "cpu":
        print("\n=====================================================================")
        print("ВНИМАНИЕ: Видеокарта (Nvidia GPU) не обнаружена! Работаем на процессоре (CPU).")
        print("Чтобы программа не зависла на часы, используем легкую модель 'base'.")
        print("=====================================================================\n")
        model_name = "base"
    else:
        model_name = "large-v3"
    
    print(f"1. Transcribing with WhisperX ({model_name}) on {device}...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=8)
    
    language = result["language"]
    del model; gc.collect(); torch.cuda.empty_cache()
    
    print(f"2. Aligning timestamps for language: {language}...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    del model_a; gc.collect(); torch.cuda.empty_cache()
    
    print("3. Diarizing speakers (Local Pyannote)...")
    pyannote_yaml = os.path.join(models_dir, "pyannote", "config.yaml")
    
    if not os.path.exists(pyannote_yaml):
        raise FileNotFoundError(f"Missing Pyannote config at {pyannote_yaml}. Please download models manually.")
        
    diarize_pipeline = Pipeline.from_pretrained(pyannote_yaml)
    diarize_pipeline.to(torch.device(device))
    
    diarize_segments = diarize_pipeline(audio_path)
    
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    del diarize_pipeline; gc.collect(); torch.cuda.empty_cache()
    
    final_segments =[]
    for seg in result["segments"]:
        final_segments.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker", "SPEAKER_00")
        })
        
    return final_segments