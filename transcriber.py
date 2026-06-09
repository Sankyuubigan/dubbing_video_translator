import os
import torch
import whisperx
import gc
import nltk
import tempfile
import yaml
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils import getter as pyannote_getter

def _ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("NLTK 'punkt_tab' not found locally, trying download...")
        try:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"CRITICAL WARNING: NLTK download failed ({e}). Install manually if alignment fails.")

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
    
    pyannote_dir = os.path.join(models_dir, "pyannote")
    pyannote_yaml = os.path.join(pyannote_dir, "config.yaml")
    
    if not os.path.exists(pyannote_yaml):
        raise FileNotFoundError(f"Missing Pyannote config at {pyannote_yaml}. Please download models manually.")
    
    with open(pyannote_yaml) as f:
        cfg = yaml.safe_load(f)
    
    seg = cfg['pipeline']['params']['segmentation']
    emb = cfg['pipeline']['params']['embedding']
    if not os.path.isabs(seg):
        cfg['pipeline']['params']['segmentation'] = os.path.join(pyannote_dir, seg)
    if not os.path.isabs(emb):
        cfg['pipeline']['params']['embedding'] = os.path.join(pyannote_dir, emb)
    
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(cfg, tmp)
    tmp_path = tmp.name
    tmp.close()
    
    real_get_plda = pyannote_getter.get_plda
    pyannote_getter.get_plda = lambda *a, **kw: None
    
    diarize_pipeline = Pipeline.from_pretrained(tmp_path)
    pyannote_getter.get_plda = real_get_plda
    os.unlink(tmp_path)
    
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