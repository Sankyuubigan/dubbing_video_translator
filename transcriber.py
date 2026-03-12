import sherpa_onnx
import os
import numpy as np
import wave
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

_recognizer = None
_speaker_extractor = None

def get_recognizer(models_dir):
    global _recognizer
    if _recognizer: return _recognizer
    
    stt_dir = os.path.join(models_dir, "stt_whisper_tiny")
    encoder = os.path.join(stt_dir, "encoder.onnx")
    decoder = os.path.join(stt_dir, "decoder.onnx")
    tokens = os.path.join(stt_dir, "tokens.txt")
    
    # Проверка наличия файлов
    if not os.path.exists(encoder): raise FileNotFoundError(f"Missing: {encoder}")
    if not os.path.exists(decoder): raise FileNotFoundError(f"Missing: {decoder}")
    if not os.path.exists(tokens): raise FileNotFoundError(f"Missing: {tokens}")

    print("Loading Sherpa-ONNX Whisper...")
    try:
        _recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            num_threads=4
        )
    except Exception as e:
        print(f"CRITICAL ERROR loading Whisper: {e}")
        print(f"Check if {tokens} is a valid text file.")
        raise e
        
    return _recognizer

def get_speaker_extractor(models_dir):
    global _speaker_extractor
    if _speaker_extractor: return _speaker_extractor
    
    spk_path = os.path.join(models_dir, "speaker_encoder", "model.onnx")
    if not os.path.exists(spk_path): raise FileNotFoundError(f"Missing: {spk_path}")
    
    print("Loading Sherpa-ONNX Speaker Extractor...")
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=spk_path,
        num_threads=4,
        debug=False,
    )
    _speaker_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    return _speaker_extractor

def read_wave(wave_filename):
    with wave.open(wave_filename, "rb") as f:
        params = f.getparams()
        samples = f.readframes(params.nframes)
        samples = np.frombuffer(samples, dtype=np.int16)
        samples = samples.astype(np.float32) / 32768.0
        return samples, params.framerate

def transcribe_with_sherpa(audio_path, models_dir):
    recognizer = get_recognizer(models_dir)
    samples, sample_rate = read_wave(audio_path)
    if sample_rate != 16000: raise ValueError("Audio must be 16kHz")

    print(f"Transcribing {os.path.basename(audio_path)}...")
    s = recognizer.create_stream()
    s.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(s)
    
    # Получаем токены и таймкоды
    # В Sherpa ONNX Whisper result.timestamps - это таймкоды токенов
    text = s.result.text
    tokens = s.result.tokens
    timestamps = s.result.timestamps 
    
    segments = []
    if not timestamps:
        segments.append({'start': 0.0, 'end': len(samples)/16000, 'text': text})
    else:
        # Разбиваем поток токенов на предложения по знакам препинания
        current_seg = {'start': timestamps[0], 'text': '', 'end': 0}
        
        for i, t in enumerate(tokens):
            # Whisper токены часто начинаются с пробела (Ġ в BPE, но sherpa декодирует)
            current_seg['text'] += t
            current_seg['end'] = timestamps[i] + 0.1 # Условная длительность
            
            # Эвристика конца предложения
            is_eos = t.strip() in ['.', '?', '!', '...']
            # Или длинная пауза между токенами
            is_pause = (i < len(timestamps)-1) and (timestamps[i+1] - timestamps[i] > 0.5)
            
            if is_eos or is_pause:
                current_seg['text'] = current_seg['text'].strip()
                if current_seg['text']:
                    segments.append(current_seg)
                if i < len(timestamps)-1:
                    current_seg = {'start': timestamps[i+1], 'text': '', 'end': 0}
        
        if current_seg['text'].strip():
            segments.append(current_seg)
            
    return segments

def diarize_segments(audio_path, segments, models_dir, num_speakers=None):
    if not segments: return segments, pd.DataFrame()
    extractor = get_speaker_extractor(models_dir)
    samples, sr = read_wave(audio_path)

    print("Diarizing...")
    embeddings = []
    valid_indices = []

    for i, seg in enumerate(segments):
        start = int(seg['start'] * sr)
        end = int(seg['end'] * sr)
        if end - start < 1600: continue # <0.1s
        if end > len(samples): end = len(samples)
        
        sub = samples[start:end]
        s = extractor.create_stream()
        s.accept_waveform(sr, sub)
        embeddings.append(extractor.compute(s))
        valid_indices.append(i)

    if not embeddings:
        for s in segments: s['speaker'] = 'SPEAKER_00'
        return segments, pd.DataFrame()

    X = np.array(embeddings)
    try:
        n = num_speakers if num_speakers else None
        thresh = 0.6 if not n else None
        clusterer = AgglomerativeClustering(n_clusters=n, distance_threshold=thresh, metric='cosine', linkage='average')
        labels = clusterer.fit_predict(X)
    except:
        labels = [0]*len(X)

    for idx, label in zip(valid_indices, labels):
        segments[idx]['speaker'] = f"SPEAKER_{label:02d}"

    for s in segments:
        if 'speaker' not in s: s['speaker'] = 'SPEAKER_00'

    return segments, pd.DataFrame()