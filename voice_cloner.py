import sherpa_onnx
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import video_processor
import ffmpeg
import shutil

# Кэш
_tts_engine = None

def get_tts_engine(models_dir):
    global _tts_engine
    if _tts_engine: return _tts_engine
    
    # Используем обновленную папку (Denis)
    tts_dir = os.path.join(models_dir, "tts_ru_denis")
    model = os.path.join(tts_dir, "model.onnx")
    tokens = os.path.join(tts_dir, "tokens.txt")
    data_json = os.path.join(tts_dir, "model.onnx.json")
    
    if not os.path.exists(model): 
        # Fallback для совместимости, если вдруг папка старая
        tts_dir_old = os.path.join(models_dir, "tts_ru_milena")
        if os.path.exists(os.path.join(tts_dir_old, "model.onnx")):
            tts_dir = tts_dir_old
            model = os.path.join(tts_dir, "model.onnx")
            tokens = os.path.join(tts_dir, "tokens.txt")
        else:
            raise FileNotFoundError(f"TTS model not found in {tts_dir}")
    
    print(f"Loading Sherpa-ONNX TTS from {tts_dir}...")
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model,
                tokens=tokens,
                data_dir=tts_dir,
            ),
            num_threads=4,
            debug=False,
            provider="cuda" if os.path.exists("/dev/nvidia0") or os.path.exists("NUL") else "cpu"
        ),
        rule_fsts="",
        max_num_sentences=1
    )
    _tts_engine = sherpa_onnx.OfflineTts(config)
    return _tts_engine

def _apply_speed_adjustment(input_path, output_path, target_duration, current_duration):
    if target_duration <= 0.05: return 
    
    ratio = current_duration / target_duration
    
    if 0.9 <= ratio <= 1.1:
        shutil.copy(input_path, output_path)
        return

    if ratio < 1.0:
        pad = target_duration - current_duration
        try:
            (ffmpeg.input(input_path)
             .filter('apad', pad_dur=f"{pad}s")
             .output(output_path, acodec='pcm_s16le', ar=22050)
             .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        except: shutil.copy(input_path, output_path)
        return

    speed = min(ratio, 2.0)
    try:
        (ffmpeg.input(input_path)
         .filter('atempo', str(speed))
         .output(output_path, acodec='pcm_s16le', ar=22050)
         .overwrite_output().run(capture_stdout=True, capture_stderr=True))
    except: shutil.copy(input_path, output_path)

def synthesize_segments(segments, models_dir, temp_dir, progress_cb=None):
    tts = get_tts_engine(models_dir)
    out_dir = os.path.join(temp_dir, "tts_segments")
    os.makedirs(out_dir, exist_ok=True)
    
    sr = tts.sample_rate 
    full_audio_parts = []
    last_end = 0.0
    
    for i, seg in enumerate(tqdm(segments, desc="TTS", disable=not progress_cb)):
        text = seg.get('translated_text', '').strip()
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        target_dur = end - start
        
        # 1. Тишина перед сегментом
        gap = start - last_end
        if gap > 0.05:
            gap_path = os.path.join(out_dir, f"gap_{i}.wav")
            video_processor.generate_silence(gap_path, gap, sr)
            full_audio_parts.append(gap_path)
            
        if not text:
            sil_seg_path = os.path.join(out_dir, f"sil_seg_{i}.wav")
            video_processor.generate_silence(sil_seg_path, target_dur, sr)
            full_audio_parts.append(sil_seg_path)
            last_end = end
            continue

        # 2. Синтез
        raw_path = os.path.join(out_dir, f"raw_{i}.wav")
        final_path = os.path.join(out_dir, f"final_{i}.wav")
        
        try:
            audio = tts.generate(text, sid=0, speed=1.0)
            sf.write(raw_path, audio.samples, audio.sample_rate)
            
            # 3. Подгонка скорости
            raw_dur = len(audio.samples) / audio.sample_rate
            _apply_speed_adjustment(raw_path, final_path, target_dur, raw_dur)
            
            full_audio_parts.append(final_path)
        except Exception as e:
            print(f"TTS Error {i}: {e}")
            video_processor.generate_silence(final_path, target_dur, sr)
            full_audio_parts.append(final_path)
            
        last_end = end

    full_path = os.path.join(temp_dir, "full_dubbed.wav")
    video_processor.merge_audio_segments(full_audio_parts, full_path)
    
    return full_path