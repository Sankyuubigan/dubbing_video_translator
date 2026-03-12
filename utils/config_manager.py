import os
import json
import platform 
import requests 
from tqdm import tqdm 
import sys 

# Константы
CONFIG_FILE_NAME = "app_settings.json" 
FFMPEG_DIR = "ffmpeg" 
FFMPEG_EXE_NAME = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
FFPROBE_EXE_NAME = "ffprobe.exe" if platform.system() == "Windows" else "ffprobe"

# ССЫЛКИ НА МОДЕЛИ (HUGGINGFACE)
MODELS_URLS = {
    # 1. STT: Whisper Tiny (Sherpa ONNX)
    "stt_whisper_tiny": {
        "folder": "stt_whisper_tiny",
        "files": {
            "encoder.onnx": "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/tiny.en-encoder.int8.onnx",
            "decoder.onnx": "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/tiny.en-decoder.int8.onnx",
            "tokens.txt": "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/tiny.en-tokens.txt"
        }
    },
    # 2. Speaker Embedding (Sherpa ONNX)
    "speaker_encoder": {
        "folder": "speaker_encoder",
        "files": {
            "model.onnx": "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
        }
    },
    # 3. TTS: Russian VITS Piper (Denis - Male, Stable)
    # Milena (female) недоступна по прямой ссылке (401), используем Denis.
    "tts_ru": {
        "folder": "tts_ru_denis",
        "files": {
            # Сохраняем как model.onnx для унификации
            "model.onnx": "https://huggingface.co/csukuangfj/vits-piper-ru_RU-denis-medium/resolve/main/ru_RU-denis-medium.onnx",
            "tokens.txt": "https://huggingface.co/csukuangfj/vits-piper-ru_RU-denis-medium/resolve/main/tokens.txt",
            "model.onnx.json": "https://huggingface.co/csukuangfj/vits-piper-ru_RU-denis-medium/resolve/main/ru_RU-denis-medium.onnx.json"
        }
    },
    # 4. MT: NLLB-200 (CTranslate2)
    "mt_nllb": {
        "folder": "mt_nllb_ct2",
        "files": {
            "model.bin": "https://huggingface.co/softcatala/nllb-200-distilled-600M-ct2-int8/resolve/main/model.bin",
            "config.json": "https://huggingface.co/softcatala/nllb-200-distilled-600M-ct2-int8/resolve/main/config.json",
            "shared_vocabulary.txt": "https://huggingface.co/softcatala/nllb-200-distilled-600M-ct2-int8/resolve/main/shared_vocabulary.txt",
            "sentencepiece.bpe.model": "https://huggingface.co/softcatala/nllb-200-distilled-600M-ct2-int8/resolve/main/sentencepiece.bpe.model"
        }
    }
}

def get_work_dir_from_config():
    if getattr(sys, 'frozen', False): base = os.path.dirname(sys.executable)
    else: base = os.getcwd()
    cfg_path = os.path.join(base, CONFIG_FILE_NAME)
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as f: return json.load(f).get('work_dir', base)
        except: pass
    return base

def _download_file(url, dest):
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        # Если файл меньше 10КБ - это скорее всего ошибка (текст 404/401), перекачиваем
        if (dest.endswith('.bin') or dest.endswith('.onnx')) and size < 10240:
            print(f"File {os.path.basename(dest)} is too small ({size} bytes). Re-downloading...")
            os.remove(dest)
        else:
            return True

    print(f"Downloading {os.path.basename(dest)}...")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        
        total = int(resp.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
            for chunk in resp.iter_content(1024*1024):
                if chunk: 
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(dest): os.remove(dest)
        return False

def check_and_download_models(work_dir):
    models_root = os.path.join(work_dir, "models_onnx")
    os.makedirs(models_root, exist_ok=True)
    
    for key, info in MODELS_URLS.items():
        folder_path = os.path.join(models_root, info["folder"])
        os.makedirs(folder_path, exist_ok=True)
        
        for filename, url in info["files"].items():
            dest_path = os.path.join(folder_path, filename)
            if not _download_file(url, dest_path):
                raise RuntimeError(f"Failed to download {filename} for {key}")

    return models_root

def initialize_paths_from_work_dir(work_dir):
    return True, True, False