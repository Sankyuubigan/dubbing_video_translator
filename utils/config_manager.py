import os
import json
import platform 
import requests 
from tqdm import tqdm 
import sys 

CONFIG_FILE_NAME = "app_settings.json" 

# ССЫЛКИ НА МОДЕЛИ
MODELS_URLS = {
    # TTS: Russian VITS Piper (Denis)
    "tts_ru": {
        "folder": "tts_ru_denis",
        "files": {
            "model.onnx": "https://huggingface.co/csukuangfj/vits-piper-ru_RU-denis-medium/resolve/main/ru_RU-denis-medium.onnx",
            "tokens.txt": "https://huggingface.co/csukuangfj/vits-piper-ru_RU-denis-medium/resolve/main/tokens.txt",
            "model.onnx.json": "https://huggingface.co/csukuangfj/vits-piper-ru_RU-denis-medium/resolve/main/ru_RU-denis-medium.onnx.json"
        }
    },
    # LLM для перевода: Qwen 3.5 4B (Unsloth 4-bit GGUF)
    "llm_translator": {
        "folder": "llm_translator",
        "files": {
            "Qwen3.5-4B-Q4_K_M.gguf": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
        }
    }
}

def get_config_path():
    if getattr(sys, 'frozen', False): return os.path.join(os.path.dirname(sys.executable), CONFIG_FILE_NAME)
    else: return os.path.join(os.getcwd(), CONFIG_FILE_NAME)

def get_setting(key, default=None):
    cfg_path = get_config_path()
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as f: return json.load(f).get(key, default)
        except: pass
    return default

def save_setting(key, value):
    cfg_path = get_config_path()
    try:
        with open(cfg_path, 'r') as f: cfg = json.load(f)
    except: cfg = {}
    cfg[key] = value
    with open(cfg_path, 'w') as f: json.dump(cfg, f, indent=4)

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
        if dest.endswith('.gguf') and size < 1024 * 1024 * 100:
            print(f"File {os.path.basename(dest)} seems corrupted. Re-downloading...")
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
    
    pyannote_dir = os.path.join(models_root, "pyannote")
    if not os.path.exists(os.path.join(pyannote_dir, "config.yaml")):
        print("WARNING: Папка pyannote пуста! Убедитесь, что вы вручную скачали config.yaml, segmentation.bin и wespeaker.bin в", pyannote_dir)
    
    for key, info in MODELS_URLS.items():
        folder_path = os.path.join(models_root, info["folder"])
        os.makedirs(folder_path, exist_ok=True)
        
        for filename, url in info["files"].items():
            dest_path = os.path.join(folder_path, filename)
            if not _download_file(url, dest_path):
                raise RuntimeError(f"Failed to download {filename} for {key}")

    return models_root