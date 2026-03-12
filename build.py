import PyInstaller.__main__
import os
import shutil
import datetime 

# --- Configuration ---
APP_BASE_NAME = "VideoDubbingTool" 
MAIN_SCRIPT = "main.py"
ICON_PATH = "icon.ico" 
BUILD_ONEDIR = False # False для --onefile

APP_VERSION_DATE = datetime.datetime.now().strftime("%d.%m.%y")
APP_NAME_WITH_VERSION = f"{APP_BASE_NAME}_v{APP_VERSION_DATE}"

print(f"INFO: Имя собираемого файла: {APP_NAME_WITH_VERSION}.exe")
print(f"INFO: Версия (дата): {APP_VERSION_DATE}")

dist_dir_base = "dist" 
dist_output_path = os.path.join(dist_dir_base, APP_NAME_WITH_VERSION if BUILD_ONEDIR else f"{APP_NAME_WITH_VERSION}.exe")
build_dir = "build"
spec_file_generated_name = f"{APP_NAME_WITH_VERSION}.spec" 

if os.path.exists(spec_file_generated_name): os.remove(spec_file_generated_name)
if os.path.exists(dist_output_path):
    if os.path.isdir(dist_output_path): shutil.rmtree(dist_output_path)
    else: 
        try: os.remove(dist_output_path)
        except OSError: pass
if os.path.exists(build_dir): shutil.rmtree(build_dir)

pyinstaller_base_options = [
    MAIN_SCRIPT,
    f"--name={APP_NAME_WITH_VERSION}", 
    "--windowed",
    f"--icon={ICON_PATH}",
    
    # --- Hidden Imports ---
    "--hidden-import=torch",
    "--hidden-import=torchaudio",
    "--hidden-import=transformers",
    "--hidden-import=TTS",
    "--hidden-import=TTS.api",
    "--hidden-import=whisperx",
    "--hidden-import=soundfile",
    "--hidden-import=huggingface_hub",
    "--hidden-import=langdetect",
    "--hidden-import=deepmultilingualpunctuation",
    "--hidden-import=srt",
    "--hidden-import=ffmpeg", 
    "--hidden-import=yt_dlp",
    "--hidden-import=pyperclip",
    "--hidden-import=pandas",
    "--hidden-import=numpy",
    
    # Новые зависимости для оффлайн диаризации
    "--hidden-import=speechbrain",
    "--hidden-import=speechbrain.inference",
    "--hidden-import=speechbrain.inference.speaker",
    "--hidden-import=speechbrain.lobes.models.ECAPA_TDNN",
    "--hidden-import=sklearn",
    "--hidden-import=sklearn.cluster",
    "--hidden-import=sklearn.neighbors.typedefs",
    "--hidden-import=sklearn.neighbors.quad_tree",
    "--hidden-import=sklearn.tree._utils",
    "--hidden-import=scipy.special.cython_special",
    
    # Удалены pyannote зависимости
    
    '--log-level=INFO',
    '--noupx',
]

if BUILD_ONEDIR: pyinstaller_options = pyinstaller_base_options + ["--onedir"]
else: pyinstaller_options = pyinstaller_base_options + ["--onefile"]

print(f"Собираем {APP_NAME_WITH_VERSION}...")
import sys
sys.setrecursionlimit(5000) 

if __name__ == "__main__":
    try:
        PyInstaller.__main__.run(pyinstaller_options)
        print(f"Сборка {APP_NAME_WITH_VERSION} завершена.")
    except Exception as e:
        print(f"Ошибка во время сборки: {e}")