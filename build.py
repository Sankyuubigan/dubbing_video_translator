import PyInstaller.__main__
import os
import shutil
import datetime 

# --- Configuration ---
APP_BASE_NAME = "VideoDubbingTool" 
MAIN_SCRIPT = "main.py"
ICON_PATH = "app.ico" 
BUILD_ONEDIR = False # False для --onefile

# Формируем версию (дату) и имя приложения с версией
APP_VERSION_DATE = datetime.datetime.now().strftime("%d.%m.%y")
APP_NAME_WITH_VERSION = f"{APP_BASE_NAME}_v{APP_VERSION_DATE}"

print(f"INFO: Имя собираемого файла: {APP_NAME_WITH_VERSION}.exe")
print(f"INFO: Версия (дата): {APP_VERSION_DATE}")


# Очистка перед сборкой
dist_dir_base = "dist" 
dist_output_path = os.path.join(dist_dir_base, APP_NAME_WITH_VERSION if BUILD_ONEDIR else f"{APP_NAME_WITH_VERSION}.exe")
build_dir = "build"
# .spec файл будет генерироваться PyInstaller с именем APP_NAME_WITH_VERSION
# Удаляем его, чтобы всегда генерировался свежий, если он существует
spec_file_generated_name = f"{APP_NAME_WITH_VERSION}.spec" 
if os.path.exists(spec_file_generated_name):
    print(f"Удаление предыдущего .spec файла: {spec_file_generated_name}")
    os.remove(spec_file_generated_name)


if os.path.exists(dist_output_path):
    print(f"Очистка предыдущего вывода: {dist_output_path}")
    if os.path.isdir(dist_output_path): 
        shutil.rmtree(dist_output_path)
    else: 
        try: os.remove(dist_output_path)
        except OSError as e: print(f"Не удалось удалить {dist_output_path}: {e}")


if os.path.exists(build_dir):
    print(f"Очистка папки build: {build_dir}")
    shutil.rmtree(build_dir)


pyinstaller_base_options = [
    MAIN_SCRIPT,
    f"--name={APP_NAME_WITH_VERSION}", 
    "--windowed",
    # f"--icon={ICON_PATH}", # Раскомментируйте, если есть иконка
    
    # Важные hidden imports (оставляем как есть)
    "--hidden-import=torch",
    "--hidden-import=torchaudio",
    "--hidden-import=transformers",
    "--hidden-import=TTS",
    "--hidden-import=TTS.api",
    "--hidden-import=TTS.tts",
    "--hidden-import=TTS.tts.configs",
    "--hidden-import=TTS.tts.configs.xtts_config",
    "--hidden-import=TTS.tts.models.xtts",
    "--hidden-import=TTS.utils.synthesizer",
    "--hidden-import=whisperx",
    "--hidden-import=whisperx.asr",
    "--hidden-import=whisperx.alignment",
    "--hidden-import=whisperx.diarize",
    "--hidden-import=soundfile",
    "--hidden-import=pyannote.audio",
    "--hidden-import=pyannote.audio.core.model",
    "--hidden-import=pyannote.audio.models.segmentation.शైतान", 
    "--hidden-import=pyannote.audio.pipelines",
    "--hidden-import=pyannote.core",
    "--hidden-import=speechbrain",
    "--hidden-import=speechbrain.pretrained",
    "--hidden-import=speechbrain.inference",
    "--hidden-import=speechbrain.inference.speaker",
    "--hidden-import=huggingface_hub",
    "--hidden-import=langdetect",
    "--hidden-import=deepmultilingualpunctuation",
    "--hidden-import=srt",
    "--hidden-import=ffmpeg", 
    "--hidden-import=yt_dlp",
    "--hidden-import=pyperclip",
    "--hidden-import=pandas",
    "--hidden-import=sklearn",
    "--hidden-import=sklearn.cluster",
    "--hidden-import=scipy",
    "--hidden-import=scipy.signal",
    "--hidden-import=scipy.ndimage",
    "--hidden-import=ctranslate2", 
    "--hidden-import=aeneas",
    "--hidden-import=aeneas.tools",
    "--hidden-import=aeneas.tools.execute_task",
    "--hidden-import=aeneas.executetask",
    "--hidden-import=aeneas.cmPlattsыргызcha", 
    "--hidden-import=pkg_resources.py2_warn",
    "--hidden-import=requests", 
    "--hidden-import=tqdm", 
    "--hidden-import=charset_normalizer", # Зависимость requests
    "--hidden-import=multiprocessing.popen_spawn_win32", # Иногда нужно для multiprocessing на Windows

    '--log-level=INFO',
    '--noupx',
]

if BUILD_ONEDIR:
    pyinstaller_options = pyinstaller_base_options + ["--onedir"]
else:
    pyinstaller_options = pyinstaller_base_options + ["--onefile"]


print(f"Собираем {APP_NAME_WITH_VERSION}...")
print(f"Команда: pyinstaller {' '.join(pyinstaller_options)}")

import sys
sys.setrecursionlimit(5000) # Оставляем на всякий случай

if __name__ == "__main__":
    try:
        PyInstaller.__main__.run(pyinstaller_options)
        print(f"Сборка {APP_NAME_WITH_VERSION} завершена. Результат в папке 'dist'.")
    except Exception as e:
        print(f"Ошибка во время сборки: {e}")
        import traceback
        traceback.print_exc()
    # Файл file_version_info.txt больше не создается и не удаляется