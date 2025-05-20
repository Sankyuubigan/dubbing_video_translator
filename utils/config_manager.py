import os
import json
import shutil
import platform 
import tempfile
import subprocess
import requests 
from tqdm import tqdm 
import zipfile 
import tarfile 
import sys 

# --- Константы для рабочей директории и инструментов ---
CONFIG_FILE_NAME = "app_settings.json" 

FFMPEG_DIR = "ffmpeg" 
FFMPEG_EXE_NAME = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
FFPROBE_EXE_NAME = "ffprobe.exe" if platform.system() == "Windows" else "ffprobe"

# Константы для ESPEAK удалены, так как мы его больше не ищем и не скачиваем активно

CURRENT_FFMPEG_PATH = None
CURRENT_FFPROBE_PATH = None
# CURRENT_ESPEAK_PATH = None # Больше не нужен, так как мы не управляем им

FFMPEG_WINDOWS_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" 
# ESPEAK_NG_WINDOWS_URL = None # Удалено


def _download_file(url, target_path, description="Скачивание"):
    # ... (без изменений)
    try:
        print(f"Скачивание {description} с {url} в {target_path}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, stream=True, timeout=300, headers=headers, allow_redirects=True)
        response.raise_for_status() 
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024*1024): 
                size = f.write(data)
                bar.update(size)
        print(f"{description} успешно скачан.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Ошибка скачивания {description} ({url}): {e}")
    except Exception as e:
        print(f"Непредвиденная ошибка при скачивании {description}: {e}")
    if os.path.exists(target_path): 
        os.remove(target_path)
    return False

def _extract_zip(zip_path, extract_to_dir, strip_components=0):
    # ... (без изменений)
    try:
        print(f"Распаковка {zip_path} в {extract_to_dir} (strip_components={strip_components})...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            member_list = zip_ref.infolist()
            if not member_list:
                print(f"Ошибка: ZIP-архив {zip_path} пуст.")
                return False

            if strip_components == 0 and len(member_list) > 0:
                top_level_paths = set()
                for member in member_list:
                    path_parts = member.filename.replace('\\', '/').split('/')
                    if path_parts and path_parts[0]: 
                        top_level_paths.add(path_parts[0])
                
                if len(top_level_paths) == 1:
                    single_top_level_name = list(top_level_paths)[0]
                    all_inside = True
                    first_member_is_dir_and_matches = False
                    
                    for m_info_idx, m_info in enumerate(member_list):
                        if not m_info.filename.strip() == '': 
                            current_member_path_norm = m_info.filename.replace('\\', '/').strip('/')
                            if current_member_path_norm == single_top_level_name and m_info.is_dir():
                                first_member_is_dir_and_matches = True
                            elif current_member_path_norm.startswith(single_top_level_name + '/') :
                                first_member_is_dir_and_matches = True 
                            break 
                    
                    if first_member_is_dir_and_matches:
                        for member in member_list:
                            if not member.filename.replace('\\', '/').startswith(single_top_level_name + '/'):
                                if member.filename.replace('\\', '/').strip('/') != single_top_level_name: 
                                    all_inside = False
                                    break
                        if all_inside:
                            print(f"INFO: Обнаружена одна корневая папка '{single_top_level_name}' в архиве. Применяется strip_components=1.")
                            strip_components = 1


            for member in member_list:
                path_parts = member.filename.replace('\\', '/').split('/')
                
                if len(path_parts) <= strip_components and strip_components > 0: 
                    continue
                
                target_filename_parts = path_parts[strip_components:]
                if not target_filename_parts or not ''.join(target_filename_parts).strip(): 
                    continue 
                    
                target_filename = os.path.join(*target_filename_parts)
                target_path = os.path.join(extract_to_dir, target_filename)
                
                if member.is_dir() or member.filename.endswith('/'): 
                    os.makedirs(target_path, exist_ok=True)
                else: 
                    parent_dir = os.path.dirname(target_path)
                    if parent_dir: 
                        os.makedirs(parent_dir, exist_ok=True)
                    
                    with zip_ref.open(member, 'r') as source, open(target_path, "wb") as target_f:
                        shutil.copyfileobj(source, target_f)
                        
        print(f"Архив {zip_path} успешно распакован в {extract_to_dir}.")
        return True
    except zipfile.BadZipFile:
        print(f"Ошибка: {zip_path} не является корректным ZIP-архивом.")
    except Exception as e:
        print(f"Ошибка при распаковке {zip_path}: {e}")
        import traceback
        traceback.print_exc()
    return False


def download_and_setup_ffmpeg(work_dir):
    # ... (без изменений)
    if platform.system() != "Windows":
        print("INFO: Автоматическое скачивание FFmpeg пока поддерживается только для Windows.")
        return False

    ffmpeg_target_install_dir = os.path.join(work_dir, FFMPEG_DIR) 
    os.makedirs(ffmpeg_target_install_dir, exist_ok=True)
    
    ffmpeg_exe_full_path = os.path.join(ffmpeg_target_install_dir, FFMPEG_EXE_NAME)
    ffprobe_exe_full_path = os.path.join(ffmpeg_target_install_dir, FFPROBE_EXE_NAME)

    if os.path.exists(ffmpeg_exe_full_path) and os.path.exists(ffprobe_exe_full_path):
        print("INFO: FFmpeg и FFprobe уже существуют в рабочей директории.")
        return True

    temp_zip_path = os.path.join(tempfile.gettempdir(), "ffmpeg_download.zip")
    
    if _download_file(FFMPEG_WINDOWS_URL, temp_zip_path, "FFmpeg Essentials"):
        temp_extract_dir = os.path.join(tempfile.gettempdir(), "ffmpeg_extract_temp_" + next(tempfile._get_candidate_names()))
        if os.path.exists(temp_extract_dir): shutil.rmtree(temp_extract_dir)
        os.makedirs(temp_extract_dir)

        extract_success = False
        if _extract_zip(temp_zip_path, temp_extract_dir, strip_components=1): 
            found_bin_dir = os.path.join(temp_extract_dir, "bin") 
            
            if os.path.isdir(found_bin_dir):
                print(f"INFO: Найдена папка bin FFmpeg: {found_bin_dir}")
                try:
                    for item in os.listdir(found_bin_dir):
                        s = os.path.join(found_bin_dir, item)
                        d = os.path.join(ffmpeg_target_install_dir, item)
                        if os.path.isfile(s):
                            if item.lower() == FFMPEG_EXE_NAME.lower() or \
                               item.lower() == FFPROBE_EXE_NAME.lower() or \
                               item.lower().endswith(".dll"): 
                                shutil.copy2(s, d)
                    extract_success = True
                    print(f"INFO: FFmpeg/FFprobe скопированы в {ffmpeg_target_install_dir}")
                except Exception as e_copy:
                    print(f"ERROR: Ошибка при копировании файлов FFmpeg: {e_copy}")
            else:
                print(f"ERROR: Папка 'bin' с {FFMPEG_EXE_NAME} и {FFPROBE_EXE_NAME} не найдена в {temp_extract_dir} (ожидалось после strip_components=1).")
        
        if os.path.exists(temp_extract_dir): shutil.rmtree(temp_extract_dir)
        if os.path.exists(temp_zip_path): os.remove(temp_zip_path)
        
        if extract_success and os.path.exists(ffmpeg_exe_full_path) and os.path.exists(ffprobe_exe_full_path):
            print("INFO: FFmpeg успешно скачан и настроен.")
            return True
        else:
            print("ERROR: FFmpeg не настроен после попытки скачивания.")
            return False
    return False

# Функция download_and_setup_espeak_ng удалена

def check_and_download_tools(work_dir_path, status_callback=None):
    """Проверяет наличие FFmpeg/FFprobe и скачивает их при необходимости."""
    if not work_dir_path or not os.path.isdir(work_dir_path):
        if status_callback: status_callback("Ошибка: Рабочая директория не указана или не существует.")
        return False, False, False # ffmpeg_ok, ffprobe_ok, espeak_ok (espeak всегда False)

    if status_callback: status_callback("Проверка FFmpeg/FFprobe...")
    ffmpeg_exe_abs_path = os.path.join(work_dir_path, FFMPEG_DIR, FFMPEG_EXE_NAME)
    ffprobe_exe_abs_path = os.path.join(work_dir_path, FFMPEG_DIR, FFPROBE_EXE_NAME)
    
    ffmpeg_ok = os.path.exists(ffmpeg_exe_abs_path) and os.access(ffmpeg_exe_abs_path, os.X_OK)
    ffprobe_ok = os.path.exists(ffprobe_exe_abs_path) and os.access(ffprobe_exe_abs_path, os.X_OK)

    if not (ffmpeg_ok and ffprobe_ok):
        if platform.system() == "Windows":
            if status_callback: status_callback("FFmpeg/FFprobe не найдены. Попытка скачивания FFmpeg...")
            if download_and_setup_ffmpeg(work_dir_path):
                ffmpeg_ok = os.path.exists(ffmpeg_exe_abs_path) and os.access(ffmpeg_exe_abs_path, os.X_OK)
                ffprobe_ok = os.path.exists(ffprobe_exe_abs_path) and os.access(ffprobe_exe_abs_path, os.X_OK)
            else:
                if status_callback: status_callback("Ошибка скачивания FFmpeg.")
        else:
            if status_callback: status_callback("FFmpeg/FFprobe не найдены. Пожалуйста, установите их системно.")
    
    if status_callback: status_callback(f"FFmpeg: {'OK' if ffmpeg_ok else 'НЕ НАЙДЕН'}")
    if status_callback: status_callback(f"FFprobe: {'OK' if ffprobe_ok else 'НЕ НАЙДЕН'}")

    # Espeak больше не проверяется на скачивание
    espeak_found_final = False 
    if status_callback: status_callback("Проверка eSpeak NG (опционально, для старых/альтернативных методов выравнивания)...")
    
    # Проверяем системный espeak для информации
    sys_espeak_cmd = "espeak-ng.exe" if platform.system() == "Windows" else "espeak-ng"
    sys_espeak_path = shutil.which(sys_espeak_cmd)
    if not sys_espeak_path and platform.system() == "Windows":
         sys_espeak_path = shutil.which("espeak") # Пробуем еще espeak без -ng

    if sys_espeak_path and os.access(sys_espeak_path, os.X_OK):
        if status_callback: status_callback(f"eSpeak NG: Найден системный ({sys_espeak_path}).")
        espeak_found_final = True
    else:
        if status_callback: status_callback("eSpeak NG: Не найден системно.")
            
    # Обновляем глобальные пути после всех проверок и возможных скачиваний
    # Передаем check_only_espeak=True, чтобы не сбрасывать пути ffmpeg/ffprobe
    # Но так как initialize_paths_from_work_dir теперь всегда проверяет espeak,
    # этот флаг становится менее значимым в данном контексте.
    # Важнее, что мы передаем work_dir_path для корректной настройки кэшей.
    ffmpeg_ready, ffprobe_ready, espeak_is_actually_ready = initialize_paths_from_work_dir(work_dir_path) 
    
    return ffmpeg_ready, ffprobe_ready, espeak_is_actually_ready


def get_config_file_path(work_dir_path=None):
    # ... (без изменений)
    if work_dir_path and os.path.isdir(work_dir_path):
        return os.path.join(work_dir_path, CONFIG_FILE_NAME)
    return None

def load_app_config(current_work_dir_path=None):
    # ... (без изменений)
    config_path = get_config_file_path(current_work_dir_path)
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"INFO: Конфигурация загружена из {config_path}: {config}")
            return config
        except Exception as e:
            print(f"ERROR: Не удалось загрузить конфигурацию из {config_path}: {e}")
    return {} 

def save_app_config(config_data, work_dir_path):
    # ... (без изменений)
    if not work_dir_path or not os.path.isdir(work_dir_path):
        print(f"ERROR: Невозможно сохранить конфигурацию. Рабочая директория не указана или некорректна: {work_dir_path}")
        return False
        
    config_path = get_config_file_path(work_dir_path)
    if not config_path: 
        print(f"ERROR: Не удалось получить путь к файлу конфигурации для сохранения в {work_dir_path}")
        return False
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        print(f"INFO: Конфигурация сохранена в {config_path}")
        return True
    except Exception as e:
        print(f"ERROR: Не удалось сохранить конфигурацию в {config_path}: {e}")
        return False

def get_work_dir_from_config():
    # ... (без изменений)
    possible_exe_or_script_dir = None
    if getattr(sys, 'frozen', False):  
        possible_exe_or_script_dir = os.path.dirname(sys.executable)
    else: 
        try:
            main_module = sys.modules['__main__']
            if hasattr(main_module, '__file__'):
                possible_exe_or_script_dir = os.path.dirname(os.path.abspath(main_module.__file__))
            else: 
                possible_exe_or_script_dir = os.getcwd()
        except (KeyError, AttributeError, NameError):
             possible_exe_or_script_dir = os.getcwd()

    if possible_exe_or_script_dir:
        potential_config_path = os.path.join(possible_exe_or_script_dir, CONFIG_FILE_NAME)
        if os.path.exists(potential_config_path):
            config = load_app_config(possible_exe_or_script_dir) 
            work_dir_from_local_config = config.get('work_dir')
            if work_dir_from_local_config and os.path.isdir(work_dir_from_local_config):
                print(f"INFO: Рабочая директория загружена из локального {potential_config_path}: {work_dir_from_local_config}")
                return work_dir_from_local_config
            elif not work_dir_from_local_config: 
                print(f"INFO: Локальный конфиг {potential_config_path} найден, но work_dir не указан. Используем директорию приложения как рабочую: {possible_exe_or_script_dir}")
                save_app_config({'work_dir': possible_exe_or_script_dir}, possible_exe_or_script_dir)
                return possible_exe_or_script_dir
        else: 
            print(f"INFO: Конфиг не найден в {possible_exe_or_script_dir}. Предполагаем, что это рабочая директория по умолчанию.")
            save_app_config({'work_dir': possible_exe_or_script_dir}, possible_exe_or_script_dir)
            return possible_exe_or_script_dir

    print("INFO: Рабочая директория не найдена в конфигурациях и не определена по умолчанию.")
    return None


def save_work_dir_to_config(work_dir_path):
    # ... (без изменений)
    if not work_dir_path or not os.path.isdir(work_dir_path):
        print(f"WARNING: Попытка сохранить невалидный путь к рабочей директории: {work_dir_path}")
        return False

    config_data = {'work_dir': work_dir_path} 
    return save_app_config(config_data, work_dir_path)


def initialize_paths_from_work_dir(work_dir_path): # Убран check_only_espeak
    global CURRENT_FFMPEG_PATH, CURRENT_FFPROBE_PATH, CURRENT_ESPEAK_PATH
    
    CURRENT_FFMPEG_PATH = None
    CURRENT_FFPROBE_PATH = None
    CURRENT_ESPEAK_PATH = None 
    if 'FFMPEG_BINARY' in os.environ: del os.environ['FFMPEG_BINARY']
    if 'FFPROBE_BINARY' in os.environ: del os.environ['FFPROBE_BINARY']

    ffmpeg_found = False
    ffprobe_found = False
    espeak_found = False # По умолчанию считаем, что не найден

    if not work_dir_path or not os.path.isdir(work_dir_path):
        print("INFO: Рабочая директория не указана/не существует. Попытка использовать системные инструменты.")
        sys_ffmpeg = shutil.which(FFMPEG_EXE_NAME if platform.system() == "Windows" else "ffmpeg")
        if sys_ffmpeg and os.access(sys_ffmpeg, os.X_OK): 
            CURRENT_FFMPEG_PATH = sys_ffmpeg
            os.environ['FFMPEG_BINARY'] = CURRENT_FFMPEG_PATH
            ffmpeg_found = True
        
        sys_ffprobe = shutil.which(FFPROBE_EXE_NAME if platform.system() == "Windows" else "ffprobe")
        if sys_ffprobe and os.access(sys_ffprobe, os.X_OK):
            CURRENT_FFPROBE_PATH = sys_ffprobe
            os.environ['FFPROBE_BINARY'] = CURRENT_FFPROBE_PATH
            ffprobe_found = True
        
        # Проверка системного espeak (если он вдруг нужен для чего-то еще, но Aeneas убран)
        # sys_espeak_cmd = "espeak-ng.exe" if platform.system() == "Windows" else "espeak-ng"
        # sys_espeak_path = shutil.which(sys_espeak_cmd)
        # if not sys_espeak_path and platform.system() == "Windows": 
        #      sys_espeak_path = shutil.which("espeak") 
        # if sys_espeak_path and os.access(sys_espeak_path, os.X_OK):
        #     CURRENT_ESPEAK_PATH = sys_espeak_path
        #     espeak_found = True
        # else: print(f"INFO: Системный espeak-ng/espeak не найден (не используется по умолчанию).")
        
        for env_var in ['HF_HOME', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TTS_HOME']:
            if env_var in os.environ:
                try: del os.environ[env_var] 
                except KeyError: pass 
        print("INFO: Используются стандартные пути для кэшей моделей (рабочая папка не задана).")
        
    else: # Если рабочая директория указана
        print(f"INFO: Инициализация путей из рабочей директории: {work_dir_path}")

        ffmpeg_tools_dir = os.path.join(work_dir_path, FFMPEG_DIR)
        potential_ffmpeg_path = os.path.join(ffmpeg_tools_dir, FFMPEG_EXE_NAME)
        if os.path.exists(potential_ffmpeg_path) and os.access(potential_ffmpeg_path, os.X_OK):
            CURRENT_FFMPEG_PATH = potential_ffmpeg_path
            os.environ['FFMPEG_BINARY'] = CURRENT_FFMPEG_PATH
            ffmpeg_found = True
        
        potential_ffprobe_path = os.path.join(ffmpeg_tools_dir, FFPROBE_EXE_NAME)
        if os.path.exists(potential_ffprobe_path) and os.access(potential_ffprobe_path, os.X_OK):
            CURRENT_FFPROBE_PATH = potential_ffprobe_path
            os.environ['FFPROBE_BINARY'] = CURRENT_FFPROBE_PATH
            ffprobe_found = True

        # Поиск eSpeak NG в рабочей директории (если пользователь скопировал вручную)
        # Эта логика может остаться, на случай если кто-то захочет использовать Aeneas с ручной настройкой
        espeak_dir_for_check = "espeak-ng" # Имя подпапки для espeak в рабочей директории
        espeak_base_dir_check = os.path.join(work_dir_path, espeak_dir_for_check)
        potential_espeak_path_check = None
        if platform.system() == "Windows":
            potential_espeak_path_check = os.path.join(espeak_base_dir_check, "espeak-ng.exe") 
        else: 
            path1_check = os.path.join(espeak_base_dir_check, "bin", "espeak-ng")
            path2_check = os.path.join(espeak_base_dir_check, "espeak-ng")
            if os.path.exists(path1_check) and os.access(path1_check, os.X_OK): potential_espeak_path_check = path1_check
            elif os.path.exists(path2_check) and os.access(path2_check, os.X_OK): potential_espeak_path_check = path2_check

        if potential_espeak_path_check and os.path.exists(potential_espeak_path_check) and os.access(potential_espeak_path_check, os.X_OK):
            CURRENT_ESPEAK_PATH = os.path.abspath(potential_espeak_path_check) 
            espeak_found = True
            # PATH для espeak больше не модифицируем здесь, т.к. он не используется напрямую приложением
        # else: print(f"INFO: Espeak не найден в рабочей директории {espeak_base_dir_check} (не используется по умолчанию).")

        # Настройка кэшей внутри рабочей директории
        cache_base_dir = os.path.join(work_dir_path, ".cache") 
        hf_cache_dir = os.path.join(cache_base_dir, "huggingface")
        tts_models_dir = os.path.join(cache_base_dir, "tts_models_coqui") 
        torch_cache_dir = os.path.join(cache_base_dir, "torch") 

        os.makedirs(hf_cache_dir, exist_ok=True)
        os.makedirs(tts_models_dir, exist_ok=True)
        os.makedirs(torch_cache_dir, exist_ok=True)
        
        os.environ['HF_HOME'] = hf_cache_dir 
        os.environ['XDG_CACHE_HOME'] = cache_base_dir 
        os.environ['TORCH_HOME'] = torch_cache_dir
        os.environ['TTS_HOME'] = tts_models_dir 
        
        print(f"INFO: HF_HOME установлен: {hf_cache_dir}")
        print(f"INFO: TTS_HOME (для Coqui TTS) установлен: {tts_models_dir}")
    
    if ffmpeg_found: print(f"INFO: FFMPEG используется из: {CURRENT_FFMPEG_PATH}")
    else: print("WARNING: FFMPEG НЕ НАСТРОЕН.")
    if ffprobe_found: print(f"INFO: FFPROBE используется из: {CURRENT_FFPROBE_PATH}")
    else: print("WARNING: FFPROBE НЕ НАСТРОЕН.")
    if espeak_found: print(f"INFO: ESPEAK найден: {CURRENT_ESPEAK_PATH} (не используется по умолчанию).")
    # else: print("INFO: ESPEAK не найден (не используется по умолчанию).") # Это сообщение будет в check_and_download_tools

    return ffmpeg_found, ffprobe_found, espeak_found 