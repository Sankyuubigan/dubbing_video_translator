import ffmpeg
import subprocess
import os
import shutil
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

def check_command_availability(command):
    # ... (без изменений)
    try:
        command_path = shutil.which(command)
        if command_path is None: return False, f"'{command}' not found in PATH."
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        version_flags = ['--version', '-version', '-V']; success = False; error_output = ""
        for flag in version_flags:
            try:
                process = subprocess.run(
                    [command_path, flag], capture_output=True, text=True, check=False,
                    startupinfo=startupinfo, timeout=5, encoding='utf-8', errors='replace'
                )
                output_cmd = process.stdout; error_output = process.stderr
                if process.returncode == 0 or any(v in output_cmd.lower() for v in ["version", "copyright", "ffmpeg", "ffprobe"]):
                     success = True; break
            except FileNotFoundError: return False, f"'{command}' found but failed to execute (FileNotFound)."
            except subprocess.TimeoutExpired: print(f"Warning: '{command} {flag}' timed out."); continue
            except Exception as e: print(f"Warning: Error executing '{command} {flag}': {e}"); continue
        if success: return True, f"'{command}' found at: {command_path}"
        else:
             last_err_msg = error_output if error_output else f"Command execution failed or produced no recognizable output with flags: {', '.join(version_flags)}"
             return False, f"'{command}' found but execution failed. Error: {last_err_msg}"
    except Exception as e: return False, f"Error checking command '{command}': {e}"

def extract_audio(video_path, output_audio_path, sample_rate=16000):
    # ... (без изменений)
    try:
        (ffmpeg.input(video_path).output(output_audio_path, format='wav', acodec='pcm_s16le', ac=1, ar=str(sample_rate))
         .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return output_audio_path
    except ffmpeg.Error as e:
        print("ERROR: ffmpeg audio extraction failed."); stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        raise RuntimeError(f"Audio extraction failed: {stderr}") from e
    except Exception as e: print(f"ERROR: An unexpected error occurred during audio extraction: {e}"); raise

def merge_audio_segments(segment_files, output_path):
    # ... (без изменений)
    print(f"Merging {len(segment_files)} audio segments into {os.path.basename(output_path)}")
    if not segment_files: raise ValueError("No segment files provided for merging.")
    temp_dir = os.path.dirname(output_path); list_filename = os.path.join(temp_dir, "concat_list.txt"); output_stream = None
    try:
        valid_segment_count = 0
        with open(list_filename, "w", encoding='utf-8') as f:
            for segment_file in segment_files:
                if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                     normalized_path = os.path.normpath(os.path.abspath(segment_file)); safe_path = normalized_path.replace('\\', '/')
                     f.write(f"file '{safe_path}'\n"); valid_segment_count += 1
                else: print(f"Warning: Segment file not found or empty, skipping: {segment_file}")
        if valid_segment_count == 0: 
            print("Warning: No valid audio segments to merge. Creating an empty WAV file.")
            (ffmpeg.input('anullsrc', format='lavfi', r=24000) 
                .output(output_path, acodec='pcm_s16le', t=0.01) 
                .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            if os.path.exists(list_filename): os.remove(list_filename)
            return output_path

        output_options = {'acodec': 'copy'}
        output_stream = (ffmpeg.input(list_filename, f='concat', safe=0).output(output_path, **output_options).overwrite_output())
        _stdout, stderr = output_stream.run(capture_stdout=True, capture_stderr=True)
        stderr_str = stderr.decode('utf-8', errors='replace')
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
             print(f"Warning: FFmpeg merging produced an empty or missing file: {output_path}.")
             if stderr_str: print(f"----- FFmpeg stderr for merge_audio_segments -----\n{stderr_str}\n-------------------------")
             (ffmpeg.input('anullsrc', format='lavfi', r=24000)
                .output(output_path, acodec='pcm_s16le', t=0.01)
                .overwrite_output().run(capture_stdout=True, capture_stderr=True))
             print(f"Created a fallback empty WAV file at {output_path}")
        elif "error" in stderr_str.lower():
             print("Warning: FFmpeg might have encountered errors during merging.")
             print(f"----- FFmpeg stderr for merge_audio_segments -----\n{stderr_str}\n-------------------------")
    except ffmpeg.Error as e:
        print("ERROR: ffmpeg audio merging failed."); stderr_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        cmd_args = output_stream.get_args() if output_stream else ["N/A"]
        print(f"----- FFmpeg arguments for merge_audio_segments -----\nffmpeg {' '.join(cmd_args)}\n--------------------------")
        print(f"----- FFmpeg stderr for merge_audio_segments -----\n{stderr_msg}\n-------------------------")
        raise RuntimeError(f"Audio merging failed. FFmpeg stderr:\n{stderr_msg}") from e
    except Exception as e:
         print(f"ERROR: An unexpected error occurred during audio merging: {e}")
         if os.path.exists(list_filename):
            try: os.remove(list_filename)
            except OSError: pass
         raise
    finally:
        if os.path.exists(list_filename):
            try: os.remove(list_filename)
            except OSError as e_rem: print(f"Warning: Could not remove temporary concat list file {os.path.basename(list_filename)}: {e_rem}")
    return output_path

def mix_and_replace_audio(video_path, original_audio_path, dubbed_audio_path, output_path, original_volume=0.1, dubbed_volume=1.0):
    # ... (без изменений)
    output_stream = None
    try:
        input_video = ffmpeg.input(video_path); input_original_audio = ffmpeg.input(original_audio_path); input_dubbed_audio = ffmpeg.input(dubbed_audio_path)
        video_stream = input_video['v']
        
        has_audio_in_video = False
        try:
            probe = ffmpeg.probe(video_path)
            if any(stream.get('codec_type') == 'audio' for stream in probe.get('streams', [])):
                has_audio_in_video = True
        except ffmpeg.Error:
            print(f"Warning: Could not probe video {video_path} for audio streams.")

        if not os.path.exists(dubbed_audio_path) or os.path.getsize(dubbed_audio_path) == 0:
            print(f"Warning: Dubbed audio file {dubbed_audio_path} is missing or empty. Using original audio only.")
            if has_audio_in_video and os.path.exists(original_audio_path) and os.path.getsize(original_audio_path) > 0:
                 audio_to_use = ffmpeg.filter(input_original_audio, 'volume', str(original_volume))
                 output_options = {'vcodec': 'copy', 'acodec': 'aac', 'audio_bitrate': '192k', 'strict': '-2'}
                 output_stream = ffmpeg.output(video_stream, audio_to_use, output_path, **output_options).overwrite_output()
            else: 
                 output_options = {'vcodec': 'copy', 'an': None} 
                 output_stream = ffmpeg.output(video_stream, output_path, **output_options).overwrite_output()
        elif not os.path.exists(original_audio_path) or os.path.getsize(original_audio_path) == 0:
            print(f"Warning: Original audio file {original_audio_path} is missing or empty. Using dubbed audio only.")
            dubbed_audio_filtered = ffmpeg.filter(input_dubbed_audio, 'volume', str(dubbed_volume))
            output_options = {'vcodec': 'copy', 'acodec': 'aac', 'audio_bitrate': '192k', 'strict': '-2'}
            output_stream = ffmpeg.output(video_stream, dubbed_audio_filtered, output_path, **output_options).overwrite_output()
        else: 
            mixed_audio = ffmpeg.filter([input_original_audio, input_dubbed_audio], 'amix', inputs=2, duration='first', dropout_transition=2, weights=f"{original_volume} {dubbed_volume}")
            output_options = {'vcodec': 'copy', 'acodec': 'aac', 'audio_bitrate': '192k', 'strict': '-2'}
            output_stream = ffmpeg.output(video_stream, mixed_audio, output_path, **output_options).overwrite_output()
        
        output_stream.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print("ERROR: ffmpeg video assembly failed."); stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        cmd_args = output_stream.get_args() if output_stream else ["N/A"]
        print(f"----- FFmpeg arguments for mix_and_replace_audio -----\nffmpeg {' '.join(cmd_args)}\n--------------------------")
        raise RuntimeError(f"Video assembly failed. FFmpeg stderr:\n{stderr}") from e
    except Exception as e: print(f"ERROR: An unexpected error occurred during video assembly: {e}"); raise

def escape_ffmpeg_path(path): # ... (без изменений)
    path = path.replace('\\', '/'); path = path.replace(':', '\\:').replace("'", "'\\''"); return path

def add_subtitles(video_path, srt_path, output_path): # ... (без изменений)
    output_stream = None
    try:
        style_options = "FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=1"
        abs_srt_path = os.path.abspath(srt_path); escaped_srt_path_for_filter = escape_ffmpeg_path(abs_srt_path)
        vf_option = f"subtitles='{escaped_srt_path_for_filter}':force_style='{style_options}'"
        output_stream = (ffmpeg.input(video_path).output(output_path, vf=vf_option, acodec='copy').overwrite_output())
        output_stream.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print("ERROR: ffmpeg subtitle adding failed."); stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        cmd_args = output_stream.get_args() if output_stream else ["N/A"]
        print(f"----- FFmpeg arguments for add_subtitles -----\nffmpeg {' '.join(cmd_args)}\n--------------------------")
        raise RuntimeError(f"Subtitle adding failed. FFmpeg stderr:\n{stderr}") from e
    except Exception as e: print(f"ERROR: An unexpected error occurred during subtitle adding: {e}"); raise

def _attempt_yt_dlp_download(ydl_opts, url):
    # ... (без изменений по сравнению с v0.6.1)
    if yt_dlp is None: raise ImportError("yt-dlp library is not available but is required for this function.")
    downloaded_files = []
    
    def download_hook(d):
        if d['status'] == 'finished':
            filepath = d.get('filename') or d.get('info_dict', {}).get('_filename')
            if filepath and filepath not in downloaded_files:
                 downloaded_files.append(filepath)
            
            requested_subs = d.get('info_dict', {}).get('requested_subtitles')
            if requested_subs:
                for lang_code, sub_info in requested_subs.items():
                    sub_filepath = sub_info.get('filepath')
                    if sub_filepath and os.path.exists(sub_filepath) and sub_filepath not in downloaded_files:
                        downloaded_files.append(sub_filepath)
    
    current_ydl_opts = ydl_opts.copy()
    current_ydl_opts['progress_hooks'] = [download_hook]
    current_ydl_opts['ignoreerrors'] = current_ydl_opts.get('ignoreerrors', True) 

    with yt_dlp.YoutubeDL(current_ydl_opts) as ydl:
        info = ydl.extract_info(url, download=not current_ydl_opts.get('skip_download', False))
        
        main_video_file = None
        if not current_ydl_opts.get('skip_download', False):
            main_video_file = ydl.prepare_filename(info) 
            if not (main_video_file and os.path.exists(main_video_file)):
                 for f_path in downloaded_files:
                     # Проверяем по расширению, которое yt-dlp должен был использовать для видео
                     # yt-dlp может вернуть info['ext'] или info_dict.get('ext')
                     video_ext = info.get('ext')
                     if video_ext and f_path.lower().endswith(f".{video_ext.lower()}"):
                         main_video_file = f_path
                         break
            if main_video_file and main_video_file not in downloaded_files:
                 downloaded_files.insert(0, main_video_file)

        return main_video_file, info, downloaded_files


def _get_subtitle_path(info_dict, lang_code_to_find, video_filepath_no_ext, downloaded_files_list, output_dir):
    # ... (без изменений по сравнению с v0.6.1)
    # 1. Проверяем в info_dict['requested_subtitles']
    if 'requested_subtitles' in info_dict and info_dict['requested_subtitles']:
        # yt-dlp может вернуть 'ru' даже если запрашивали 'ru-RU' или 'ru-*'
        # Поэтому ищем по основному коду языка
        base_lang_to_find = lang_code_to_find.split('-')[0]
        for actual_lang_key, sub_info in info_dict['requested_subtitles'].items():
            if actual_lang_key.startswith(base_lang_to_find):
                 if sub_info.get('filepath') and os.path.exists(sub_info['filepath']):
                    return sub_info['filepath'], sub_info.get('ext', 'srt')

    possible_exts = ['vtt', 'srt'] # VTT часто предпочтительнее для YouTube
    # 2. Ищем среди всех скачанных файлов
    for f_path in downloaded_files_list:
        f_name_lower = os.path.basename(f_path).lower()
        # Проверяем, содержит ли имя файла код языка и одно из расширений
        for ext_iter in possible_exts:
            # Ищем точное совпадение или начало (например, ru-RU.vtt для lang_code_to_find='ru')
            if f".{lang_code_to_find}.{ext_iter}" in f_name_lower or \
               (lang_code_to_find.split('-')[0] + "." + ext_iter in f_name_lower and f_name_lower.startswith(os.path.basename(video_filepath_no_ext).lower())):
                if os.path.exists(f_path):
                    return f_path, ext_iter
    
    # 3. Формируем ожидаемые пути
    for ext_iter in possible_exts:
        # Точный путь с кодом языка
        expected_path_lang = f"{video_filepath_no_ext}.{lang_code_to_find}.{ext_iter}"
        if os.path.exists(expected_path_lang):
            return expected_path_lang, ext_iter
        
        # Путь с базовым кодом языка (например, file.ru.vtt для lang_code_to_find='ru-RU')
        base_lang_to_find = lang_code_to_find.split('-')[0]
        if base_lang_to_find != lang_code_to_find:
            expected_path_base_lang = f"{video_filepath_no_ext}.{base_lang_to_find}.{ext_iter}"
            if os.path.exists(expected_path_base_lang):
                return expected_path_base_lang, ext_iter

        # Поиск по шаблону в директории (на случай нестандартных имен)
        for f_in_dir in os.listdir(output_dir):
            f_in_dir_lower = f_in_dir.lower()
            if f_in_dir_lower.startswith(os.path.basename(video_filepath_no_ext).lower()) and \
               f".{base_lang_to_find}." in f_in_dir_lower and f_in_dir_lower.endswith(f".{ext_iter}"):
                full_path = os.path.join(output_dir, f_in_dir)
                if os.path.exists(full_path):
                     return full_path, ext_iter
    return None, None

def download_youtube_video(url, output_dir, quality='1080p', 
                           preferred_sub_lang='ru', fallback_sub_lang='en'):
    if yt_dlp is None: 
        raise ImportError("yt-dlp library not found. Please install it using: pip install yt-dlp")

    video_filename_final = None; subtitle_filename_final = None; actual_sub_lang_found = None
    
    preferred_sub_langs_list = [preferred_sub_lang] if preferred_sub_lang else []
    if preferred_sub_lang and '-' in preferred_sub_lang: # ru-RU -> ['ru-RU', 'ru']
        preferred_sub_langs_list.append(preferred_sub_lang.split('-')[0])
    preferred_sub_langs_list = sorted(list(set(filter(None, preferred_sub_langs_list))), key=len, reverse=True)

    fallback_sub_langs_list = [fallback_sub_lang] if fallback_sub_lang else []
    if fallback_sub_lang and '-' in fallback_sub_lang:
        fallback_sub_langs_list.append(fallback_sub_lang.split('-')[0])
    fallback_sub_langs_list = sorted(list(set(filter(None, fallback_sub_langs_list))), key=len, reverse=True)

    base_ydl_opts = {
        'format': f'bestvideo[height<={quality[:-1]}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality[:-1]}][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'), 
        'noplaylist': True, 'merge_output_format': 'mp4',
        'quiet': True, 'no_warnings': True, # Возвращаем обратно для чистоты логов, если отладка yt-dlp не нужна
        # 'verbose': True, # Можно включить для максимальной отладки yt-dlp
        'writesubtitles': True,
        'writeautomaticsub': True, 
        'subtitlesformat': 'vtt/srt', # VTT часто более доступен на YouTube
        'ignoreerrors': True, 
        'no_color': True,
        'cachedir': False, # Отключаем кэш yt-dlp явно (эквивалент --no-cache-dir)
        # '--no-cache-dir': True, # Прямая передача аргумента командной строки (может не работать через API)
                                 # Вместо этого используем 'cachedir': False
    }
    print(f"Processing YouTube URL: {url}")
    all_downloaded_files_session = [] 

    # --- Попытка 1: Видео + Предпочитаемые субтитры ---
    if preferred_sub_langs_list:
        print(f"Attempting to download video and preferred subtitles ({', '.join(preferred_sub_langs_list)})...")
        current_ydl_opts_pref = base_ydl_opts.copy()
        current_ydl_opts_pref['subtitleslangs'] = [f"{lang}*" for lang in preferred_sub_langs_list] # Добавляем * для поиска поддиалектов
        
        try:
            video_file_pref, info_dict_pref, downloaded_pref = _attempt_yt_dlp_download(current_ydl_opts_pref, url)
            all_downloaded_files_session.extend(x for x in downloaded_pref if x not in all_downloaded_files_session)

            if video_file_pref and os.path.exists(video_file_pref):
                video_filename_final = video_file_pref
                print(f"Video downloaded: {os.path.basename(video_filename_final)}")
                video_filepath_no_ext = os.path.splitext(video_filename_final)[0]
                
                for pref_lang_code_iter in preferred_sub_langs_list:
                    sub_path, sub_ext = _get_subtitle_path(info_dict_pref, pref_lang_code_iter, video_filepath_no_ext, all_downloaded_files_session, output_dir)
                    if sub_path:
                        subtitle_filename_final = sub_path
                        actual_sub_lang_found = pref_lang_code_iter.split('-')[0] # Используем основной код языка
                        print(f"Preferred subtitles ({actual_sub_lang_found}, ext: {sub_ext}) found: {os.path.basename(subtitle_filename_final)}")
                        break 
            else:
                 print(f"Video download failed or file not found from preferred sub attempt: {video_file_pref}")

        except yt_dlp.utils.DownloadError as e_pref:
            print(f"yt-dlp DownloadError during preferred subtitle attempt: {e_pref}")
            if "Did not get any data blocks" in str(e_pref):
                 print("This specific error often means the subtitle stream was announced but failed to download.")
        except Exception as e_pref_other:
            print(f"Unexpected error during preferred subtitle attempt: {e_pref_other}")
    
    # --- Попытка 2: Только видео или Запасные субтитры ---
    if not video_filename_final or (video_filename_final and not subtitle_filename_final and fallback_sub_langs_list):
        action_desc = "video only" if not video_filename_final else f"fallback subtitles ({', '.join(fallback_sub_langs_list)})"
        print(f"Attempting to download {action_desc}...")
        
        current_ydl_opts_fallback = base_ydl_opts.copy()
        if video_filename_final: 
            current_ydl_opts_fallback['skip_download'] = True 
            current_ydl_opts_fallback['outtmpl'] = os.path.splitext(video_filename_final)[0] + '.%(ext)s' 
            if fallback_sub_langs_list:
                current_ydl_opts_fallback['subtitleslangs'] = [f"{lang}*" for lang in fallback_sub_langs_list]
            else: # Нет запасных языков, отключаем субтитры
                current_ydl_opts_fallback['writesubtitles'] = False
                current_ydl_opts_fallback['writeautomaticsub'] = False

        else: # Видео еще не скачано
            if fallback_sub_langs_list:
                 current_ydl_opts_fallback['subtitleslangs'] = [f"{lang}*" for lang in fallback_sub_langs_list]
            else: # Нет ни предпочтительных, ни запасных, качаем без сабов
                 current_ydl_opts_fallback['writesubtitles'] = False
                 current_ydl_opts_fallback['writeautomaticsub'] = False

        try:
            video_file_fb, info_dict_fb, downloaded_fb = _attempt_yt_dlp_download(current_ydl_opts_fallback, url)
            all_downloaded_files_session.extend(x for x in downloaded_fb if x not in all_downloaded_files_session)

            if not video_filename_final and video_file_fb and os.path.exists(video_file_fb):
                video_filename_final = video_file_fb
                print(f"Video downloaded (during fallback/video-only attempt): {os.path.basename(video_filename_final)}")

            if video_filename_final and not subtitle_filename_final and fallback_sub_langs_list: 
                video_filepath_no_ext = os.path.splitext(video_filename_final)[0]
                for fall_lang_code_iter in fallback_sub_langs_list:
                    sub_path_fb, sub_ext_fb = _get_subtitle_path(info_dict_fb, fall_lang_code_iter, video_filepath_no_ext, all_downloaded_files_session, output_dir)
                    if sub_path_fb:
                        subtitle_filename_final = sub_path_fb
                        actual_sub_lang_found = fall_lang_code_iter.split('-')[0]
                        print(f"Fallback subtitles ({actual_sub_lang_found}, ext: {sub_ext_fb}) found: {os.path.basename(subtitle_filename_final)}")
                        break 
        except yt_dlp.utils.DownloadError as e_fb:
            print(f"yt-dlp DownloadError during fallback/video-only attempt: {e_fb}")
        except Exception as e_fb_other:
            print(f"Unexpected error during fallback/video-only attempt: {e_fb_other}")

    if not video_filename_final or not os.path.exists(video_filename_final):
        found_video_in_all_files = None
        for f_path in all_downloaded_files_session:
             try:
                 video_ext_check = ('.mp4', '.mkv', '.webm', '.flv', '.mov', '.avi')
                 if f_path.lower().endswith(video_ext_check):
                     probe_info = ffmpeg.probe(f_path)
                     if any(s.get('codec_type') == 'video' for s in probe_info.get('streams',[])):
                         found_video_in_all_files = f_path
                         break
             except Exception:
                 pass 
        
        if found_video_in_all_files:
            video_filename_final = found_video_in_all_files
            print(f"Video identified from all downloaded files: {os.path.basename(video_filename_final)}")
        else:
            raise RuntimeError(f"Failed to download or identify the YouTube video file after all attempts. URL: {url}")

    if not subtitle_filename_final:
         print(f"Subtitles for requested languages ('{preferred_sub_lang}', '{fallback_sub_lang}') were not found/downloaded after all attempts.")
            
    return video_filename_final, subtitle_filename_final, actual_sub_lang_found