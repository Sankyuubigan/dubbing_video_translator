import ffmpeg
import subprocess
import os
import shutil
import traceback 

try:
    import yt_dlp
    from yt_dlp.utils import OUTTMPL_TYPES 
except ImportError:
    yt_dlp = None
    OUTTMPL_TYPES = () 

def check_command_availability(command): # ... (без изменений) ...
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

def extract_audio(video_path, output_audio_path, sample_rate=16000): # ... (без изменений) ...
    try:
        (ffmpeg.input(video_path).output(output_audio_path, format='wav', acodec='pcm_s16le', ac=1, ar=str(sample_rate))
         .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return output_audio_path
    except ffmpeg.Error as e:
        print("ERROR: ffmpeg audio extraction failed."); stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        raise RuntimeError(f"Audio extraction failed: {stderr}") from e
    except Exception as e: print(f"ERROR: An unexpected error occurred during audio extraction: {e}"); raise

def merge_audio_segments(segment_files, output_path): # ... (без изменений) ...
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

def mix_and_replace_audio(video_path, original_audio_path, dubbed_audio_path, output_path, original_volume=0.1, dubbed_volume=1.0): # ... (без изменений) ...
    output_stream = None
    try:
        input_video = ffmpeg.input(video_path); input_original_audio = ffmpeg.input(original_audio_path); input_dubbed_audio = ffmpeg.input(dubbed_audio_path)
        video_stream = input_video['v']
        has_audio_in_video = False
        try:
            probe = ffmpeg.probe(video_path)
            if any(stream.get('codec_type') == 'audio' for stream in probe.get('streams', [])):
                has_audio_in_video = True
        except ffmpeg.Error: print(f"Warning: Could not probe video {video_path} for audio streams.")
        if not os.path.exists(dubbed_audio_path) or os.path.getsize(dubbed_audio_path) == 0:
            print(f"Warning: Dubbed audio file {dubbed_audio_path} is missing or empty. Using original audio only (if video has it).")
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

def escape_ffmpeg_path(path): # ... (без изменений) ...
    path = path.replace('\\', '/'); path = path.replace(':', '\\:').replace("'", "'\\''"); return path

def add_subtitles(video_path, srt_path, output_path): # ... (без изменений) ...
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
    if yt_dlp is None: raise ImportError("yt-dlp library is not available but is required for this function.")
    
    downloaded_files_from_hook = [] 
    
    current_ydl_opts = ydl_opts.copy()
    current_ydl_opts['verbose'] = True 
    current_ydl_opts['quiet'] = False
    current_ydl_opts['no_warnings'] = False

    # current_output_dir берется из ydl_opts['outtmpl'], который должен быть строкой с полным путем
    # outtmpl_setting извлекается из словаря ydl_opts
    outtmpl_setting_from_opts = current_ydl_opts.get('outtmpl')
    
    # Определяем effective_outtmpl_str как строку
    if isinstance(outtmpl_setting_from_opts, dict) and 'default' in outtmpl_setting_from_opts and isinstance(outtmpl_setting_from_opts['default'], str):
        effective_outtmpl_str = outtmpl_setting_from_opts['default']
    elif isinstance(outtmpl_setting_from_opts, str):
        effective_outtmpl_str = outtmpl_setting_from_opts
    else:
        print(f"  yt-dlp: CRITICAL - outtmpl has unexpected type/structure: {outtmpl_setting_from_opts}. Cannot determine output directory.")
        # Это не должно происходить, так как мы формируем outtmpl как строку в download_youtube_video
        # Но на всякий случай, если это произойдет, используем заглушку
        effective_outtmpl_str = os.path.join('.', '%(title)s.%(id)s.%(ext)s') 

    current_output_dir = os.path.dirname(effective_outtmpl_str)
    if not os.path.isabs(current_output_dir) and current_output_dir : 
        current_output_dir = os.path.abspath(current_output_dir)
    elif not current_output_dir : 
        current_output_dir = os.path.abspath('.') 
    
    # print(f"  yt-dlp: Determined current_output_dir for download: {current_output_dir}")

    def ytdlp_hook(d): # ... (без изменений) ...
        status = d.get('status'); filename_hook = d.get('filename', d.get('info_dict', {}).get('_filename'))
        if status == 'downloading':
            progress_str = d.get('_percent_str', d.get('_total_bytes_str', '')); speed_str = d.get('_speed_str', ''); eta_str = d.get('_eta_str', '')
            if filename_hook: print(f"  yt-dlp: Downloading {os.path.basename(filename_hook)}: {progress_str} ({speed_str}, ETA: {eta_str})")
        elif status == 'finished':
            if filename_hook:
                if filename_hook not in downloaded_files_from_hook: downloaded_files_from_hook.append(filename_hook)
            requested_subs = d.get('info_dict', {}).get('requested_subtitles')
            if requested_subs:
                for lang_code, sub_info in requested_subs.items():
                    sub_filepath = sub_info.get('filepath')
                    if sub_filepath and os.path.exists(sub_filepath) and sub_filepath not in downloaded_files_from_hook: downloaded_files_from_hook.append(sub_filepath)
        elif status == 'error':
            if filename_hook: print(f"  yt-dlp: Error downloading (hook): {os.path.basename(filename_hook)}.")

    current_ydl_opts['progress_hooks'] = [ytdlp_hook]
    current_ydl_opts['ignoreerrors'] = current_ydl_opts.get('ignoreerrors', True) 

    info_dict_after_download = None
    main_video_file_path = None

    with yt_dlp.YoutubeDL(current_ydl_opts) as ydl:
        try:
            info_dict_after_download = ydl.extract_info(url, download=not current_ydl_opts.get('skip_download', False))
            if not current_ydl_opts.get('skip_download', False) and info_dict_after_download:
                final_path_from_info = info_dict_after_download.get('filepath') or \
                                       info_dict_after_download.get('_filename') 
                if final_path_from_info and os.path.exists(final_path_from_info):
                    main_video_file_path = final_path_from_info
                else:
                    try:
                        expected_final_path = ydl.prepare_filename(info_dict_after_download)
                        if expected_final_path and os.path.exists(expected_final_path):
                            main_video_file_path = expected_final_path
                    except Exception as e_prep_fn:
                        print(f"  yt-dlp: Error calling ydl.prepare_filename: {e_prep_fn}. Trying manual construction based on '{current_output_dir}'.")
                        video_title = info_dict_after_download.get('title', 'video'); video_id = info_dict_after_download.get('id', 'unknown_id')
                        file_extension = info_dict_after_download.get('ext', 'mp4')
                        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '_', '-')).rstrip()
                        manually_constructed_filename = f"{safe_title}.{video_id}.{file_extension}"
                        manually_constructed_path = os.path.join(current_output_dir, manually_constructed_filename)
                        if os.path.exists(manually_constructed_path):
                            main_video_file_path = manually_constructed_path
                if not main_video_file_path:
                    video_exts = ('.mp4', '.mkv', '.webm', '.flv', '.avi', '.mov')
                    for f_path_hook in downloaded_files_from_hook:
                        if f_path_hook.lower().endswith(video_exts):
                            try: 
                                probe_test = ffmpeg.probe(f_path_hook)
                                if any(s.get('codec_type') == 'video' for s in probe_test.get('streams',[])):
                                    main_video_file_path = f_path_hook; break
                            except Exception: pass
                if main_video_file_path and main_video_file_path not in downloaded_files_from_hook:
                     downloaded_files_from_hook.append(main_video_file_path)
            elif current_ydl_opts.get('skip_download', False):
                 print(f"  yt-dlp: Video download was skipped.")
        except yt_dlp.utils.DownloadError as e_dl_err: print(f"  yt-dlp: DownloadError occurred: {e_dl_err}")
        except Exception as e_generic: print(f"  yt-dlp: Generic exception during extract_info for URL '{url}': {e_generic}\n{traceback.format_exc()}")
    return main_video_file_path, info_dict_after_download, downloaded_files_from_hook

def _get_subtitle_path(info_dict, lang_code_to_find, video_filepath_no_ext, downloaded_files_list, output_dir): # ... (без изменений) ...
    if info_dict and 'requested_subtitles' in info_dict and info_dict['requested_subtitles']:
        base_lang_to_find = lang_code_to_find.split('-')[0]
        for actual_lang_key, sub_info in info_dict['requested_subtitles'].items():
            if actual_lang_key.startswith(base_lang_to_find): 
                 sub_filepath = sub_info.get('filepath')
                 if sub_filepath and os.path.exists(sub_filepath) and os.path.getsize(sub_filepath) > 0 :
                    return sub_filepath, sub_info.get('ext', 'vtt')
    possible_exts = ['vtt', 'srt'] 
    for f_path in downloaded_files_list: 
        f_name_lower = os.path.basename(f_path).lower()
        video_base_name_lower_for_sub = ""
        if video_filepath_no_ext: video_base_name_lower_for_sub = os.path.basename(video_filepath_no_ext).lower()
        elif info_dict and info_dict.get('title') and info_dict.get('id'):
             sanitized_title = "".join(c for c in info_dict['title'] if c.isalnum() or c in (' ', '_', '-')).rstrip()
             video_base_name_lower_for_sub = f"{sanitized_title}.{info_dict['id']}".lower()
        for ext_iter in possible_exts:
            name_prefix_matches = (video_base_name_lower_for_sub and f_name_lower.startswith(video_base_name_lower_for_sub)) or \
                                  (not video_base_name_lower_for_sub)
            lang_tag_matches = (f".{lang_code_to_find}.{ext_iter}" in f_name_lower or \
                                f".{lang_code_to_find.split('-')[0]}.{ext_iter}" in f_name_lower)
            if name_prefix_matches and lang_tag_matches and \
               os.path.exists(f_path) and os.path.getsize(f_path) > 0:
                return f_path, ext_iter
    if video_filepath_no_ext:
        for ext_iter in possible_exts:
            expected_paths_to_check = [
                f"{video_filepath_no_ext}.{lang_code_to_find}.{ext_iter}",
                f"{video_filepath_no_ext}.{lang_code_to_find.split('-')[0]}.{ext_iter}" ]
            for expected_path in set(expected_paths_to_check):
                if os.path.exists(expected_path) and os.path.getsize(expected_path) > 0:
                    return expected_path, ext_iter
    return None, None

def download_youtube_video(url, output_dir, quality='1080p', 
                           preferred_sub_lang='ru', fallback_sub_lang='en'): # Убран cookie_file_path
    # ... (остальной код без изменений, кроме удаления cookie_file_path из base_ydl_opts) ...
    if yt_dlp is None: raise ImportError("yt-dlp library not found. Please install it using: pip install yt-dlp")
    video_filename_final = None; subtitle_filename_final = None; actual_sub_lang_found = None
    unique_sub_langs = []
    if preferred_sub_lang: unique_sub_langs.append(preferred_sub_lang)
    if fallback_sub_lang and fallback_sub_lang not in unique_sub_langs: unique_sub_langs.append(fallback_sub_lang)
    for lang in list(unique_sub_langs):
        if '-' in lang:
            base_lang = lang.split('-')[0]
            if base_lang not in unique_sub_langs: unique_sub_langs.append(base_lang)
    outtmpl_str = os.path.join(output_dir, '%(title)s.%(id)s.%(ext)s')
    base_ydl_opts = {
        'format': f'bestvideo[height<={quality[:-1]}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality[:-1]}][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': outtmpl_str, 
        'noplaylist': True, 'merge_output_format': 'mp4',
        'writesubtitles': True, 'writeautomaticsub': True, 
        'subtitlesformat': 'vtt/srt', 
        'ignoreerrors': True, 'no_color': True, 'cachedir': False,
    }
    # print(f"Processing YouTube URL: {url}") # Уже логируется в _attempt_yt_dlp_download
    all_downloaded_files_session = [] 
    current_ydl_opts_dl = base_ydl_opts.copy()
    if unique_sub_langs:
        current_ydl_opts_dl['subtitleslangs'] = [f"{lang}*" for lang in unique_sub_langs]
    else:
        current_ydl_opts_dl['writesubtitles'] = False
        current_ydl_opts_dl['writeautomaticsub'] = False
    # print(f"Attempting to download video and subtitles for languages: {current_ydl_opts_dl.get('subtitleslangs', 'None')}...")
    info_dict_main_dl = None
    try:
        video_file_dl, info_dict_main_dl, downloaded_dl_files = _attempt_yt_dlp_download(current_ydl_opts_dl, url)
        all_downloaded_files_session.extend(x for x in downloaded_dl_files if x not in all_downloaded_files_session)
        if video_file_dl and os.path.exists(video_file_dl):
            video_filename_final = video_file_dl
            video_filepath_no_ext = os.path.splitext(video_filename_final)[0] if video_filename_final else None
            search_order_langs = []
            if preferred_sub_lang: search_order_langs.append(preferred_sub_lang)
            if fallback_sub_lang and fallback_sub_lang != preferred_sub_lang : search_order_langs.append(fallback_sub_lang)
            for lang_code_iter in search_order_langs:
                sub_path, sub_ext = _get_subtitle_path(info_dict_main_dl, lang_code_iter, video_filepath_no_ext, all_downloaded_files_session, output_dir)
                if sub_path:
                    subtitle_filename_final = sub_path
                    actual_sub_lang_found = lang_code_iter.split('-')[0] 
                    break 
    except yt_dlp.utils.DownloadError as e_dl: print(f"yt-dlp DownloadError during main download attempt for '{url}': {e_dl}")
    except Exception as e_dl_other: print(f"Unexpected error during main download attempt for '{url}': {e_dl_other}\n{traceback.format_exc()}")
    if not video_filename_final or not os.path.exists(video_filename_final):
        video_exts_check = ('.mp4', '.mkv', '.webm', '.flv', '.mov', '.avi')
        for f_path_check in all_downloaded_files_session: 
             if f_path_check.lower().endswith(video_exts_check):
                 try:
                     probe_info_check = ffmpeg.probe(f_path_check)
                     if any(s.get('codec_type') == 'video' for s in probe_info_check.get('streams',[])):
                         video_filename_final = f_path_check; break
                 except Exception: pass
        if not video_filename_final: raise RuntimeError(f"Failed to download or identify the YouTube video file after all attempts. URL: {url}")
    if not subtitle_filename_final: print(f"Subtitles for requested languages ('{preferred_sub_lang}', '{fallback_sub_lang}') were not found/downloaded after all attempts.")
    return video_filename_final, subtitle_filename_final, actual_sub_lang_found