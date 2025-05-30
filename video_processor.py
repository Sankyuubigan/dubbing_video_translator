import ffmpeg
import subprocess
import os
import shutil
import traceback
import soundfile as sf 

try:
    import yt_dlp
    from yt_dlp.utils import OUTTMPL_TYPES
except ImportError:
    yt_dlp = None
    OUTTMPL_TYPES = ()

def check_command_availability(command): 
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

def get_video_duration(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        duration_str = probe.get('format', {}).get('duration')
        if duration_str is None:
            for stream in probe.get('streams', []):
                if stream.get('codec_type') == 'video' and stream.get('duration'):
                    duration_str = stream.get('duration')
                    break
        if duration_str:
            return float(duration_str)
        else:
            print(f"Warning: Could not determine duration from format or video streams for {os.path.basename(video_path)}.")
            return 0
    except ffmpeg.Error as e:
        stderr_decoded = e.stderr.decode('utf8', 'ignore') if e.stderr else "N/A"
        print(f"Error probing video duration for {os.path.basename(video_path)}: {stderr_decoded}")
        return 0
    except Exception as e:
        print(f"Unexpected error probing video duration for {os.path.basename(video_path)}: {e}")
        return 0

def extract_audio(video_path, output_audio_path, sample_rate=16000, start_time_seconds=None, duration_seconds=None): 
    try:
        input_options = {}
        if start_time_seconds is not None:
            input_options['ss'] = str(start_time_seconds)
        
        input_stream = ffmpeg.input(video_path, **input_options)
        
        output_options_ffmpeg = {'format': 'wav', 'acodec': 'pcm_s16le', 'ac': 1, 'ar': str(sample_rate)}
        if duration_seconds is not None:
            output_options_ffmpeg['t'] = str(duration_seconds)
        
        (input_stream
            .output(output_audio_path, **output_options_ffmpeg)
            .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                
        return output_audio_path
    except ffmpeg.Error as e:
        print(f"ERROR: ffmpeg audio extraction failed for video '{os.path.basename(video_path)}'. Start: {start_time_seconds}, Duration: {duration_seconds}"); 
        stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        raise RuntimeError(f"Audio extraction failed: {stderr}") from e
    except Exception as e: 
        print(f"ERROR: An unexpected error occurred during audio extraction from '{os.path.basename(video_path)}': {e}"); 
        raise

def merge_audio_segments(segment_files, output_path, target_samplerate=24000, target_channels=1, target_codec='pcm_s16le', log_prefix="  (Merge) "):
    if log_prefix: print(f"{log_prefix}Attempting to merge {len(segment_files)} audio segments into {os.path.basename(output_path)}")
    if not segment_files:
        if log_prefix: print(f"{log_prefix}Warning: No segment files provided for merging. Creating an empty WAV file.")
        try:
            (ffmpeg.input('anullsrc', format='lavfi', r=target_samplerate, channel_layout=f"{target_channels}c" if target_channels > 1 else "mono")
             .output(output_path, acodec=target_codec, t=0.01, ar=target_samplerate, ac=target_channels)
             .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        except Exception as e_empty:
            if log_prefix: print(f"{log_prefix}  Error creating empty WAV for merge: {e_empty}")
        return output_path

    temp_dir_for_normalized_segments = os.path.join(os.path.dirname(output_path), "normalized_for_merge_" + os.path.splitext(os.path.basename(output_path))[0])
    os.makedirs(temp_dir_for_normalized_segments, exist_ok=True)
    
    list_filename = os.path.join(temp_dir_for_normalized_segments, "concat_list.txt")
    valid_normalized_segment_files = []
    
    if log_prefix: print(f"{log_prefix}Normalizing {len(segment_files)} segments to SR={target_samplerate}, Channels={target_channels}, Codec={target_codec}...")
    for idx, segment_file_original in enumerate(segment_files):
        base_name = os.path.basename(segment_file_original)
        normalized_segment_path = os.path.join(temp_dir_for_normalized_segments, f"norm_{idx:04d}_{base_name}")

        if not os.path.exists(segment_file_original) or os.path.getsize(segment_file_original) == 0:
            original_duration_fallback = 0.01 
            try:
                (ffmpeg.input('anullsrc', format='lavfi', r=target_samplerate, channel_layout=f"{target_channels}c" if target_channels > 1 else "mono")
                 .output(normalized_segment_path, t=original_duration_fallback, acodec=target_codec, ar=target_samplerate, ac=target_channels)
                 .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            except Exception as e_norm_silence:
                if log_prefix: print(f"{log_prefix}  Error creating normalized silence for missing/empty {base_name}: {e_norm_silence}")
                continue 
        else:
            try:
                (ffmpeg.input(segment_file_original)
                 .output(normalized_segment_path, ar=target_samplerate, ac=target_channels, acodec=target_codec)
                 .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            except ffmpeg.Error as e_norm:
                stderr_norm = e_norm.stderr.decode('utf-8', errors='replace') if e_norm.stderr else "N/A"
                if log_prefix: print(f"{log_prefix}  Error normalizing segment {base_name}: {stderr_norm}. Skipping.")
                if os.path.exists(normalized_segment_path): os.remove(normalized_segment_path)
                continue 
            except Exception as e_gen_norm:
                if log_prefix: print(f"{log_prefix}  Generic error during normalization of {base_name}: {e_gen_norm}. Skipping.")
                if os.path.exists(normalized_segment_path): os.remove(normalized_segment_path)
                continue

        if os.path.exists(normalized_segment_path) and os.path.getsize(normalized_segment_path) > 0:
            valid_normalized_segment_files.append(normalized_segment_path)
        elif log_prefix:
            print(f"{log_prefix}  Normalized segment {os.path.basename(normalized_segment_path)} is missing or empty. Skipping.")
    
    if log_prefix:
        print(f"{log_prefix}Normalization complete. {len(valid_normalized_segment_files)} valid segments remaining for merge.")

    if not valid_normalized_segment_files:
        if log_prefix: print(f"{log_prefix}Warning: No valid segments left after normalization. Creating an empty WAV file.")
        try:
            (ffmpeg.input('anullsrc', format='lavfi', r=target_samplerate, channel_layout=f"{target_channels}c" if target_channels > 1 else "mono")
             .output(output_path, acodec=target_codec, t=0.01, ar=target_samplerate, ac=target_channels)
             .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        except Exception as e_empty_final:
            if log_prefix: print(f"{log_prefix}  Error creating final empty WAV: {e_empty_final}")
        if os.path.exists(temp_dir_for_normalized_segments): shutil.rmtree(temp_dir_for_normalized_segments)
        return output_path

    with open(list_filename, "w", encoding='utf-8') as f:
        for norm_file in valid_normalized_segment_files:
            safe_path = os.path.normpath(os.path.abspath(norm_file)).replace('\\', '/')
            f.write(f"file '{safe_path}'\n")

    try:
        if log_prefix: print(f"{log_prefix}Starting final merge of {len(valid_normalized_segment_files)} normalized segments...")
        (ffmpeg.input(list_filename, format='concat', safe=0)
            .output(output_path, acodec='copy') 
            .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        if log_prefix: print(f"{log_prefix}Audio segments merged successfully to {os.path.basename(output_path)}")
    except ffmpeg.Error as e:
        if log_prefix: print(f"{log_prefix}CRITICAL ERROR: ffmpeg audio merging failed (ffmpeg.Error).");
        stderr_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        if log_prefix: print(f"{log_prefix}----- FFmpeg stderr for merge_audio_segments -----\n{stderr_msg}\n-------------------------")
        try:
            (ffmpeg.input('anullsrc', format='lavfi', r=target_samplerate, channel_layout=f"{target_channels}c" if target_channels > 1 else "mono")
                .output(output_path, acodec=target_codec, t=0.01, ar=target_samplerate, ac=target_channels)
                .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            if log_prefix: print(f"{log_prefix}Created a fallback empty WAV file at {output_path} due to ffmpeg.Error.")
        except Exception as e_fb_ffmpeg_err:
             if log_prefix: print(f"{log_prefix}  Failed to create fallback empty WAV after ffmpeg.Error: {e_fb_ffmpeg_err}")
    except Exception as e_gen:
         if log_prefix: print(f"{log_prefix}CRITICAL ERROR: An unexpected error occurred during audio merging: {e_gen}")
         if log_prefix: traceback.print_exc()
         try:
            (ffmpeg.input('anullsrc', format='lavfi', r=target_samplerate, channel_layout=f"{target_channels}c" if target_channels > 1 else "mono")
                .output(output_path, acodec=target_codec, t=0.01, ar=target_samplerate, ac=target_channels)
                .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            if log_prefix: print(f"{log_prefix}Created a fallback empty WAV file at {output_path} due to generic error.")
         except Exception as e_fb_gen_err:
            if log_prefix: print(f"{log_prefix}  Failed to create fallback empty WAV after generic error: {e_fb_gen_err}")
    finally:
        if os.path.exists(temp_dir_for_normalized_segments):
            try: shutil.rmtree(temp_dir_for_normalized_segments)
            except OSError as e_rem:
                if log_prefix: print(f"{log_prefix}Warning: Could not remove temp dir {temp_dir_for_normalized_segments}: {e_rem}")
    return output_path

def concatenate_video_chunks(chunk_files, output_path, temp_dir):
    if not chunk_files:
        print("No video chunks to concatenate.")
        return None
    if len(chunk_files) == 1:
        try:
            shutil.copy(chunk_files[0], output_path)
            print(f"Single video chunk copied to {output_path}")
            return output_path
        except Exception as e_copy:
            print(f"Error copying single video chunk: {e_copy}")
            return None

    list_filename = os.path.join(temp_dir, "concat_video_list.txt")
    with open(list_filename, "w", encoding='utf-8') as f:
        for chunk_file in chunk_files:
            safe_path = os.path.normpath(os.path.abspath(chunk_file)).replace('\\', '/')
            f.write(f"file '{safe_path}'\n")
    
    print(f"Attempting to concatenate {len(chunk_files)} video chunks into {os.path.basename(output_path)}")
    try:
        (ffmpeg
         .input(list_filename, format='concat', safe=0, auto_convert=1)
         .output(output_path, vcodec='copy', acodec='copy', **{'bsf:a': 'aac_adtstoasc'} if any(c.lower().endswith('.mp4') for c in chunk_files) else {})
         .overwrite_output()
         .run(capture_stdout=True, capture_stderr=True))
        print(f"Video chunks successfully concatenated (codec copy) to {output_path}")
        return output_path
    except ffmpeg.Error as e:
        stderr_str = e.stderr.decode('utf8', 'ignore') if e.stderr else "N/A"
        print(f"Warning: Concatenating video chunks with codec copy failed. Retrying with re-encoding.\nFFmpeg stderr: {stderr_str[:500]}...")
        try:
            (ffmpeg
             .input(list_filename, format='concat', safe=0, auto_convert=1)
             .output(output_path, vcodec='libx264', preset='medium', crf=23, acodec='aac', audio_bitrate='192k')
             .overwrite_output()
             .run(capture_stdout=True, capture_stderr=True))
            print(f"Video chunks successfully re-encoded and concatenated to {output_path}")
            return output_path
        except ffmpeg.Error as e2:
            stderr_str2 = e2.stderr.decode('utf8', 'ignore') if e2.stderr else "N/A"
            print(f"ERROR: Re-encoding and concatenating video chunks also failed.\nFFmpeg stderr: {stderr_str2[:500]}")
            return None
    except Exception as ex:
        print(f"ERROR: Unexpected error during video concatenation: {ex}")
        return None
    finally:
        if os.path.exists(list_filename):
            try: os.remove(list_filename)
            except OSError: pass

def mix_and_replace_audio(video_path, original_audio_path, dubbed_audio_path, output_path, 
                          original_volume=0.1, dubbed_volume=1.0,
                          video_start_time=None, video_duration=None): 
    output_stream = None
    try:
        input_video_options = {}
        if video_start_time is not None:
            input_video_options['ss'] = str(video_start_time)
        
        input_video = ffmpeg.input(video_path, **input_video_options)
        video_stream_final = input_video['v']
        if video_duration is not None:
             video_stream_final = ffmpeg.trim(video_stream_final, duration=video_duration)

        input_original_audio = None
        if original_audio_path and os.path.exists(original_audio_path) and os.path.getsize(original_audio_path) > 0:
            input_original_audio = ffmpeg.input(original_audio_path)

        input_dubbed_audio = None
        if dubbed_audio_path and os.path.exists(dubbed_audio_path) and os.path.getsize(dubbed_audio_path) > 0:
            input_dubbed_audio = ffmpeg.input(dubbed_audio_path)

        audio_output_stream = None
        if input_dubbed_audio and input_original_audio: 
            filtered_original = ffmpeg.filter(input_original_audio, 'volume', str(original_volume))
            filtered_dubbed = ffmpeg.filter(input_dubbed_audio, 'volume', str(dubbed_volume))
            audio_output_stream = ffmpeg.filter(
                [filtered_original, filtered_dubbed], 
                'amix', inputs=2, duration='first', dropout_transition=2 
            )
        elif input_dubbed_audio: 
            audio_output_stream = ffmpeg.filter(input_dubbed_audio, 'volume', str(dubbed_volume))
        elif input_original_audio: 
            audio_output_stream = ffmpeg.filter(input_original_audio, 'volume', str(original_volume))
        
        output_options_ffmpeg = {'vcodec': 'copy', 'acodec': 'aac', 'audio_bitrate': '192k', 'strict': '-2'}
        
        if audio_output_stream:
            output_stream = ffmpeg.output(video_stream_final, audio_output_stream, output_path, **output_options_ffmpeg).overwrite_output()
        else: 
            output_stream = ffmpeg.output(video_stream_final, output_path, vcodec='copy', an=None).overwrite_output()

        output_stream.run(capture_stdout=True, capture_stderr=True)
        log_msg_suffix = f" (Chunk: start={video_start_time}s, dur={video_duration}s)" if video_start_time is not None else ""
        print(f"Video assembly for '{os.path.basename(output_path)}'{log_msg_suffix} completed.")

    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        print(f"ERROR: ffmpeg video assembly failed for '{os.path.basename(video_path)}'. FFmpeg stderr:\n{stderr_msg[:1000]}")
        raise RuntimeError(f"Video assembly failed. FFmpeg stderr:\n{stderr_msg}") from e
    except Exception as e: 
        print(f"ERROR: An unexpected error occurred during video assembly of '{os.path.basename(video_path)}': {e}"); 
        raise

def escape_ffmpeg_path(path): 
    path = path.replace('\\', '/'); path = path.replace(':', '\\:').replace("'", "'\\''"); return path

def add_subtitles(video_path, srt_path, output_path): 
    output_stream = None
    try:
        style_options = "FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=1"
        abs_srt_path = os.path.abspath(srt_path); escaped_srt_path_for_filter = escape_ffmpeg_path(abs_srt_path)
        vf_option = f"subtitles='{escaped_srt_path_for_filter}':force_style='{style_options}'"
        output_stream = (ffmpeg.input(video_path).output(output_path, vf=vf_option, acodec='copy').overwrite_output())
        output_stream.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print("ERROR: ffmpeg subtitle adding failed."); stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        cmd_args_str = "N/A"
        if output_stream and hasattr(output_stream, 'get_args'):
            try: cmd_args_str = ' '.join(output_stream.get_args())
            except: pass
        print(f"----- FFmpeg arguments for add_subtitles -----\nffmpeg {cmd_args_str}\n--------------------------")
        raise RuntimeError(f"Subtitle adding failed. FFmpeg stderr:\n{stderr}") from e
    except Exception as e: print(f"ERROR: An unexpected error occurred during subtitle adding: {e}"); raise

def _attempt_yt_dlp_download(ydl_opts, url): 
    if yt_dlp is None: raise ImportError("yt-dlp library is not available but is required for this function.")
    downloaded_files_from_hook = []
    current_ydl_opts = ydl_opts.copy(); current_ydl_opts['verbose'] = True; current_ydl_opts['quiet'] = False; current_ydl_opts['no_warnings'] = False
    outtmpl_setting_from_opts = current_ydl_opts.get('outtmpl')
    if isinstance(outtmpl_setting_from_opts, dict) and 'default' in outtmpl_setting_from_opts and isinstance(outtmpl_setting_from_opts['default'], str): effective_outtmpl_str = outtmpl_setting_from_opts['default']
    elif isinstance(outtmpl_setting_from_opts, str): effective_outtmpl_str = outtmpl_setting_from_opts
    else: effective_outtmpl_str = os.path.join('.', '%(title)s.%(id)s.%(ext)s')
    current_output_dir = os.path.dirname(effective_outtmpl_str)
    if not os.path.isabs(current_output_dir) and current_output_dir : current_output_dir = os.path.abspath(current_output_dir)
    elif not current_output_dir : current_output_dir = os.path.abspath('.')
    def ytdlp_hook(d):
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
    current_ydl_opts['progress_hooks'] = [ytdlp_hook]; current_ydl_opts['ignoreerrors'] = current_ydl_opts.get('ignoreerrors', True)
    info_dict_after_download = None; main_video_file_path = None
    with yt_dlp.YoutubeDL(current_ydl_opts) as ydl:
        try:
            info_dict_after_download = ydl.extract_info(url, download=not current_ydl_opts.get('skip_download', False))
            if not current_ydl_opts.get('skip_download', False) and info_dict_after_download:
                final_path_from_info = info_dict_after_download.get('filepath') or info_dict_after_download.get('_filename')
                if final_path_from_info and os.path.exists(final_path_from_info): main_video_file_path = final_path_from_info
                else:
                    try:
                        expected_final_path = ydl.prepare_filename(info_dict_after_download)
                        if expected_final_path and os.path.exists(expected_final_path): main_video_file_path = expected_final_path
                    except Exception as e_prep_fn:
                        video_title = info_dict_after_download.get('title', 'video'); video_id = info_dict_after_download.get('id', 'unknown_id'); file_extension = info_dict_after_download.get('ext', 'mp4')
                        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '_', '-')).rstrip(); manually_constructed_filename = f"{safe_title}.{video_id}.{file_extension}"
                        manually_constructed_path = os.path.join(current_output_dir, manually_constructed_filename)
                        if os.path.exists(manually_constructed_path): main_video_file_path = manually_constructed_path
                if not main_video_file_path:
                    video_exts = ('.mp4', '.mkv', '.webm', '.flv', '.avi', '.mov')
                    for f_path_hook in downloaded_files_from_hook:
                        if f_path_hook.lower().endswith(video_exts):
                            try:
                                probe_test = ffmpeg.probe(f_path_hook)
                                if any(s.get('codec_type') == 'video' for s in probe_test.get('streams',[])): main_video_file_path = f_path_hook; break
                            except Exception: pass
                if main_video_file_path and main_video_file_path not in downloaded_files_from_hook: downloaded_files_from_hook.append(main_video_file_path)
            elif current_ydl_opts.get('skip_download', False): print(f"  yt-dlp: Video download was skipped.")
        except yt_dlp.utils.DownloadError as e_dl_err: print(f"  yt-dlp: DownloadError occurred: {e_dl_err}")
        except Exception as e_generic: print(f"  yt-dlp: Generic exception during extract_info for URL '{url}': {e_generic}\n{traceback.format_exc()}")
    return main_video_file_path, info_dict_after_download, downloaded_files_from_hook

def _get_subtitle_path(info_dict, lang_code_to_find, video_filepath_no_ext, downloaded_files_list, output_dir): 
    if info_dict and 'requested_subtitles' in info_dict and info_dict['requested_subtitles']:
        base_lang_to_find = lang_code_to_find.split('-')[0]
        for actual_lang_key, sub_info in info_dict['requested_subtitles'].items():
            if actual_lang_key.startswith(base_lang_to_find):
                 sub_filepath = sub_info.get('filepath')
                 if sub_filepath and os.path.exists(sub_filepath) and os.path.getsize(sub_filepath) > 0 : return sub_filepath, sub_info.get('ext', 'vtt')
    possible_exts = ['vtt', 'srt']
    for f_path in downloaded_files_list:
        f_name_lower = os.path.basename(f_path).lower(); video_base_name_lower_for_sub = ""
        if video_filepath_no_ext: video_base_name_lower_for_sub = os.path.basename(video_filepath_no_ext).lower()
        elif info_dict and info_dict.get('title') and info_dict.get('id'):
             sanitized_title = "".join(c for c in info_dict['title'] if c.isalnum() or c in (' ', '_', '-')).rstrip(); video_base_name_lower_for_sub = f"{sanitized_title}.{info_dict['id']}".lower()
        for ext_iter in possible_exts:
            name_prefix_matches = (video_base_name_lower_for_sub and f_name_lower.startswith(video_base_name_lower_for_sub)) or (not video_base_name_lower_for_sub)
            lang_tag_matches = (f".{lang_code_to_find}.{ext_iter}" in f_name_lower or f".{lang_code_to_find.split('-')[0]}.{ext_iter}" in f_name_lower)
            if name_prefix_matches and lang_tag_matches and os.path.exists(f_path) and os.path.getsize(f_path) > 0: return f_path, ext_iter
    if video_filepath_no_ext:
        for ext_iter in possible_exts:
            expected_paths_to_check = [ f"{video_filepath_no_ext}.{lang_code_to_find}.{ext_iter}", f"{video_filepath_no_ext}.{lang_code_to_find.split('-')[0]}.{ext_iter}" ]
            for expected_path in set(expected_paths_to_check):
                if os.path.exists(expected_path) and os.path.getsize(expected_path) > 0: return expected_path, ext_iter
    return None, None

def download_youtube_video(url, output_dir, quality='1080p', preferred_sub_lang='ru', fallback_sub_lang='en'): 
    if yt_dlp is None: raise ImportError("yt-dlp library not found. Please install it using: pip install yt-dlp")
    video_filename_final = None; subtitle_filename_final = None; actual_sub_lang_found = None
    
    unique_sub_langs = []
    if preferred_sub_lang: unique_sub_langs.append(preferred_sub_lang)
    if fallback_sub_lang and fallback_sub_lang not in unique_sub_langs: unique_sub_langs.append(fallback_sub_lang)
    
    # Исправление UnboundLocalError
    temp_langs_to_add = []
    for lang in list(unique_sub_langs): # Итерируемся по копии, чтобы изменять оригинал
        if '-' in lang:
            base_lang = lang.split('-')[0] # base_lang определяется здесь
            if base_lang not in unique_sub_langs and base_lang not in temp_langs_to_add:
                temp_langs_to_add.append(base_lang)
    unique_sub_langs.extend(temp_langs_to_add)
    # Конец исправления

    outtmpl_str = os.path.join(output_dir, '%(title)s.%(id)s.%(ext)s')
    base_ydl_opts = {
        'format': f'bestvideo[height<={quality[:-1]}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality[:-1]}][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': outtmpl_str, 'noplaylist': True, 'merge_output_format': 'mp4',
        'writesubtitles': True, 'writeautomaticsub': True, 'subtitlesformat': 'vtt/srt',
        'ignoreerrors': True, 'no_color': True, 'cachedir': False,
    }
    all_downloaded_files_session = []; current_ydl_opts_dl = base_ydl_opts.copy()
    if unique_sub_langs: current_ydl_opts_dl['subtitleslangs'] = [f"{lang}*" for lang in unique_sub_langs]
    else: current_ydl_opts_dl['writesubtitles'] = False; current_ydl_opts_dl['writeautomaticsub'] = False
    info_dict_main_dl = None
    try:
        video_file_dl, info_dict_main_dl, downloaded_dl_files = _attempt_yt_dlp_download(current_ydl_opts_dl, url)
        all_downloaded_files_session.extend(x for x in downloaded_dl_files if x not in all_downloaded_files_session)
        if video_file_dl and os.path.exists(video_file_dl):
            video_filename_final = video_file_dl; video_filepath_no_ext = os.path.splitext(video_filename_final)[0] if video_filename_final else None
            search_order_langs = []
            if preferred_sub_lang: search_order_langs.append(preferred_sub_lang)
            if fallback_sub_lang and fallback_sub_lang != preferred_sub_lang : search_order_langs.append(fallback_sub_lang)
            for lang_code_iter in search_order_langs:
                sub_path, sub_ext = _get_subtitle_path(info_dict_main_dl, lang_code_iter, video_filepath_no_ext, all_downloaded_files_session, output_dir)
                if sub_path: subtitle_filename_final = sub_path; actual_sub_lang_found = lang_code_iter.split('-')[0]; break
    except yt_dlp.utils.DownloadError as e_dl: print(f"yt-dlp DownloadError during main download attempt for '{url}': {e_dl}")
    except Exception as e_dl_other: print(f"Unexpected error during main download attempt for '{url}': {e_dl_other}\n{traceback.format_exc()}")
    if not video_filename_final or not os.path.exists(video_filename_final):
        video_exts_check = ('.mp4', '.mkv', '.webm', '.flv', '.mov', '.avi')
        for f_path_check in all_downloaded_files_session:
             if f_path_check.lower().endswith(video_exts_check):
                 try:
                     probe_info_check = ffmpeg.probe(f_path_check)
                     if any(s.get('codec_type') == 'video' for s in probe_info_check.get('streams',[])): video_filename_final = f_path_check; break
                 except Exception: pass
        if not video_filename_final: raise RuntimeError(f"Failed to download or identify the YouTube video file after all attempts. URL: {url}")
    if not subtitle_filename_final: print(f"Subtitles for requested languages ('{preferred_sub_lang}', '{fallback_sub_lang}') were not found/downloaded after all attempts.")
    return video_filename_final, subtitle_filename_final, actual_sub_lang_found