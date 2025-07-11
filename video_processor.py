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

MIN_AUDIO_DURATION_SECONDS = 0.01

def check_command_availability(command): 
    try:
        command_path = shutil.which(command)
        if command_path is None: return False, f"'{command}' not found in PATH."
        startupinfo = None
        if os.name == 'nt': 
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE 
        
        version_flags = ['--version', '-version', '-V']
        success = False; error_output_combined = ""
        for flag in version_flags:
            process_result = None
            try:
                process_result = subprocess.run(
                    [command_path, flag], capture_output=True, text=True, check=False,
                    startupinfo=startupinfo, timeout=5, encoding='utf-8', errors='replace'
                )
            except FileNotFoundError: 
                return False, f"'{command}' found by shutil.which but execution failed (FileNotFound)."
            except subprocess.TimeoutExpired:
                print(f"Warning: '{command} {flag}' timed out.")
                error_output_combined += f"Timeout with flag {flag}.\n"; continue
            except Exception as e_run: 
                print(f"Warning: Error executing '{command} {flag}': {e_run}")
                error_output_combined += f"Error with flag {flag}: {e_run}\n"; continue
            if process_result: 
                output_cmd_text = process_result.stdout; error_output_cmd_text = process_result.stderr
                combined_output_for_check = output_cmd_text + error_output_cmd_text 
                if process_result.returncode == 0 or \
                   any(v_kw in combined_output_for_check.lower() for v_kw in ["version", "copyright", "ffmpeg", "ffprobe", "libavutil"]):
                     success = True; break 
                else:
                    if error_output_cmd_text: error_output_combined += f"Stderr with flag {flag}: {error_output_cmd_text.strip()}\n"
                    if output_cmd_text: error_output_combined += f"Stdout with flag {flag}: {output_cmd_text.strip()}\n"
        if success: return True, f"'{command}' found at: {command_path}"
        else:
             final_err_msg = error_output_combined if error_output_combined else f"Command execution failed or produced no recognizable output with flags: {', '.join(version_flags)}"
             return False, f"'{command}' found but version check failed. Details:\n{final_err_msg.strip()}"
    except Exception as e_outer: return False, f"Error checking command '{command}': {e_outer}"

def get_video_duration(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found for duration check: {video_path}"); return 0.0 
    probe_data = None; error_stderr = None
    try: probe_data = ffmpeg.probe(video_path)
    except ffmpeg.Error as e_probe: error_stderr = e_probe.stderr.decode('utf8', 'ignore') if e_probe.stderr else "N/A"
    except Exception as e_generic_probe: print(f"Unexpected error probing video duration for {os.path.basename(video_path)}: {e_generic_probe}"); return 0.0
    if probe_data:
        duration_str = probe_data.get('format', {}).get('duration')
        if duration_str is not None:
            try: return float(duration_str)
            except ValueError: print(f"Warning: Duration string '{duration_str}' from format is not a valid float for {os.path.basename(video_path)}."); duration_str = None 
        if duration_str is None: 
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'video' and stream.get('duration'):
                    duration_str_stream = stream.get('duration')
                    try: return float(duration_str_stream)
                    except ValueError: print(f"Warning: Duration string '{duration_str_stream}' from video stream is not a valid float for {os.path.basename(video_path)}.")
            print(f"Warning: Could not determine duration from format or video streams for {os.path.basename(video_path)}."); return 0.0 
    else: 
        if error_stderr: print(f"Error probing video duration for {os.path.basename(video_path)}. FFmpeg stderr: {error_stderr}")
        else: print(f"Failed to get probe data for {os.path.basename(video_path)}."); return 0.0
    return 0.0 # Добавлено для полноты, хотя предыдущие return должны сработать

def extract_audio(video_path, output_audio_path, sample_rate=16000, start_time_seconds=None, duration_seconds=None): 
    if not os.path.exists(video_path): print(f"ERROR: Video file not found for audio extraction: {video_path}"); return None 
    input_options_dict = {}; output_options_ffmpeg_dict = {}
    if start_time_seconds is not None: input_options_dict['ss'] = str(start_time_seconds)
    output_options_ffmpeg_dict = {'format': 'wav', 'acodec': 'pcm_s16le', 'ac': 1, 'ar': str(sample_rate)}
    if duration_seconds is not None: output_options_ffmpeg_dict['t'] = str(duration_seconds)
    process_ran_ok = False; ffmpeg_stderr_output = ""
    try:
        process = (ffmpeg.input(video_path, **input_options_dict)
            .output(output_audio_path, **output_options_ffmpeg_dict)
            .overwrite_output().run_async(pipe_stdout=True, pipe_stderr=True))
        _out, err = process.communicate(); ffmpeg_stderr_output = err.decode('utf-8', errors='replace') if err else ""
        if process.returncode == 0:
            if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0: process_ran_ok = True
            else: ffmpeg_stderr_output += "\nOutput file missing or empty after successful ffmpeg run."
    except ffmpeg.Error as e_ffmpeg_run: ffmpeg_stderr_output = e_ffmpeg_run.stderr.decode('utf-8', errors='replace') if e_ffmpeg_run.stderr else str(e_ffmpeg_run)
    except Exception as e_generic_run: ffmpeg_stderr_output = f"Unexpected error during ffmpeg run: {e_generic_run}"
    if process_ran_ok: return output_audio_path
    else:
        print(f"ERROR: ffmpeg audio extraction failed for '{os.path.basename(video_path)}'. Start: {start_time_seconds}, Dur: {duration_seconds}")
        print(f"FFmpeg stderr:\n{ffmpeg_stderr_output}")
        if os.path.exists(output_audio_path):
            try: os.remove(output_audio_path)
            except OSError: pass # Игнорируем ошибку удаления временного файла
        return None

def merge_audio_segments(segment_files, output_path, target_samplerate=24000, target_channels=1, target_codec='pcm_s16le', log_prefix="  (Merge) "):
    if log_prefix: print(f"{log_prefix}Attempting to merge {len(segment_files)} audio segments into {os.path.basename(output_path)}")
    if not segment_files:
        if log_prefix: print(f"{log_prefix}Warning: No segment files provided. Creating empty WAV.")
        empty_created = False
        try:
            (ffmpeg.input('anullsrc', format='lavfi', r=target_samplerate, channel_layout=f"{target_channels}c" if target_channels > 1 else "mono")
             .output(output_path, acodec=target_codec, t=MIN_AUDIO_DURATION_SECONDS, ar=target_samplerate, ac=target_channels)
             .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            if os.path.exists(output_path): empty_created = True
        except Exception as e_empty_create:
            if log_prefix: print(f"{log_prefix}  Error creating empty WAV: {e_empty_create}")
        return output_path if empty_created else None

    temp_dir_norm = os.path.join(os.path.dirname(output_path), "normalized_for_merge_" + os.path.splitext(os.path.basename(output_path))[0])
    if not os.path.isdir(temp_dir_norm): os.makedirs(temp_dir_norm, exist_ok=True)
    
    list_file_path = os.path.join(temp_dir_norm, "concat_list.txt")
    valid_norm_files = []; norm_failed_crit = False

    if log_prefix: print(f"{log_prefix}Normalizing {len(segment_files)} segments...")
    for idx, seg_orig_path in enumerate(segment_files):
        base_seg_name = os.path.basename(seg_orig_path)
        norm_out_path = os.path.join(temp_dir_norm, f"norm_{idx:04d}_{base_seg_name}")
        norm_ok = False; ff_norm_err = ""
        if not os.path.exists(seg_orig_path) or os.path.getsize(seg_orig_path) == 0:
            try:
                (ffmpeg.input('anullsrc', format='lavfi', r=target_samplerate, channel_layout="mono")
                 .output(norm_out_path, t=MIN_AUDIO_DURATION_SECONDS, acodec=target_codec, ar=target_samplerate, ac=target_channels)
                 .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                if os.path.exists(norm_out_path): norm_ok = True
            except Exception as e_norm_silent: ff_norm_err = f"Error creating norm silence for {base_seg_name}: {e_norm_silent}"
            if not norm_ok: norm_failed_crit = True; print(f"{log_prefix}Critical error creating normalized silence: {ff_norm_err}"); break
        else:
            try:
                proc_norm = (ffmpeg.input(seg_orig_path)
                                .output(norm_out_path, ar=target_samplerate, ac=target_channels, acodec=target_codec)
                                .overwrite_output().run_async(pipe_stderr=True))
                _, err_n = proc_norm.communicate(); ff_norm_err = err_n.decode('utf-8', errors='replace') if err_n else ""
                if proc_norm.returncode == 0 and os.path.exists(norm_out_path) and os.path.getsize(norm_out_path) > 0: norm_ok = True
            except Exception as e_gen_norm: ff_norm_err = f"Generic error normalizing {base_seg_name}: {e_gen_norm}"
            if not norm_ok:
                if log_prefix: print(f"{log_prefix}  Error normalizing {base_seg_name}. FFmpeg: {ff_norm_err}. Skipping.")
                if os.path.exists(norm_out_path): 
                    # Вместо try-catch для os.remove
                    if os.path.isfile(norm_out_path): # Проверяем, что это файл
                        try: os.remove(norm_out_path) # Оставляем try для самой операции удаления
                        except OSError as e_rem_norm: print(f"Warning: Could not remove norm_out_path {norm_out_path}: {e_rem_norm}")
                continue
        if os.path.exists(norm_out_path) and os.path.getsize(norm_out_path) > 0: valid_norm_files.append(norm_out_path)
        elif log_prefix: print(f"{log_prefix}  Normalized segment {os.path.basename(norm_out_path)} missing/empty. Skipping.")
    
    if norm_failed_crit:
        if log_prefix: print(f"{log_prefix}Critical error during segment normalization. Merge aborted.")
        if os.path.exists(temp_dir_norm): shutil.rmtree(temp_dir_norm, ignore_errors=True)
        return None

    if log_prefix: print(f"{log_prefix}Normalization complete. {len(valid_norm_files)} valid segments for merge.")
    if not valid_norm_files:
        if log_prefix: print(f"{log_prefix}Warning: No valid segments after normalization. Creating empty WAV.")
        empty_created_post_norm = False
        try:
            (ffmpeg.input('anullsrc', format='lavfi', r=target_samplerate, channel_layout="mono")
             .output(output_path, t=MIN_AUDIO_DURATION_SECONDS, acodec=target_codec, ar=target_samplerate, ac=target_channels)
             .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            if os.path.exists(output_path): empty_created_post_norm = True
        except Exception as e_empty_pn:
            if log_prefix: print(f"{log_prefix}  Error creating empty WAV post-norm: {e_empty_pn}")
        if os.path.exists(temp_dir_norm): shutil.rmtree(temp_dir_norm, ignore_errors=True)
        return output_path if empty_created_post_norm else None

    write_list_ok = False; f_list_io = None
    try:
        f_list_io = open(list_file_path, "w", encoding='utf-8')
        for norm_file in valid_norm_files:
            f_list_io.write(f"file '{os.path.normpath(os.path.abspath(norm_file)).replace(chr(92), '/')}'\n")
        write_list_ok = True
    except IOError as e_w_list:
        if log_prefix: print(f"{log_prefix}ERROR: Could not write concat list {list_file_path}: {e_w_list}")
    finally:
        if f_list_io: f_list_io.close()
        
    if not write_list_ok:
        if os.path.exists(temp_dir_norm): shutil.rmtree(temp_dir_norm, ignore_errors=True)
        return None

    merge_ok = False; ff_merge_err = ""
    try:
        if log_prefix: print(f"{log_prefix}Merging {len(valid_norm_files)} normalized segments...")
        proc_merge = (ffmpeg.input(list_file_path, format='concat', safe=0)
                        .output(output_path, acodec='copy').overwrite_output().run_async(pipe_stderr=True))
        _, err_m = proc_merge.communicate(); ff_merge_err = err_m.decode('utf-8', errors='replace') if err_m else ""
        if proc_merge.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0: merge_ok = True
    except Exception as e_gen_merge: ff_merge_err = f"Generic error during merge: {e_gen_merge}"
    
    if merge_ok:
        if log_prefix: print(f"{log_prefix}Audio segments merged to {os.path.basename(output_path)}")
    else:
        if log_prefix: print(f"{log_prefix}CRITICAL ERROR: ffmpeg audio merge failed. FFmpeg: {ff_merge_err}")
        fallback_empty_created_merge = False
        try:
            (ffmpeg.input('anullsrc', format='lavfi', r=target_samplerate, channel_layout="mono")
                .output(output_path, t=MIN_AUDIO_DURATION_SECONDS, acodec=target_codec, ar=target_samplerate, ac=target_channels)
                .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            if os.path.exists(output_path): fallback_empty_created_merge = True
        except Exception as e_fb_merge:
             if log_prefix: print(f"{log_prefix}  Failed to create fallback empty WAV after merge error: {e_fb_merge}")
        if os.path.exists(temp_dir_norm): shutil.rmtree(temp_dir_norm, ignore_errors=True)
        return output_path if fallback_empty_created_merge else None
        
    if os.path.exists(temp_dir_norm): shutil.rmtree(temp_dir_norm, ignore_errors=True)
    return output_path

def concatenate_video_chunks(chunk_files, output_path, temp_dir):
    if not chunk_files: print("No video chunks to concatenate."); return None
    if len(chunk_files) == 1:
        try: shutil.copy(chunk_files[0], output_path); print(f"Single video chunk copied to {output_path}"); return output_path
        except Exception as e: print(f"Error copying single video chunk: {e}"); return None
    
    list_fn = os.path.join(temp_dir, "concat_video_list.txt")
    f_concat_list = None; write_ok = False
    try:
        f_concat_list = open(list_fn, "w", encoding='utf-8')
        for cf in chunk_files: f_concat_list.write(f"file '{os.path.normpath(os.path.abspath(cf)).replace(chr(92), '/')}'\n")
        write_ok = True
    except IOError as e: print(f"ERR writing concat list {list_fn}: {e}")
    finally:
        if f_concat_list: f_concat_list.close()
    if not write_ok: return None

    print(f"Concatenating {len(chunk_files)} video chunks to {os.path.basename(output_path)}")
    is_mp4 = any(c.lower().endswith('.mp4') for c in chunk_files)
    bsf_opt = {'bsf:a': 'aac_adtstoasc'} if is_mp4 else {}
    
    concat_success = False; ffmpeg_error_details = ""
    # Попытка 1: Копирование кодеков
    try:
        process1 = (ffmpeg.input(list_fn, format='concat', safe=0, auto_convert=1)
                   .output(output_path, vcodec='copy', acodec='copy', **bsf_opt)
                   .overwrite_output().run_async(pipe_stderr=True))
        _, err1 = process1.communicate()
        ffmpeg_error_details = err1.decode('utf8', 'ignore') if err1 else ""
        if process1.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            concat_success = True
            print(f"Video chunks concatenated (codec copy) to {output_path}")
    except ffmpeg.Error as e1_ff: ffmpeg_error_details = e1_ff.stderr.decode('utf8', 'ignore') if e1_ff.stderr else str(e1_ff)
    except Exception as e1_gen: ffmpeg_error_details = f"Unexpected error (copy codecs): {e1_gen}"

    if not concat_success:
        print(f"WARN: Concat video (codec copy) failed. Retrying with re-encode. FFmpeg (attempt 1): {ffmpeg_error_details[:500]}...")
        ffmpeg_error_details = "" # Сбрасываем для второй попытки
        try: # Попытка 2: Перекодирование
            process2 = (ffmpeg.input(list_fn, format='concat', safe=0, auto_convert=1)
                       .output(output_path, vcodec='libx264', preset='medium', crf=23, acodec='aac', audio_bitrate='192k')
                       .overwrite_output().run_async(pipe_stderr=True))
            _, err2 = process2.communicate()
            ffmpeg_error_details = err2.decode('utf8', 'ignore') if err2 else ""
            if process2.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                concat_success = True
                print(f"Video chunks re-encoded and concatenated to {output_path}")
        except ffmpeg.Error as e2_ff: ffmpeg_error_details = e2_ff.stderr.decode('utf8', 'ignore') if e2_ff.stderr else str(e2_ff)
        except Exception as e2_gen: ffmpeg_error_details = f"Unexpected error (re-encode): {e2_gen}"

    if os.path.exists(list_fn):
        try: os.remove(list_fn)
        except OSError as e_rm_list: print(f"Warning: Could not remove concat list {list_fn}: {e_rm_list}")
        
    if concat_success: return output_path
    else: print(f"ERR: All video concatenation attempts failed. Last FFmpeg error: {ffmpeg_error_details[:500]}"); return None


def mix_and_replace_audio(video_path, original_audio_path, dubbed_audio_path, output_path, 
                          original_volume=0.1, dubbed_volume=1.0,
                          video_start_time=None, video_duration=None): 
    if not os.path.exists(video_path): print(f"ERR: Input video for mix_replace not found: {video_path}"); return False
    in_vid_opts = {}; 
    if video_start_time is not None: in_vid_opts['ss'] = str(video_start_time)
    in_vid = ffmpeg.input(video_path, **in_vid_opts); vid_stream = in_vid['v']
    if video_duration is not None: vid_stream = ffmpeg.trim(vid_stream, duration=video_duration)
    has_orig_aud = original_audio_path and os.path.exists(original_audio_path) and os.path.getsize(original_audio_path) > 0
    has_dub_aud = dubbed_audio_path and os.path.exists(dubbed_audio_path) and os.path.getsize(dubbed_audio_path) > 0
    aud_out_stream = None
    if has_dub_aud and has_orig_aud:
        in_orig_aud = ffmpeg.input(original_audio_path); in_dub_aud = ffmpeg.input(dubbed_audio_path)
        filt_orig = ffmpeg.filter(in_orig_aud, 'volume', str(original_volume))
        filt_dub = ffmpeg.filter(in_dub_aud, 'volume', str(dubbed_volume))
        aud_out_stream = ffmpeg.filter([filt_orig, filt_dub], 'amix', inputs=2, duration='first', dropout_transition=2)
    elif has_dub_aud: aud_out_stream = ffmpeg.filter(ffmpeg.input(dubbed_audio_path), 'volume', str(dubbed_volume))
    elif has_orig_aud: aud_out_stream = ffmpeg.filter(ffmpeg.input(original_audio_path), 'volume', str(original_volume))
    out_streams_ff = [vid_stream]; 
    if aud_out_stream: out_streams_ff.append(aud_out_stream)
    out_opts_ff = {'vcodec': 'copy', 'acodec': 'aac', 'audio_bitrate': '192k', 'strict': '-2'}
    if not aud_out_stream: out_opts_ff = {'vcodec': 'copy', 'an': None}
    mix_ok = False; ff_mix_err = ""
    try:
        proc_mix = (ffmpeg.output(*out_streams_ff, output_path, **out_opts_ff)
                        .overwrite_output().run_async(pipe_stderr=True))
        _, err_mx = proc_mix.communicate(); ff_mix_err = err_mx.decode('utf-8', errors='replace') if err_mx else ""
        if proc_mix.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0: mix_ok = True
    except Exception as e_gen_mix: ff_mix_err = f"Unexpected error during video assembly: {e_gen_mix}" # Оставляем try-except для run_async
    if mix_ok:
        log_suffix = f" (Chunk: start={video_start_time}s, dur={video_duration}s)" if video_start_time is not None else ""
        print(f"Video assembly for '{os.path.basename(output_path)}'{log_suffix} completed."); return True
    else:
        print(f"ERR: ffmpeg video assembly failed for '{os.path.basename(video_path)}'. FFmpeg: {ff_mix_err[:1000]}")
        if os.path.exists(output_path): 
            try: os.remove(output_path)
            except OSError: pass # Игнорируем ошибку удаления
        return False

def escape_ffmpeg_path(path): 
    path = path.replace('\\', '/'); path = path.replace(':', '\\:'); path = path.replace("'", "'\\''"); return path

def add_subtitles(video_path, srt_path, output_path): 
    if not os.path.exists(video_path) or not os.path.exists(srt_path):
        print(f"ERR (add_subtitles): Video or SRT not found. Vid: {video_path}, SRT: {srt_path}"); return False
    style_opts = "FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=1"
    abs_srt = os.path.abspath(srt_path); esc_srt = escape_ffmpeg_path(abs_srt)
    vf_opt = f"subtitles='{esc_srt}':force_style='{style_opts}'"
    sub_add_ok = False; ff_sub_err = ""; cmd_obj = None
    try:
        cmd_obj = (ffmpeg.input(video_path).output(output_path, vf=vf_opt, acodec='copy').overwrite_output())
        proc_sub = cmd_obj.run_async(pipe_stderr=True)
        _, err_s = proc_sub.communicate(); ff_sub_err = err_s.decode('utf-8', errors='replace') if err_s else ""
        if proc_sub.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0: sub_add_ok = True
    except Exception as e_gen_sub:
        ff_sub_err = f"Unexpected error adding subtitles: {e_gen_sub}"
        if cmd_obj and hasattr(cmd_obj, 'get_args'):
            try: ff_sub_err += f"\nFFmpeg cmd (approx): ffmpeg {' '.join(cmd_obj.get_args())}"
            except Exception: pass
    if sub_add_ok: print(f"Subtitles added to {output_path}"); return True
    else:
        print(f"ERR: ffmpeg subtitle add failed. FFmpeg: {ff_sub_err}")
        if os.path.exists(output_path): 
            try: os.remove(output_path)
            except OSError: pass
        return False

def _attempt_yt_dlp_download(ydl_opts_input, url, format_selector): 
    if yt_dlp is None: return None, None, [], "yt-dlp library not available"
    downloaded_files_hook = []; current_ydl_opts = ydl_opts_input.copy()
    current_ydl_opts['format'] = format_selector 
    current_ydl_opts['verbose'] = True; current_ydl_opts['quiet'] = False; current_ydl_opts['no_warnings'] = False
    def ytdlp_progress_hook_local(d):
        status_h = d.get('status'); filename_h = d.get('filename', d.get('info_dict', {}).get('_filename'))
        if status_h == 'downloading':
            prog_s = d.get('_percent_str', d.get('_total_bytes_str', '')); speed_s = d.get('_speed_str', ''); eta_s = d.get('_eta_str', '')
            if filename_h: print(f"  yt-dlp: Downloading {os.path.basename(filename_h)} ({format_selector}): {prog_s} ({speed_s}, ETA: {eta_s})")
        elif status_h == 'finished':
            if filename_h and filename_h not in downloaded_files_hook: downloaded_files_hook.append(filename_h)
            req_subs_h = d.get('info_dict', {}).get('requested_subtitles')
            if req_subs_h:
                for _, sub_info_h in req_subs_h.items():
                    sub_fp_h = sub_info_h.get('filepath')
                    if sub_fp_h and os.path.exists(sub_fp_h) and not sub_fp_h.endswith('.part') and sub_fp_h not in downloaded_files_hook:
                        downloaded_files_hook.append(sub_fp_h)
        elif status_h == 'error':
            if filename_h: print(f"  yt-dlp (hook): Error for {os.path.basename(filename_h)} with format {format_selector}.")
    current_ydl_opts['progress_hooks'] = [ytdlp_progress_hook_local]
    info_dict_res = None; main_vid_path_res = None; error_msg_res = ""; ydl_inst = None
    try:
        ydl_inst = yt_dlp.YoutubeDL(current_ydl_opts)
        info_dict_res = ydl_inst.extract_info(url, download=not current_ydl_opts.get('skip_download', False))
    except yt_dlp.utils.DownloadError as e_dl_yt: error_msg_res = f"yt-dlp DownloadError (format: {format_selector}): {e_dl_yt}"
    except yt_dlp.utils.ExtractorError as e_extr_yt: error_msg_res = f"yt-dlp ExtractorError (format: {format_selector}): {e_extr_yt}"
    except Exception as e_gen_yt: error_msg_res = f"yt-dlp Generic (format: {format_selector}): {e_gen_yt}\n{traceback.format_exc()}"
    if info_dict_res:
        path_from_idict = info_dict_res.get('filepath') or info_dict_res.get('_filename')
        if path_from_idict and os.path.exists(path_from_idict) and not path_from_idict.endswith('.part'): main_vid_path_res = path_from_idict
        if not main_vid_path_res and ydl_inst:
            try:
                exp_path = ydl_inst.prepare_filename(info_dict_res)
                if exp_path and os.path.exists(exp_path) and not exp_path.endswith('.part'): main_vid_path_res = exp_path
            except Exception as e_prep:
                if not error_msg_res: error_msg_res = f"Error preparing filename: {e_prep}"
        if not main_vid_path_res:
            vid_exts = ('.mp4', '.mkv', '.webm', '.flv', '.avi', '.mov')
            for fph_item in downloaded_files_hook:
                if fph_item.lower().endswith(vid_exts) and os.path.exists(fph_item) and not fph_item.endswith('.part'):
                    try:
                        pr_test = ffmpeg.probe(fph_item)
                        if any(s.get('codec_type') == 'video' for s in pr_test.get('streams',[])): main_vid_path_res = fph_item; break 
                    except Exception: pass 
        if main_vid_path_res and main_vid_path_res not in downloaded_files_hook: downloaded_files_hook.append(main_vid_path_res)
        if not main_vid_path_res and current_ydl_opts.get('skip_download', False):
            if not error_msg_res: error_msg_res = "Video download skipped."
        elif not main_vid_path_res and not error_msg_res: error_msg_res = "Main video file not identified after download."
    elif not error_msg_res: error_msg_res = "yt-dlp returned no info and no error."
    final_dl_files = [f for f in downloaded_files_hook if os.path.exists(f) and not f.endswith('.part')]
    return main_vid_path_res, info_dict_res, final_dl_files, error_msg_res

def _get_subtitle_path(info_dict_subs, lang_code, vid_fp_no_ext, dl_files_list, _out_dir_unused): 
    if info_dict_subs and 'requested_subtitles' in info_dict_subs and info_dict_subs['requested_subtitles']:
        base_lang = lang_code.split('-')[0]
        for actual_lang, sub_info in info_dict_subs['requested_subtitles'].items():
            if isinstance(sub_info, dict) and actual_lang.startswith(base_lang):
                 sub_fp = sub_info.get('filepath')
                 if sub_fp and os.path.exists(sub_fp) and os.path.getsize(sub_fp) > 0 : return sub_fp, sub_info.get('ext', 'vtt')
    sub_exts = ['vtt', 'srt']; vid_base_name_search = ""
    if vid_fp_no_ext: vid_base_name_search = os.path.basename(vid_fp_no_ext).lower()
    elif info_dict_subs and info_dict_subs.get('title') and info_dict_subs.get('id'):
         s_title = "".join(c for c in info_dict_subs['title'] if c.isalnum() or c in (' ', '_', '-')).rstrip()
         vid_base_name_search = f"{s_title}.{info_dict_subs['id']}".lower()
    for fp_sub in dl_files_list:
        if not os.path.exists(fp_sub) or os.path.getsize(fp_sub) == 0: continue
        fn_low_sub = os.path.basename(fp_sub).lower()
        for ext_s in sub_exts:
            tag_short = f".{lang_code.split('-')[0]}.{ext_s}"; tag_full = f".{lang_code}.{ext_s}"
            match_name = True if not vid_base_name_search else fn_low_sub.startswith(vid_base_name_search)
            if match_name and (tag_short in fn_low_sub or tag_full in fn_low_sub): return fp_sub, ext_s
    if vid_fp_no_ext:
        for ext_s_g in sub_exts:
            paths_check = [f"{vid_fp_no_ext}.{lang_code.split('-')[0]}.{ext_s_g}", f"{vid_fp_no_ext}.{lang_code}.{ext_s_g}"]
            for exp_path_g in set(paths_check):
                if os.path.exists(exp_path_g) and os.path.getsize(exp_path_g) > 0: return exp_path_g, ext_s_g
    return None, None

def download_youtube_video(url, output_dir, quality='1080p', preferred_sub_lang='ru', fallback_sub_lang='en'): 
    if yt_dlp is None: 
        print("ERROR: yt-dlp library not found. Cannot download YouTube video.")
        return None, None, None, "yt-dlp library not found."
    vid_fn_final = None; sub_fn_final = None; actual_sub_lang = None; err_msg_dl_final = ""
    unique_subs = []
    if preferred_sub_lang: unique_subs.append(preferred_sub_lang)
    if fallback_sub_lang and fallback_sub_lang not in unique_subs: unique_subs.append(fallback_sub_lang)
    temp_langs_add = []
    for lang_item in list(unique_subs):
        if '-' in lang_item:
            base_l = lang_item.split('-')[0]
            if base_l not in unique_subs and base_l not in temp_langs_add: temp_langs_add.append(base_l)
    unique_subs.extend(temp_langs_add)
    base_outtmpl_path = os.path.join(output_dir, '%(title)s.%(id)s') 
    ydl_opts_base = {
        'outtmpl': {'default': base_outtmpl_path + '.%(ext)s'}, 
        'noplaylist': True, 'merge_output_format': 'mp4',
        'writesubtitles': bool(unique_subs), 'writeautomaticsub': bool(unique_subs),
        'subtitlesformat': 'vtt/srt', 'ignoreerrors': True, 
        'no_color': True, 'cachedir': False,
    }
    if unique_subs: ydl_opts_base['subtitleslangs'] = [f"{lang}*" for lang in unique_subs]
    format_selectors_to_try = [
        f'bestvideo[height<={quality[:-1]}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={quality[:-1]}][ext=mp4]+bestaudio/best[height<={quality[:-1]}][ext=mp4]/best[ext=mp4]/best',
        'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 
        'bestvideo+bestaudio/best' 
    ]
    all_dl_files_session = []
    for fs_idx, format_sel in enumerate(format_selectors_to_try):
        print(f"Attempting YouTube download with format: {format_sel} (Attempt {fs_idx+1}/{len(format_selectors_to_try)})")
        vid_path_att, info_dict_att, dl_files_att, err_msg_att = _attempt_yt_dlp_download(ydl_opts_base.copy(), url, format_sel)
        for f_att in dl_files_att:
            if f_att not in all_dl_files_session: all_dl_files_session.append(f_att)
        if vid_path_att and os.path.exists(vid_path_att):
            vid_fn_final = vid_path_att; err_msg_dl_final = "" 
            vid_fp_no_ext_subs = os.path.splitext(vid_fn_final)[0]
            search_langs_subs = []
            if preferred_sub_lang: search_langs_subs.append(preferred_sub_lang)
            if fallback_sub_lang and fallback_sub_lang != preferred_sub_lang : search_langs_subs.append(fallback_sub_lang)
            for lang_c_iter in search_langs_subs:
                sub_path_f, _ = _get_subtitle_path(info_dict_att, lang_c_iter, vid_fp_no_ext_subs, all_dl_files_session, output_dir)
                if sub_path_f: sub_fn_final = sub_path_f; actual_sub_lang = lang_c_iter.split('-')[0]; break
            break 
        else: 
            err_msg_dl_final = err_msg_att 
            print(f"Download attempt {fs_idx+1} failed: {err_msg_dl_final}")
            if "yt-dlp library not available" in err_msg_dl_final: break 
    if not vid_fn_final:
        if not err_msg_dl_final: err_msg_dl_final = f"Failed to download YouTube video file after all attempts. URL: {url}"
    if not sub_fn_final and vid_fn_final : 
        subs_warn_msg = f"Subtitles for requested languages ({preferred_sub_lang}, {fallback_sub_lang}) not found/downloaded."
        if not err_msg_dl_final: err_msg_dl_final = subs_warn_msg
        else: err_msg_dl_final = f"{err_msg_dl_final}. {subs_warn_msg}"
        print(subs_warn_msg)
    return vid_fn_final, sub_fn_final, actual_sub_lang, err_msg_dl_final
