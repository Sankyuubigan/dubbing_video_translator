import ffmpeg
import subprocess
import os
import shutil

def check_command_availability(command):
    """Checks if a command is available and executable in the system's PATH."""
    try:
        command_path = shutil.which(command)
        if command_path is None:
            return False, f"'{command}' not found in PATH."

        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        version_flags = ['--version', '-version', '-V']
        success = False
        # output = "" # Не используется
        error = ""
        for flag in version_flags:
            try:
                process = subprocess.run(
                    [command_path, flag], capture_output=True, text=True, check=False,
                    startupinfo=startupinfo, timeout=5
                )
                output_cmd = process.stdout # Переименовали, чтобы не конфликтовать с output выше
                error = process.stderr
                if process.returncode == 0 or any(v in output_cmd.lower() for v in ["version", "copyright", "ffmpeg", "ffprobe"]):
                     # print(f"'{command} {flag}' executed successfully.") # Убираем этот лог
                     success = True
                     break
            except FileNotFoundError: return False, f"'{command}' found but failed to execute (FileNotFound)."
            except subprocess.TimeoutExpired: print(f"Warning: '{command} {flag}' timed out."); continue
            except Exception as e: print(f"Warning: Error executing '{command} {flag}': {e}"); continue

        if success: return True, f"'{command}' found at: {command_path}"
        else:
             last_err_msg = error if error else f"Command execution failed or produced no recognizable output with flags: {', '.join(version_flags)}"
             return False, f"'{command}' found but execution failed. Error: {last_err_msg}"

    except Exception as e:
        return False, f"Error checking command '{command}': {e}"


def extract_audio(video_path, output_audio_path, sample_rate=16000):
    """Extracts audio from video using ffmpeg-python, resamples, and saves as WAV."""
    # print(f"Extracting audio to {output_audio_path} at {sample_rate} Hz") # Этот лог уже есть в main.py
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, format='wav', acodec='pcm_s16le', ac=1, ar=str(sample_rate))
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True) # Оставляем захват, но не выводим если нет ошибки
        )
        # print("Audio extraction successful.") # Этот лог уже есть в main.py
        return output_audio_path
    except ffmpeg.Error as e:
        print("ERROR: ffmpeg audio extraction failed.")
        stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        # print(f"----- FFmpeg stderr -----\n{stderr}\n-------------------------") # stderr будет в общем логе ошибки
        raise RuntimeError(f"Audio extraction failed: {stderr}") from e
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during audio extraction: {e}")
        raise


def merge_audio_segments(segment_files, output_path):
    """Merges multiple audio files into one using ffmpeg concat demuxer."""
    print(f"Merging {len(segment_files)} audio segments into {os.path.basename(output_path)}") # Сокращаем путь
    if not segment_files:
        raise ValueError("No segment files provided for merging.")

    temp_dir = os.path.dirname(output_path)
    list_filename = os.path.join(temp_dir, "concat_list.txt")
    output_stream = None

    try:
        valid_segment_count = 0
        with open(list_filename, "w", encoding='utf-8') as f:
            for segment_file in segment_files:
                if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                     normalized_path = os.path.normpath(os.path.abspath(segment_file))
                     safe_path = normalized_path.replace('\\', '/')
                     f.write(f"file '{safe_path}'\n")
                     valid_segment_count += 1
                else:
                    print(f"Warning: Segment file not found or empty, skipping: {segment_file}")

        if valid_segment_count == 0:
             raise RuntimeError("Concat list file is empty or contains no valid segments.")
        # print(f"Generated concat list file {os.path.basename(list_filename)} with {valid_segment_count} valid segments.") # Можно убрать

        output_options = {'acodec': 'copy'}
        output_stream = (
            ffmpeg
            .input(list_filename, f='concat', safe=0)
            .output(output_path, **output_options)
            .overwrite_output()
        )

        # args = output_stream.get_args() # Не выводим команду, если все ок
        # print(f"Running FFmpeg command: ffmpeg {' '.join(args)}")
        _stdout, stderr = output_stream.run(capture_stdout=True, capture_stderr=True) # stdout не используем

        stderr_str = stderr.decode('utf-8', errors='replace')
        if "error" in stderr_str.lower() or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
             print("Warning: FFmpeg might have encountered errors during merging or output file is invalid.")
             print(f"----- FFmpeg stderr for merge_audio_segments -----\n{stderr_str}\n-------------------------")
             # raise RuntimeError(f"Audio merging possibly failed. FFmpeg stderr:\n{stderr_str}") # Не обязательно падать

        # print("Audio segments merged successfully.") # Этот лог уже есть в voice_cloner
    except ffmpeg.Error as e:
        print("ERROR: ffmpeg audio merging failed.")
        stderr_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
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
            try:
                os.remove(list_filename)
                # print(f"Removed temporary concat list file: {os.path.basename(list_filename)}") # Можно убрать
            except OSError as e_rem:
                print(f"Warning: Could not remove temporary concat list file {os.path.basename(list_filename)}: {e_rem}")
    return output_path


def mix_and_replace_audio(video_path, original_audio_path, dubbed_audio_path, output_path, original_volume=0.1, dubbed_volume=1.0):
    """
    Mixes original and dubbed audio, then replaces the original video's audio.
    """
    # print(f"Mixing audio: Original ('{os.path.basename(original_audio_path)}') Vol={original_volume}, Dubbed ('{os.path.basename(dubbed_audio_path)}') Vol={dubbed_volume}") # Лог в main.py
    # print(f"Replacing audio in '{os.path.basename(video_path)}' -> '{os.path.basename(output_path)}'") # Лог в main.py
    output_stream = None

    try:
        input_video = ffmpeg.input(video_path)
        input_original_audio = ffmpeg.input(original_audio_path)
        input_dubbed_audio = ffmpeg.input(dubbed_audio_path)
        video_stream = input_video['v']
        mixed_audio = ffmpeg.filter(
            [input_original_audio, input_dubbed_audio],
            'amix', inputs=2, duration='first', dropout_transition=2,
            weights=f"{original_volume} {dubbed_volume}"
        )
        output_options = {
            'vcodec': 'copy', 'acodec': 'aac',
            'audio_bitrate': '192k', 'strict': '-2'
        }
        output_stream = ffmpeg.output(
            video_stream, mixed_audio, output_path, **output_options
        ).overwrite_output()

        # args = output_stream.get_args() # Не выводим команду, если все ок
        # print(f"Running FFmpeg command: ffmpeg {' '.join(args)}")
        output_stream.run(capture_stdout=True, capture_stderr=True) # Оставляем захват, но не выводим если нет ошибки
        # print("Video assembly with mixed audio successful.") # Лог в main.py
    except ffmpeg.Error as e:
        print("ERROR: ffmpeg video assembly failed.")
        stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        cmd_args = output_stream.get_args() if output_stream else ["N/A"]
        print(f"----- FFmpeg arguments for mix_and_replace_audio -----\nffmpeg {' '.join(cmd_args)}\n--------------------------") # Выводим команду только при ошибке
        # print(f"----- FFmpeg stderr -----\n{stderr}\n-------------------------") # stderr будет в общем логе ошибки
        raise RuntimeError(f"Video assembly failed. FFmpeg stderr:\n{stderr}") from e
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during video assembly: {e}")
        raise


def add_subtitles(video_path, srt_path, output_path):
    """Adds subtitles to a video using ffmpeg-python."""
    # print(f"Adding subtitles '{os.path.basename(srt_path)}' to '{os.path.basename(video_path)}' -> '{os.path.basename(output_path)}'") # Лог в main.py
    output_stream = None

    try:
        style_options = "FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=1"
        safe_srt_path = os.path.normpath(srt_path).replace('\\', '/')
        vf_option = f"subtitles='{safe_srt_path}':force_style='{style_options}'"
        output_stream = (
            ffmpeg
            .input(video_path)
            .output(output_path, vf=vf_option, acodec='copy') # Видео будет перекодировано из-за vf
            .overwrite_output()
        )
        # args = output_stream.get_args() # Не выводим команду, если все ок
        # print(f"Running FFmpeg command: ffmpeg {' '.join(args)}")
        output_stream.run(capture_stdout=True, capture_stderr=True) # Оставляем захват, но не выводим если нет ошибки
        # print("Subtitles added successfully.") # Лог в main.py
    except ffmpeg.Error as e:
        print("ERROR: ffmpeg subtitle adding failed.")
        stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        cmd_args = output_stream.get_args() if output_stream else ["N/A"]
        print(f"----- FFmpeg arguments for add_subtitles -----\nffmpeg {' '.join(cmd_args)}\n--------------------------") # Выводим команду только при ошибке
        # print(f"----- FFmpeg stderr -----\n{stderr}\n-------------------------") # stderr будет в общем логе ошибки
        raise RuntimeError(f"Subtitle adding failed. FFmpeg stderr:\n{stderr}") from e
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during subtitle adding: {e}")
        raise