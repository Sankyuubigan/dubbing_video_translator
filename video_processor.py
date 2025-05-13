import ffmpeg
import os
import subprocess
import shutil
import time # Для замера времени

def check_command_availability(cmd):
    # ... (без изменений, как в предыдущем ответе) ...
    command_path = shutil.which(cmd)
    if not command_path:
        print(f"Error: Command '{cmd}' not found in PATH.")
        return False, f"Command '{cmd}' not found in PATH."
    print(f"Command '{cmd}' found at: {command_path}")
    try:
        result = subprocess.run([command_path, "-version"], capture_output=True, text=True, check=True, timeout=10, shell=(os.name == 'nt'))
        print(f"'{cmd} -version' executed successfully.")
        return True, None
    except FileNotFoundError:
        print(f"Error: Command '{cmd}' found by shutil.which but failed to execute (FileNotFoundError).")
        return False, f"Command '{cmd}' found but failed to execute (FileNotFoundError)."
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{cmd} -version' failed with exit code {e.returncode}.")
        return False, f"Command '{cmd} -version' failed with exit code {e.returncode}."
    except subprocess.TimeoutExpired:
        print(f"Error: Command '{cmd} -version' timed out.")
        return False, f"Command '{cmd} -version' timed out."
    except PermissionError:
         print(f"Error: Permission denied when trying to execute '{cmd}'.")
         return False, f"Permission denied when trying to execute '{cmd}'."
    except Exception as e:
         print(f"Error: An unexpected error occurred while checking '{cmd}': {e}")
         return False, f"An unexpected error occurred while checking '{cmd}': {e}"

def extract_audio(video_path, output_audio_path, sample_rate=16000):
    # ... (без изменений) ...
    print(f"Extracting audio to {output_audio_path} at {sample_rate} Hz")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ar=str(sample_rate))
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("Audio extraction successful.")
        return output_audio_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
        raise RuntimeError(f"FFmpeg error during audio extraction: {stderr}")
    except FileNotFoundError:
         raise FileNotFoundError("ffmpeg command not found. Ensure FFmpeg is installed and in PATH.")

def extract_speaker_clip(audio_path, start_time, end_time, output_clip_path):
    # ... (без изменений) ...
    print(f"Extracting speaker clip: {output_clip_path} from {start_time:.2f}s to {end_time:.2f}s")
    try:
        duration = end_time - start_time
        if duration <= 0:
            # XTTS может иногда справиться с очень короткими, но лучше избегать
            print(f"Warning: Speaker clip duration is very short or zero ({duration:.2f}s). This might affect cloning quality.")
            if duration <= 0.1: # Слишком короткий, чтобы быть полезным
                 raise ValueError("Speaker clip duration too short for extraction.")

        (
            ffmpeg
            .input(audio_path, ss=start_time, t=duration)
            .output(output_clip_path, acodec='pcm_s16le', ar='24000') # XTTS требует 24kHz
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("Speaker clip extracted.")
        return output_clip_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
        raise RuntimeError(f"FFmpeg error during speaker clip extraction: {stderr}")
    except FileNotFoundError:
         raise FileNotFoundError("ffmpeg command not found.")

def merge_audio_segments(segment_files, output_path):
    # ... (без изменений) ...
    print(f"Merging {len(segment_files)} audio segments into {output_path}")
    if not segment_files:
        raise ValueError("No audio segments provided to merge.")

    list_file_path = os.path.join(os.path.dirname(output_path), "concat_list.txt")
    with open(list_file_path, 'w', encoding='utf-8') as f:
        for wav_file in segment_files:
            escaped_path = wav_file.replace('\\', '/').replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    try:
        (
            ffmpeg
            .input(list_file_path, format='concat', safe=0)
            .output(output_path, acodec='aac', ar='24000')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("Audio segments merged successfully.")
        os.remove(list_file_path)
        return output_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
        try: os.remove(list_file_path)
        except OSError: pass
        raise RuntimeError(f"FFmpeg error during audio merge: {stderr}")
    except Exception as e:
         try: os.remove(list_file_path)
         except OSError: pass
         raise RuntimeError(f"Error during audio merge process: {e}")


def mix_and_replace_audio(video_path, original_audio_path_for_mixing, dubbed_audio_path, output_video_path, original_volume=0.1, dubbed_volume=0.9):
    """
    Микширует оригинальное аудио (с пониженной громкостью) с дублированным аудио
    и заменяет им аудиодорожку в оригинальном видео.
    """
    print(f"Mixing original audio (vol:{original_volume}) with dubbed audio (vol:{dubbed_volume})")
    mixed_audio_output_path = os.path.join(os.path.dirname(dubbed_audio_path), "mixed_final_audio.m4a") # Используем m4a/aac

    try:
        original_audio_stream = ffmpeg.input(original_audio_path_for_mixing).audio.filter('volume', volume=original_volume)
        dubbed_audio_stream = ffmpeg.input(dubbed_audio_path).audio.filter('volume', volume=dubbed_volume)

        (
            ffmpeg
            .filter([original_audio_stream, dubbed_audio_stream], 'amix', inputs=2, duration='first', dropout_transition=2)
            .output(mixed_audio_output_path, acodec='aac', ar='44100') # Стандартная частота для видео
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Audio mixed successfully to: {mixed_audio_output_path}")

        # Теперь заменяем аудио в видео на смикшированное
        print(f"Replacing audio in {video_path} with mixed audio")
        input_video = ffmpeg.input(video_path)
        mixed_audio_input = ffmpeg.input(mixed_audio_output_path) # Используем только что созданный файл
        (
            ffmpeg
            .output(input_video['v'], mixed_audio_input['a'], output_video_path, vcodec='copy', acodec='copy', shortest=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Final video with mixed audio created: {output_video_path}")
        return output_video_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
        raise RuntimeError(f"FFmpeg error during audio mixing/replacement: {stderr}")
    except FileNotFoundError:
         raise FileNotFoundError("ffmpeg command not found.")
    finally:
        # Удаляем промежуточный смикшированный аудиофайл
        if os.path.exists(mixed_audio_output_path):
            try:
                os.remove(mixed_audio_output_path)
                print(f"Removed intermediate mixed audio file: {mixed_audio_output_path}")
            except Exception as e_rem:
                print(f"Could not remove intermediate mixed audio file {mixed_audio_output_path}: {e_rem}")


def add_subtitles(video_path, srt_path, output_video_path):
    # ... (без изменений) ...
    print(f"Adding subtitles from {srt_path} to {video_path}")
    escaped_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_video_path,
                    vf=f"subtitles='{escaped_srt_path}'",
                    vcodec='libx264',
                    acodec='copy')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Subtitles added successfully. Output: {output_video_path}")
        return output_video_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
        if "subtitles filter requires libass" in stderr.lower() or "library not found" in stderr.lower():
             raise RuntimeError(f"FFmpeg error: libass library not found or FFmpeg not compiled with libass support. Subtitle filter requires libass. FFmpeg stderr: {stderr}")
        raise RuntimeError(f"FFmpeg error during subtitle burning: {stderr}")
    except FileNotFoundError:
         raise FileNotFoundError("ffmpeg command not found.")
