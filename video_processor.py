# Содержимое файла video_processor.py остается БЕЗ ИЗМЕНЕНИЙ
# по сравнению с предыдущим ответом, где мы его создали.
# Включает: check_command_availability, extract_audio, extract_speaker_clip,
# merge_audio_segments, replace_audio, add_subtitles.

import ffmpeg
import os
import subprocess
import shutil

def check_command_availability(cmd):
    """Проверяет, доступна ли команда в PATH и может ли она быть выполнена."""
    command_path = shutil.which(cmd)
    if not command_path:
        print(f"Error: Command '{cmd}' not found in PATH.")
        return False, f"Command '{cmd}' not found in PATH."
    print(f"Command '{cmd}' found at: {command_path}")
    try:
        # Используем shell=True на Windows для лучшей обработки PATH, но это менее безопасно
        # Используем check=True для вызова исключения при ошибке
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
    """Извлекает аудио из видео с заданной частотой дискретизации."""
    print(f"Extracting audio to {output_audio_path} at {sample_rate} Hz")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ar=str(sample_rate)) # Whisper(X) предпочитает 16kHz
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
    """Извлекает короткий клип из аудиофайла."""
    print(f"Extracting speaker clip: {output_clip_path} from {start_time:.2f}s to {end_time:.2f}s")
    try:
        duration = end_time - start_time
        if duration <= 0:
            raise ValueError("Duration must be positive for speaker clip extraction.")
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
    """
    Объединяет несколько аудиофайлов в один.
    segment_files: список путей к файлам WAV для объединения.
    """
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
            .output(output_path, acodec='aac', ar='24000') # Кодируем в AAC 24kHz для видео
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
    except FileNotFoundError:
         raise FileNotFoundError("ffmpeg command not found.")
    except Exception as e:
         try: os.remove(list_file_path)
         except OSError: pass
         raise RuntimeError(f"Error during audio merge process: {e}")

def replace_audio(video_path, new_audio_path, output_video_path):
    """Заменяет аудиодорожку в видео."""
    print(f"Replacing audio in {video_path} with {new_audio_path}")
    try:
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(new_audio_path)
        (
            ffmpeg
            .output(input_video['v'], input_audio['a'], output_video_path, vcodec='copy', acodec='aac', shortest=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Audio replaced successfully. Output: {output_video_path}")
        return output_video_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
        raise RuntimeError(f"FFmpeg error during audio replacement: {stderr}")
    except FileNotFoundError:
         raise FileNotFoundError("ffmpeg command not found.")


def add_subtitles(video_path, srt_path, output_video_path):
    """Добавляет (прожигает) субтитры в видео."""
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
