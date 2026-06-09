import ffmpeg
import subprocess
import os
import shutil
import traceback
import soundfile as sf 

# (Оставляем все вспомогательные функции без изменений, добавляем generate_silence)

def generate_silence(output_path, duration, samplerate=24000):
    try:
        (ffmpeg.input('anullsrc', format='lavfi', r=samplerate, channel_layout='mono')
            .output(output_path, t=duration, acodec='pcm_s16le')
            .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return True
    except: return False

def get_video_duration(video_path):
    # (Код из предыдущего ответа)
    if not os.path.exists(video_path): return 0.0
    try:
        probe = ffmpeg.probe(video_path)
        return float(probe['format']['duration'])
    except: return 0.0

def extract_audio(video_path, output_audio_path, sample_rate=16000, start_time_seconds=None, duration_seconds=None):
    # (Код из предыдущего ответа)
    input_opts = {}
    if start_time_seconds is not None: input_opts['ss'] = str(start_time_seconds)
    output_opts = {'ar': str(sample_rate), 'ac': 1, 'format': 'wav'}
    if duration_seconds is not None: output_opts['t'] = str(duration_seconds)
    
    try:
        (ffmpeg.input(video_path, **input_opts)
         .output(output_audio_path, **output_opts)
         .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return output_audio_path
    except Exception as e:
        print(f"Extract audio error: {e}")
        return None

def merge_audio_segments(segment_files, output_path, log_prefix=None):
    # (Упрощенная версия merge для экономии места, логика та же - concat demuxer)
    if not segment_files: return None
    list_path = output_path + ".txt"
    with open(list_path, 'w', encoding='utf-8') as f:
        for seg in segment_files: f.write(f"file '{seg}'\n")
    
    try:
        (ffmpeg.input(list_path, format='concat', safe=0)
         .output(output_path, acodec='copy')
         .overwrite_output().run(capture_stdout=True, capture_stderr=True))
    except: return None
    finally:
        if os.path.exists(list_path): os.remove(list_path)
    return output_path

def mix_and_replace_audio(video_path, original_audio, dubbed_audio, output_path, video_start_time=None, video_duration=None, orig_vol=0.1, dub_vol=1.0):
    # (Код из предыдущего ответа)
    vid_opts = {}
    if video_start_time: vid_opts['ss'] = str(video_start_time)
    if video_duration: vid_opts['t'] = str(video_duration)
    
    vid = ffmpeg.input(video_path, **vid_opts).video
    aud_orig = ffmpeg.input(original_audio).filter('volume', orig_vol)
    aud_dub = ffmpeg.input(dubbed_audio).filter('volume', dub_vol)
    
    # Amix
    mix = ffmpeg.filter([aud_orig, aud_dub], 'amix', inputs=2, duration='first')
    
    try:
        (ffmpeg.output(vid, mix, output_path, vcodec='libx264', acodec='aac', strict='experimental')
         .overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return True
    except Exception as e:
        print(f"Mix error: {e}")
        return False

def embed_soft_subtitles(video_path, srt_orig, srt_translated, output_path):
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-i', srt_orig,
            '-i', srt_translated,
            '-map', '0:v', '-map', '0:a', '-map', '1', '-map', '2',
            '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text',
            '-metadata:s:s:0', 'language=eng',
            '-metadata:s:s:1', 'language=rus',
            '-y', output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Embed subtitles error: {e}")
        return False

def concatenate_video_chunks(chunks, output_path, temp_dir):
    # (Код из предыдущего ответа)
    list_path = os.path.join(temp_dir, "concat.txt")
    with open(list_path, 'w') as f:
        for c in chunks: f.write(f"file '{c}'\n")
        
    try:
        (ffmpeg.input(list_path, format='concat', safe=0)
         .output(output_path, c='copy')
         .overwrite_output().run(capture_stdout=True, capture_stderr=True))
    except: pass
