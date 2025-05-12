import os
import tempfile
import shutil
from deep_translator import GoogleTranslator
import pyttsx3
import speech_recognition as sr # Оставляем для AudioFile, но не для распознавания
import ffmpeg
import pysubs2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import traceback
import shutil
import whisper # <--- Добавлен импорт Whisper

# --- Настройки ---
OUTPUT_VIDEO_DIR = os.path.expanduser("~/Desktop/VideoTranslator")
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
# FFMPEG_PATH = "C:/path/to/ffmpeg/bin/ffmpeg.exe" # Раскомментируйте, если нужно

# Определяем модель Whisper для использования (например, "base", "small", "medium")
# "base" - самая быстрая и легкая, "medium" - более точная, но медленнее и тяжелее.
# "tiny.en", "base.en", "small.en", "medium.en" - только для английского (быстрее).
# Используем "base.en", так как исходное аудио у нас английское.
WHISPER_MODEL = "base.en"
whisper_model_cache = None # Для кэширования загруженной модели

# --- Вспомогательные функции ---

def create_temp_dir():
    return tempfile.mkdtemp(prefix="video_translator_")

def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Temporary directory {temp_dir} cleaned up.")
        except Exception as e:
            print(f"Error cleaning up temp directory {temp_dir}: {e}")

def split_text(text, max_length=4500):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# --- Функции обработки медиа с FFmpeg (без изменений) ---

def extract_audio_ffmpeg(video_path, output_audio_path):
    print(f"Extracting audio from {video_path} to {output_audio_path}")
    try:
        cmd = 'ffmpeg'
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ar='16000') # Whisper предпочитает 16kHz
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("Audio extraction successful.")
        return output_audio_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
        raise RuntimeError(f"FFmpeg error during audio extraction: {stderr}")
    except FileNotFoundError:
         raise FileNotFoundError("ffmpeg command not found.")

def replace_audio_ffmpeg(video_path, new_audio_path, output_video_path):
    print(f"Replacing audio in {video_path} with {new_audio_path}")
    try:
        cmd = 'ffmpeg'
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(new_audio_path)
        (
            ffmpeg
            .output(input_video['v'], input_audio['a'], output_video_path, vcodec='copy', acodec='aac', shortest=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Audio replaced successfully. Output: {output_video_path}")
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
        raise RuntimeError(f"FFmpeg error during audio replacement: {stderr}")
    except FileNotFoundError:
         raise FileNotFoundError("ffmpeg command not found.")

def add_subtitles_ffmpeg(video_path, srt_path, output_video_path):
    print(f"Adding subtitles from {srt_path} to {video_path}")
    escaped_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
    try:
        cmd = 'ffmpeg'
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
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
        if "subtitles filter requires libass" in stderr.lower() or "library not found" in stderr.lower():
             raise RuntimeError(f"FFmpeg error: libass library not found or FFmpeg not compiled with libass support. Subtitle filter requires libass. FFmpeg stderr: {stderr}")
        raise RuntimeError(f"FFmpeg error during subtitle burning: {stderr}")
    except FileNotFoundError:
         raise FileNotFoundError("ffmpeg command not found.")

# --- Функция распознавания речи с Whisper ---

def extract_text_with_whisper(audio_file_path):
    """Распознает текст из аудиофайла с помощью OpenAI Whisper."""
    global whisper_model_cache # Используем глобальный кэш модели
    print(f"Recognizing text from {audio_file_path} using Whisper model '{WHISPER_MODEL}'...")

    try:
        # Загружаем модель один раз и кэшируем ее
        if whisper_model_cache is None:
            print(f"Loading Whisper model '{WHISPER_MODEL}'... (This may take a while the first time)")
            # Добавим проверку доступности GPU, если есть torch с поддержкой CUDA
            device = None
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    print("CUDA is available. Using GPU for Whisper.")
                else:
                    print("CUDA not available. Using CPU for Whisper.")
            except ImportError:
                 print("PyTorch not found or no CUDA support. Using CPU for Whisper.")

            whisper_model_cache = whisper.load_model(WHISPER_MODEL, device=device)
            print("Whisper model loaded.")
        else:
            print("Using cached Whisper model.")

        # Распознаем речь
        # fp16=False может помочь на CPU или если с fp16 есть проблемы
        result = whisper_model_cache.transcribe(audio_file_path, fp16=False, language='en') # Указываем язык
        recognized_text = result["text"]
        print("Whisper transcription successful.")
        return recognized_text

    except Exception as e:
        # Указываем, что ошибка произошла именно в Whisper
        raise RuntimeError(f"Ошибка при распознавании текста с Whisper: {e}\n{traceback.format_exc()}")


# --- Функции перевода и синтеза (без изменений) ---

def translate_text(text, max_chars=4500):
    print("Translating text...")
    try:
        translator_obj = GoogleTranslator(source="en", target="ru")
        text_parts = split_text(text, max_chars)
        translated_parts = []
        for i, part in enumerate(text_parts):
             print(f"Translating part {i+1}/{len(text_parts)}")
             if part.strip():
                 translated_parts.append(translator_obj.translate(part))
        result = ' '.join(filter(None, translated_parts))
        print("Translation successful.")
        return result
    except Exception as e:
        raise RuntimeError(f"Ошибка при переводе текста: {e}")

def synthesize_speech(text, temp_speech_file_path):
    print(f"Synthesizing speech to {temp_speech_file_path}")
    try:
        engine_obj = pyttsx3.init()
        voices = engine_obj.getProperty('voices')
        ru_voice_id = None
        for voice in voices:
             if any(lang in check.lower() for lang in ['ru', 'russian'] for check in [voice.id, voice.name]):
                 ru_voice_id = voice.id
                 print(f"Found Russian voice: {voice.name} ({voice.id})")
                 break
        if ru_voice_id:
             engine_obj.setProperty('voice', ru_voice_id)
        else:
             print("Russian voice not found, using default.")

        engine_obj.setProperty('rate', 150)
        engine_obj.save_to_file(text, temp_speech_file_path)
        engine_obj.runAndWait()
        if not os.path.exists(temp_speech_file_path) or os.path.getsize(temp_speech_file_path) == 0:
            raise RuntimeError("Speech synthesis failed to create a non-empty audio file.")
        print("Speech synthesis successful.")
        return temp_speech_file_path
    except Exception as e:
        raise RuntimeError(f"Ошибка при синтезе речи: {e}")


# --- Основная логика обработки (теперь использует Whisper) ---

def translate_video_logic(file_video, temp_dir):
    output_path = None
    try:
        # 1. Извлечь аудио (с частотой 16kHz для Whisper)
        temp_audio_file = os.path.join(temp_dir, "extracted_audio.wav")
        extract_audio_ffmpeg(file_video, temp_audio_file) # FFmpeg извлекает аудио

        # 2. Распознать текст из аудио с помощью Whisper
        extracted_text = extract_text_with_whisper(temp_audio_file) # Используем Whisper
        if not extracted_text or not extracted_text.strip():
            raise ValueError("Whisper не смог извлечь текст из аудио.")
        print(f"Whisper extracted text: {extracted_text[:100]}...")

        # 3. Перевести текст
        translated_text = translate_text(extracted_text)
        if not translated_text or not translated_text.strip():
            raise ValueError("Переведенный текст пуст.")
        print(f"Translated text: {translated_text[:100]}...")

        # 4. Синтезировать речь
        temp_speech_path = os.path.join(temp_dir, "translated_speech.wav")
        synthesize_speech(translated_text, temp_speech_path)

        # 5. Заменить аудио в оригинальном видео
        video_name = os.path.splitext(os.path.basename(file_video))[0]
        output_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_translated.mp4")
        replace_audio_ffmpeg(file_video, temp_speech_path, output_path)

        return output_path
    except Exception as e:
        raise RuntimeError(f"Ошибка в translate_video_logic: {e}")


def add_subtitles_to_video_logic(file_video, file_srt, temp_dir):
    try:
        video_name = os.path.splitext(os.path.basename(file_video))[0]
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_with_subs.mp4")
        add_subtitles_ffmpeg(file_video, file_srt, output_video_path)
        return output_video_path
    except Exception as e:
        raise RuntimeError(f"Ошибка в add_subtitles_to_video_logic: {e}")


def translate_video_with_subtitles_logic(file_video, file_srt, temp_dir):
    intermediate_dir = os.path.join(temp_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    temp_translated_path = None
    try:
        # 1. Переводим видео (аудио) с Whisper
        print("Step 1: Translating video audio using Whisper...")
        video_name = os.path.splitext(os.path.basename(file_video))[0]
        temp_translated_path = os.path.join(intermediate_dir, f"{video_name}_translated_temp.mp4")

        temp_audio_file = os.path.join(temp_dir, "extracted_audio_trans.wav")
        extract_audio_ffmpeg(file_video, temp_audio_file)
        extracted_text = extract_text_with_whisper(temp_audio_file) # Используем Whisper
        if not extracted_text or not extracted_text.strip(): raise ValueError("Whisper extracted no text.")
        translated_text = translate_text(extracted_text)
        if not translated_text or not translated_text.strip(): raise ValueError("Translation is empty.")
        temp_speech_path = os.path.join(temp_dir, "translated_speech_trans.wav")
        synthesize_speech(translated_text, temp_speech_path)
        replace_audio_ffmpeg(file_video, temp_speech_path, temp_translated_path)
        print(f"Intermediate translated video saved to: {temp_translated_path}")

        # 2. Добавляем субтитры к переведенному видео
        print("Step 2: Adding subtitles to translated video...")
        final_output_name = f"{video_name}_translated_with_subs.mp4"
        final_output_path = os.path.join(OUTPUT_VIDEO_DIR, final_output_name)
        add_subtitles_ffmpeg(temp_translated_path, file_srt, final_output_path)

        return final_output_path

    except Exception as e:
        raise RuntimeError(f"Ошибка в translate_video_with_subtitles_logic: {e}")


# --- GUI Part (без изменений) ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor (FFmpeg + Whisper)") # Изменил заголовок
        self.root.geometry("600x450")

        self.video_path = tk.StringVar()
        self.srt_path = tk.StringVar()
        self.operation_options = {
            "Translate Audio Only": "Translate Only",
            "Add Subtitles Only": "Add Subtitles Only",
            "Translate Audio and Add Subtitles": "Translate and Add Subtitles"
        }
        self.operation = tk.StringVar(value=list(self.operation_options.keys())[0])

        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", padding=6)
        style.configure("TEntry", padding=6)
        style.configure("TRadiobutton", padding=6)

        video_frame = ttk.LabelFrame(root, text="Video File", padding=(10, 5))
        video_frame.pack(padx=10, pady=5, fill="x")
        ttk.Entry(video_frame, textvariable=self.video_path, width=60, state="readonly").pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(video_frame, text="Browse", command=self.browse_video).pack(side=tk.LEFT)

        srt_frame = ttk.LabelFrame(root, text="SRT File (Needed for Subtitle Ops)", padding=(10, 5))
        srt_frame.pack(padx=10, pady=5, fill="x")
        ttk.Entry(srt_frame, textvariable=self.srt_path, width=60, state="readonly").pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(srt_frame, text="Browse", command=self.browse_srt).pack(side=tk.LEFT)

        ops_frame = ttk.LabelFrame(root, text="Operation", padding=(10,5))
        ops_frame.pack(padx=10, pady=5, fill="x")
        for op_text in self.operation_options.keys():
            ttk.Radiobutton(ops_frame, text=op_text, variable=self.operation, value=op_text).pack(anchor=tk.W)

        self.process_button = ttk.Button(root, text="Process Video", command=self.start_processing_thread)
        self.process_button.pack(pady=10)

        status_frame = ttk.LabelFrame(root, text="Status", padding=(10,5))
        status_frame.pack(padx=10, pady=5, fill="both", expand=True)
        self.status_text = tk.Text(status_frame, height=8, wrap=tk.WORD, state="disabled", bg="#f0f0f0")
        self.status_text.pack(side=tk.LEFT, fill="both", expand=True)

        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text['yscrollcommand'] = scrollbar.set

    def _update_status(self, message, append=True):
        def update_gui():
            self.status_text.config(state="normal")
            if append:
                self.status_text.insert(tk.END, str(message) + "\n")
            else:
                self.status_text.delete("1.0", tk.END)
                self.status_text.insert("1.0", str(message) + "\n")
            self.status_text.config(state="disabled")
            self.status_text.see(tk.END)
        self.root.after(0, update_gui)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm *.flv")])
        if path:
            self.video_path.set(path)
            self._update_status(f"Video selected: {path}", append=False)

    def browse_srt(self):
        path = filedialog.askopenfilename(filetypes=[("SRT Files", "*.srt")])
        if path:
            self.srt_path.set(path)
            self._update_status(f"SRT selected: {path}", append=True)

    def start_processing_thread(self):
        video_file = self.video_path.get()
        srt_file = self.srt_path.get()
        op_key = self.operation.get()
        op_choice = self.operation_options.get(op_key)

        if not video_file:
            messagebox.showerror("Error", "Please select a video file.")
            return

        if "Subtitles" in op_key and not srt_file:
            messagebox.showerror("Error", "Please select an SRT file for this operation.")
            return

        self.process_button.config(state="disabled")
        self._update_status(f"Starting operation: {op_key}...", append=False)

        thread = threading.Thread(target=self.actual_processing, args=(video_file, srt_file, op_choice, op_key), daemon=True)
        thread.start()

    def actual_processing(self, video_file, srt_file, op_choice, op_key):
        global whisper_model_cache # Доступ к кэшу модели
        current_temp_dir = create_temp_dir()
        self._update_status(f"Using temporary directory: {current_temp_dir}")
        output_path = None
        success = False
        try:
            # Загружаем модель Whisper заранее, если она еще не загружена, чтобы поймать ошибки загрузки
            if whisper_model_cache is None and "Translate" in op_key: # Только если операция требует Whisper
                 self._update_status(f"Pre-loading Whisper model '{WHISPER_MODEL}'...")
                 # Вызываем функцию загрузки, чтобы она обработала ошибки и закэшировала модель
                 extract_text_with_whisper(None) # Передаем None, т.к. нам нужна только загрузка модели
                 self._update_status(f"Whisper model '{WHISPER_MODEL}' pre-loaded.")

            # Выполняем выбранную операцию
            if op_choice == "Translate Only":
                output_path = translate_video_logic(video_file, current_temp_dir)
            elif op_choice == "Add Subtitles Only":
                output_path = add_subtitles_to_video_logic(video_file, srt_file, current_temp_dir)
            elif op_choice == "Translate and Add Subtitles":
                 output_path = translate_video_with_subtitles_logic(video_file, srt_file, current_temp_dir)

            if output_path and os.path.exists(output_path):
                self._update_status(f"✅ Operation '{op_key}' finished! Output: {output_path}")
                messagebox.showinfo("Success", f"Video processed successfully!\nOutput: {output_path}")
                success = True
            else:
                 self._update_status(f"⚠️ Operation '{op_key}' finished, but output file was not found or not returned.")

        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"❌ Error during '{op_key}': {e}\n--- Traceback ---\n{tb_str}-----------------"
            self._update_status(error_message)
            messagebox.showerror("Error", f"An error occurred during '{op_key}':\n{e}\n\n(See status window for details)")
            print(f"--- ERROR during '{op_key}' ---")
            print(tb_str)
            print("----------------------------")
        finally:
            cleanup_temp_dir(current_temp_dir)
            self.root.after(0, self.enable_process_button)
            if not success:
                 self.root.after(0, lambda: self._update_status(f"Operation '{op_key}' failed.", append=True))

    def enable_process_button(self):
        self.process_button.config(state="normal")


# --- Проверка зависимостей и точка входа ---

def check_command_availability(cmd):
    """Проверяет, доступна ли команда в PATH и может ли она быть выполнена."""
    command_path = shutil.which(cmd)
    if not command_path:
        print(f"Error: Command '{cmd}' not found in PATH.")
        return False, f"Command '{cmd}' not found in PATH."
    print(f"Command '{cmd}' found at: {command_path}")
    try:
        result = subprocess.run([command_path, "-version"], capture_output=True, text=True, check=True, timeout=10)
        print(f"'{cmd} -version' executed successfully.")
        return True, None
    except Exception as e:
         print(f"Error: Failed to execute '{cmd} -version'. Error: {e}")
         return False, f"Failed to execute '{cmd} -version': {e}"

if __name__ == "__main__":
    # Проверяем доступность FFmpeg
    ffmpeg_available, ffmpeg_error = check_command_availability('ffmpeg')
    ffprobe_available, ffprobe_error = check_command_availability('ffprobe') # ffprobe может быть не нужен, но проверим

    error_messages = []
    if not ffmpeg_available:
        error_messages.append(f"FFmpeg: {ffmpeg_error}")
    # if not ffprobe_available: # Можно сделать опциональным
    #     messagebox.showwarning("FFprobe Warning", f"ffprobe: {ffprobe_error}")

    if error_messages:
        full_error_message = "Required command(s) not found or failed to execute:\n\n" + "\n".join(error_messages)
        full_error_message += "\n\nPlease ensure FFmpeg (and ffprobe) are installed correctly and added to your system's PATH."
        messagebox.showerror("Dependency Error", full_error_message)
        print(full_error_message)
        exit(1)

    # Запускаем GUI
    root = tk.Tk()
    app = App(root)
    root.mainloop()