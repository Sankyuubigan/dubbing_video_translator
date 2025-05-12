import os
import tempfile
import shutil
from deep_translator import GoogleTranslator
import pyttsx3
import speech_recognition as sr
import pysubs2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import ffmpeg  # Новая зависимость для замены moviepy

# Создание директории для хранения видео на рабочем столе
OUTPUT_VIDEO_DIR = os.path.expanduser("~/Desktop/VideoTranslator")
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

def create_temp_dir():
    """Создает уникальную временную директорию для одной операции."""
    return tempfile.mkdtemp(prefix="video_translator_")

def cleanup_temp_dir(temp_dir):
    """Удаляет временную директорию, если она существует."""
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Temporary directory {temp_dir} cleaned up.")
        except Exception as e:
            print(f"Error cleaning up temp directory {temp_dir}: {e}")

def split_text(text, max_length=4500):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def extract_audio_ffmpeg(video_path, output_audio_path):
    """Извлекает аудио из видео с помощью FFmpeg."""
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ar='44100')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return output_audio_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8') if e.stderr else "N/A"
        raise RuntimeError(f"FFmpeg error during audio extraction: {stderr}")

def extract_audio_text(video_path, temp_audio_file_path):
    """Извлекает текст из аудио, используя FFmpeg для извлечения аудио."""
    try:
        extract_audio_ffmpeg(video_path, temp_audio_file_path)
        r = sr.Recognizer()
        with sr.AudioFile(temp_audio_file_path) as source:
            audio = r.record(source)
        try:
            text = r.recognize_google(audio, language="en-US")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio, trying Sphinx")
            try:
                text = r.recognize_sphinx(audio)
            except sr.UnknownValueError:
                raise ValueError("Sphinx could not understand audio")
            except sr.RequestError as e:
                raise RuntimeError(f"Sphinx error; {e}")
        except sr.RequestError as e:
            raise RuntimeError(f"Could not request results from Google Speech Recognition service; {e}")
        return text
    except Exception as e:
        raise RuntimeError(f"Ошибка при извлечении/распознавании текста: {e}")

def translate_text(text, max_chars=4500):
    try:
        translator_obj = GoogleTranslator(source="en", target="ru")
        text_parts = split_text(text, max_chars)
        translated_parts = []
        for part in text_parts:
            if part.strip():
                translated_parts.append(translator_obj.translate(part))
        return ' '.join(filter(None, translated_parts))
    except Exception as e:
        raise RuntimeError(f"Ошибка при переводе текста: {e}")

def synthesize_speech(text, temp_speech_file_path):
    try:
        engine_obj = pyttsx3.init()
        voices = engine_obj.getProperty('voices')
        ru_voice_id = None
        for voice in voices:
            if "russian" in voice.name.lower() or "ru-ru" in voice.id.lower():
                ru_voice_id = voice.id
                break
        if ru_voice_id:
            engine_obj.setProperty('voice', ru_voice_id)
        else:
            print("Русский голос не найден, будет использован голос по умолчанию.")
        engine_obj.save_to_file(text, temp_speech_file_path)
        engine_obj.runAndWait()
        return temp_speech_file_path
    except Exception as e:
        raise RuntimeError(f"Ошибка при синтезе речи: {e}")

def replace_audio_ffmpeg(video_path, new_audio_path, output_video_path):
    """Заменяет аудиодорожку в видео с помощью FFmpeg."""
    try:
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(new_audio_path)
        (
            ffmpeg
            .output(input_video['v'], input_audio['a'], output_video_path, vcodec='copy', acodec='aac', shortest=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8') if e.stderr else "N/A"
        raise RuntimeError(f"FFmpeg error during audio replacement: {stderr}")

def add_subtitles_ffmpeg(video_path, srt_path, output_video_path):
    """Добавляет субтитры из SRT в видео с помощью FFmpeg."""
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_video_path, vf=f"subtitles='{srt_path}'", vcodec='libx264', acodec='copy')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8') if e.stderr else "N/A"
        raise RuntimeError(f"FFmpeg error during subtitle burning: {stderr} \nMake sure FFmpeg is compiled with libass support for SRT burning.")

def translate_video_logic(file_video, temp_dir):
    try:
        temp_audio_file = os.path.join(temp_dir, "extracted_audio.wav")
        extracted_text = extract_audio_text(file_video, temp_audio_file)
        if not extracted_text.strip():
            raise ValueError("Не удалось извлечь текст из аудио.")
        translated_text = translate_text(extracted_text)
        if not translated_text.strip():
            raise ValueError("Переведенный текст пуст.")
        temp_speech_path = os.path.join(temp_dir, "translated_speech.wav")
        synthesize_speech(translated_text, temp_speech_path)
        video_name = os.path.splitext(os.path.basename(file_video))[0]
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_translated.mp4")
        replace_audio_ffmpeg(file_video, temp_speech_path, output_video_path)
        return output_video_path
    except Exception as e:
        raise RuntimeError(f"Ошибка при переводе видео: {e}")

def add_subtitles_to_video_logic(file_video, file_srt, temp_dir):
    try:
        video_name = os.path.splitext(os.path.basename(file_video))[0]
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_with_subs.mp4")
        add_subtitles_ffmpeg(file_video, file_srt, output_video_path)
        return output_video_path
    except Exception as e:
        raise RuntimeError(f"Ошибка при добавлении субтитров (логика): {e}")

def translate_video_with_subtitles_logic(file_video, file_srt, temp_dir):
    translated_video_path = translate_video_logic(file_video, temp_dir)
    final_output_path = add_subtitles_to_video_logic(translated_video_path, file_srt, temp_dir)
    return final_output_path

# --- GUI Part ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor")
        self.root.geometry("600x450")
        self.video_path = tk.StringVar()
        self.srt_path = tk.StringVar()
        self.operation = tk.StringVar(value="Translate")
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", padding=6)
        style.configure("TEntry", padding=6)
        style.configure("TRadiobutton", padding=6)
        video_frame = ttk.LabelFrame(root, text="Video File", padding=(10, 5))
        video_frame.pack(padx=10, pady=5, fill="x")
        ttk.Entry(video_frame, textvariable=self.video_path, width=60, state="readonly").pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(video_frame, text="Browse", command=self.browse_video).pack(side=tk.LEFT)
        srt_frame = ttk.LabelFrame(root, text="SRT File (Optional)", padding=(10, 5))
        srt_frame.pack(padx=10, pady=5, fill="x")
        ttk.Entry(srt_frame, textvariable=self.srt_path, width=60, state="readonly").pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(srt_frame, text="Browse", command=self.browse_srt).pack(side=tk.LEFT)
        ops_frame = ttk.LabelFrame(root, text="Operation", padding=(10,5))
        ops_frame.pack(padx=10, pady=5, fill="x")
        operations = [
            "Translate Only",
            "Add Subtitles Only",
            "Translate and Add Subtitles"
        ]
        for op_text in operations:
            ttk.Radiobutton(ops_frame, text=op_text, variable=self.operation, value=op_text).pack(anchor=tk.W)
        self.process_button = ttk.Button(root, text="Process Video", command=self.start_processing_thread)
        self.process_button.pack(pady=10)
        status_frame = ttk.LabelFrame(root, text="Status", padding=(10,5))
        status_frame.pack(padx=10, pady=5, fill="both", expand=True)
        self.status_text = tk.Text(status_frame, height=8, wrap=tk.WORD, state="disabled")
        self.status_text.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self.status_text, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text['yscrollcommand'] = scrollbar.set

    def _update_status(self, message, append=False):
        self.status_text.config(state="normal")
        if append:
            self.status_text.insert(tk.END, message + "\n")
        else:
            self.status_text.delete("1.0", tk.END)
            self.status_text.insert("1.0", message + "\n")
        self.status_text.config(state="disabled")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")])
        if path:
            self.video_path.set(path)
            self._update_status(f"Video selected: {path}", append=True)

    def browse_srt(self):
        path = filedialog.askopenfilename(filetypes=[("SRT Files", "*.srt")])
        if path:
            self.srt_path.set(path)
            self._update_status(f"SRT selected: {path}", append=True)

    def start_processing_thread(self):
        video_file = self.video_path.get()
        srt_file = self.srt_path.get()
        op_choice = self.operation.get()
        if not video_file:
            messagebox.showerror("Error", "Please select a video file.")
            return
        if "Subtitles" in op_choice and not srt_file:
            messagebox.showerror("Error", "Please select an SRT file for subtitle operations.")
            return
        self.process_button.config(state="disabled")
        self._update_status("Processing started...", append=False)
        thread = threading.Thread(target=self.actual_processing, args=(video_file, srt_file, op_choice), daemon=True)
        thread.start()

    def actual_processing(self, video_file, srt_file, op_choice):
        current_temp_dir = create_temp_dir()
        self._update_status(f"Using temporary directory: {current_temp_dir}", append=True)
        output_path = None
        try:
            if op_choice == "Translate Only":
                self._update_status("Translating video...", append=True)
                output_path = translate_video_logic(video_file, current_temp_dir)
            elif op_choice == "Add Subtitles Only":
                self._update_status("Adding subtitles...", append=True)
                output_path = add_subtitles_to_video_logic(video_file, srt_file, current_temp_dir)
            elif op_choice == "Translate and Add Subtitles":
                self._update_status("Translating and adding subtitles...", append=True)
                output_path = translate_video_with_subtitles_logic(video_file, srt_file, current_temp_dir)
            if output_path:
                self._update_status(f"Processing finished! Output: {output_path}", append=True)
                messagebox.showinfo("Success", f"Video processed successfully!\nOutput: {output_path}")
            else:
                self._update_status("Processing finished, but no output path returned.", append=True)
        except Exception as e:
            error_message = f"Error during processing: {e}\nTraceback: {traceback.format_exc()}"
            self._update_status(error_message, append=True)
            messagebox.showerror("Error", f"An error occurred: {e}")
            import traceback
            print("--- ERROR TRACEBACK ---")
            traceback.print_exc()
            print("-----------------------")
        finally:
            cleanup_temp_dir(current_temp_dir)
            self.root.after(0, self.enable_process_button)

    def enable_process_button(self):
        self.process_button.config(state="normal")

if __name__ == "__main__":
    import traceback
    root = tk.Tk()
    app = App(root)
    root.mainloop()