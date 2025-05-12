import os
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import traceback
import subprocess # Для проверки команд
import shutil # Для проверки команд

# Импортируем наши модули
import video_processor
import transcriber
import translator
import voice_cloner

# --- Вспомогательные функции ---

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

# --- GUI Part ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Dubbing Tool (WhisperX+XTTS)")
        self.root.geometry("600x600") # Оставляем увеличенную высоту

        self.video_path = tk.StringVar()
        self.srt_path = tk.StringVar()
        self.operation_options = {
            "Translate & Dub (Voice Clone)": "translate_dub",
            "Add Subtitles Only": "add_subtitles"
        }
        self.operation = tk.StringVar(value=list(self.operation_options.keys())[0])

        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", padding=6)
        style.configure("TEntry", padding=6)
        style.configure("TRadiobutton", padding=6)

        # --- Фреймы для ввода ---
        video_frame = ttk.LabelFrame(root, text="Video File", padding=(10, 5))
        video_frame.pack(padx=10, pady=5, fill="x")
        ttk.Entry(video_frame, textvariable=self.video_path, width=60, state="readonly").pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(video_frame, text="Browse", command=self.browse_video).pack(side=tk.LEFT)

        srt_frame = ttk.LabelFrame(root, text="SRT File (Only for 'Add Subtitles' op)", padding=(10, 5))
        srt_frame.pack(padx=10, pady=5, fill="x")
        ttk.Entry(srt_frame, textvariable=self.srt_path, width=60, state="readonly").pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(srt_frame, text="Browse", command=self.browse_srt).pack(side=tk.LEFT)

        ops_frame = ttk.LabelFrame(root, text="Operation", padding=(10,5))
        ops_frame.pack(padx=10, pady=5, fill="x")
        for op_text in self.operation_options.keys():
            ttk.Radiobutton(ops_frame, text=op_text, variable=self.operation, value=op_text).pack(anchor=tk.W)

        # --- Кнопка и Прогресс-бар ---
        self.process_button = ttk.Button(root, text="Process Video", command=self.start_processing_thread)
        self.process_button.pack(pady=5)

        self.progress_frame = ttk.Frame(root, padding=(10, 0))
        self.progress_label = ttk.Label(self.progress_frame, text="Progress:", width=10)
        self.progress_label.pack(side=tk.LEFT, padx=(0,5))
        # --- ИЗМЕНЕНИЕ: Определенный прогресс-бар ---
        self.progressbar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', maximum=100)
        self.progressbar.pack(side=tk.LEFT, fill="x", expand=True)
        self.progress_frame.pack_forget() # Скрываем изначально

        # --- Статус ---
        status_frame = ttk.LabelFrame(root, text="Status / Log", padding=(10,5))
        status_frame.pack(padx=10, pady=5, fill="both", expand=True)
        self.status_text = tk.Text(status_frame, height=10, wrap=tk.WORD, state="disabled", bg="#f0f0f0")
        self.status_text.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text['yscrollcommand'] = scrollbar.set

    # --- Методы GUI ---
    def _update_status(self, message, append=True):
        def update_gui():
            # ... (код без изменений) ...
            self.status_text.config(state="normal")
            if append: self.status_text.insert(tk.END, str(message) + "\n")
            else: self.status_text.delete("1.0", tk.END); self.status_text.insert("1.0", str(message) + "\n")
            self.status_text.config(state="disabled")
            self.status_text.see(tk.END)
        self.root.after(0, update_gui)

    # --- ИЗМЕНЕНИЕ: Метод для обновления прогресс-бара ---
    def _update_progress(self, value):
        """Обновляет значение прогресс-бара (0-100)."""
        clamped_value = max(0, min(100, value)) # Ограничиваем значение от 0 до 100
        def update_gui():
            self.progressbar['value'] = clamped_value
        # Используем after для потокобезопасности
        self.root.after(0, update_gui)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm *.flv")])
        if path: self.video_path.set(path); self._update_status(f"Video selected: {path}", append=False)

    def browse_srt(self):
        path = filedialog.askopenfilename(filetypes=[("SRT Files", "*.srt")])
        if path: self.srt_path.set(path); self._update_status(f"SRT selected: {path}", append=True)

    # --- Обработка в потоке ---
    def start_processing_thread(self):
        video_file = self.video_path.get()
        srt_file = self.srt_path.get()
        op_key = self.operation.get()
        op_choice = self.operation_options.get(op_key)

        if not video_file: messagebox.showerror("Error", "Please select a video file."); return
        if op_choice == "add_subtitles" and not srt_file: messagebox.showerror("Error", "Please select an SRT file."); return

        self.process_button.config(state="disabled")
        self._update_status(f"Starting operation: {op_key}...", append=False)
        self.progress_frame.pack(padx=10, pady=2, fill="x") # Показываем прогресс-бар
        # self.progressbar.start(10) # Больше не используем indeterminate
        self._update_progress(0) # Сбрасываем прогресс

        thread = threading.Thread(target=self.actual_processing, args=(video_file, srt_file, op_choice, op_key), daemon=True)
        thread.start()

    def actual_processing(self, video_file, srt_file, op_choice, op_key):
        current_temp_dir = create_temp_dir()
        self._update_status(f"Using temporary directory: {current_temp_dir}")
        output_path = None
        success = False
        # --- ИЗМЕНЕНИЕ: Определяем шаги для прогресс-бара ---
        total_steps_dub = 5 # Для операции дубляжа
        total_steps_subs = 1 # Для операции добавления субтитров
        current_step = 0

        try:
            if op_choice == "translate_dub":
                # Шаг 1: Извлечение аудио (Примерно 10% времени)
                current_step = 1
                self._update_status(f"Step {current_step}/{total_steps_dub}: Extracting audio...")
                self._update_progress(current_step / total_steps_dub * 90) # Обновляем до 18% (оставляем запас на последний шаг)
                temp_audio_path = video_processor.extract_audio(video_file, os.path.join(current_temp_dir, "original_audio.wav"))

                # Шаг 2: Транскрибация и диаризация (Самый долгий шаг, дадим ему больше веса ~50%)
                current_step = 2
                self._update_status(f"Step {current_step}/{total_steps_dub}: Transcribing & Diarizing (WhisperX)... (can take time)")
                # Здесь можно было бы использовать неопределенный прогресс, если бы знали как...
                # Пока просто ставим прогресс на начало этого шага
                self._update_progress(current_step / total_steps_dub * 90) # ~36%
                transcriber.load_stt_diarization_models()
                aligned_segments = transcriber.transcribe_and_diarize(temp_audio_path)
                if not aligned_segments: raise ValueError("Transcription/Diarization failed.")

                # Шаг 3: Перевод (Относительно быстрый, ~10%)
                current_step = 3
                self._update_status(f"Step {current_step}/{total_steps_dub}: Translating text...")
                self._update_progress(current_step / total_steps_dub * 90) # ~54%
                translator.load_translator_model()
                translated_segments = translator.translate_segments(aligned_segments)

                # Шаг 4: Синтез и клонирование речи (Может быть долгим, ~20%)
                current_step = 4
                self._update_status(f"Step {current_step}/{total_steps_dub}: Synthesizing & Cloning Voice (XTTS-v2)... (can take time)")
                self._update_progress(current_step / total_steps_dub * 90) # ~72%
                voice_cloner.load_tts_model()
                final_audio_path = voice_cloner.synthesize_speech_segments(translated_segments, temp_audio_path, current_temp_dir)

                # Шаг 5: Сборка видео (Обычно быстро, ~10%)
                current_step = 5
                self._update_status(f"Step {current_step}/{total_steps_dub}: Assembling final video...")
                self._update_progress(current_step / total_steps_dub * 90) # ~90%
                input_dir = os.path.dirname(video_file)
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                output_path = os.path.join(input_dir, f"{video_name}_dubbed_ru.mp4")
                video_processor.replace_audio(video_file, final_audio_path, output_path)
                self._update_progress(100) # Завершаем

            elif op_choice == "add_subtitles":
                self._update_status("Adding subtitles...")
                self._update_progress(50) # Пример, можно сделать 0 -> 100
                input_dir = os.path.dirname(video_file)
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                output_path = os.path.join(input_dir, f"{video_name}_with_subs.mp4")
                video_processor.add_subtitles(video_file, srt_file, output_path)
                self._update_progress(100)

            # Проверка результата
            if output_path and os.path.exists(output_path):
                self._update_status(f"✅ Operation '{op_key}' finished! Output:\n{output_path}")
                messagebox.showinfo("Success", f"Video processed successfully!\nOutput: {output_path}")
                success = True
            else:
                 self._update_status(f"⚠️ Operation '{op_key}' finished, but output file was not found or not returned.")

        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"❌ Error during '{op_key}': {e}\n--- Traceback ---\n{tb_str}-----------------"
            self._update_status(error_message)
            messagebox.showerror("Error", f"An error occurred during '{op_key}':\n{e}\n\n(See status window for details)")
            print(f"--- ERROR during '{op_key}' ---"); print(tb_str); print("----------------------------")
        finally:
            self.root.after(0, self.stop_and_hide_progressbar)
            cleanup_temp_dir(current_temp_dir)
            self.root.after(0, self.enable_process_button)
            if not success:
                 self.root.after(0, lambda: self._update_status(f"Operation '{op_key}' failed.", append=True))

    def stop_and_hide_progressbar(self):
        # self.progressbar.stop() # Больше не нужно для determinate
        self.progressbar['value'] = 0 # Сбрасываем значение
        self.progress_frame.pack_forget()

    def enable_process_button(self):
        self.process_button.config(state="normal")

# --- Проверка зависимостей и точка входа ---
if __name__ == "__main__":
    print("Checking dependencies...")
    # Проверяем доступность FFmpeg
    ffmpeg_ok, ffmpeg_err = video_processor.check_command_availability('ffmpeg')
    ffprobe_ok, ffprobe_err = video_processor.check_command_availability('ffprobe')

    errors = []
    if not ffmpeg_ok: errors.append(f"FFmpeg: {ffmpeg_err}")
    if not ffprobe_ok: errors.append(f"ffprobe: {ffprobe_err}")

    if errors:
        msg = "Required command(s) not found or failed:\n\n" + "\n".join(errors)
        msg += "\n\nPlease ensure FFmpeg (with ffprobe) is installed and in PATH."
        messagebox.showerror("Dependency Error", msg)
        print(msg)
        exit(1)
    else:
        print("FFmpeg/ffprobe check passed.")

    # Проверка PyTorch/CUDA
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available(): print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed. Required for WhisperX and Coqui TTS.")
        messagebox.showerror("Dependency Error", "PyTorch not found. Please install PyTorch.")
        exit(1)
    except Exception as e:
        print(f"Error during PyTorch/CUDA check: {e}")
        messagebox.showwarning("Warning", f"Could not fully check PyTorch/CUDA: {e}")

    # Запускаем GUI
    root = tk.Tk()
    app = App(root)
    root.mainloop()