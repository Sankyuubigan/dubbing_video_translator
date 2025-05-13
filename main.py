import os
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import traceback
import subprocess
import shutil
import time
import torch # Импортируем здесь для проверки и для обходного пути

# --- Попытка решить проблему с загрузкой модели Coqui TTS и PyTorch 2.1+ ---
# Это нужно сделать до первого импорта TTS или вызова функций, которые его используют
if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
    try:
        # Пытаемся импортировать класс, который вызывает проблему
        # Путь к классу может отличаться в зависимости от структуры TTS
        # Проверьте точный путь из вашего сообщения об ошибке
        from TTS.tts.configs.xtts_config import XttsConfig
        torch.serialization.add_safe_globals([XttsConfig])
        print("INFO: Added TTS.tts.configs.xtts_config.XttsConfig to torch safe globals.")
    except ImportError:
        print("WARNING: Could not import XttsConfig to add to torch safe globals. TTS model loading might still fail if it's a new model format.")
    except Exception as e_sg:
        print(f"WARNING: Error trying to add XttsConfig to torch safe globals: {e_sg}")
else:
    print("INFO: torch.serialization.add_safe_globals not available (likely older PyTorch or this specific issue doesn't apply).")

# --- Автоматическое принятие лицензии Coqui CPML ---
# Установите эту переменную окружения перед импортом TTS или voice_cloner
# Это может не сработать для всех версий TTS, но стоит попробовать.
os.environ["COQUI_AGREED_TO_CPML"] = "1"
print("INFO: Attempting to auto-agree to Coqui CPML via environment variable.")


# Импортируем наши модули ПОСЛЕ установки переменных окружения и обходных путей
import video_processor
import transcriber
import translator
import voice_cloner


# --- Вспомогательные функции ---
def create_temp_dir(): return tempfile.mkdtemp(prefix="video_translator_")
def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        try: shutil.rmtree(temp_dir); print(f"Temp dir {temp_dir} cleaned.")
        except Exception as e: print(f"Error cleaning temp dir {temp_dir}: {e}")

# --- GUI Part ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Dubbing Tool (WhisperX+XTTS) - v0.3") # Обновим версию
        self.root.geometry("650x550")

        self.video_path = tk.StringVar()
        self.srt_path = tk.StringVar()
        self.operation_options = {
            "Translate & Dub (Voice Clone)": "translate_dub",
            "Add Subtitles Only": "add_subtitles"
        }
        self.operation = tk.StringVar(value=list(self.operation_options.keys())[0])

        self.processing_times = {}
        self.total_start_time = 0

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background="#e0e0e0", foreground="black")
        style.map("TButton", background=[('active', '#c0c0c0')])
        style.configure("TLabel", padding=6, background="#f0f0f0")
        style.configure("TEntry", padding=6)
        style.configure("TRadiobutton", padding=6, background="#f0f0f0")
        style.configure("LabeledProgress.Horizontal.TProgressbar", troughcolor ='#E0E0E0', background='#4CAF50', text="0%", relief='sunken', thickness=20)
        self.root.configure(bg="#f0f0f0")

        input_outer_frame = ttk.Frame(root, padding=10)
        input_outer_frame.pack(fill="x")

        video_frame = ttk.LabelFrame(input_outer_frame, text="Video File", padding=(10, 5))
        video_frame.pack(padx=10, pady=5, fill="x")
        ttk.Entry(video_frame, textvariable=self.video_path, width=60, state="readonly").pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(video_frame, text="Browse", command=self.browse_video).pack(side=tk.LEFT)

        srt_frame = ttk.LabelFrame(input_outer_frame, text="SRT File (Only for 'Add Subtitles' op)", padding=(10, 5))
        srt_frame.pack(padx=10, pady=5, fill="x")
        ttk.Entry(srt_frame, textvariable=self.srt_path, width=60, state="readonly").pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(srt_frame, text="Browse", command=self.browse_srt).pack(side=tk.LEFT)

        ops_frame = ttk.LabelFrame(input_outer_frame, text="Operation", padding=(10,5))
        ops_frame.pack(padx=10, pady=5, fill="x")
        for op_text in self.operation_options.keys():
            ttk.Radiobutton(ops_frame, text=op_text, variable=self.operation, value=op_text).pack(anchor=tk.W)

        self.process_button = ttk.Button(root, text="Process Video", command=self.start_processing_thread, style="Accent.TButton")
        self.process_button.pack(pady=10)

        self.progress_frame = ttk.Frame(root, padding=(10,0))
        self.progress_label_text = tk.StringVar(value="Progress: 0%")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_label_text, width=15, anchor="e")
        self.progress_label.pack(side=tk.LEFT, padx=(0,5))
        self.progressbar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', maximum=100)
        self.progressbar.pack(side=tk.LEFT, fill="x", expand=True)
        self.progress_frame.pack_forget()

        status_frame = ttk.LabelFrame(root, text="Status / Log", padding=(10,5))
        status_frame.pack(padx=10, pady=5, fill="both", expand=True)
        self.status_text = tk.Text(status_frame, height=12, wrap=tk.WORD, state="disabled", bg="#ffffff", relief="sunken", borderwidth=1)
        self.status_text.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text['yscrollcommand'] = scrollbar.set

    def log_time(self, step_name, start_time):
        duration = time.time() - start_time
        self.processing_times[step_name] = duration
        self._update_status(f"⏱️ {step_name} took {duration:.2f} seconds.", append=True)

    def _update_status(self, message, append=True):
        def update_gui():
            self.status_text.config(state="normal")
            if append: self.status_text.insert(tk.END, str(message) + "\n")
            else: self.status_text.delete("1.0", tk.END); self.status_text.insert("1.0", str(message) + "\n")
            self.status_text.config(state="disabled")
            self.status_text.see(tk.END)
        self.root.after(0, update_gui)

    def _update_progress(self, value, step_name=""):
        clamped_value = max(0, min(100, int(value)))
        def update_gui():
            self.progressbar['value'] = clamped_value
            self.progress_label_text.set(f"{step_name}: {clamped_value}%")
            if clamped_value == 100 and step_name: self.progress_label_text.set(f"{step_name}: Done!")
            elif not step_name and clamped_value == 0: self.progress_label_text.set("Progress: 0%")
        self.root.after(0, update_gui)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm *.flv")])
        if path: self.video_path.set(path); self._update_status(f"Video selected: {path}", append=False)

    def browse_srt(self):
        path = filedialog.askopenfilename(filetypes=[("SRT Files", "*.srt")])
        if path: self.srt_path.set(path); self._update_status(f"SRT selected: {path}", append=True)

    def start_processing_thread(self):
        # ... (код без изменений, как в предыдущем ответе) ...
        video_file = self.video_path.get()
        srt_file = self.srt_path.get()
        op_key = self.operation.get()
        op_choice = self.operation_options.get(op_key)

        if not video_file: messagebox.showerror("Error", "Please select a video file."); return
        if op_choice == "add_subtitles" and not srt_file: messagebox.showerror("Error", "SRT file needed for subtitles."); return

        self.process_button.config(state="disabled")
        self._update_status(f"🚀 Starting operation: {op_key}...", append=False)
        self.progress_frame.pack(padx=10, pady=5, fill="x")
        self._update_progress(0, "Initializing")
        self.processing_times = {}
        self.total_start_time = time.time()

        thread = threading.Thread(target=self.actual_processing, args=(video_file, srt_file, op_choice, op_key), daemon=True)
        thread.start()

    def actual_processing(self, video_file, srt_file, op_choice, op_key):
        # ... (код с логированием времени и обновлением прогресса, как в предыдущем ответе) ...
        current_temp_dir = create_temp_dir()
        self._update_status(f"Using temp dir: {current_temp_dir}")
        output_path = None
        success = False
        dub_steps = {
            "Audio Extraction": 5, "Transcription & Diarization": 50,
            "Translation": 10, "Voice Synthesis": 30, "Video Assembly": 5
        }
        total_weight_dub = sum(dub_steps.values())
        current_progress = 0

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu" # Определяем устройство здесь
            self._update_status(f"⚡ Using device: {device.upper()} for ML models.", append=True)

            if op_choice == "translate_dub":
                step_name = "Audio Extraction"
                self._update_status(f"➡️ {step_name}..."); self._update_progress(current_progress / total_weight_dub * 100, step_name); start_t = time.time()
                original_audio_for_mixing = video_processor.extract_audio(video_file, os.path.join(current_temp_dir, "original_for_mix.wav"), sample_rate=44100)
                stt_audio_path = video_processor.extract_audio(video_file, os.path.join(current_temp_dir, "audio_for_stt.wav"), sample_rate=16000)
                self.log_time(step_name, start_t); current_progress += dub_steps[step_name]

                step_name = "Transcription & Diarization"
                self._update_status(f"➡️ {step_name} (WhisperX)..."); self._update_progress(current_progress / total_weight_dub * 100, step_name); start_t = time.time()
                transcriber.load_stt_diarization_models(device=device) # Передаем device
                aligned_segments = transcriber.transcribe_and_diarize(stt_audio_path)
                if not aligned_segments: raise ValueError("Transcription/Diarization failed.")
                self.log_time(step_name, start_t); current_progress += dub_steps[step_name]

                step_name = "Translation"
                self._update_status(f"➡️ {step_name} (Helsinki-NLP)..."); self._update_progress(current_progress / total_weight_dub * 100, step_name); start_t = time.time()
                translator.load_translator_model(device=device) # Передаем device
                translated_segments = translator.translate_segments(aligned_segments)
                self.log_time(step_name, start_t); current_progress += dub_steps[step_name]

                step_name = "Voice Synthesis"
                self._update_status(f"➡️ {step_name} (Coqui XTTS-v2)...");
                def synthesis_progress_callback(fraction_done):
                    base_prog = (current_progress / total_weight_dub * 100)
                    step_prog = (dub_steps[step_name] / total_weight_dub * 100) * fraction_done
                    self._update_progress(base_prog + step_prog, f"{step_name} ({int(fraction_done*100)}%)")
                self._update_progress(current_progress / total_weight_dub * 100, step_name); start_t = time.time()
                voice_cloner.load_tts_model(device=device) # Передаем device
                final_dubbed_audio_path = voice_cloner.synthesize_speech_segments(
                    translated_segments, stt_audio_path, current_temp_dir,
                    progress_callback=synthesis_progress_callback
                )
                self.log_time(step_name, start_t); current_progress += dub_steps[step_name]

                step_name = "Video Assembly"
                self._update_status(f"➡️ {step_name} (FFmpeg)..."); self._update_progress(current_progress / total_weight_dub * 100, step_name); start_t = time.time()
                input_dir = os.path.dirname(video_file)
                video_name_base = os.path.splitext(os.path.basename(video_file))[0]
                output_path = os.path.join(input_dir, f"{video_name_base}_dubbed_ru_mixed.mp4")
                video_processor.mix_and_replace_audio(video_file, original_audio_for_mixing, final_dubbed_audio_path, output_path, original_volume=0.10, dubbed_volume=0.95)
                self.log_time(step_name, start_t); self._update_progress(100, "Completed")

            elif op_choice == "add_subtitles":
                # ... (код без изменений) ...
                step_name = "Adding Subtitles"
                self._update_status(f"➡️ {step_name}..."); self._update_progress(0, step_name); start_t = time.time()
                input_dir = os.path.dirname(video_file); video_name_base = os.path.splitext(os.path.basename(video_file))[0]
                output_path = os.path.join(input_dir, f"{video_name_base}_with_subs.mp4")
                video_processor.add_subtitles(video_file, srt_file, output_path)
                self.log_time(step_name, start_t); self._update_progress(100, "Completed")


            if output_path and os.path.exists(output_path):
                total_time = time.time() - self.total_start_time
                self._update_status(f"✅🎉 Total processing time: {total_time:.2f} seconds.")
                self._update_status(f"✅ Operation '{op_key}' finished! Output:\n{output_path}", append=True)
                messagebox.showinfo("Success", f"Video processed in {total_time:.2f}s!\nOutput: {output_path}")
                success = True
            else: self._update_status(f"⚠️ Operation '{op_key}' finished, but output was not found.")
        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"❌ Error during '{op_key}': {e}\n--- Traceback ---\n{tb_str}-----------------"
            self._update_status(error_message)
            messagebox.showerror("Error", f"An error occurred: {e}\n(See status for details)")
            print(f"--- ERROR during '{op_key}' ---"); print(tb_str); print("----------------------------")
        finally:
            self.root.after(0, self.stop_and_hide_progressbar)
            cleanup_temp_dir(current_temp_dir)
            self.root.after(0, self.enable_process_button)
            if not success: self.root.after(0, lambda: self._update_status(f"Operation '{op_key}' failed.", append=True))

    def stop_and_hide_progressbar(self):
        self.progressbar['value'] = 0
        self.progress_label_text.set("Progress: 0%")
        self.progress_frame.pack_forget()

    def enable_process_button(self):
        self.process_button.config(state="normal")

# --- Проверка зависимостей и точка входа ---
if __name__ == "__main__":
    # Проверка PyTorch/CUDA в первую очередь, т.к. от этого зависит выбор device для моделей
    try:
        # import torch # Уже импортирован глобально
        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda if torch.version.cuda else 'N/A (might be ROCm build)'}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            # Проверка, что PyTorch не CPU-only сборка, если CUDA доступна
            if "+cpu" in torch.__version__ and cuda_available:
                 messagebox.showwarning("PyTorch Warning", "CUDA is available, but PyTorch seems to be a CPU-only build. GPU acceleration will not be used.\nPlease install PyTorch with CUDA support from pytorch.org.")
        elif "+cpu" not in torch.__version__ and not cuda_available:
            messagebox.showwarning("PyTorch Warning", "PyTorch seems to be a GPU-enabled build, but CUDA is not available/detected on your system. ML models will use CPU.")

    except ImportError:
        messagebox.showerror("Dependency Error", "PyTorch not found. Please install PyTorch.\nIt is required for WhisperX and Coqui TTS.")
        exit(1)
    except Exception as e_torch:
        messagebox.showwarning("Warning", f"Could not fully check PyTorch/CUDA status: {e_torch}")

    print("Checking FFmpeg/ffprobe...")
    ffmpeg_ok, ffmpeg_err = video_processor.check_command_availability('ffmpeg')
    ffprobe_ok, ffprobe_err = video_processor.check_command_availability('ffprobe')
    errors = []
    if not ffmpeg_ok: errors.append(f"FFmpeg: {ffmpeg_err}")
    if not ffprobe_ok: errors.append(f"ffprobe: {ffprobe_err}")
    if errors:
        msg = "Required command(s) not found or failed:\n\n" + "\n".join(errors)
        msg += "\n\nPlease ensure FFmpeg (with ffprobe) is installed and in PATH."
        messagebox.showerror("Dependency Error", msg); print(msg); exit(1)
    else: print("FFmpeg/ffprobe check passed.")

    root = tk.Tk()
    app = App(root)
    root.mainloop()