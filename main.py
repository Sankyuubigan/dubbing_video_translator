import os
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import traceback
import subprocess
# import shutil # Уже импортирован выше
import time
import torch # Импортируем здесь для проверки и для обходного пути

# --- Попытка решить проблему с загрузкой модели Coqui TTS и PyTorch 2.1+ ---
# Это нужно сделать до первого импорта TTS или вызова функций, которые его используют
if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
    problematic_classes_tts = []
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        problematic_classes_tts.append(XttsConfig)
    except ImportError: print("WARNING: Could not import TTS.tts.configs.xtts_config.XttsConfig")
    try:
        from TTS.tts.models.xtts import XttsAudioConfig
        problematic_classes_tts.append(XttsAudioConfig)
    except ImportError: print("WARNING: Could not import TTS.tts.models.xtts.XttsAudioConfig")
    try:
        from TTS.config.shared_configs import BaseDatasetConfig
        problematic_classes_tts.append(BaseDatasetConfig)
    except ImportError: print("WARNING: Could not import TTS.config.shared_configs.BaseDatasetConfig")
    except AttributeError: print("WARNING: Could not import BaseDatasetConfig from TTS.config.shared_configs (AttributeError).")
    try:
        from TTS.tts.models.xtts import XttsArgs
        problematic_classes_tts.append(XttsArgs)
    except ImportError: print("WARNING: Could not import TTS.tts.models.xtts.XttsArgs")

    if problematic_classes_tts:
        try:
            unique_problematic_classes = list(set(problematic_classes_tts))
            torch.serialization.add_safe_globals(unique_problematic_classes)
            class_names = [cls.__name__ for cls in unique_problematic_classes]
            print(f"INFO: Added {len(unique_problematic_classes)} TTS classes to torch safe globals: {', '.join(class_names)}")
        except Exception as e_sg:
            print(f"WARNING: Error trying to add TTS classes to torch safe globals: {e_sg}")
    else:
        print("INFO: No problematic TTS classes found to add to torch safe globals (or they couldn't be imported).")
else:
    print("INFO: torch.serialization.add_safe_globals not available (older PyTorch or this specific issue doesn't apply).")

# --- Автоматическое принятие лицензии Coqui CPML ---
os.environ["COQUI_AGREED_TO_CPML"] = "1"
print("INFO: Attempting to auto-agree to Coqui CPML via environment variable.")

# Импортируем наши модули ПОСЛЕ установки переменных окружения и обходных путей
import video_processor
import transcriber
import translator
import voice_cloner
# Импорт `ffmpeg` из модуля `ffmpeg-python` нужен для обработки ошибок
try:
    import ffmpeg
except ImportError:
    # Обработка случая, если ffmpeg-python не установлен (хотя должен быть)
    print("WARNING: ffmpeg-python library not found. Install it with 'pip install ffmpeg-python'")
    ffmpeg = None # Установим в None, чтобы проверки ниже не падали

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
        self.root.title("Video Dubbing Tool (WhisperX+XTTS) - v0.4.3") # Обновим версию
        self.root.geometry("650x580") # Немного увеличим высоту по умолчанию

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
        style.configure("TButton", padding=6, relief="raised", background="#d9d9d9", foreground="black")
        style.map("TButton",
            background=[('pressed', '#c0c0c0'), ('active', '#e8e8e8')],
            relief=[('pressed', 'sunken')])
        style.configure("TLabel", padding=6, background="#f0f0f0")
        style.configure("TEntry", padding=6)
        style.configure("TRadiobutton", padding=6, background="#f0f0f0")
        self.root.configure(bg="#f0f0f0")

        # --- Фрейм для ввода ---
        input_outer_frame = ttk.Frame(root, padding=10)
        input_outer_frame.pack(fill="x", padx=5, pady=5)

        video_frame = ttk.LabelFrame(input_outer_frame, text="Video File", padding=(10, 5))
        video_frame.pack(padx=10, pady=(5,0), fill="x")
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

        # --- Кнопка запуска ---
        self.process_button = ttk.Button(root, text="Process Video", command=self.start_processing_thread)
        self.process_button.pack(pady=10)

        # --- Фрейм прогресса (изначально скрыт) ---
        self.progress_frame = ttk.Frame(root, padding=(10,5))
        # Не будем паковать его здесь, будем паковать/распаковывать в start/stop
        self.progress_label_text = tk.StringVar(value="Progress: 0%")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_label_text, width=25, anchor="w")
        self.progress_label.pack(side=tk.LEFT, padx=(0,5))
        self.progressbar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', maximum=100)
        self.progressbar.pack(side=tk.LEFT, fill="x", expand=True)

        # --- Фрейм статуса/лога ---
        status_frame = ttk.LabelFrame(root, text="Status / Log", padding=(10,5))
        status_frame.pack(padx=15, pady=(0,10), fill="both", expand=True) # Добавили отступы
        self.status_text = tk.Text(status_frame, height=10, wrap=tk.WORD, state="disabled", bg="#ffffff", relief="sunken", borderwidth=1, font=("Consolas", 9))
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
            if not self.status_text.winfo_exists(): return
            self.status_text.config(state="normal")
            if append: self.status_text.insert(tk.END, str(message) + "\n")
            else: self.status_text.delete("1.0", tk.END); self.status_text.insert("1.0", str(message) + "\n")
            self.status_text.config(state="disabled")
            self.status_text.see(tk.END)
        if self.root.winfo_exists():
             self.root.after_idle(update_gui)

    def _update_progress(self, value, step_name=""):
        clamped_value = max(0, min(100, int(value)))
        def update_gui():
            if not (self.root.winfo_exists() and self.progressbar.winfo_exists() and self.progress_label.winfo_exists()): return
            self.progressbar['value'] = clamped_value
            progress_text = f"{step_name}: {clamped_value}%" if step_name else f"Progress: {clamped_value}%"
            if clamped_value == 100 and step_name: progress_text = f"{step_name}: Done!"
            elif not step_name and clamped_value == 0: progress_text = "Progress: 0%"
            self.progress_label_text.set(progress_text)
        if self.root.winfo_exists():
            self.root.after_idle(update_gui)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm *.flv")])
        if path: self.video_path.set(path); self._update_status(f"Video selected: {path}", append=False)

    def browse_srt(self):
        path = filedialog.askopenfilename(filetypes=[("SRT Files", "*.srt")])
        if path: self.srt_path.set(path); self._update_status(f"SRT selected: {path}", append=True)

    def start_processing_thread(self):
        video_file = self.video_path.get()
        srt_file = self.srt_path.get()
        op_key = self.operation.get()
        op_choice = self.operation_options.get(op_key)

        if not video_file: messagebox.showerror("Error", "Please select a video file."); return
        if op_choice == "add_subtitles" and not srt_file: messagebox.showerror("Error", "SRT file needed for 'Add Subtitles Only' operation."); return

        self.process_button.config(state="disabled")
        self._update_status(f"🚀 Starting operation: {op_key}...", append=False)

        # --- Отображение прогрессбара ---
        self.progress_frame.pack(padx=15, pady=(0, 5), fill="x", before=self.process_button.master.pack_slaves()[-1]) # Пакуем перед последним элементом (лог)
        self.progressbar['value'] = 0 # Сбрасываем значение
        self.progress_label_text.set("Initializing: 0%") # Устанавливаем начальный текст
        self.root.update_idletasks() # <--- ВАЖНО: Обновляем GUI, чтобы прогрессбар отрисовался

        self.processing_times = {}
        self.total_start_time = time.time()

        thread = threading.Thread(target=self.actual_processing, args=(video_file, srt_file, op_choice, op_key), daemon=True)
        thread.start()

    def actual_processing(self, video_file, srt_file, op_choice, op_key):
        current_temp_dir = None
        output_path = None
        success = False
        dub_steps = {
            "Audio Extraction": 5, "Transcription & Diarization": 45,
            "Translation": 10, "Voice Synthesis": 35, "Video Assembly": 5
        }
        total_weight_dub = sum(dub_steps.values()) if dub_steps else 1

        try:
            current_temp_dir = create_temp_dir()
            self._update_status(f"Using temp dir: {current_temp_dir}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._update_status(f"⚡ Using device: {device.upper()} for ML models.", append=True)

            current_progress = 0

            if op_choice == "translate_dub":
                total_weight_dub = sum(dub_steps.values()) if dub_steps else 1

                # --- Шаг 1: Извлечение аудио ---
                step_name = "Audio Extraction"
                if step_name in dub_steps:
                    self._update_status(f"➡️ {step_name}..."); self._update_progress(current_progress / total_weight_dub * 100, step_name); start_t = time.time()
                    original_audio_for_mixing = video_processor.extract_audio(video_file, os.path.join(current_temp_dir, "original_for_mix.wav"), sample_rate=44100)
                    stt_audio_path = video_processor.extract_audio(video_file, os.path.join(current_temp_dir, "audio_for_stt.wav"), sample_rate=16000)
                    self.log_time(step_name, start_t); current_progress += dub_steps[step_name]
                else: self._update_status(f"Skipping step: {step_name}")

                # --- Шаг 2: Транскрипция и Диаизация ---
                step_name = "Transcription & Diarization"
                # ... (код шага без изменений) ...
                if step_name in dub_steps:
                    self._update_status(f"➡️ {step_name} (WhisperX)..."); self._update_progress(current_progress / total_weight_dub * 100, step_name); start_t = time.time()
                    transcriber.load_stt_diarization_models(device=device)
                    aligned_segments = transcriber.transcribe_and_diarize(stt_audio_path)
                    if not aligned_segments: raise ValueError("Transcription/Diarization failed or returned no segments.")
                    self._update_status(f"Transcription found {len(aligned_segments)} segments.", append=True)
                    self.log_time(step_name, start_t); current_progress += dub_steps[step_name]
                else: self._update_status(f"Skipping step: {step_name}")


                # --- Шаг 3: Перевод ---
                step_name = "Translation"
                # ... (код шага без изменений) ...
                if step_name in dub_steps:
                    self._update_status(f"➡️ {step_name} (Helsinki-NLP)..."); self._update_progress(current_progress / total_weight_dub * 100, step_name); start_t = time.time()
                    translator.load_translator_model(device=device)
                    translated_segments = translator.translate_segments(aligned_segments)
                    self._update_status(f"Translation complete for {len(translated_segments)} segments.", append=True)
                    self.log_time(step_name, start_t); current_progress += dub_steps[step_name]
                else: self._update_status(f"Skipping step: {step_name}")

                # --- Шаг 4: Синтез речи ---
                step_name = "Voice Synthesis"
                # ... (код шага без изменений) ...
                if step_name in dub_steps:
                    self._update_status(f"➡️ {step_name} (Coqui XTTS-v2)...");
                    def synthesis_progress_callback(fraction_done):
                        base_prog = (current_progress / total_weight_dub * 100) if total_weight_dub else 0
                        step_prog = (dub_steps.get(step_name, 0) / total_weight_dub * 100) * fraction_done if total_weight_dub else 0
                        self._update_progress(int(base_prog + step_prog), f"{step_name} ({int(fraction_done*100)}%)")
                    self._update_progress(current_progress / total_weight_dub * 100, step_name); start_t = time.time()
                    voice_cloner.load_tts_model(device=device)
                    final_dubbed_audio_path = voice_cloner.synthesize_speech_segments(
                        translated_segments, stt_audio_path, current_temp_dir,
                        progress_callback=synthesis_progress_callback
                    )
                    self._update_status(f"Voice synthesis complete. Output: {final_dubbed_audio_path}", append=True)
                    self.log_time(step_name, start_t); current_progress += dub_steps[step_name]
                else: self._update_status(f"Skipping step: {step_name}")

                # --- Шаг 5: Сборка видео ---
                step_name = "Video Assembly"
                if step_name in dub_steps:
                    self._update_status(f"➡️ {step_name} (FFmpeg)..."); self._update_progress(current_progress / total_weight_dub * 100, step_name); start_t = time.time()
                    input_dir = os.path.dirname(video_file)
                    video_name_base = os.path.splitext(os.path.basename(video_file))[0]
                    output_path = os.path.join(input_dir, f"{video_name_base}_dubbed_ru_mixed.mp4")
                    self._update_status(f"Assembling final video: {output_path}", append=True)
                    if not os.path.exists(original_audio_for_mixing): raise FileNotFoundError(f"Original audio for mixing not found: {original_audio_for_mixing}")
                    if not os.path.exists(final_dubbed_audio_path): raise FileNotFoundError(f"Final dubbed audio not found: {final_dubbed_audio_path}")
                    # Вызов функции сборки (изменения будут в video_processor.py)
                    video_processor.mix_and_replace_audio(video_file, original_audio_for_mixing, final_dubbed_audio_path, output_path, original_volume=0.10, dubbed_volume=0.95)
                    self.log_time(step_name, start_t);
                    self._update_progress(100, "Completed")
                else: self._update_status(f"Skipping step: {step_name}")

            elif op_choice == "add_subtitles":
                step_name = "Adding Subtitles"
                self._update_status(f"➡️ {step_name} (FFmpeg)..."); self._update_progress(0, step_name); start_t = time.time()
                input_dir = os.path.dirname(video_file); video_name_base = os.path.splitext(os.path.basename(video_file))[0]
                output_path = os.path.join(input_dir, f"{video_name_base}_with_subs.mp4")
                self._update_status(f"Adding subtitles to: {output_path}", append=True)
                if not os.path.exists(srt_file): raise FileNotFoundError(f"SRT file not found: {srt_file}")
                 # Вызов функции добавления субтитров (изменения могут быть и там для логирования)
                video_processor.add_subtitles(video_file, srt_file, output_path)
                self.log_time(step_name, start_t); self._update_progress(100, "Completed")

            # --- Проверка результата и завершение ---
            if output_path and os.path.exists(output_path):
                total_time = time.time() - self.total_start_time
                self._update_status(f"\n✅🎉 Total processing time: {total_time:.2f} seconds.")
                self._update_status(f"✅ Operation '{op_key}' finished successfully!", append=True)
                self._update_status(f"Output file:\n{output_path}", append=True)
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Operation '{op_key}' completed in {total_time:.2f}s!\n\nOutput: {output_path}"))
                success = True
            else:
                if output_path: self._update_status(f"⚠️ Operation '{op_key}' finished, but the expected output file was not found:\n{output_path}", append=True)
                else: self._update_status(f"⚠️ Operation '{op_key}' finished, but no output path was generated.", append=True)
                # Покажем предупреждение только если файл должен был быть, но его нет
                if output_path:
                    self.root.after(0, lambda: messagebox.showwarning("Warning", f"Operation '{op_key}' finished, but the output file seems to be missing."))

        # --- Обработка исключений ---
        except Exception as e:
            tb_str = traceback.format_exc()
            failed_step = "Initialization or Unknown Step"
            current_max_progress = 0
            if op_choice == "translate_dub":
                for name, weight in dub_steps.items():
                     if current_progress >= current_max_progress and current_progress < current_max_progress + weight:
                         failed_step = name
                         break
                     current_max_progress += weight
            elif op_choice == "add_subtitles": failed_step = "Adding Subtitles"

            # Логируем ошибку ffmpeg если она есть
            ffmpeg_stderr = ""
            if ffmpeg and isinstance(e, ffmpeg.Error):
                try:
                    # Пытаемся получить stderr из исключения ffmpeg.Error
                    ffmpeg_stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "Stderr not captured."
                    ffmpeg_stderr = f"\n--- FFmpeg Output (stderr) ---\n{ffmpeg_stderr}\n------------------------------"
                except Exception as stderr_e:
                     ffmpeg_stderr = f"\n--- Could not decode FFmpeg stderr: {stderr_e} ---"

            error_message = f"\n❌❌❌ ERROR during '{failed_step}' step of '{op_key}':\n{type(e).__name__}: {e}{ffmpeg_stderr}\n--- Traceback ---\n{tb_str}-----------------"
            print(error_message)
            self._update_status(error_message, append=True)
            # Показываем ошибку пользователю
            display_error = f"An error occurred during the '{failed_step}' step:\n\n{e}"
            if ffmpeg_stderr: display_error += "\n\n(Check logs for FFmpeg output details)"
            self.root.after(0, lambda f_step=failed_step, e_msg=display_error: messagebox.showerror("Processing Error", e_msg))


        finally:
            self.root.after(0, self.stop_and_hide_progressbar)
            cleanup_temp_dir(current_temp_dir)
            self.root.after(0, self.enable_process_button)
            if not success:
                self.root.after(0, lambda: self._update_status(f"\n❌ Operation '{op_key}' failed.", append=True))

    def stop_and_hide_progressbar(self):
        if self.root.winfo_exists():
            try:
                if self.progress_frame.winfo_ismapped():
                     self.progress_frame.pack_forget() # Скрываем рамку
                # Сбрасываем значения на случай следующего запуска
                self.progressbar['value'] = 0
                self.progress_label_text.set("Progress: 0%")
            except tk.TclError as e:
                print(f"Warning: Could not reset/hide progress bar: {e}")

    def enable_process_button(self):
        if self.root.winfo_exists():
             try:
                 self.process_button.config(state="normal")
             except tk.TclError as e:
                 print(f"Warning: Could not enable process button: {e}")

# --- Проверка зависимостей и точка входа ---
if __name__ == "__main__":
    # ... (код проверки зависимостей без изменений) ...
    try:
        # Проверка PyTorch и CUDA
        print(f"PyTorch version: {torch.__version__}")
        cuda_available_startup = torch.cuda.is_available()
        print(f"CUDA available at startup: {cuda_available_startup}")
        cuda_version_str = "N/A"
        gpu_name = "N/A"
        show_pytorch_warning = False
        pytorch_warning_msg = ""
        pytorch_msg_type = "info" # "info" или "warning"

        if cuda_available_startup:
            cuda_version_str = torch.version.cuda if torch.version.cuda else 'N/A (check build)'
            try: gpu_name = torch.cuda.get_device_name(0)
            except Exception as e_gpu: gpu_name = f"Error getting name: {e_gpu}"
            print(f"CUDA version: {cuda_version_str}")
            print(f"GPU Name: {gpu_name}")
            if "+cpu" in torch.__version__:
                 show_pytorch_warning = True
                 pytorch_msg_type = "warning"
                 pytorch_warning_msg = "CUDA is detected, but PyTorch is a CPU-only build.\nGPU acceleration will NOT be used.\n\nInstall PyTorch with CUDA support from pytorch.org for significant speed improvements."
            else: print("PyTorch build seems compatible with CUDA.")
        else:
            if "+cpu" not in torch.__version__ and "cuda" in torch.__version__:
                 show_pytorch_warning = True
                 pytorch_msg_type = "warning"
                 pytorch_warning_msg = "PyTorch seems to be a GPU-enabled build, but CUDA was not detected or is incompatible.\nML models will use the CPU (which can be very slow).\n\nEnsure CUDA Toolkit is installed and compatible with your PyTorch build, or install a CPU-only PyTorch version if you don't have a compatible GPU."
            else:
                 show_pytorch_warning = True
                 pytorch_msg_type = "info"
                 pytorch_warning_msg = "PyTorch is using the CPU.\nFor faster processing on NVIDIA GPUs, consider installing PyTorch with CUDA support and ensuring you have a compatible CUDA Toolkit installed."

    except ImportError:
        messagebox.showerror("Fatal Dependency Error", "PyTorch was not found.\nThis application requires PyTorch to function.\nPlease install it (e.g., 'pip install torch torchvision torchaudio').")
        print("FATAL ERROR: PyTorch not found. Exiting.")
        exit(1)
    except Exception as e_torch:
        messagebox.showwarning("Dependency Check Warning", f"Could not fully check PyTorch/CUDA status:\n{e_torch}\nThe application will attempt to continue.")
        print(f"WARNING: Could not fully check PyTorch/CUDA status: {e_torch}")
        cuda_available_startup = False
        show_pytorch_warning = False

    # Проверка FFmpeg/ffprobe
    print("Checking FFmpeg/ffprobe...")
    ffmpeg_ok, ffmpeg_err = video_processor.check_command_availability('ffmpeg')
    ffprobe_ok, ffprobe_err = video_processor.check_command_availability('ffprobe')
    cli_errors = []
    if not ffmpeg_ok: cli_errors.append(f"❌ FFmpeg: {ffmpeg_err}")
    else: print("✅ FFmpeg found.")
    if not ffprobe_ok: cli_errors.append(f"❌ ffprobe: {ffprobe_err}")
    else: print("✅ ffprobe found.")

    if cli_errors:
        msg = "Required command-line tool(s) not found or failed execution:\n\n" + "\n".join(cli_errors)
        msg += "\n\nPlease ensure FFmpeg (which includes ffprobe) is installed correctly and its location is added to your system's PATH environment variable."
        messagebox.showerror("Fatal Dependency Error", msg)
        print(f"FATAL ERROR: Required CLI tools missing.\n{msg}")
        exit(1)
    else: print("FFmpeg/ffprobe check passed.")

    # Запуск GUI
    print("Launching GUI...")
    root = tk.Tk()
    app = App(root)

    if show_pytorch_warning:
        if pytorch_msg_type == "info": root.after(200, lambda: messagebox.showinfo("System Info", pytorch_warning_msg))
        elif pytorch_msg_type == "warning": root.after(200, lambda: messagebox.showwarning("System Configuration Warning", pytorch_warning_msg))

    try: root.mainloop()
    except Exception as e_gui:
        print(f"\n--- GUI Error ---")
        traceback.print_exc()
        messagebox.showerror("Application Error", f"A critical error occurred in the application's main loop:\n\n{e_gui}\n\nThe application might need to close.")
    finally: print("Application closing.")
