import os
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import traceback
import subprocess
import time
import torch
import re 

# --- Попытка решить проблему с загрузкой модели Coqui TTS и PyTorch 2.1+ ---
if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
    problematic_classes_tts = []
    try: from TTS.tts.configs.xtts_config import XttsConfig; problematic_classes_tts.append(XttsConfig)
    except ImportError: print("WARNING: Could not import TTS.tts.configs.xtts_config.XttsConfig")
    try: from TTS.tts.models.xtts import XttsAudioConfig; problematic_classes_tts.append(XttsAudioConfig)
    except ImportError: print("WARNING: Could not import TTS.tts.models.xtts.XttsAudioConfig")
    try: from TTS.config.shared_configs import BaseDatasetConfig; problematic_classes_tts.append(BaseDatasetConfig)
    except ImportError: print("WARNING: Could not import TTS.config.shared_configs.BaseDatasetConfig")
    except AttributeError: print("WARNING: Could not import BaseDatasetConfig from TTS.config.shared_configs (AttributeError).")
    try: from TTS.tts.models.xtts import XttsArgs; problematic_classes_tts.append(XttsArgs)
    except ImportError: print("WARNING: Could not import TTS.tts.models.xtts.XttsArgs")
    if problematic_classes_tts:
        try: torch.serialization.add_safe_globals(list(set(problematic_classes_tts)))
        except Exception as e_sg: print(f"WARNING: Error trying to add TTS classes to torch safe globals: {e_sg}")
os.environ["COQUI_AGREED_TO_CPML"] = "1"

import video_processor
import transcriber 
import translator
import voice_cloner
try: import ffmpeg
except ImportError: print("WARNING: ffmpeg-python library not found."); ffmpeg = None
try: import yt_dlp
except ImportError: print("WARNING: yt-dlp library not found. YouTube download will not work. Run 'pip install yt-dlp' to fix."); yt_dlp = None
try: import srt 
except ImportError: print("WARNING: srt library not found. SRT parsing will fail. Run 'pip install srt' to fix."); srt = None


def create_temp_dir(): return tempfile.mkdtemp(prefix="video_translator_")
def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        try: shutil.rmtree(temp_dir)
        except Exception as e: print(f"Error cleaning temp dir {temp_dir}: {e}")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Dubbing Tool (WhisperX/SRT + XTTS) - v0.6.2") 
        self.root.geometry("700x580") 

        self.video_source_text = tk.StringVar() 
        self.srt_path = tk.StringVar() 
        self.output_dir_path = tk.StringVar() 
        
        self.processing_times = {}; self.total_start_time = 0
        self.downloaded_video_path = None; self.downloaded_srt_path = None
        self.downloaded_srt_lang = None 

        style = ttk.Style(); style.theme_use('clam')
        style.configure("TButton", padding=6, relief="raised", background="#d9d9d9", foreground="black")
        style.map("TButton", background=[('pressed', '#c0c0c0'), ('active', '#e8e8e8')], relief=[('pressed', 'sunken')])
        style.configure("TLabel", padding=6, background="#f0f0f0")
        style.configure("TEntry", padding=6)
        self.root.configure(bg="#f0f0f0")
        
        self.make_context_menu_for_entry_fields() 

        source_frame = ttk.LabelFrame(root, text="Video Source (File Path or YouTube URL)", padding=(10, 5))
        source_frame.pack(padx=10, pady=5, fill="x")
        self.video_source_entry = ttk.Entry(source_frame, textvariable=self.video_source_text, width=60)
        self.video_source_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        
        self.video_source_entry.bind("<Button-3>", self.show_entry_context_menu) 
        
        # --- ИЗМЕНЕНИЕ: Используем стандартные имена событий Tkinter для хоткеев ---
        # Для Windows/Linux
        self.video_source_entry.bind_class("TEntry", "<Control-v>", self._handle_paste_event) # Привязка к классу TEntry
        self.video_source_entry.bind_class("TEntry", "<Control-V>", self._handle_paste_event) # На всякий случай
        # Для macOS (Command клавиша обычно соответствует Control в Tkinter на macOS, но можно добавить явные)
        self.video_source_entry.bind_class("TEntry", "<Command-v>", self._handle_paste_event)
        self.video_source_entry.bind_class("TEntry", "<Command-V>", self._handle_paste_event) 
        # Shift-Insert
        self.video_source_entry.bind_class("TEntry", "<Shift-Insert>", self._handle_paste_event)
        # Также, чтобы стандартная вставка работала, если наш обработчик не сработает,
        # можно попробовать не возвращать "break" или привязаться к самому виджету, а не классу.
        # Пока оставим так, с привязкой к классу.
        # self.video_source_entry.bind("<Control-v>", self._handle_paste_event) # Можно попробовать это, если bind_class не сработает


        self.video_browse_button = ttk.Button(source_frame, text="Browse File...", command=self.browse_video_for_source_entry)
        self.video_browse_button.pack(side=tk.LEFT)
        self.video_clear_button = ttk.Button(source_frame, text="Clear", command=self.clear_video_source_text)
        self.video_clear_button.pack(side=tk.LEFT, padx=(5,0))
        
        yt_subs_info_label = ttk.Label(source_frame, text="For YouTube URLs, subtitles (RU then EN) are auto-downloaded if available.", font=("Segoe UI", 8))
        yt_subs_info_label.pack(side=tk.BOTTOM, anchor=tk.W, padx=0, pady=(5,0))

        self.srt_outer_frame = ttk.LabelFrame(root, text="Subtitles File (Optional: if provided, used instead of WhisperX/YT subs)", padding=(10, 5))
        self.srt_outer_frame.pack(padx=10, pady=5, fill="x")
        srt_input_line_frame = ttk.Frame(self.srt_outer_frame)
        srt_input_line_frame.pack(fill="x") 
        self.srt_path_entry = ttk.Entry(srt_input_line_frame, textvariable=self.srt_path, width=60, state="readonly")
        self.srt_path_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        self.srt_browse_button = ttk.Button(srt_input_line_frame, text="Browse...", command=self.browse_srt) 
        self.srt_browse_button.pack(side=tk.LEFT)
        self.srt_clear_button = ttk.Button(srt_input_line_frame, text="Clear", command=self.clear_srt_path) 
        self.srt_clear_button.pack(side=tk.LEFT, padx=(5,0))
        
        output_dir_frame = ttk.LabelFrame(root, text="Output Directory (Optional: defaults to subfolder in CWD or near original video)", padding=(10,5))
        output_dir_frame.pack(padx=10, pady=5, fill="x")
        self.output_dir_entry = ttk.Entry(output_dir_frame, textvariable=self.output_dir_path, width=60, state="readonly")
        self.output_dir_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(output_dir_frame, text="Browse...", command=self.browse_output_dir).pack(side=tk.LEFT)
        ttk.Button(output_dir_frame, text="Clear", command=self.clear_output_dir).pack(side=tk.LEFT, padx=(5,0))
        
        self.process_button = ttk.Button(root, text="Translate & Dub Video", command=self.start_processing_thread)
        self.process_button.pack(pady=15)

        self.progress_frame = ttk.Frame(root, padding=(10,5))
        self.progress_label_text = tk.StringVar(value="Progress: 0%")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_label_text, width=30, anchor="w")
        self.progress_label.pack(side=tk.LEFT, padx=(0,5))
        self.progressbar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', maximum=100)
        self.progressbar.pack(side=tk.LEFT, fill="x", expand=True)
        status_frame = ttk.LabelFrame(root, text="Status / Log", padding=(10,5))
        status_frame.pack(padx=10, pady=(0,10), fill="both", expand=True)
        self.status_text = tk.Text(status_frame, height=12, wrap=tk.WORD, state="disabled", bg="#ffffff", relief="sunken", borderwidth=1, font=("Consolas", 9))
        self.status_text.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text['yscrollcommand'] = scrollbar.set

    def _handle_paste_event(self, event):
        # Эта функция будет вызвана для ВСЕХ TEntry, если используется bind_class.
        # Нам нужно убедиться, что мы работаем с правильным виджетом, если это важно.
        # В данном случае, мы хотим, чтобы вставка работала для поля URL.
        
        focused_widget = self.root.focus_get()
        if not isinstance(focused_widget, (ttk.Entry, tk.Entry)) or focused_widget.cget("state") == 'readonly':
            self._update_status(f"DEBUG: Paste event on non-Entry or readonly widget. Ignoring. Focused: {focused_widget}", append=True)
            return 

        # Убедимся, что событие пришло к полю, которое сейчас в фокусе
        if event.widget != focused_widget:
            self._update_status(f"DEBUG: Paste event on widget {event.widget} but focus is on {focused_widget}. Trying to operate on focused_widget.", append=True)
            # Если событие пришло не к тому виджету (например, из-за bind_class),
            # но мы знаем, какой виджет в фокусе, попробуем работать с ним.
            # Это может быть не всегда правильно, но для отладки попробуем.
            # widget_to_paste_into = focused_widget
        else:
            widget_to_paste_into = event.widget


        self._update_status(f"DEBUG: Paste hotkey triggered for widget: {widget_to_paste_into}. Event keysym: {event.keysym}, Type: {event.type}", append=True)
        
        try:
            # Стандартный способ вставки в Tkinter - генерация виртуального события <<Paste>>
            # Это позволяет виджету самому обработать вставку (например, заменить выделенный текст)
            widget_to_paste_into.event_generate("<<Paste>>")
            self._update_status(f"DEBUG: <<Paste>> event generated for {widget_to_paste_into}.", append=True)
            
            # Логируем текст ПОСЛЕ генерации события (с задержкой)
            # Это поможет понять, сработала ли вставка
            widget_to_paste_into.after(20, lambda w=widget_to_paste_into: 
                self._update_status(f"DEBUG: Text in widget after <<Paste>>: '{w.get()}'", append=True)
            )
            return "break" # Останавливаем дальнейшую обработку этого события, т.к. мы его "обработали"
        except tk.TclError as e:
            self._update_status(f"DEBUG: TclError during paste event generation (clipboard empty?): {e}", append=True)
            # Если ошибка (например, буфер пуст), все равно вернем break, 
            # чтобы не было стандартной обработки пустого буфера, если она есть.
            return "break"
        except Exception as e_paste:
            self._update_status(f"DEBUG: Generic error during paste handling: {e_paste}", append=True)
            # В случае другой ошибки, не возвращаем "break", чтобы стандартный обработчик мог попытаться.
            # Но это маловероятно, если event_generate сам по себе не вызывает исключений.


    def make_context_menu_for_entry_fields(self):
        self.entry_context_menu = tk.Menu(self.root, tearoff=0)
        self.entry_context_menu.add_command(label="Cut", command=lambda: self.root.focus_get().event_generate("<<Cut>>"))
        self.entry_context_menu.add_command(label="Copy", command=lambda: self.root.focus_get().event_generate("<<Copy>>"))
        self.entry_context_menu.add_command(label="Paste", command=lambda: self.root.focus_get().event_generate("<<Paste>>"))
        self.entry_context_menu.add_separator()
        self.entry_context_menu.add_command(label="Select All", command=lambda: self.root.focus_get().event_generate("<<SelectAll>>"))

    def show_entry_context_menu(self, event): 
        focused_widget = self.root.focus_get() 
        if isinstance(focused_widget, (ttk.Entry, tk.Entry)) and focused_widget.cget("state") != 'readonly':
            self.entry_context_menu.tk_popup(event.x_root, event.y_root)
    
    def is_youtube_url(self, url_string): 
        if not url_string: return False
        youtube_regex = re.compile(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
        return bool(youtube_regex.match(url_string))

    def log_time(self, step_name, start_time): duration = time.time() - start_time; self.processing_times[step_name] = duration; self._update_status(f"⏱️ {step_name} took {duration:.2f} seconds.", append=True)
    
    def _update_status(self, message, append=True): 
        def update_gui():
            if not self.status_text.winfo_exists(): return
            self.status_text.config(state="normal")
            if append: self.status_text.insert(tk.END, str(message) + "\n")
            else: self.status_text.delete("1.0", tk.END); self.status_text.insert("1.0", str(message) + "\n")
            self.status_text.config(state="disabled"); self.status_text.see(tk.END)
        if self.root.winfo_exists(): self.root.after_idle(update_gui)

    def _update_progress(self, value, step_name=""): 
        clamped_value = max(0, min(100, int(value)))
        def update_gui():
            if not (self.root.winfo_exists() and self.progressbar.winfo_exists() and self.progress_label.winfo_exists()): return
            self.progressbar['value'] = clamped_value
            progress_text = f"{step_name}: {clamped_value}%" if step_name else f"Progress: {clamped_value}%"
            if clamped_value == 100 and step_name: progress_text = f"{step_name}: Done!"
            elif not step_name and clamped_value == 0: progress_text = "Progress: 0%"
            self.progress_label_text.set(progress_text)
        if self.root.winfo_exists(): self.root.after_idle(update_gui)

    def browse_video_for_source_entry(self): 
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm *.flv")])
        if path: self.video_source_text.set(path); self._update_status(f"Video file selected: {path}", append=False)

    def clear_video_source_text(self): self.video_source_text.set(""); self._update_status("Video source cleared.", append=False)
    
    def browse_srt(self): 
        path = filedialog.askopenfilename(filetypes=[("SRT/VTT Files", "*.srt *.vtt")]) 
        if path: self.srt_path.set(path); self._update_status(f"Subtitles file selected: {path}. If provided, this will be used instead of WhisperX/YT subs.", append=True)

    def clear_srt_path(self): self.srt_path.set(""); self._update_status("SRT/VTT file selection cleared. WhisperX or downloaded YouTube subs will be used.", append=True)
    
    def browse_output_dir(self): 
        path = filedialog.askdirectory(); 
        if path: self.output_dir_path.set(path); self._update_status(f"Output directory set to: {path}", append=True)

    def clear_output_dir(self): self.output_dir_path.set(""); self._update_status("Output directory cleared. Will use default.", append=True)

    def start_processing_thread(self):
        # ... (без изменений)
        source_input = self.video_source_text.get().strip()
        if not source_input: messagebox.showerror("Error", "Please provide a video source (file path or YouTube URL)."); return
        is_youtube = self.is_youtube_url(source_input)
        video_file_to_process = None if is_youtube else source_input
        youtube_url_to_process = source_input if is_youtube else None
        if is_youtube and yt_dlp is None: messagebox.showerror("Error", "yt-dlp library is not installed..."); return
        if not is_youtube and not os.path.exists(video_file_to_process): messagebox.showerror("Error", f"Local video file not found: {video_file_to_process}"); return
        srt_file_input_path = self.srt_path.get(); output_dir_user_specified = self.output_dir_path.get().strip() 
        if srt_file_input_path and srt is None: messagebox.showerror("Error", "The 'srt' library is not installed..."); return
        op_key = "Translate & Dub Video" 
        self.process_button.config(state="disabled"); self._update_status(f"🚀 Starting operation: {op_key}...", append=False)
        self.progress_frame.pack(padx=10, pady=(5, 5), fill="x", after=self.process_button)
        self._update_progress(0, "Initializing"); self.root.update_idletasks()
        self.processing_times = {}; self.total_start_time = time.time()
        self.downloaded_video_path = None; self.downloaded_srt_path = None; self.downloaded_srt_lang = None
        thread = threading.Thread(target=self.actual_processing, 
                                  args=(video_file_to_process, srt_file_input_path, op_key, 
                                        is_youtube, youtube_url_to_process, 
                                        output_dir_user_specified), 
                                  daemon=True)
        thread.start()

    def actual_processing(self, video_file_path, local_srt_path, op_key, 
                          is_youtube, youtube_url_str, 
                          user_output_dir):
        # ... (содержимое этой функции остается без изменений по сравнению с предыдущим ответом)
        current_temp_dir = None; output_path = None; success = False
        dub_steps = { 
            "YouTube Download & Subs": 8, "Audio Extraction": 5, "SRT Parsing": 2, "Diarization for SRT": 8,
            "Transcription & Diarization": 32, "Translation": 10, "Voice Synthesis": 30, "Video Assembly": 5
        }
        video_to_use = video_file_path; srt_to_use_for_transcription = local_srt_path 
        subs_source_tag = ""; diarization_data_for_cloning = None; skip_translation_step = False
        try:
            current_temp_dir = create_temp_dir(); self._update_status(f"Using temp dir: {current_temp_dir}")
            device = "cuda" if torch.cuda.is_available() else "cpu"; current_progress = 0
            total_weight_op = sum(dub_steps.values()) if dub_steps else 1
            if is_youtube:
                download_step_name = "YouTube Download & Subs"
                if download_step_name in dub_steps:
                    self._update_status(f"➡️ {download_step_name} from {youtube_url_str}...");
                    self._update_progress(current_progress / total_weight_op * 100, download_step_name)
                    start_t_dl = time.time(); download_output_dir_vid = os.path.join(current_temp_dir, "youtube_download"); os.makedirs(download_output_dir_vid, exist_ok=True)
                    self.downloaded_video_path, self.downloaded_srt_path, self.downloaded_srt_lang = video_processor.download_youtube_video(
                        url=youtube_url_str, output_dir=download_output_dir_vid, preferred_sub_lang='ru', fallback_sub_lang='en')
                    video_to_use = self.downloaded_video_path 
                    self.log_time(download_step_name, start_t_dl); current_progress += dub_steps[download_step_name]
                    if self.downloaded_srt_path and os.path.exists(self.downloaded_srt_path):
                        self._update_status(f"✅ YouTube subtitles ({self.downloaded_srt_lang or 'unknown'}) downloaded: {os.path.basename(self.downloaded_srt_path)}", append=True)
                        if not srt_to_use_for_transcription: # Если пользователь не указал свои сабы
                            srt_to_use_for_transcription = self.downloaded_srt_path 
                            subs_source_tag = f"_ytbsubs-{self.downloaded_srt_lang}" if self.downloaded_srt_lang else "_ytbsubs"
                            if self.downloaded_srt_lang and 'ru' in self.downloaded_srt_lang.lower(): 
                                skip_translation_step = True; self._update_status("INFO: Downloaded Russian YouTube subtitles. Translation step will be skipped.", append=True)
                        # Если пользователь указал свои, но мы скачали русские, а пользовательские НЕ русские - это сложный случай.
                        # Пока что, если пользователь указал свои, они в приоритете, даже если скачались русские.
                        # Можно добавить опцию "предпочитать скачанные русские сабы".
                    else: self._update_status(f"⚠️ YouTube subtitles for specified languages were not found/downloaded.", append=True)
            
            if not video_to_use or not os.path.exists(video_to_use): raise FileNotFoundError(f"Video file for processing not found: {video_to_use or 'None'}")
            
            aligned_segments = None; stt_audio_path_for_cloning = None; original_audio_for_mixing = None; whisperx_srt_output_path = None 
            
            if user_output_dir and os.path.isdir(user_output_dir): final_output_dir = user_output_dir
            elif video_file_path and not is_youtube : final_output_dir = os.path.dirname(video_file_path)
            else: final_output_dir = os.path.join(os.getcwd(), "Translated_Videos"); os.makedirs(final_output_dir, exist_ok=True)
            self._update_status(f"Final outputs will be saved to: {final_output_dir}", append=True)
            
            video_name_base_for_outputs = os.path.splitext(os.path.basename(video_to_use))[0]
            
            step_name_ae = "Audio Extraction"
            if step_name_ae in dub_steps:
                self._update_status(f"➡️ {step_name_ae} from {os.path.basename(video_to_use)}..."); 
                self._update_progress(current_progress / total_weight_op * 100, step_name_ae); start_t_ae = time.time()
                original_audio_for_mixing = video_processor.extract_audio(video_to_use, os.path.join(current_temp_dir, "original_for_mix.wav"), sample_rate=44100)
                stt_audio_path_for_cloning = video_processor.extract_audio(video_to_use, os.path.join(current_temp_dir, "audio_for_diar_clone.wav"), sample_rate=16000) 
                self.log_time(step_name_ae, start_t_ae); current_progress += dub_steps[step_name_ae]
            
            if stt_audio_path_for_cloning is None or original_audio_for_mixing is None: raise RuntimeError("Audio for cloning or mixing was not prepared.")
            
            if srt_to_use_for_transcription: 
                if not os.path.exists(srt_to_use_for_transcription):
                     self._update_status(f"⚠️ Specified/downloaded subtitle file not found: {srt_to_use_for_transcription}. Proceeding with WhisperX transcription.", append=True)
                     srt_to_use_for_transcription = None 
                else:
                    if not subs_source_tag: subs_source_tag = "_customsubs"
                    step_name_parse = "SRT Parsing" 
                    if step_name_parse in dub_steps:
                        self._update_status(f"➡️ {step_name_parse} from {os.path.basename(srt_to_use_for_transcription)}...");
                        self._update_progress(current_progress / total_weight_op * 100, step_name_parse); start_t = time.time()
                        parsed_srt_segments = transcriber.parse_srt_file(srt_to_use_for_transcription)
                        if not parsed_srt_segments: 
                            self._update_status(f"⚠️ SRT parsing failed for {os.path.basename(srt_to_use_for_transcription)}. Will attempt WhisperX transcription.", append=True)
                            srt_to_use_for_transcription = None 
                        else:
                            self.log_time(step_name_parse, start_t); current_progress += dub_steps[step_name_parse]
                            # Проверяем, нужно ли пропускать перевод, если это русские сабы
                            # Это условие должно быть здесь, после успешного парсинга.
                            if subs_source_tag.startswith("_ytbsubs-") and 'ru' in subs_source_tag.lower():
                                skip_translation_step = True; self._update_status("INFO: Using parsed Russian YouTube subtitles. Translation step will be skipped.", append=True)
                            elif local_srt_path and 'ru' in os.path.basename(local_srt_path).lower() and not subs_source_tag.startswith("_ytbsubs-"): # Если пользовательские сабы русские
                                # Нужно решить, как определять язык пользовательских сабов. Пока по имени файла.
                                # Либо добавить выбор языка для пользовательских сабов.
                                self._update_status("INFO: Assuming provided custom subtitles are Russian. Checking if translation can be skipped.", append=True)
                                # Здесь можно добавить более сложную логику определения языка или опцию в GUI
                                # Для простоты, если в имени файла есть 'ru', считаем их русскими
                                if '.ru.' in os.path.basename(local_srt_path).lower() or os.path.basename(local_srt_path).lower().endswith('.ru.srt') or os.path.basename(local_srt_path).lower().endswith('.ru.vtt'):
                                     skip_translation_step = True; self._update_status("INFO: Custom subtitles seem to be Russian. Translation step will be skipped.", append=True)


                            step_name_diar_srt = "Diarization for SRT"
                            if step_name_diar_srt in dub_steps:
                                self._update_status(f"➡️ {step_name_diar_srt} using audio: {os.path.basename(stt_audio_path_for_cloning)}..."); 
                                self._update_progress(current_progress / total_weight_op * 100, step_name_diar_srt); start_t_ds = time.time()
                                diarization_data_for_cloning = transcriber.perform_diarization_only(stt_audio_path_for_cloning, device=device) 
                                if diarization_data_for_cloning is not None and not diarization_data_for_cloning.empty:
                                    aligned_segments = transcriber.assign_srt_segments_to_speakers(parsed_srt_segments, diarization_data_for_cloning)
                                    self._update_status(f"Assigned speakers to {len(aligned_segments)} SRT segments.", append=True)
                                else: 
                                    self._update_status("⚠️ Diarization for SRT failed or no speakers found. Using default speaker for all SRT segments.", append=True)
                                    aligned_segments = parsed_srt_segments # Используем просто распарсенные сегменты
                                self.log_time(step_name_diar_srt, start_t_ds); current_progress += dub_steps[step_name_diar_srt]
            
            if not aligned_segments: 
                subs_source_tag = "_whspsubs"
                step_name_td = "Transcription & Diarization" 
                if step_name_td in dub_steps:
                    self._update_status(f"➡️ {step_name_td} (WhisperX) using audio: {os.path.basename(stt_audio_path_for_cloning)}..."); 
                    self._update_progress(current_progress / total_weight_op * 100, step_name_td); start_t_td = time.time()
                    whisperx_srt_output_path = os.path.join(final_output_dir, f"{video_name_base_for_outputs}{subs_source_tag}_transcription.srt") 
                    aligned_segments, diarization_data_for_cloning = transcriber.transcribe_and_diarize(
                        stt_audio_path_for_cloning, 
                        output_srt_path=whisperx_srt_output_path,
                        return_diarization_df=True 
                    )
                    if not aligned_segments: raise ValueError("Transcription/Diarization failed or returned no segments.")
                    if os.path.exists(whisperx_srt_output_path): self._update_status(f"WhisperX transcription saved to: {whisperx_srt_output_path}", append=True)
                    self.log_time(step_name_td, start_t_td); current_progress += dub_steps[step_name_td]
            
            if aligned_segments is None: raise RuntimeError("Failed to obtain segments for translation/synthesis.")
            
            # Убедимся, что в aligned_segments тексты очищены от VTT тегов, если они пришли из parse_srt_file
            for seg_idx, seg_val in enumerate(aligned_segments):
                cleaned_text = seg_val.get('text', '')
                cleaned_text = re.sub(r'<(\d{2}:\d{2}:\d{2}\.\d{3})><c>', '', cleaned_text)
                cleaned_text = re.sub(r'</c>', '', cleaned_text)
                cleaned_text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', cleaned_text).strip()
                aligned_segments[seg_idx]['text'] = cleaned_text
            
            translated_segments = aligned_segments
            if not skip_translation_step:
                step_name_trans = "Translation" 
                if step_name_trans in dub_steps:
                    self._update_status(f"➡️ {step_name_trans} (Helsinki-NLP)..."); self._update_progress(current_progress / total_weight_op * 100, step_name_trans); start_t_tr = time.time()
                    translator.load_translator_model(device=device); translated_segments = translator.translate_segments(aligned_segments) 
                    self.log_time(step_name_trans, start_t_tr); current_progress += dub_steps[step_name_trans]
            else: 
                self._update_status("➡️ Translation step skipped (using pre-translated/Russian text).", append=True); 
                for seg_idx, seg_val in enumerate(translated_segments): # Используем translated_segments т.к. aligned_segments мог измениться
                    translated_segments[seg_idx]['translated_text'] = seg_val.get('text', '') # текст уже должен быть очищен
            
            step_name_vs = "Voice Synthesis"
            if step_name_vs in dub_steps:
                self._update_status(f"➡️ {step_name_vs} (Coqui XTTS-v2)...");
                def synthesis_progress_callback(fraction_done): 
                    base_prog = (current_progress / total_weight_op * 100) if total_weight_op else 0
                    step_prog = (dub_steps.get(step_name_vs, 0) / total_weight_op * 100) * fraction_done if total_weight_op else 0
                    self._update_progress(int(base_prog + step_prog), f"{step_name_vs} ({int(fraction_done*100)}%)")
                
                self._update_progress(current_progress / total_weight_op * 100, step_name_vs); start_t_vs = time.time()
                voice_cloner.load_tts_model(device=device)
                
                final_dubbed_audio_path = voice_cloner.synthesize_speech_segments(
                    translated_segments, stt_audio_path_for_cloning, current_temp_dir, 
                    diarization_result_df=diarization_data_for_cloning, 
                    progress_callback=synthesis_progress_callback 
                )
                self.log_time(step_name_vs, start_t_vs); current_progress += dub_steps[step_name_vs]
            
            step_name_va = "Video Assembly" 
            if step_name_va in dub_steps:
                self._update_status(f"➡️ {step_name_va} (FFmpeg)..."); self._update_progress(current_progress / total_weight_op * 100, step_name_va); start_t_va = time.time()
                output_path = os.path.join(final_output_dir, f"{video_name_base_for_outputs}_dubbed_ru{subs_source_tag}.mp4") 
                if not os.path.exists(original_audio_for_mixing): raise FileNotFoundError(f"Original audio for mixing not found: {original_audio_for_mixing}")
                if not final_dubbed_audio_path or not os.path.exists(final_dubbed_audio_path): raise FileNotFoundError(f"Final dubbed audio not found or not generated: {final_dubbed_audio_path}")
                video_processor.mix_and_replace_audio(video_to_use, original_audio_for_mixing, final_dubbed_audio_path, output_path, original_volume=0.10, dubbed_volume=0.95)
                self.log_time(step_name_va, start_t_va); self._update_progress(100, "Completed")
            
            if output_path and os.path.exists(output_path): 
                total_time_seconds = time.time() - self.total_start_time; total_time_minutes = total_time_seconds / 60.0 
                self._update_status(f"\n✅🎉 Total processing time: {total_time_minutes:.2f} minutes.") 
                self._update_status(f"✅ Operation '{op_key}' finished successfully!", append=True); self._update_status(f"Output file:\n{output_path}", append=True)
                if whisperx_srt_output_path and os.path.exists(whisperx_srt_output_path): self._update_status(f"WhisperX transcription saved to:\n{whisperx_srt_output_path}", append=True)
                success_message = f"Operation '{op_key}' completed in {total_time_minutes:.2f} minutes!\n\nOutput: {output_path}"
                if whisperx_srt_output_path and os.path.exists(whisperx_srt_output_path): success_message += f"\nTranscription SRT: {whisperx_srt_output_path}"
                self.root.after(0, lambda msg=success_message: messagebox.showinfo("Success", msg)); success = True
            else: 
                err_msg_out = f"⚠️ Operation '{op_key}' finished, but the expected output file was not found"
                if output_path: err_msg_out += f":\n{output_path}"
                else: err_msg_out += ", as no output path was generated."
                self._update_status(err_msg_out, append=True)
                if output_path: self.root.after(0, lambda: messagebox.showwarning("Warning", f"Operation '{op_key}' finished, but the output file seems to be missing."))
        except Exception as e:  
            tb_str = traceback.format_exc(); failed_step = "Initialization or Unknown Step" 
            current_max_progress_calc = 0; active_steps_map_exc = dub_steps
            for name, weight in active_steps_map_exc.items():
                if current_progress >= current_max_progress_calc and current_progress < current_max_progress_calc + weight: failed_step = name; break
                current_max_progress_calc += weight
            if is_youtube and failed_step == "Initialization or Unknown Step" and current_progress < active_steps_map_exc.get("YouTube Download & Subs",0): failed_step = "YouTube Download & Subs"
            if failed_step == "Voice Synthesis" and isinstance(e, NameError) and 'shutil' in str(e):
                 failed_step = "Voice Synthesis (shutil error)"

            ffmpeg_stderr = ""; 
            if ffmpeg and isinstance(e, ffmpeg.Error):
                try: ffmpeg_stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "Stderr not captured."; ffmpeg_stderr = f"\n--- FFmpeg Output (stderr) ---\n{ffmpeg_stderr}\n------------------------------"
                except Exception as stderr_e: ffmpeg_stderr = f"\n--- Could not decode FFmpeg stderr: {stderr_e} ---"
            elif yt_dlp and isinstance(e, RuntimeError) and ("Failed to download YouTube video" in str(e) or "yt-dlp download error" in str(e)): 
                ffmpeg_stderr = f"\n--- YouTube Download Error Details ---\n{e}\n------------------------------------"
            error_message = f"\n❌❌❌ ERROR during '{failed_step}' step of '{op_key}':\n{type(e).__name__}: {e}{ffmpeg_stderr}\n--- Traceback ---\n{tb_str}-----------------"
            print(error_message); self._update_status(error_message, append=True)
            display_error = f"An error occurred during the '{failed_step}' step:\n\n{e}"
            if ffmpeg_stderr and "YouTube Download Error Details" not in ffmpeg_stderr : display_error += "\n\n(Check logs for FFmpeg output details)"
            elif "YouTube Download Error Details" in ffmpeg_stderr: display_error = f"Error downloading YouTube video/subs:\n\n{e}\n\n(Check logs for details)"
            self.root.after(0, lambda f_step=failed_step, e_msg=display_error: messagebox.showerror("Processing Error", e_msg))
        finally: 
            self.root.after(0, self.stop_and_hide_progressbar); cleanup_temp_dir(current_temp_dir) 
            self.root.after(0, self.enable_process_button)
            if not success: self.root.after(0, lambda: self._update_status(f"\n❌ Operation '{op_key}' failed.", append=True))

    def stop_and_hide_progressbar(self): 
        # ... (без изменений)
        if self.root.winfo_exists():
            try:
                if self.progress_frame.winfo_ismapped(): self.progress_frame.pack_forget() 
                self.progressbar['value'] = 0; self.progress_label_text.set("Progress: 0%")
            except tk.TclError as e: print(f"Warning: Could not reset/hide progress bar: {e}")

    def enable_process_button(self): 
        # ... (без изменений)
        if self.root.winfo_exists():
             try: self.process_button.config(state="normal")
             except tk.TclError as e: print(f"Warning: Could not enable process button: {e}")

if __name__ == "__main__": 
    # ... (без изменений)
    try:
        cuda_available_startup = torch.cuda.is_available()
        show_pytorch_warning = False; pytorch_warning_msg = ""; pytorch_msg_type = "info"
        if cuda_available_startup:
            if "+cpu" in torch.__version__: show_pytorch_warning = True; pytorch_msg_type = "warning"; pytorch_warning_msg = "CUDA is detected, but PyTorch is a CPU-only build...\nInstall PyTorch with CUDA support..."
        else:
            if "+cpu" not in torch.__version__ and "cuda" in torch.__version__: show_pytorch_warning = True; pytorch_msg_type = "warning"; pytorch_warning_msg = "PyTorch seems to be a GPU-enabled build, but CUDA was not detected...\nML models will use CPU..."
            else: show_pytorch_warning = True; pytorch_msg_type = "info"; pytorch_warning_msg = "PyTorch is using the CPU...\nFor faster processing on NVIDIA GPUs..."
    except ImportError: messagebox.showerror("Fatal Dependency Error", "PyTorch was not found...\nPlease install it."); exit(1)
    except Exception as e_torch: messagebox.showwarning("Dependency Check Warning", f"Could not fully check PyTorch/CUDA status:\n{e_torch}"); show_pytorch_warning = False
    ffmpeg_ok, _ = video_processor.check_command_availability('ffmpeg'); ffprobe_ok, _ = video_processor.check_command_availability('ffprobe')
    if not (ffmpeg_ok and ffprobe_ok): messagebox.showerror("Fatal Dependency Error", "FFmpeg and/or ffprobe not found or not executable.\nPlease ensure they are installed and in your system's PATH."); exit(1)
    if yt_dlp is None: print("WARNING: yt-dlp is not installed. YouTube download functionality will be disabled.")
    root = tk.Tk(); app = App(root)
    if show_pytorch_warning:
        if pytorch_msg_type == "info": root.after(200, lambda: messagebox.showinfo("System Info", pytorch_warning_msg))
        elif pytorch_msg_type == "warning": root.after(200, lambda: messagebox.showwarning("System Configuration Warning", pytorch_warning_msg))
    try: root.mainloop()
    except Exception as e_gui: print(f"\n--- GUI Error ---"); traceback.print_exc(); messagebox.showerror("Application Error", f"A critical error occurred in the application's main loop:\n\n{e_gui}")
