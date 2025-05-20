import os
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import threading
import traceback
import subprocess
import time
import torch
import re
import pyperclip
import platform 
import datetime 
import sys 
import json 

from utils import config_manager 

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
except ImportError: print("WARNING: srt library not found. SRT parsing/generation will fail. Run 'pip install srt' to fix."); srt = None
try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("WARNING: langdetect library not found. Language detection for custom SRT will rely on filename only. Run 'pip install langdetect' to improve.")
    detect = None
    LangDetectException = None


def create_temp_dir(): return tempfile.mkdtemp(prefix="video_translator_")
def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        try: shutil.rmtree(temp_dir)
        except Exception as e: print(f"Error cleaning temp dir {temp_dir}: {e}")

class App:
    def __init__(self, root_tk):
        self.root_tk = root_tk
        
        self.app_display_version = datetime.date.today().strftime('%d.%m.%y')
        self.root_tk.title(f"Video Dubbing Tool - v{self.app_display_version}") 
        self.root_tk.geometry("700x680")
        
        self.current_work_dir = config_manager.get_work_dir_from_config()
        if not self.current_work_dir:
            default_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
            chosen_dir = filedialog.askdirectory(
                title="Выберите или создайте рабочую директорию",
                initialdir=default_dir
            )
            if chosen_dir:
                self.current_work_dir = chosen_dir
                os.makedirs(self.current_work_dir, exist_ok=True) 
                config_manager.save_work_dir_to_config(self.current_work_dir) 
                print(f"INFO: Рабочая директория установлена: {self.current_work_dir}")
            else: 
                messagebox.showerror("Ошибка", "Рабочая директория не выбрана. Приложение не может продолжить.")
                self.root_tk.destroy() 
                return
        
        self.work_dir_path_var = tk.StringVar(value=self.current_work_dir)
        
        self.ffmpeg_configured = False
        self.ffprobe_configured = False

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
        self.root_tk.configure(bg="#f0f0f0")
        self.make_context_menu_for_entry_fields()

        work_dir_frame = ttk.LabelFrame(self.root_tk, text="Рабочая директория (для моделей и инструментов)", padding=(10, 5))
        work_dir_frame.pack(padx=10, pady=5, fill="x")
        
        self.work_dir_entry = ttk.Entry(work_dir_frame, textvariable=self.work_dir_path_var, width=50) 
        self.work_dir_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        
        self.work_dir_browse_button = ttk.Button(work_dir_frame, text="Выбрать...", command=self.browse_work_dir)
        self.work_dir_browse_button.pack(side=tk.LEFT, padx=(0,5))
        
        self.work_dir_apply_button = ttk.Button(work_dir_frame, text="Применить и Проверить", command=self.apply_and_check_work_dir_gui_wrapper)
        self.work_dir_apply_button.pack(side=tk.LEFT, padx=(0,5))

        self.download_tools_button = ttk.Button(work_dir_frame, text="Скачать FFmpeg", command=self.trigger_download_tools_gui_wrapper, state=tk.NORMAL if self.current_work_dir else tk.DISABLED)
        self.download_tools_button.pack(side=tk.LEFT)

        source_frame = ttk.LabelFrame(self.root_tk, text="Video Source (File Path or YouTube URL)", padding=(10, 5))
        source_frame.pack(padx=10, pady=5, fill="x")
        self.video_source_entry = ttk.Entry(source_frame, textvariable=self.video_source_text, width=60)
        self.video_source_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        self.video_source_entry.bind("<Button-3>", self.show_entry_context_menu)
        try: self.video_source_entry.unbind("<<Paste>>")
        except tk.TclError: pass
        self.video_source_entry.bind("<Control-v>", self._force_paste_event); self.video_source_entry.bind("<Control-V>", self._force_paste_event)
        if self.root_tk.tk.call('tk', 'windowingsystem') == 'aqua': self.video_source_entry.bind("<Command-v>", self._force_paste_event); self.video_source_entry.bind("<Command-V>", self._force_paste_event)
        self.video_source_entry.bind("<Shift-Insert>", self._force_paste_event)
        self.video_browse_button = ttk.Button(source_frame, text="Browse File...", command=self.browse_video_for_source_entry); self.video_browse_button.pack(side=tk.LEFT)
        self.video_clear_button = ttk.Button(source_frame, text="Clear", command=self.clear_video_source_text); self.video_clear_button.pack(side=tk.LEFT, padx=(5,0))
        yt_subs_info_label = ttk.Label(source_frame, text="For YouTube URLs, subtitles (RU then EN) are auto-downloaded if available (used if no local SRT).", font=("Segoe UI", 8)); yt_subs_info_label.pack(side=tk.BOTTOM, anchor=tk.W, padx=0, pady=(5,0))
        
        self.srt_outer_frame = ttk.LabelFrame(self.root_tk, text="Subtitles File (RU text, timings will be used & post-processed)", padding=(10, 5)); self.srt_outer_frame.pack(padx=10, pady=5, fill="x")
        srt_input_line_frame = ttk.Frame(self.srt_outer_frame); srt_input_line_frame.pack(fill="x")
        self.srt_path_entry = ttk.Entry(srt_input_line_frame, textvariable=self.srt_path, width=60, state="readonly"); self.srt_path_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        self.srt_browse_button = ttk.Button(srt_input_line_frame, text="Browse...", command=self.browse_srt) ; self.srt_browse_button.pack(side=tk.LEFT)
        self.srt_clear_button = ttk.Button(srt_input_line_frame, text="Clear", command=self.clear_srt_path) ; self.srt_clear_button.pack(side=tk.LEFT, padx=(5,0))
        
        output_dir_frame = ttk.LabelFrame(self.root_tk, text="Output Directory (Optional: defaults to subfolder in CWD or near original video)", padding=(10,5)); output_dir_frame.pack(padx=10, pady=5, fill="x")
        self.output_dir_entry = ttk.Entry(output_dir_frame, textvariable=self.output_dir_path, width=60, state="readonly"); self.output_dir_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(output_dir_frame, text="Browse...", command=self.browse_output_dir).pack(side=tk.LEFT)
        ttk.Button(output_dir_frame, text="Clear", command=self.clear_output_dir).pack(side=tk.LEFT, padx=(5,0))
        
        self.process_button = ttk.Button(self.root_tk, text="Translate & Dub Video", command=self.start_processing_thread); self.process_button.pack(pady=10)
        self.progress_frame = ttk.Frame(self.root_tk, padding=(10,5))
        self.progress_label_text = tk.StringVar(value="Progress: 0%"); self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_label_text, width=30, anchor="w"); self.progress_label.pack(side=tk.LEFT, padx=(0,5))
        self.progressbar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', maximum=100); self.progressbar.pack(side=tk.LEFT, fill="x", expand=True)
        status_frame_outer = ttk.LabelFrame(self.root_tk, text="Status / Log", padding=(10,5)); status_frame_outer.pack(padx=10, pady=(0,5), fill="both", expand=True)
        self.status_text = scrolledtext.ScrolledText(status_frame_outer, height=10, wrap=tk.WORD, state="disabled", bg="#ffffff", relief="sunken", borderwidth=1, font=("Consolas", 9)); self.status_text.pack(side=tk.TOP, fill="both", expand=True)
        self.copy_log_button = ttk.Button(status_frame_outer, text="Copy Log to Clipboard", command=self.copy_log_to_clipboard); self.copy_log_button.pack(side=tk.TOP, pady=(5,0))

        if self.current_work_dir: 
            self.apply_and_check_work_dir_gui_wrapper(show_success_message=False, initial_setup=True)

    def browse_work_dir(self):
        path = filedialog.askdirectory(title="Выберите рабочую директорию")
        if path:
            self.current_work_dir = path 
            self.work_dir_path_var.set(path)
            self.apply_and_check_work_dir_gui_wrapper() 

    def apply_and_check_work_dir_gui_wrapper(self, show_success_message=True, initial_setup=False):
        work_dir_to_check = self.work_dir_path_var.get().strip() 
        
        if not work_dir_to_check : 
             if not initial_setup: messagebox.showwarning("Рабочая директория", "Путь к рабочей директории не может быть пустым.")
             if self.current_work_dir: self.work_dir_path_var.set(self.current_work_dir)
             else: 
                self.ffmpeg_configured = self.ffprobe_configured = False
                if hasattr(self, 'download_tools_button'): self.download_tools_button.config(state=tk.DISABLED)
                self.ffmpeg_configured, self.ffprobe_configured, _ = config_manager.initialize_paths_from_work_dir(None)
                status_messages = []
                if self.ffmpeg_configured: status_messages.append(f"FFmpeg (системный): OK ({os.path.basename(config_manager.CURRENT_FFMPEG_PATH)})")
                else: status_messages.append("FFmpeg (системный): НЕ НАЙДЕН.")
                if self.ffprobe_configured: status_messages.append(f"FFprobe (системный): OK ({os.path.basename(config_manager.CURRENT_FFPROBE_PATH)})")
                else: status_messages.append("FFprobe (системный): НЕ НАЙДЕН.")
                status_text_joined = "\n".join(status_messages)
                self._update_status(f"Статус системных инструментов (рабочая папка не задана):\n{status_text_joined}", append=True)
                return self.ffmpeg_configured and self.ffprobe_configured

             work_dir_to_check = self.current_work_dir 

        if work_dir_to_check != self.current_work_dir and os.path.isdir(work_dir_to_check):
            self.current_work_dir = work_dir_to_check 
        elif work_dir_to_check != self.current_work_dir and not os.path.isdir(work_dir_to_check): 
            try:
                if messagebox.askyesno("Создать директорию?", f"Директория '{work_dir_to_check}' не существует. Создать её?"):
                    os.makedirs(work_dir_to_check, exist_ok=True)
                    self._update_status(f"INFO: Создана рабочая директория: {work_dir_to_check}", append=True)
                    self.current_work_dir = work_dir_to_check 
                else: 
                    self.work_dir_path_var.set(self.current_work_dir or "") 
                    self._update_status(f"INFO: Создание директории {work_dir_to_check} отменено.", append=True)
                    return self.ffmpeg_configured and self.ffprobe_configured 
            except Exception as e:
                 messagebox.showerror("Ошибка", f"Не удалось создать директорию: {work_dir_to_check}\n{e}")
                 self.work_dir_path_var.set(self.current_work_dir or "")
                 return False
        
        if not self.current_work_dir: 
             messagebox.showerror("Критическая ошибка", "Рабочая директория не определена.")
             if hasattr(self, 'download_tools_button'): self.download_tools_button.config(state=tk.DISABLED)
             return False

        self._update_status(f"INFO: Применение рабочей директории: {self.current_work_dir}", append=True)
        self.ffmpeg_configured, self.ffprobe_configured, _espeak_status_ignored = config_manager.initialize_paths_from_work_dir(self.current_work_dir)
        config_manager.save_work_dir_to_config(self.current_work_dir) 

        status_messages = []
        if self.ffmpeg_configured: status_messages.append(f"FFmpeg: OK ({os.path.basename(config_manager.CURRENT_FFMPEG_PATH)})")
        else: status_messages.append("FFmpeg: НЕ НАЙДЕН в раб. папке (попробуйте 'Скачать FFmpeg').")
        
        if self.ffprobe_configured: status_messages.append(f"FFprobe: OK ({os.path.basename(config_manager.CURRENT_FFPROBE_PATH)})")
        else: status_messages.append("FFprobe: НЕ НАЙДЕН в раб. папке (попробуйте 'Скачать FFmpeg').")

        status_text_for_log = "\n".join(status_messages) 
        self._update_status(f"Статус инструментов в рабочей папке '{self.current_work_dir}':\n{status_text_for_log}", append=True)

        if hasattr(self, 'download_tools_button'): self.download_tools_button.config(state=tk.NORMAL if self.current_work_dir else tk.DISABLED)

        if show_success_message:
            final_message_parts = [f"Рабочая директория: {self.current_work_dir}"]
            all_critical_tools_ok = self.ffmpeg_configured and self.ffprobe_configured
            
            if all_critical_tools_ok:
                final_message_parts.append("FFmpeg и FFprobe настроены.")
            else:
                final_message_parts.append("ВНИМАНИЕ: FFmpeg и/или FFprobe НЕ настроены из этой папки!")
            
            final_message = "\n".join(final_message_parts)
            if not all_critical_tools_ok : 
                final_message += "\n\nИспользуйте кнопку 'Скачать FFmpeg' или скопируйте их вручную в подпапку 'ffmpeg' рабочей директории."
                messagebox.showwarning("Рабочая директория", final_message)
            else:
                messagebox.showinfo("Рабочая директория", final_message)

        return self.ffmpeg_configured and self.ffprobe_configured

    def trigger_download_tools_gui_wrapper(self):
        if not self.current_work_dir or not os.path.isdir(self.current_work_dir):
            messagebox.showerror("Ошибка", "Сначала выберите или создайте корректную рабочую директорию.")
            return
        
        self._update_status("INFO: Запуск скачивания FFmpeg...", append=True) 
        if hasattr(self, 'process_button'): self.process_button.config(state=tk.DISABLED)
        if hasattr(self, 'download_tools_button'): self.download_tools_button.config(state=tk.DISABLED)
        if hasattr(self, 'work_dir_apply_button'): self.work_dir_apply_button.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self._download_tools_thread_with_feedback, args=(self.current_work_dir,), daemon=True)
        thread.start()

    def _download_tools_thread_with_feedback(self, work_dir_for_download): 
        def gui_update_status(msg):
            self.root_tk.after(0, lambda: self._update_status(msg, append=True))

        gui_update_status("Начало скачивания инструментов (FFmpeg)...")
        ffmpeg_ok, ffprobe_ok, _ = config_manager.check_and_download_tools(work_dir_for_download, status_callback=gui_update_status)
        gui_update_status("Процесс скачивания инструментов завершен.")
        self.root_tk.after(0, lambda: self.apply_and_check_work_dir_gui_wrapper(show_success_message=True)) 
        
        if hasattr(self, 'process_button'): self.root_tk.after(0, lambda: self.process_button.config(state=tk.NORMAL))
        if hasattr(self, 'download_tools_button'): self.root_tk.after(0, lambda: self.download_tools_button.config(state=tk.NORMAL))
        if hasattr(self, 'work_dir_apply_button'): self.root_tk.after(0, lambda: self.work_dir_apply_button.config(state=tk.NORMAL))
    
    def copy_log_to_clipboard(self):
        try: log_content = self.status_text.get("1.0", tk.END); pyperclip.copy(log_content); self._update_status("Log copied to clipboard.", append=True); messagebox.showinfo("Log Copied", "The log content has been copied to your clipboard.")
        except pyperclip.PyperclipException as e: self._update_status(f"Error copying log: {e}. Pyperclip might not be configured correctly.", append=True); messagebox.showerror("Copy Error", f"Could not copy log: {e}\nInstall xclip or xsel (Linux).")
        except Exception as e_general: self._update_status(f"Unexpected error copying log: {e_general}", append=True); messagebox.showerror("Copy Error", f"Unexpected error: {e_general}")

    def _force_paste_event(self, event):
        widget = event.widget;
        if widget != self.video_source_entry: return
        if widget.cget("state") == 'readonly': return "break"
        try:
            widget.event_generate("<<Paste>>");
            return "break"
        except tk.TclError as e: self._update_status(f"DEBUG: TclError during <<Paste>> event generation: {e}", append=True); return "break"
        except Exception as e_paste: self._update_status(f"DEBUG: Generic error during hotkey paste handling: {e_paste}\n{traceback.format_exc()}", append=True); return "break"

    def _log_text_after_paste(self, widget, expected_clipboard_content):
        current_text_after_paste = widget.get(); self._update_status(f"DEBUG: Text in widget AFTER <<Paste>>: '{current_text_after_paste}'", append=True)
        if expected_clipboard_content and expected_clipboard_content not in current_text_after_paste: self._update_status(f"WARNING: Expected clipboard content ('{expected_clipboard_content[:30]}...') not found in widget after paste. Something might be wrong.", append=True)

    def make_context_menu_for_entry_fields(self):
        self.entry_context_menu = tk.Menu(self.root_tk, tearoff=0)
        self.entry_context_menu.add_command(label="Cut", command=lambda: self.root_tk.focus_get().event_generate("<<Cut>>"))
        self.entry_context_menu.add_command(label="Copy", command=lambda: self.root_tk.focus_get().event_generate("<<Copy>>"))
        self.entry_context_menu.add_command(label="Paste", command=lambda: self.root_tk.focus_get().event_generate("<<Paste>>"))
        self.entry_context_menu.add_separator()
        self.entry_context_menu.add_command(label="Select All", command=lambda: self.root_tk.focus_get().event_generate("<<SelectAll>>"))

    def show_entry_context_menu(self, event):
        widget = event.widget
        if isinstance(widget, (ttk.Entry, tk.Entry, scrolledtext.ScrolledText)) and widget.cget("state") != 'readonly':
            self.entry_context_menu.tk_popup(event.x_root, event.y_root)

    def is_youtube_url(self, url_string):
        if not url_string: return False
        youtube_regex = re.compile(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
        return bool(youtube_regex.match(url_string))

    def log_time(self, step_name, start_time):
        duration = time.time() - start_time; self.processing_times[step_name] = duration; self._update_status(f"⏱️ {step_name} took {duration:.2f} seconds.", append=True)

    def _update_status(self, message, append=True):
        def update_gui():
            if not hasattr(self, 'status_text') or not self.status_text.winfo_exists(): return 
            current_state = self.status_text.cget("state")
            self.status_text.config(state="normal")
            if append: self.status_text.insert(tk.END, str(message) + "\n")
            else: self.status_text.delete("1.0", tk.END); self.status_text.insert("1.0", str(message) + "\n")
            self.status_text.config(state=current_state); self.status_text.see(tk.END)
        if hasattr(self, 'root_tk') and self.root_tk.winfo_exists(): self.root_tk.after_idle(update_gui) 

    def _update_progress(self, value, step_name=""):
        clamped_value = max(0, min(100, int(value)))
        def update_gui():
            if not (hasattr(self, 'root_tk') and self.root_tk.winfo_exists() and \
                    hasattr(self, 'progressbar') and self.progressbar.winfo_exists() and \
                    hasattr(self, 'progress_label') and self.progress_label.winfo_exists()): return
            self.progressbar['value'] = clamped_value
            progress_text = f"{step_name}: {clamped_value}%" if step_name else f"Progress: {clamped_value}%"
            if clamped_value == 100 and step_name == "Completed": progress_text = f"Process: Done!"
            elif clamped_value == 100 and step_name and step_name != "Completed": progress_text = f"{step_name}: Done!"
            elif not step_name and clamped_value == 0: progress_text = "Progress: 0%"
            if hasattr(self, 'progress_label_text'): self.progress_label_text.set(progress_text)
        if hasattr(self, 'root_tk') and self.root_tk.winfo_exists(): self.root_tk.after_idle(update_gui) 

    def browse_video_for_source_entry(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm *.flv")])
        if path: self.video_source_text.set(path); self._update_status(f"Video file selected: {path}", append=False)

    def clear_video_source_text(self):
        self.video_source_text.set(""); self._update_status("Video source cleared.", append=False)

    def browse_srt(self):
        path = filedialog.askopenfilename(filetypes=[("SRT/VTT Files", "*.srt *.vtt")])
        if path: self.srt_path.set(path); self._update_status(f"Subtitles file selected: {path}. Its TEXT and TIMINGS will be used (after post-processing).", append=True)

    def clear_srt_path(self):
        self.srt_path.set(""); self._update_status("SRT/VTT file selection cleared. WhisperX will transcribe audio from scratch.", append=True)

    def browse_output_dir(self):
        path = filedialog.askdirectory();
        if path: self.output_dir_path.set(path); self._update_status(f"Output directory set to: {path}", append=True)

    def clear_output_dir(self):
        self.output_dir_path.set(""); self._update_status("Output directory cleared. Will use default.", append=True)
        
    def start_processing_thread(self): 
        if not (self.ffmpeg_configured and self.ffprobe_configured):
            if not self.apply_and_check_work_dir_gui_wrapper(show_success_message=True):
                 messagebox.showerror("Ошибка конфигурации", "Инструменты FFmpeg/FFprobe не настроены. Пожалуйста, укажите корректную рабочую директорию, где они находятся, или скачайте их с помощью кнопки 'Скачать FFmpeg'.")
                 return
        
        if not config_manager.CURRENT_FFMPEG_PATH or not config_manager.CURRENT_FFPROBE_PATH:
            messagebox.showerror("Ошибка FFmpeg/FFprobe", "Пути к FFmpeg/FFprobe не установлены. Проверьте настройки рабочей директории или системные пути.")
            return

        source_input = self.video_source_text.get().strip()
        if not source_input: messagebox.showerror("Error", "Please provide a video source (file path or YouTube URL)."); return
        is_youtube = self.is_youtube_url(source_input)
        video_file_to_process = None if is_youtube else source_input
        youtube_url_to_process = source_input if is_youtube else None
        if is_youtube and yt_dlp is None: messagebox.showerror("Error", "yt-dlp library is not installed..."); return
        if not is_youtube and not os.path.exists(video_file_to_process): messagebox.showerror("Error", f"Local video file not found: {video_file_to_process}"); return
        
        local_srt_path_input = self.srt_path.get().strip() 
        if local_srt_path_input and srt is None: messagebox.showerror("Error", "The 'srt' library is not installed..."); return
        
        output_dir_user_specified = self.output_dir_path.get().strip()
        op_key = "Translate & Dub Video"
        self.process_button.config(state="disabled"); self._update_status(f"🚀 Starting operation: {op_key}...", append=False)
        self.progress_frame.pack(padx=10, pady=(5, 5), fill="x", after=self.process_button)
        self._update_progress(0, "Initializing"); self.root_tk.update_idletasks()
        self.processing_times = {}; self.total_start_time = time.time()
        self.downloaded_video_path = None; self.downloaded_srt_path = None; self.downloaded_srt_lang = None
        
        thread = threading.Thread(target=self.actual_processing,
                                  args=(video_file_to_process, local_srt_path_input, op_key,
                                        is_youtube, youtube_url_to_process,
                                        output_dir_user_specified), 
                                  daemon=True)
        thread.start()
        
    def actual_processing(self, video_file_path, local_srt_path_input, op_key,
                          is_youtube, youtube_url_str,
                          user_output_dir):
        current_temp_dir = None; output_path = None; success = False
        
        # Адаптируем веса шагов
        processing_external_srt = bool(local_srt_path_input and os.path.exists(local_srt_path_input))
        
        dub_steps = { 
            "YouTube Download & Subs": 8 if is_youtube else 0, 
            "Audio Extraction": 5, 
            "SRT Processing (External)": 15 if processing_external_srt else 0,
            "Transcription & Alignment (WhisperX)": 0 if processing_external_srt else 45, # Запускается только если нет внешнего SRT
            "Translation": 10, 
            "Voice Synthesis": 27, 
            "Video Assembly": 5 
        }
        total_weight_op = sum(w for w in dub_steps.values() if w > 0) 

        video_to_use = video_file_path
        subs_source_tag = ""; diarization_data_for_cloning = None; skip_translation_step = False
        total_raw_tts_duration = 0.0
        total_adjusted_segments_duration = 0.0
        debug_srt_path = None 
        
        segments_for_tts = [] 

        try:
            current_temp_dir = create_temp_dir(); self._update_status(f"Using temp dir: {current_temp_dir}")
            device = "cuda" if torch.cuda.is_available() else "cpu"; current_progress = 0
            
            # --- YouTube Download (если нужно) ---
            download_step_name = "YouTube Download & Subs"
            if is_youtube:
                self._update_status(f"➡️ {download_step_name} from {youtube_url_str}...");
                start_t_dl = time.time(); 
                download_output_dir_vid = os.path.join(current_temp_dir, "youtube_download")
                os.makedirs(download_output_dir_vid, exist_ok=True)
                
                self.downloaded_video_path, self.downloaded_srt_path, self.downloaded_srt_lang = video_processor.download_youtube_video(
                    url=youtube_url_str, output_dir=download_output_dir_vid,
                    preferred_sub_lang='ru', fallback_sub_lang='en')
                video_to_use = self.downloaded_video_path
                self.log_time(download_step_name, start_t_dl)

                if not local_srt_path_input and self.downloaded_srt_path and os.path.exists(self.downloaded_srt_path) and self.downloaded_srt_lang == 'ru':
                    self._update_status(f"✅ Russian YouTube subtitles downloaded and will be used: {os.path.basename(self.downloaded_srt_path)}", append=True)
                    local_srt_path_input = self.downloaded_srt_path # Этот файл теперь будет обработан как внешний SRT
                    # subs_source_tag и skip_translation_step будут установлены ниже, при обработке local_srt_path_input
            if download_step_name in dub_steps: current_progress += dub_steps[download_step_name]
            self._update_progress(current_progress / total_weight_op * 100, "Preparation")
            
            if not video_to_use or not os.path.exists(video_to_use): 
                raise FileNotFoundError(f"Video file for processing not found: {video_to_use or 'None'}")
            
            stt_audio_path_for_cloning = None; original_audio_for_mixing = None
            step_name_ae = "Audio Extraction"
            self._update_status(f"➡️ {step_name_ae} from {os.path.basename(video_to_use)}..."); start_t_ae = time.time()
            original_audio_for_mixing = video_processor.extract_audio(video_to_use, os.path.join(current_temp_dir, "original_for_mix.wav"), sample_rate=44100)
            stt_audio_path_for_cloning = video_processor.extract_audio(video_to_use, os.path.join(current_temp_dir, "audio_for_diar_clone.wav"), sample_rate=16000) 
            self.log_time(step_name_ae, start_t_ae)
            if stt_audio_path_for_cloning is None or original_audio_for_mixing is None: 
                raise RuntimeError("Audio for cloning or mixing was not prepared.")
            current_progress += dub_steps[step_name_ae]
            self._update_progress(current_progress / total_weight_op * 100, step_name_ae)

            if user_output_dir and os.path.isdir(user_output_dir): final_output_dir = user_output_dir
            elif video_file_path and not is_youtube : final_output_dir = os.path.dirname(video_file_path)
            else: final_output_dir = os.path.join(os.getcwd(), "Translated_Videos"); os.makedirs(final_output_dir, exist_ok=True)
            self._update_status(f"Final outputs will be saved to: {final_output_dir}", append=True)
            video_name_base_for_outputs = os.path.splitext(os.path.basename(video_to_use))[0]

            # --- ОБРАБОТКА SRT ИЛИ ТРАНСКРИПЦИЯ ---
            step_name_srt_proc = "SRT Processing (External)"
            step_name_transcribe = "Transcription & Alignment (WhisperX)"

            if local_srt_path_input and os.path.exists(local_srt_path_input):
                self._update_status(f"➡️ {step_name_srt_proc}: {os.path.basename(local_srt_path_input)}...");
                self._update_progress(current_progress / total_weight_op * 100, step_name_srt_proc)
                start_t_srt_proc = time.time()
                
                segments_from_srt = transcriber.parse_srt_file(local_srt_path_input)
                if not segments_from_srt:
                    raise ValueError(f"Provided SRT file {local_srt_path_input} could not be parsed or is empty.")
                self._update_status(f"  Parsed {len(segments_from_srt)} segments from SRT.", append=True)
                subs_source_tag = "_customsrt"

                texts_for_punct = [s.get('text', '') for s in segments_from_srt]
                restored_texts = transcriber._restore_punctuation(texts_for_punct)
                if len(restored_texts) == len(segments_from_srt):
                    for i, seg in enumerate(segments_from_srt): seg['text'] = restored_texts[i]
                
                self._update_status(f"  Performing diarization on audio to assign speakers to SRT segments...", append=True)
                diarization_data_for_cloning = transcriber.perform_diarization_only(stt_audio_path_for_cloning, device=device)
                if diarization_data_for_cloning is not None and not diarization_data_for_cloning.empty:
                    segments_for_tts = transcriber.assign_srt_segments_to_speakers(segments_from_srt, diarization_data_for_cloning, trust_srt_speaker_field=False)
                else:
                    self._update_status("  Warning: Diarization returned no data. Assigning default speaker.", append=True)
                    for seg in segments_from_srt: seg['speaker'] = 'SPEAKER_00'
                    segments_for_tts = segments_from_srt
                
                self._update_status(f"  Post-processing {len(segments_for_tts)} SRT segments...", append=True)
                segments_for_tts = transcriber._postprocess_srt_segments(segments_for_tts, is_external_srt=True) # Указываем, что это внешний SRT
                self._update_status(f"  After post-processing: {len(segments_for_tts)} segments for TTS.", append=True)

                sample_text_for_lang_detect = " ".join([s.get('text','') for s in segments_for_tts[:10] if s.get('text','').strip()])
                if detect and sample_text_for_lang_detect:
                    try:
                        detected_lang_srt = detect(sample_text_for_lang_detect)
                        self._update_status(f"  Langdetect: Detected language for SRT text: '{detected_lang_srt}'", append=True)
                        if detected_lang_srt == 'ru': skip_translation_step = True
                    except: pass # Игнорируем ошибки langdetect
                elif ('.ru.' in os.path.basename(local_srt_path_input).lower() or \
                        os.path.basename(local_srt_path_input).lower().endswith(('.ru.srt', '.ru.vtt'))):
                    skip_translation_step = True
                
                if skip_translation_step: self._update_status("INFO: Text from SRT seems Russian. Translation step will be skipped.", append=True)

                self.log_time(step_name_srt_proc, start_t_srt_proc)
                current_progress += dub_steps[step_name_srt_proc]
            
            else: # Если внешний SRT не предоставлен, транскрибируем аудио
                self._update_status(f"➡️ {step_name_transcribe}...");
                self._update_progress(current_progress / total_weight_op * 100, step_name_transcribe)
                start_t_transcribe = time.time()
                
                # Аудио английское, поэтому язык для STT - 'en'
                segments_from_whisper, diarization_data_for_cloning = transcriber.transcribe_and_diarize_audio(
                    stt_audio_path_for_cloning, language_for_stt='en', return_diarization_df=True
                ) 
                if not segments_from_whisper:
                    raise ValueError("WhisperX transcription failed or returned no segments.")
                self._update_status(f"  WhisperX produced {len(segments_from_whisper)} segments.", append=True)
                subs_source_tag = "_whspxsubs"
                
                self._update_status(f"  Post-processing {len(segments_from_whisper)} WhisperX segments...", append=True)
                segments_for_tts = transcriber._postprocess_srt_segments(segments_from_whisper, is_external_srt=False) # Это не внешний SRT
                self._update_status(f"  After post-processing: {len(segments_for_tts)} segments for TTS.", append=True)
                
                skip_translation_step = False # Текст от WhisperX (английский) точно нужно переводить
                self.log_time(step_name_transcribe, start_t_transcribe)
                current_progress += dub_steps[step_name_transcribe]
            
            # Обновляем прогресс после этапа SRT/транскрипции
            current_progress_key = step_name_srt_proc if processing_external_srt else step_name_transcribe
            self._update_progress(current_progress / total_weight_op * 100, current_progress_key)


            if not segments_for_tts: 
                raise RuntimeError("No segments available for TTS after SRT processing or transcription.")

            final_srt_for_tts_path = os.path.join(final_output_dir, f"{video_name_base_for_outputs}{subs_source_tag}_final_for_tts.srt")
            try:
                srt_content_to_save = transcriber.segments_to_srt(segments_for_tts) 
                with open(final_srt_for_tts_path, 'w', encoding='utf-8') as f_debug:
                    f_debug.write(srt_content_to_save)
                self._update_status(f"DEBUG: Final SRT for TTS saved to: {final_srt_for_tts_path}", append=True)
                debug_srt_path = final_srt_for_tts_path 
            except Exception as e_save_srt:
                self._update_status(f"WARNING: Could not save final debug SRT for TTS: {e_save_srt}", append=True)

            for seg_idx, seg_val in enumerate(segments_for_tts):
                cleaned_text = seg_val.get('text', ''); cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text).strip(); segments_for_tts[seg_idx]['text'] = cleaned_text

            # --- Translation (если нужно) ---
            translated_segments = list(segments_for_tts) 
            step_name_trans = "Translation"
            if not skip_translation_step:
                self._update_status(f"➡️ {step_name_trans} (Helsinki-NLP)..."); 
                self._update_progress(current_progress / total_weight_op * 100, step_name_trans)
                start_t_tr = time.time()
                translator.load_translator_model(device=device)
                translated_segments = translator.translate_segments(segments_for_tts) 
                self.log_time(step_name_trans, start_t_tr)
            else:
                self._update_status("➡️ Translation step skipped.", append=True);
                for seg_idx, seg_val in enumerate(translated_segments): 
                    translated_segments[seg_idx]['translated_text'] = seg_val.get('text', '')
            current_progress += dub_steps[step_name_trans]
            self._update_progress(current_progress / total_weight_op * 100, step_name_trans)

            # --- Voice Synthesis ---
            step_name_vs = "Voice Synthesis"
            self._update_status(f"➡️ {step_name_vs} (Coqui XTTS-v2)...");
            progress_before_synthesis_step = current_progress
            def synthesis_progress_callback(fraction_done):
                # total_weight_op уже посчитан с учетом условных шагов
                base_prog_val = (progress_before_synthesis_step / total_weight_op * 100)
                step_prog_val = (dub_steps.get(step_name_vs, 0) / total_weight_op * 100) * fraction_done
                self._update_progress(int(base_prog_val + step_prog_val), f"{step_name_vs} ({int(fraction_done*100)}%)")
            
            start_t_vs = time.time()
            voice_cloner.load_tts_model(device=device)
            
            final_dubbed_audio_path, total_raw_tts_duration, total_adjusted_segments_duration = voice_cloner.synthesize_speech_segments(
                translated_segments, stt_audio_path_for_cloning, current_temp_dir,
                diarization_result_df=diarization_data_for_cloning, progress_callback=synthesis_progress_callback,
                language='ru' 
            ) 
            self.log_time(step_name_vs, start_t_vs)
            self._update_status(f"📊 Voice Synthesis Stats: Total raw TTS duration: {total_raw_tts_duration:.2f}s, Total final segments duration (incl. silence): {total_adjusted_segments_duration:.2f}s", append=True)
            current_progress += dub_steps[step_name_vs]
            self._update_progress(current_progress / total_weight_op * 100, step_name_vs)

            # --- Video Assembly ---
            step_name_va = "Video Assembly"
            self._update_status(f"➡️ {step_name_va} (FFmpeg)..."); 
            start_t_va = time.time()
            output_path = os.path.join(final_output_dir, f"{video_name_base_for_outputs}_dubbed_ru{subs_source_tag}.mp4")

            if os.path.exists(output_path):
                self._update_status(f"INFO: Output file {os.path.basename(output_path)} already exists. Attempting to delete it.", append=True)
                try: os.remove(output_path); self._update_status(f"INFO: Successfully deleted existing file: {os.path.basename(output_path)}", append=True)
                except OSError as e_del: self._update_status(f"WARNING: Could not delete existing output file {os.path.basename(output_path)}: {e_del}.", append=True)

            if not os.path.exists(original_audio_for_mixing): raise FileNotFoundError(f"Original audio for mixing not found: {original_audio_for_mixing}")
            if not final_dubbed_audio_path or not os.path.exists(final_dubbed_audio_path): raise FileNotFoundError(f"Final dubbed audio not found or not generated: {final_dubbed_audio_path}")
            
            video_processor.mix_and_replace_audio(video_to_use, original_audio_for_mixing, final_dubbed_audio_path, output_path, original_volume=0.10, dubbed_volume=0.95)
            self.log_time(step_name_va, start_t_va);
            current_progress += dub_steps[step_name_va]
            self._update_progress(100, "Completed") 
            
            if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                total_time_seconds = time.time() - self.total_start_time; total_time_minutes = total_time_seconds / 60.0
                self._update_status(f"\n✅🎉 Total processing time: {total_time_minutes:.2f} minutes.")
                
                final_video_duration_str = "N/A"
                try:
                    final_video_info = ffmpeg.probe(output_path); final_video_duration = float(final_video_info['format']['duration'])
                    final_video_duration_str = f"{final_video_duration:.2f}s"
                    self._update_status(f"  Output video duration: {final_video_duration_str}", append=True)
                except Exception as e_probe: self._update_status(f"  Could not probe output video duration: {e_probe}", append=True)
                
                self._update_status(f"📊 Final Audio Durations: Raw TTS Sum: {total_raw_tts_duration:.2f}s, Final Segments Sum (incl. silence): {total_adjusted_segments_duration:.2f}s, Video: {final_video_duration_str}", append=True)
                self._update_status(f"✅ Operation '{op_key}' finished successfully!", append=True); self._update_status(f"Output file:\n{output_path}", append=True)
                
                success_message_parts = [f"Operation '{op_key}' completed in {total_time_minutes:.2f} minutes!\n\nOutput Video: {output_path}"]
                if debug_srt_path and os.path.exists(debug_srt_path): 
                     success_message_parts.append(f"Processed Subtitles used for TTS: {debug_srt_path}")
                
                self.root_tk.after(0, lambda msg="\n".join(success_message_parts): messagebox.showinfo("Success", msg)); success = True
            else:
                err_msg_out = f"⚠️ Operation '{op_key}' finished, but the expected output file was not found or is empty"
                if output_path: err_msg_out += f":\n{output_path}"
                self._update_status(err_msg_out, append=True)
                if output_path: self.root_tk.after(0, lambda: messagebox.showwarning("Warning", f"Operation '{op_key}' finished, but the output file seems missing or empty."))
        except Exception as e:
            tb_str = traceback.format_exc(); failed_step = "Initialization or Unknown Step"
            active_steps = {k:v for k,v in dub_steps.items() if v > 0} # Учитываем только активные шаги
            
            temp_cumulative_weight = 0
            for name, weight in active_steps.items():
                if current_progress < temp_cumulative_weight + weight: 
                    failed_step = name
                    break
                temp_cumulative_weight += weight
            if failed_step == "Initialization or Unknown Step" and current_progress == 0 : 
                if is_youtube: failed_step = "YouTube Download & Subs"
                else: failed_step = "Audio Extraction"

            error_message_detail = str(e)
            ffmpeg_stderr = "";
            if ffmpeg and isinstance(e, ffmpeg.Error):
                try: ffmpeg_stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "Stderr not captured."; ffmpeg_stderr = f"\n--- FFmpeg Output (stderr) ---\n{ffmpeg_stderr}\n------------------------------"
                except Exception as stderr_e: ffmpeg_stderr = f"\n--- Could not decode FFmpeg stderr: {stderr_e} ---"
            elif yt_dlp and isinstance(e, RuntimeError) and ("Failed to download YouTube video" in str(e) or "yt-dlp download error" in str(e)):
                ffmpeg_stderr = f"\n--- YouTube Download Error Details ---\n{error_message_detail}\n------------------------------------"
            
            error_message_full = f"\n❌❌❌ ERROR during '{failed_step}' step of '{op_key}':\n{type(e).__name__}: {error_message_detail}{ffmpeg_stderr}\n--- Traceback ---\n{tb_str}-----------------"
            print(error_message_full); self._update_status(error_message_full, append=True)
            
            display_error_user = f"An error occurred during the '{failed_step}' step:\n\n{error_message_detail}"
            if ffmpeg_stderr and "YouTube Download Error Details" not in ffmpeg_stderr : display_error_user += "\n\n(Check logs for FFmpeg output details)"
            elif "YouTube Download Error Details" in ffmpeg_stderr: display_error_user = f"Error downloading YouTube video/subs:\n\n{error_message_detail}\n\n(Check logs for details)"
            self.root_tk.after(0, lambda f_step=failed_step, e_msg=display_error_user: messagebox.showerror("Processing Error", e_msg))
        finally:
            self.root_tk.after(0, self.stop_and_hide_progressbar); cleanup_temp_dir(current_temp_dir)
            self.root_tk.after(0, self.enable_process_button)
            if not success: self.root_tk.after(0, lambda: self._update_status(f"\n❌ Operation '{op_key}' failed.", append=True))

    def stop_and_hide_progressbar(self):
        if hasattr(self, 'root_tk') and self.root_tk.winfo_exists():
            try:
                if hasattr(self, 'progress_frame') and self.progress_frame.winfo_ismapped(): 
                    self.progress_frame.pack_forget()
                if hasattr(self, 'progressbar'): self.progressbar['value'] = 0
                if hasattr(self, 'progress_label_text'): self.progress_label_text.set("Progress: 0%")
            except tk.TclError as e: print(f"Warning: Could not reset/hide progress bar: {e}")

    def enable_process_button(self):
        if hasattr(self, 'root_tk') and self.root_tk.winfo_exists():
             try: 
                if hasattr(self, 'process_button'): self.process_button.config(state=tk.NORMAL)
             except tk.TclError as e: print(f"Warning: Could not enable process button: {e}")


if __name__ == "__main__":
    initial_work_dir = config_manager.get_work_dir_from_config()
    ffmpeg_ready, ffprobe_ready, _espeak_status = config_manager.initialize_paths_from_work_dir(initial_work_dir) 

    if not pyperclip: messagebox.showwarning("Dependency Warning", "The 'pyperclip' library is not installed. Log copying will not work. Please install it using: pip install pyperclip")
    if detect is None: messagebox.showwarning("Dependency Warning", "The 'langdetect' library is not installed. Language detection of custom SRTs will rely on filename only. Install with: pip install langdetect")
    if transcriber.PunctuationModel is None: messagebox.showwarning("Dependency Warning", "The 'deepmultilingualpunctuation' library not found or failed to load. Punctuation restoration will be skipped. Install with 'pip install deepmultilingualpunctuation'")

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
    
    if not config_manager.CURRENT_FFMPEG_PATH or not config_manager.CURRENT_FFPROBE_PATH :
        ffmpeg_ok_startup, ffmpeg_msg_startup = video_processor.check_command_availability('ffmpeg') 
        ffprobe_ok_startup, ffprobe_msg_startup = video_processor.check_command_availability('ffprobe')
        if not (ffmpeg_ok_startup and ffprobe_ok_startup):
             messagebox.showerror("Fatal Dependency Error", 
                                 f"FFmpeg and/or ffprobe not found or not executable.\n"
                                 f"FFmpeg: {ffmpeg_msg_startup}\nFFprobe: {ffprobe_msg_startup}\n"
                                 f"Please ensure they are in your system's PATH or configure a valid work directory with these tools (and use 'Apply and Check' or 'Download FFmpeg').")
    
    if srt is None: messagebox.showwarning("Dependency Warning", "The 'srt' library is not installed. SRT file parsing and generation will be unavailable. Please install it using: pip install srt")
    if yt_dlp is None: print("WARNING: yt-dlp is not installed. YouTube download functionality will be disabled.")
    
    root_tk_main = tk.Tk(); 
    app_instance = None
    try:
        app_instance = App(root_tk_main)
        if not root_tk_main.winfo_exists(): 
            sys.exit("Приложение закрыто из-за ошибки инициализации рабочей директории.")
    except Exception as e_app_init:
        messagebox.showerror("Ошибка инициализации приложения", f"Произошла ошибка при создании окна приложения:\n{e_app_init}")
        traceback.print_exc() 
        sys.exit(f"Ошибка инициализации App: {e_app_init}")

    
    if initial_work_dir: 
        if not (ffmpeg_ready and ffprobe_ready):
            if app_instance: app_instance._update_status("WARNING: Рабочая директория загружена, но FFmpeg/FFprobe не найдены в ней или системно. Проверьте настройки или используйте 'Скачать FFmpeg'.", append=True)
    elif not (ffmpeg_ready and ffprobe_ready): 
         if app_instance: app_instance._update_status("WARNING: Рабочая директория не настроена, и системные FFmpeg/FFprobe не найдены. Пожалуйста, настройте рабочую директорию и используйте 'Скачать FFmpeg'.", append=True)


    if show_pytorch_warning:
        if pytorch_msg_type == "info": root_tk_main.after(200, lambda: messagebox.showinfo("System Info", pytorch_warning_msg))
        elif pytorch_msg_type == "warning": root_tk_main.after(200, lambda: messagebox.showwarning("System Configuration Warning", pytorch_warning_msg))
    try: root_tk_main.mainloop()
    except Exception as e_gui: print(f"\n--- GUI Error ---"); traceback.print_exc(); messagebox.showerror("Application Error", f"A critical error occurred in the application's main loop:\n\n{e_gui}")
