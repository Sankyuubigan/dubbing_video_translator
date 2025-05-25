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
    # ... (импорты для torch safe globals без изменений)
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
# ... (импорты библиотек без изменений)
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
        
        try:
            icon_path = "icon.ico" 
            if os.path.exists(icon_path):
                self.root_tk.iconbitmap(default=icon_path)
                print(f"INFO: Application icon set from {icon_path}")
            else:
                if getattr(sys, 'frozen', False): # PyInstaller
                    icon_path_alt = os.path.join(sys._MEIPASS, icon_path)
                    if os.path.exists(icon_path_alt):
                        self.root_tk.iconbitmap(default=icon_path_alt)
                        print(f"INFO: Application icon set from MEIPASS: {icon_path_alt}")
                    else:
                        print(f"WARNING: Icon file not found at {os.path.abspath(icon_path)} or in MEIPASS.")
                else:
                    print(f"WARNING: Icon file not found at {os.path.abspath(icon_path)}")
        except Exception as e_icon: 
            print(f"WARNING: Could not set application icon: {e_icon}")

        self.root_tk.geometry("700x720")
        
        # ... (инициализация переменных без изменений) ...
        self.current_work_dir = config_manager.get_work_dir_from_config()
        if not self.current_work_dir:
            default_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
            chosen_dir = filedialog.askdirectory(title="Выберите или создайте рабочую директорию", initialdir=default_dir)
            if chosen_dir:
                self.current_work_dir = chosen_dir
                os.makedirs(self.current_work_dir, exist_ok=True) 
                config_manager.save_work_dir_to_config(self.current_work_dir) 
            else: 
                messagebox.showerror("Ошибка", "Рабочая директория не выбрана. Приложение не может продолжить.")
                self.root_tk.destroy(); return
        self.work_dir_path_var = tk.StringVar(value=self.current_work_dir)
        self.ffmpeg_configured = False; self.ffprobe_configured = False
        self.video_source_text = tk.StringVar()
        self.srt_path = tk.StringVar()
        self.processing_times = {}; self.total_start_time = 0
        self.downloaded_video_path = None; self.downloaded_srt_path = None; self.downloaded_srt_lang = None
        self.process_in_chunks_var = tk.BooleanVar(value=False)
        self.chunk_duration_var = tk.StringVar(value="60")
        self.start_time_offset_var = tk.StringVar(value="0:00") 
        self.processed_video_chunks = [] 
        self.processed_original_audio_chunks = []
        self.processed_dubbed_audio_chunks = []
        self.all_segments_for_processing = [] 
        self.total_video_duration = 0.0
        self.chunk_processing_active = False 
        self.fully_processed_duration = 0.0
        self.current_temp_dir_for_operation = None
        self.current_final_output_dir_for_operation = None

        style = ttk.Style(); style.theme_use('clam')
        style.configure("TButton", padding=6, relief="raised", background="#d9d9d9", foreground="black")
        style.map("TButton", background=[('pressed', '#c0c0c0'), ('active', '#e8e8e8')], relief=[('pressed', 'sunken')])
        style.configure("TLabel", padding=6, background="#f0f0f0")
        style.configure("TEntry", padding=6)
        
        self.make_ui() 

        if self.current_work_dir: 
            self.apply_and_check_work_dir_gui_wrapper(show_success_message=False, initial_setup=True)

    def make_ui(self):
        self.root_tk.configure(bg="#f0f0f0")

        self.entry_context_menu = tk.Menu(self.root_tk, tearoff=0)
        self.entry_context_menu.add_command(label="Вырезать", command=self.do_cut)
        self.entry_context_menu.add_command(label="Копировать", command=self.do_copy)
        self.entry_context_menu.add_command(label="Вставить", command=self.do_paste)
        self.entry_context_menu.add_separator()
        self.entry_context_menu.add_command(label="Выделить всё", command=self.do_select_all)
        
        # Убираем глобальное логирование KeyPress, т.к. оно слишком шумное
        # self.root_tk.bind_all("<KeyPress>", self.log_very_global_keypress_event, add="+")
        # self.root_tk.bind_all("<KeyRelease>", self.log_very_global_keyrelease_event, add="+")

        # --- Рабочая директория ---
        work_dir_frame = ttk.LabelFrame(self.root_tk, text="Рабочая директория", padding=(10, 5))
        work_dir_frame.pack(padx=10, pady=5, fill="x")
        self.work_dir_entry = ttk.Entry(work_dir_frame, textvariable=self.work_dir_path_var, width=50) 
        self.work_dir_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        self.work_dir_entry.bind("<Button-3>", self.show_context_menu) 
        self.work_dir_entry.bind("<KeyPress>", self.handle_keypress_for_paste_and_log) 
        # ... (остальные кнопки для work_dir_frame)
        self.work_dir_browse_button = ttk.Button(work_dir_frame, text="Выбрать...", command=self.browse_work_dir) 
        self.work_dir_browse_button.pack(side=tk.LEFT, padx=(0,5))
        self.work_dir_apply_button = ttk.Button(work_dir_frame, text="Применить и Проверить", command=self.apply_and_check_work_dir_gui_wrapper)
        self.work_dir_apply_button.pack(side=tk.LEFT, padx=(0,5))
        self.download_tools_button = ttk.Button(work_dir_frame, text="Скачать FFmpeg", command=self.trigger_download_tools_gui_wrapper, state=tk.NORMAL if self.current_work_dir else tk.DISABLED)
        self.download_tools_button.pack(side=tk.LEFT)

        # --- Источник видео ---
        source_frame = ttk.LabelFrame(self.root_tk, text="Video Source (File Path or YouTube URL)", padding=(10, 5))
        source_frame.pack(padx=10, pady=5, fill="x")
        self.video_source_entry = ttk.Entry(source_frame, textvariable=self.video_source_text, width=60)
        self.video_source_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        self.video_source_entry.bind("<Button-3>", self.show_context_menu) 
        self.video_source_entry.bind("<KeyPress>", self.handle_keypress_for_paste_and_log)
        
        # ... (остальные элементы source_frame)
        self.video_browse_button = ttk.Button(source_frame, text="Browse File...", command=self.browse_video_for_source_entry); self.video_browse_button.pack(side=tk.LEFT)
        self.video_clear_button = ttk.Button(source_frame, text="Clear", command=self.clear_video_source_text); self.video_clear_button.pack(side=tk.LEFT, padx=(5,0))
        yt_subs_info_label = ttk.Label(source_frame, text="For YouTube URLs, subtitles (RU then EN) are auto-downloaded if available (used if no local SRT).", font=("Segoe UI", 8)); yt_subs_info_label.pack(side=tk.BOTTOM, anchor=tk.W, padx=0, pady=(5,0))
        
        # --- Файл субтитров ---
        self.srt_outer_frame = ttk.LabelFrame(self.root_tk, text="Subtitles File (RU text, timings will be used & post-processed)", padding=(10, 5)); self.srt_outer_frame.pack(padx=10, pady=5, fill="x")
        srt_input_line_frame = ttk.Frame(self.srt_outer_frame); srt_input_line_frame.pack(fill="x")
        self.srt_path_entry = ttk.Entry(srt_input_line_frame, textvariable=self.srt_path, width=60, state="readonly"); self.srt_path_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        self.srt_path_entry.bind("<Button-3>", self.show_context_menu) 
        self.srt_browse_button = ttk.Button(srt_input_line_frame, text="Browse...", command=self.browse_srt) 
        self.srt_browse_button.pack(side=tk.LEFT)
        self.srt_clear_button = ttk.Button(srt_input_line_frame, text="Clear", command=self.clear_srt_path) ; self.srt_clear_button.pack(side=tk.LEFT, padx=(5,0))
        
        # --- Опции обработки чанков и времени начала ---
        processing_options_frame = ttk.LabelFrame(self.root_tk, text="Processing Options", padding=(10, 5))
        processing_options_frame.pack(padx=10, pady=5, fill="x")
        
        chunk_line_frame = ttk.Frame(processing_options_frame)
        chunk_line_frame.pack(fill="x", pady=(0,5))
        self.chunk_checkbox = ttk.Checkbutton(chunk_line_frame, text="Test: Process only the first chunk ", variable=self.process_in_chunks_var)
        self.chunk_checkbox.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(chunk_line_frame, text="Chunk Duration (sec):").pack(side=tk.LEFT)
        self.chunk_duration_entry = ttk.Entry(chunk_line_frame, textvariable=self.chunk_duration_var, width=5)
        self.chunk_duration_entry.pack(side=tk.LEFT)
        self.chunk_duration_entry.bind("<Button-3>", self.show_context_menu)
        self.chunk_duration_entry.bind("<KeyPress>", self.handle_keypress_for_paste_and_log)

        start_time_line_frame = ttk.Frame(processing_options_frame)
        start_time_line_frame.pack(fill="x")
        ttk.Label(start_time_line_frame, text="Start processing from (mm:ss or seconds):").pack(side=tk.LEFT, padx=(0,5))
        self.start_time_offset_entry = ttk.Entry(start_time_line_frame, textvariable=self.start_time_offset_var, width=10)
        self.start_time_offset_entry.pack(side=tk.LEFT, padx=(0,10))
        self.start_time_offset_entry.bind("<Button-3>", self.show_context_menu)
        self.start_time_offset_entry.bind("<KeyPress>", self.handle_keypress_for_paste_and_log)

        # --- Кнопки управления ---
        # ... (без изменений)
        self.process_button = ttk.Button(self.root_tk, text="Translate & Dub Video", command=self.start_processing_thread); self.process_button.pack(pady=(10,0))
        self.save_processed_button = ttk.Button(self.root_tk, text="Save Processed Part", command=self.save_processed_part_gui_wrapper, state=tk.DISABLED)
        self.save_processed_button.pack(pady=(5,10))

        # --- Прогресс и Лог ---
        # ... (без изменений)
        self.progress_frame = ttk.Frame(self.root_tk, padding=(10,5))
        self.progress_label_text = tk.StringVar(value="Progress: 0%"); self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_label_text, width=30, anchor="w"); self.progress_label.pack(side=tk.LEFT, padx=(0,5))
        self.progressbar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', maximum=100); self.progressbar.pack(side=tk.LEFT, fill="x", expand=True)
        status_frame_outer = ttk.LabelFrame(self.root_tk, text="Status / Log", padding=(10,5)); status_frame_outer.pack(padx=10, pady=(0,5), fill="both", expand=True)
        self.status_text = scrolledtext.ScrolledText(status_frame_outer, height=10, wrap=tk.WORD, state="disabled", bg="#ffffff", relief="sunken", borderwidth=1, font=("Consolas", 9)); self.status_text.pack(side=tk.TOP, fill="both", expand=True)
        self.copy_log_button = ttk.Button(status_frame_outer, text="Copy Log to Clipboard", command=self.copy_log_to_clipboard); self.copy_log_button.pack(side=tk.TOP, pady=(5,0))

    def log_very_global_keypress_event(self, event):
        # РАСКОММЕНТИРУЙТЕ ЭТУ СТРОКУ ДЛЯ САМОГО ПОДРОБНОГО ЛОГА
        # self._update_status(f"GLOBAL KeyPress: Focus='{self.root_tk.focus_get()}', Widget='{event.widget}', KeySym='{event.keysym}', State='{event.state:#06x}', Char='{repr(event.char)}'", True)
        pass
        
    def log_very_global_keyrelease_event(self, event):
        # РАСКОММЕНТИРУЙТЕ ЭТУ СТРОКУ ДЛЯ САМОГО ПОДРОБНОГО ЛОГА
        # self._update_status(f"GLOBAL KeyRelease: Focus='{self.root_tk.focus_get()}', Widget='{event.widget}', KeySym='{event.keysym}', State='{event.state:#06x}'", True)
        pass

    def handle_keypress_for_paste_and_log(self, event):
        """Обрабатывает KeyPress на виджете, логирует И ПЫТАЕТСЯ ВСТАВИТЬ."""
        widget = event.widget
        widget_name = getattr(widget, 'winfo_name', lambda: str(widget))()
        keysym = event.keysym.lower()
        char = event.char
        state = event.state # Это битовая маска состояния модификаторов

        # Подробное логирование КАЖДОГО нажатия на этом виджете
        self._update_status(f"Widget '{widget_name}': KeyPress ks='{keysym}', char='{repr(char)}', state='{state:#06x}'", True)

        # Флаги состояния (стандартные для Tkinter)
        CONTROL_MASK = 0x0004  # Control key
        SHIFT_MASK = 0x0001    # Shift key
        # Для macOS Command может быть сложнее, но Tkinter часто мапит его на Control или специфичный бит
        # На основе вашего лога (state='0x002c' для Ctrl+V, где 0x0004 присутствует)
        
        paste_triggered = False

        # Проверка для Ctrl+V (или Command+V на Mac, если он мапится на Control state)
        # Ваш лог показал char='\x16' и state содержал Control (0x0004)
        if char == '\x16' and (state & CONTROL_MASK):
            self._update_status(f"DEBUG: Detected char '\\x16' (SYN) with Control state on '{widget_name}'. Assuming Ctrl+V.", True)
            paste_triggered = True
        elif (state & CONTROL_MASK) and keysym == 'v': # Стандартная проверка, если char не \x16
            self._update_status(f"DEBUG: Detected Control + 'v' keysym on '{widget_name}'. Assuming Ctrl+V.", True)
            paste_triggered = True
        elif (state & SHIFT_MASK) and keysym == 'insert':
            self._update_status(f"DEBUG: Detected Shift + 'Insert' on '{widget_name}'. Assuming Shift+Insert paste.", True)
            paste_triggered = True
        
        # Дополнительная проверка для macOS, если предыдущие не сработали
        elif platform.system() == "Darwin":
            # На Mac, Command часто имеет state bit 0x0008 (Mod1) или 0x0100 (иногда Command)
            # или keysym 'Command_L' / 'Command_R'
            is_command_modifier_mac = (state & (0x0008 | 0x0100)) or \
                                      event.keysym in ['Command_L', 'Command_R', 'Meta_L', 'Meta_R']
            if is_command_modifier_mac and keysym == 'v':
                self._update_status(f"DEBUG: Detected Command + 'v' (macOS specific) on '{widget_name}'.", True)
                paste_triggered = True


        if paste_triggered:
            self._update_status(f"DEBUG: Paste hotkey attempt for '{widget_name}'.", True)
            if isinstance(widget, (ttk.Entry, tk.Entry)) and widget.cget("state") != 'readonly':
                try:
                    clipboard_content = pyperclip.paste()
                    self._update_status(f"DEBUG: Pyperclip got: '{clipboard_content[:50]}...'", True)
                    
                    if clipboard_content is not None:
                        if widget.selection_present():
                            widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
                        widget.insert(tk.INSERT, clipboard_content)
                        self._update_status(f"Pasted to '{widget_name}' via hotkey.", True)
                    else:
                         self._update_status("DEBUG: Pyperclip returned None. Nothing to paste.", True)
                    return "break" 
                except pyperclip.PyperclipException as e_clip:
                    self._update_status(f"ERROR: Pyperclip paste failed: {e_clip}. Try Tkinter fallback.", True)
                    try:
                        tk_clipboard = widget.clipboard_get() # Может вызвать TclError
                        if tk_clipboard:
                             if widget.selection_present(): widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
                             widget.insert(tk.INSERT, tk_clipboard)
                             self._update_status(f"Pasted to '{widget_name}' via hotkey (Tk fallback).", True)
                        return "break"
                    except tk.TclError:
                        self._update_status("DEBUG: Tk clipboard_get also failed (TclError).", True)
                    except Exception as e_tk_f:
                        self._update_status(f"DEBUG: Tk clipboard_get generic error: {e_tk_f}", True)
                    return "break"
                except Exception as e_generic:
                    self._update_status(f"ERROR: Generic paste error: {e_generic}", True)
                    return "break"
            else:
                self._update_status(f"DEBUG: Paste hotkey ignored, widget not editable: {widget_name}", True)
        return None 

    # --- Команды для контекстного меню (остаются без изменений, как в предыдущем ответе) ---
    def do_cut(self):
        focused_widget = self.root_tk.focus_get()
        if isinstance(focused_widget, (ttk.Entry, tk.Entry)) and focused_widget.cget("state") != 'readonly':
            if focused_widget.selection_present():
                try:
                    selected_text = focused_widget.selection_get()
                    pyperclip.copy(selected_text) 
                    focused_widget.delete(tk.SEL_FIRST, tk.SEL_LAST) 
                    self._update_status(f"Cut from {focused_widget.winfo_class()} via menu.", True)
                except pyperclip.PyperclipException as e:
                     self._update_status(f"ERROR: pyperclip cut failed: {e}", True)
                     focused_widget.event_generate("<<Cut>>") 
                except tk.TclError: 
                    focused_widget.event_generate("<<Cut>>") 
    def do_copy(self):
        focused_widget = self.root_tk.focus_get()
        if isinstance(focused_widget, (ttk.Entry, tk.Entry, scrolledtext.ScrolledText)):
            if focused_widget.selection_present():
                try:
                    selected_text = focused_widget.selection_get()
                    pyperclip.copy(selected_text)
                    self._update_status(f"Copied from {focused_widget.winfo_class()} via menu.", True)
                except pyperclip.PyperclipException as e:
                    self._update_status(f"ERROR: pyperclip copy failed: {e}", True)
                    focused_widget.event_generate("<<Copy>>") 
                except tk.TclError:
                     focused_widget.event_generate("<<Copy>>")
    def do_paste(self):
        focused_widget = self.root_tk.focus_get()
        if isinstance(focused_widget, (ttk.Entry, tk.Entry)) and focused_widget.cget("state") != 'readonly':
            self._update_status(f"DEBUG: Context menu Paste on {focused_widget.winfo_class()}", True)
            try:
                clipboard_content = pyperclip.paste()
                if clipboard_content is not None:
                    if focused_widget.selection_present():
                        focused_widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
                    focused_widget.insert(tk.INSERT, clipboard_content)
                else:
                    self._update_status("DEBUG: pyperclip.paste() (menu) returned None.", True)
            except pyperclip.PyperclipException as e_pyperclip:
                self._update_status(f"ERROR: pyperclip exception during menu paste: {e_pyperclip}.", True)
                try: 
                    clipboard_content_tk = focused_widget.clipboard_get()
                    if clipboard_content_tk:
                        if focused_widget.selection_present(): focused_widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
                        focused_widget.insert(tk.INSERT, clipboard_content_tk)
                except tk.TclError:
                     self._update_status("Paste (menu) failed - Tk clipboard_get TclError.", True)
            except Exception as e:
                self._update_status(f"DEBUG: Error during menu paste: {e}", True)
    def do_select_all(self):
        focused_widget = self.root_tk.focus_get()
        if isinstance(focused_widget, (ttk.Entry, tk.Entry)):
            self._update_status(f"DEBUG: Context menu Select All on {focused_widget.winfo_class()}", append=True)
            focused_widget.select_range(0, tk.END)
            focused_widget.icursor(tk.END)
        elif isinstance(focused_widget, scrolledtext.ScrolledText): 
             self._update_status(f"DEBUG: Context menu Select All on ScrolledText", append=True)
             focused_widget.tag_add(tk.SEL, "1.0", tk.END)
             focused_widget.mark_set(tk.INSERT, "1.0") 
             focused_widget.see(tk.INSERT) 
    def show_context_menu(self, event):
        widget = event.widget
        is_entry = isinstance(widget, (ttk.Entry, tk.Entry))
        is_text_area = isinstance(widget, scrolledtext.ScrolledText) 
        is_editable = (is_entry or is_text_area) and widget.cget("state") != 'readonly'
        selection_present = False
        if is_entry or is_text_area:
            try:
                if widget.selection_present(): selection_present = True
            except tk.TclError: pass
            except AttributeError: 
                if is_text_area:
                    if widget.tag_ranges(tk.SEL): selection_present = True
        self.entry_context_menu.entryconfigure("Вырезать", state=tk.NORMAL if is_editable and selection_present else tk.DISABLED)
        self.entry_context_menu.entryconfigure("Копировать", state=tk.NORMAL if (is_entry or is_text_area) and selection_present else tk.DISABLED)
        self.entry_context_menu.entryconfigure("Вставить", state=tk.NORMAL if is_editable else tk.DISABLED)
        self.entry_context_menu.entryconfigure("Выделить всё", state=tk.NORMAL if (is_entry or is_text_area) else tk.DISABLED)
        try: self.entry_context_menu.tk_popup(event.x_root, event.y_root)
        except tk.TclError as e_popup: self._update_status(f"DEBUG: Error showing context menu: {e_popup}", True)
    
    # ... (остальные методы browse_*, apply_*, trigger_*, copy_log_to_clipboard, is_youtube_url, log_time, _update_status, _update_progress, save_*, start_processing_thread, actual_processing, stop_progressbar, enable_process_button - БЕЗ ИЗМЕНЕНИЙ)
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
            if all_critical_tools_ok: final_message_parts.append("FFmpeg и FFprobe настроены.")
            else: final_message_parts.append("ВНИМАНИЕ: FFmpeg и/или FFprobe НЕ настроены из этой папки!")
            final_message = "\n".join(final_message_parts)
            if not all_critical_tools_ok : 
                final_message += "\n\nИспользуйте кнопку 'Скачать FFmpeg' или скопируйте их вручную в подпапку 'ffmpeg' рабочей директории."
                messagebox.showwarning("Рабочая директория", final_message)
            else: messagebox.showinfo("Рабочая директория", final_message)
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
    def save_processed_part_gui_wrapper(self):
        if not self.chunk_processing_active and not self.processed_video_chunks:
             messagebox.showinfo("Info", "No processed parts available to save or processing not active.")
             return
        final_output_dir_for_part = self.current_final_output_dir_for_operation 
        if not final_output_dir_for_part or not os.path.isdir(final_output_dir_for_part):
            messagebox.showerror("Error", "Output directory is not set or invalid. Cannot save part.")
            return
        self.process_button.config(state=tk.DISABLED)
        self.save_processed_button.config(state=tk.DISABLED)
        self._update_status("--- Starting: Save Processed Part ---", append=True)
        thread = threading.Thread(target=self._save_processed_part_thread, 
                                  args=(list(self.processed_video_chunks), 
                                        list(self.processed_original_audio_chunks),
                                        list(self.processed_dubbed_audio_chunks),
                                        final_output_dir_for_part), 
                                  daemon=True)
        thread.start()
    def _save_processed_part_thread(self, video_chunks_to_save, original_audio_chunks_to_save, dubbed_audio_chunks_to_save, final_output_dir_for_part):
        if not video_chunks_to_save:
            self.root_tk.after(0, lambda: messagebox.showinfo("Nothing to save", "No video parts have been processed yet."))
            self.root_tk.after(0, self.enable_process_button) 
            if self.chunk_processing_active and hasattr(self, 'save_processed_button'): 
                self.root_tk.after(0, lambda: self.save_processed_button.config(state=tk.NORMAL))
            return
        video_source_name = self.video_source_text.get().strip()
        if not video_source_name: video_source_name = "unknown_video"
        base_name = os.path.splitext(os.path.basename(video_source_name))[0]
        parts_output_subdir = os.path.join(final_output_dir_for_part, "Translated_Parts")
        os.makedirs(parts_output_subdir, exist_ok=True)
        part_video_output_path = os.path.join(parts_output_subdir, f"{base_name}_processed_part_{len(video_chunks_to_save)}chunks.mp4")
        temp_concat_dir_name = f"concat_save_part_temp_{int(time.time())}"
        if not self.current_temp_dir_for_operation or not os.path.isdir(self.current_temp_dir_for_operation):
            self.current_temp_dir_for_operation = os.path.join(self.current_work_dir, "temp_op_fallback")
            os.makedirs(self.current_temp_dir_for_operation, exist_ok=True)
            self._update_status(f"Warning: Main operation temp dir not found, created fallback: {self.current_temp_dir_for_operation}", append=True)
        temp_concat_dir = os.path.join(self.current_temp_dir_for_operation, temp_concat_dir_name)
        os.makedirs(temp_concat_dir, exist_ok=True)
        self._update_status(f"Saving processed part... ({len(video_chunks_to_save)} video chunks) to {part_video_output_path}", append=True)
        try:
            self._update_status("Concatenating video chunks...", append=True)
            concatenated_video_path = video_processor.concatenate_video_chunks(
                video_chunks_to_save, 
                os.path.join(temp_concat_dir, "temp_video_part.mp4"),
                temp_concat_dir
            )
            if not concatenated_video_path:
                raise RuntimeError("Failed to concatenate video chunks for saving part.")
            concatenated_original_audio_path = None
            if original_audio_chunks_to_save:
                self._update_status("Concatenating original audio chunks...", append=True)
                concatenated_original_audio_path = video_processor.merge_audio_segments(
                    original_audio_chunks_to_save,
                    os.path.join(temp_concat_dir, "temp_original_audio_part.wav"),
                    log_prefix=None 
                )
            self._update_status("Concatenating dubbed audio chunks...", append=True)
            concatenated_dubbed_audio_path = video_processor.merge_audio_segments(
                dubbed_audio_chunks_to_save,
                os.path.join(temp_concat_dir, "temp_dubbed_audio_part.wav"),
                log_prefix=None
            )
            if not concatenated_dubbed_audio_path:
                raise RuntimeError("Failed to concatenate dubbed audio chunks for saving part.")
            self._update_status("Mixing audio and assembling final part video...", append=True)
            original_vol = 0.1 
            dubbed_vol = 0.95   
            video_processor.mix_and_replace_audio(
                video_path=concatenated_video_path, 
                original_audio_path=concatenated_original_audio_path, 
                dubbed_audio_path=concatenated_dubbed_audio_path,
                output_path=part_video_output_path,
                original_volume=original_vol, 
                dubbed_volume=dubbed_vol
            )
            self._update_status(f"SUCCESS: Processed part saved to: {part_video_output_path}", append=True)
            self.root_tk.after(0, lambda p=part_video_output_path: messagebox.showinfo("Success", f"Processed part saved to:\n{p}"))
        except Exception as e:
            error_msg_save = f"Failed to save processed part: {e}\n{traceback.format_exc()}"
            print(error_msg_save)
            self._update_status(f"ERROR: {error_msg_save}", append=True)
            self.root_tk.after(0, lambda em=str(e): messagebox.showerror("Error Saving Part", f"Failed to save processed part:\n{em}"))
        finally:
            if not self.chunk_processing_active: 
                 self.root_tk.after(0, self.enable_process_button)
            if hasattr(self, 'save_processed_button'): 
                self.root_tk.after(0, lambda: self.save_processed_button.config(state=tk.NORMAL if self.processed_video_chunks else tk.DISABLED))
            if os.path.exists(temp_concat_dir):
                try: shutil.rmtree(temp_concat_dir)
                except Exception as e_clean: print(f"Warning: Could not clean temp concat dir for saving part: {e_clean}")
            self._update_status("--- Finished: Save Processed Part ---", append=True)
    def start_processing_thread(self): 
        if not (self.ffmpeg_configured and self.ffprobe_configured):
            if not self.apply_and_check_work_dir_gui_wrapper(show_success_message=True):
                 messagebox.showerror("Ошибка конфигурации", "Инструменты FFmpeg/FFprobe не настроены...")
                 return
        if not config_manager.CURRENT_FFMPEG_PATH or not config_manager.CURRENT_FFPROBE_PATH:
            messagebox.showerror("Ошибка FFmpeg/FFprobe", "Пути к FFmpeg/FFprobe не установлены...")
            return
        source_input = self.video_source_text.get().strip()
        if not source_input: messagebox.showerror("Error", "Please provide a video source..."); return
        is_youtube_url_val = self.is_youtube_url(source_input)
        if is_youtube_url_val and yt_dlp is None: messagebox.showerror("Error", "yt-dlp library is not installed..."); return
        if not is_youtube_url_val and not os.path.exists(source_input): messagebox.showerror("Error", f"Local video file not found: {source_input}"); return
        local_srt_path_input_val = self.srt_path.get().strip() 
        if local_srt_path_input_val and srt is None: messagebox.showerror("Error", "The 'srt' library is not installed..."); return
        op_key = "Translate & Dub Video"
        self.process_button.config(state="disabled"); self.save_processed_button.config(state="disabled")
        self._update_status(f"🚀 Starting operation: {op_key}...", append=False)
        self.progress_frame.pack(padx=10, pady=(5, 5), fill="x", after=self.save_processed_button) 
        self._update_progress(0, "Initializing"); self.root_tk.update_idletasks()
        self.processing_times = {}; self.total_start_time = time.time()
        self.downloaded_video_path = None; self.downloaded_srt_path = None; self.downloaded_srt_lang = None
        self.processed_video_chunks = [] 
        self.processed_original_audio_chunks = []
        self.processed_dubbed_audio_chunks = []
        self.chunk_processing_active = True 
        self.fully_processed_duration = 0.0
        self.current_final_output_dir_for_operation = self.current_work_dir 
        video_source_path_for_output_calc = source_input
        if is_youtube_url_val: pass
        elif os.path.isfile(source_input):
            video_dir = os.path.dirname(source_input)
            self.current_final_output_dir_for_operation = os.path.join(video_dir, "Translated_Videos_Output")
        else: 
            self.current_final_output_dir_for_operation = os.path.join(self.current_work_dir, "Translated_Videos_Output")
        os.makedirs(self.current_final_output_dir_for_operation, exist_ok=True)
        self._update_status(f"INFO: Output files will be saved in/near: {self.current_final_output_dir_for_operation}", append=True)
        process_first_chunk_only_test_flag = self.process_in_chunks_var.get()
        chunk_duration_s_config = 60.0 
        try: chunk_duration_s_config = float(self.chunk_duration_var.get())
        except ValueError: self._update_status("Warning: Invalid chunk duration, using 60s.", append=True)
        if chunk_duration_s_config <= 0: chunk_duration_s_config = 60.0
        user_start_offset_str_config = self.start_time_offset_var.get()
        thread = threading.Thread(target=self.actual_processing,
                                  args=(source_input, 
                                        local_srt_path_input_val, 
                                        op_key,
                                        is_youtube_url_val, 
                                        source_input if is_youtube_url_val else None, 
                                        process_first_chunk_only_test_flag, 
                                        chunk_duration_s_config,
                                        user_start_offset_str_config), 
                                  daemon=True)
        thread.start()
    def actual_processing(self, video_file_path_or_url, local_srt_path_input, op_key,
                          is_youtube, youtube_url_str,
                          process_first_chunk_only_test_flag, 
                          chunk_duration_seconds_config,
                          user_start_offset_str_config 
                          ):
        # ... (без изменений)
        current_temp_dir = None; final_assembled_video_path = None; success_overall = False
        self.current_temp_dir_for_operation = create_temp_dir() 
        current_temp_dir = self.current_temp_dir_for_operation
        self._update_status(f"Using temp dir: {current_temp_dir}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dub_steps_weights = { 
            "YouTube Download & Subs": 8 if is_youtube else 0, 
            "Initial Setup & SRT Load": 5, 
            "Chunk Processing Loop": 82,
            "Final Assembly": 5 
        }
        total_weight_op_overall = sum(w for w in dub_steps_weights.values() if w > 0) 
        current_overall_progress_value = 0
        video_to_use_for_chunks = None
        if is_youtube:
            self._update_status(f"➡️ Downloading YouTube video and subs from {youtube_url_str}..."); start_t_dl = time.time()
            download_output_dir_vid = os.path.join(current_temp_dir, "youtube_download")
            os.makedirs(download_output_dir_vid, exist_ok=True)
            self.downloaded_video_path, self.downloaded_srt_path, self.downloaded_srt_lang = video_processor.download_youtube_video(
                url=youtube_url_str, output_dir=download_output_dir_vid, preferred_sub_lang='ru', fallback_sub_lang='en')
            video_to_use_for_chunks = self.downloaded_video_path
            self.log_time("YouTube Download & Subs", start_t_dl)
            if not local_srt_path_input and self.downloaded_srt_path and self.downloaded_srt_lang == 'ru':
                local_srt_path_input = self.downloaded_srt_path 
            if video_to_use_for_chunks and os.path.dirname(self.current_final_output_dir_for_operation) == self.current_work_dir:
                 new_output_dir = os.path.join(os.path.dirname(video_to_use_for_chunks), "Translated_Videos_Output")
                 if self.current_final_output_dir_for_operation != new_output_dir:
                     self.current_final_output_dir_for_operation = new_output_dir
                     os.makedirs(self.current_final_output_dir_for_operation, exist_ok=True)
                     self._update_status(f"INFO: Output for YouTube video refined to: {self.current_final_output_dir_for_operation}", append=True)
        else:
            video_to_use_for_chunks = video_file_path_or_url
        current_overall_progress_value += dub_steps_weights.get("YouTube Download & Subs", 0)
        self._update_progress(current_overall_progress_value / total_weight_op_overall * 100, "Initial Download")
        if not video_to_use_for_chunks or not os.path.exists(video_to_use_for_chunks): 
            raise FileNotFoundError(f"Video file for processing not found: {video_to_use_for_chunks or 'None'}")
        self.total_video_duration = video_processor.get_video_duration(video_to_use_for_chunks)
        if self.total_video_duration <= 0: raise ValueError("Could not determine video duration or video is empty.")
        processing_start_offset_seconds = 0.0
        if ':' in user_start_offset_str_config:
            try: m, s = map(int, user_start_offset_str_config.split(':')); processing_start_offset_seconds = m * 60 + s
            except ValueError: self._update_status(f"Warning: Invalid start time format '{user_start_offset_str_config}'. Starting from 00:00.", append=True)
        elif user_start_offset_str_config.strip():
            try: processing_start_offset_seconds = float(user_start_offset_str_config)
            except ValueError: self._update_status(f"Warning: Invalid start time '{user_start_offset_str_config}'. Starting from 0s.", append=True)
        processing_start_offset_seconds = max(0, min(processing_start_offset_seconds, self.total_video_duration - 0.1 if self.total_video_duration > 0.1 else 0))
        self._update_status(f"Video Total Duration: {self.total_video_duration:.2f}s. Processing will start from {processing_start_offset_seconds:.2f}s.", append=True)
        chunks_to_process_primary_queue = [] 
        chunks_to_process_secondary_queue = [] 
        current_pos = processing_start_offset_seconds
        while current_pos < self.total_video_duration:
            remaining_duration = self.total_video_duration - current_pos
            current_chunk_dur = min(chunk_duration_seconds_config, remaining_duration)
            if current_chunk_dur < 0.5 : break 
            chunks_to_process_primary_queue.append({'start': current_pos, 'duration': current_chunk_dur, 'id': len(chunks_to_process_primary_queue)})
            current_pos += current_chunk_dur
        if processing_start_offset_seconds > 0.5: 
            current_pos_secondary = 0.0
            while current_pos_secondary < processing_start_offset_seconds - 0.1: 
                effective_end_secondary = processing_start_offset_seconds
                remaining_duration_secondary = effective_end_secondary - current_pos_secondary
                current_chunk_dur_secondary = min(chunk_duration_seconds_config, remaining_duration_secondary)
                if current_chunk_dur_secondary < 0.5 : break
                chunks_to_process_secondary_queue.append({'start': current_pos_secondary, 'duration': current_chunk_dur_secondary, 'id': len(chunks_to_process_primary_queue) + len(chunks_to_process_secondary_queue)})
                current_pos_secondary += current_chunk_dur_secondary
        all_chunks_ordered_for_processing = chunks_to_process_primary_queue + chunks_to_process_secondary_queue
        if process_first_chunk_only_test_flag and all_chunks_ordered_for_processing:
            all_chunks_ordered_for_processing = [all_chunks_ordered_for_processing[0]]
            self._update_status("--- Test Mode: Processing only the very first defined chunk. ---", append=True)
        self._update_status(f"Total chunks to process: {len(all_chunks_ordered_for_processing)}.", append=True)
        if not all_chunks_ordered_for_processing:
            raise ValueError("No chunks to process based on video duration, start offset and chunk duration.")
        all_srt_segments_loaded = None; initial_diarization_for_srt = None
        skip_translation_based_on_srt = False; subs_source_tag_overall = ""
        if local_srt_path_input and os.path.exists(local_srt_path_input):
            self._update_status(f"➡️ Processing External SRT: {os.path.basename(local_srt_path_input)}...", append=True); start_t_srt_load = time.time()
            all_srt_segments_loaded = transcriber.parse_srt_file(local_srt_path_input)
            if not all_srt_segments_loaded: self._update_status(f"Warning: Provided SRT file {local_srt_path_input} parsed as empty.", append=True)
            else:
                subs_source_tag_overall = "_customsrt"
                texts_for_punct = [s.get('text', '') for s in all_srt_segments_loaded]; restored_texts = transcriber._restore_punctuation(texts_for_punct)
                if len(restored_texts) == len(all_srt_segments_loaded):
                    for i_s, s_data in enumerate(all_srt_segments_loaded): s_data['text'] = restored_texts[i_s]
                temp_audio_for_srt_diar_path = os.path.join(current_temp_dir, "temp_audio_for_srt_diar.wav")
                diar_audio_start = 0; diar_audio_dur = min(300, self.total_video_duration) 
                video_processor.extract_audio(video_to_use_for_chunks, temp_audio_for_srt_diar_path, 16000, diar_audio_start, diar_audio_dur)
                if os.path.exists(temp_audio_for_srt_diar_path):
                    initial_diarization_for_srt = transcriber.perform_diarization_only(temp_audio_for_srt_diar_path, device=device)
                sample_text_srt = " ".join([s.get('text','') for s in all_srt_segments_loaded[:20] if s.get('text','').strip()])
                if detect and sample_text_srt:
                    try:
                        if detect(sample_text_srt) == 'ru': skip_translation_based_on_srt = True
                    except: pass 
                if skip_translation_based_on_srt: self._update_status("INFO: External SRT seems Russian. Translation will be skipped for its segments.", append=True)
            self.log_time("SRT Pre-processing", start_t_srt_load)
        current_overall_progress_value += dub_steps_weights.get("Initial Setup & SRT Load", 0)
        self._update_progress(current_overall_progress_value / total_weight_op_overall * 100, "Setup/SRT")
        base_progress_val_for_chunks = current_overall_progress_value 
        weight_for_chunk_loop = dub_steps_weights.get("Chunk Processing Loop", 82)
        final_output_dir_for_operation = self.current_final_output_dir_for_operation 
        try:
            for chunk_idx, chunk_info in enumerate(all_chunks_ordered_for_processing):
                chunk_start_s = chunk_info['start']; chunk_dur_s = chunk_info['duration']; chunk_id_str = f"chunk_{chunk_info['id']}"
                self._update_status(f"\n--- Processing {chunk_id_str} (Time: {chunk_start_s:.2f}s - {chunk_start_s + chunk_dur_s:.2f}s) ---", append=True)
                chunk_temp_dir = os.path.join(current_temp_dir, chunk_id_str); os.makedirs(chunk_temp_dir, exist_ok=True)
                audio_for_stt_chunk_path = video_processor.extract_audio(video_to_use_for_chunks, os.path.join(chunk_temp_dir, "stt_audio.wav"), 16000, chunk_start_s, chunk_dur_s)
                original_audio_for_mix_chunk_path = video_processor.extract_audio(video_to_use_for_chunks, os.path.join(chunk_temp_dir, "original_mix_audio.wav"), 44100, chunk_start_s, chunk_dur_s)
                if not audio_for_stt_chunk_path: raise RuntimeError(f"Failed to extract STT audio for {chunk_id_str}")
                segments_for_tts_this_chunk = []; diarization_for_this_chunk = None; skip_translation_this_chunk = False
                if all_srt_segments_loaded:
                    for srt_seg in all_srt_segments_loaded:
                        seg_start_abs = srt_seg['start']; seg_end_abs = srt_seg['end']
                        overlap_start = max(seg_start_abs, chunk_start_s); overlap_end = min(seg_end_abs, chunk_start_s + chunk_dur_s)
                        if overlap_end > overlap_start: 
                            new_seg_for_chunk = srt_seg.copy()
                            new_seg_for_chunk['start'] = overlap_start - chunk_start_s
                            new_seg_for_chunk['end'] = overlap_end - chunk_start_s
                            if new_seg_for_chunk['end'] > new_seg_for_chunk['start'] + 0.01: 
                                segments_for_tts_this_chunk.append(new_seg_for_chunk)
                    if segments_for_tts_this_chunk:
                        diarization_for_this_chunk = transcriber.perform_diarization_only(audio_for_stt_chunk_path, device=device)
                        segments_for_tts_this_chunk = transcriber.assign_srt_segments_to_speakers(segments_for_tts_this_chunk, diarization_for_this_chunk, trust_srt_speaker_field=True)
                        segments_for_tts_this_chunk = transcriber._postprocess_srt_segments(segments_for_tts_this_chunk, is_external_srt=True)
                        skip_translation_this_chunk = skip_translation_based_on_srt
                        self._update_status(f"  Using {len(segments_for_tts_this_chunk)} segments from external SRT for {chunk_id_str}.", append=True)
                    else: self._update_status(f"  No external SRT segments for {chunk_id_str}. Transcribing audio.", append=True)
                if not segments_for_tts_this_chunk: 
                    segments_from_whisper, diarization_for_this_chunk = transcriber.transcribe_and_diarize_audio(audio_for_stt_chunk_path, 'en', return_diarization_df=True)
                    if segments_from_whisper:
                        segments_for_tts_this_chunk = transcriber._postprocess_srt_segments(segments_from_whisper, is_external_srt=False)
                    skip_translation_this_chunk = False 
                dubbed_audio_chunk_path = None
                if not segments_for_tts_this_chunk:
                    self._update_status(f"  No text segments for TTS in {chunk_id_str}. Creating silent audio for this chunk.", append=True)
                    dubbed_audio_chunk_path = os.path.join(chunk_temp_dir, "dubbed_audio_silent.wav")
                    (ffmpeg.input('anullsrc', format='lavfi', r=24000).output(dubbed_audio_chunk_path, t=chunk_dur_s, acodec='pcm_s16le')
                     .overwrite_output().run(capture_stdout=True, capture_stderr=True))
                else:
                    translated_segments_for_chunk = list(segments_for_tts_this_chunk)
                    if not skip_translation_this_chunk:
                        translated_segments_for_chunk = translator.translate_segments(segments_for_tts_this_chunk)
                    else:
                        for s_idx, s_val in enumerate(translated_segments_for_chunk): s_val['translated_text'] = s_val.get('text', '')
                    dubbed_audio_chunk_path, _, _ = voice_cloner.synthesize_speech_segments(
                        translated_segments_for_chunk, audio_for_stt_chunk_path, chunk_temp_dir,
                        diarization_result_df=diarization_for_this_chunk, progress_callback=None, language='ru',
                        overall_audio_start_time_offset=chunk_start_s
                    )
                if not dubbed_audio_chunk_path or not os.path.exists(dubbed_audio_chunk_path):
                    raise RuntimeError(f"Failed to synthesize or create dubbed audio for {chunk_id_str}")
                video_chunk_processed_path = os.path.join(chunk_temp_dir, "video_chunk_processed.mp4")
                video_processor.mix_and_replace_audio(
                    video_path=video_to_use_for_chunks, 
                    original_audio_path=original_audio_for_mix_chunk_path,
                    dubbed_audio_path=dubbed_audio_chunk_path,
                    output_path=video_chunk_processed_path,
                    video_start_time=chunk_start_s, 
                    video_duration=chunk_dur_s
                )
                if not os.path.exists(video_chunk_processed_path):
                    self._update_status(f"WARNING: Failed to assemble video for {chunk_id_str}. This part might be missing.", append=True)
                else:
                    self.processed_video_chunks.append(video_chunk_processed_path)
                if original_audio_for_mix_chunk_path and os.path.exists(original_audio_for_mix_chunk_path):
                    self.processed_original_audio_chunks.append(original_audio_for_mix_chunk_path)
                self.processed_dubbed_audio_chunks.append(dubbed_audio_chunk_path)
                self.fully_processed_duration += chunk_dur_s
                self._update_status(f"{chunk_id_str} processed. Total processed: {self.fully_processed_duration:.2f}s / {self.total_video_duration:.2f}s", append=True)
                progress_in_chunk_loop = ( (chunk_idx + 1) / len(all_chunks_ordered_for_processing) ) * weight_for_chunk_loop
                self._update_progress((base_progress_val_for_chunks + progress_in_chunk_loop) / total_weight_op_overall * 100, f"Chunk {chunk_idx+1}/{len(all_chunks_ordered_for_processing)}")
                if self.processed_video_chunks and hasattr(self, 'save_processed_button'):
                    self.root_tk.after(0, lambda: self.save_processed_button.config(state=tk.NORMAL))
            success_overall = True
            video_base_name_full_output = os.path.splitext(os.path.basename(video_to_use_for_chunks))[0]
            if success_overall and not process_first_chunk_only_test_flag and self.processed_video_chunks:
                self._update_status("\n--- Assembling Final Full Video ---", append=True)
                final_assembled_video_path = os.path.join(final_output_dir_for_operation, f"{video_base_name_full_output}_dubbed_ru{subs_source_tag_overall}_FULL.mp4")
                full_dubbed_audio = video_processor.merge_audio_segments(
                    self.processed_dubbed_audio_chunks, 
                    os.path.join(current_temp_dir, "FULL_dubbed_audio.wav"), log_prefix=None)
                full_original_audio = None
                if self.processed_original_audio_chunks:
                     full_original_audio = video_processor.merge_audio_segments(
                        self.processed_original_audio_chunks, 
                        os.path.join(current_temp_dir, "FULL_original_audio.wav"), log_prefix=None)
                if not full_dubbed_audio: raise RuntimeError("Failed to assemble full dubbed audio track.")
                final_original_vol = 0.1 
                final_dubbed_vol = 0.95   
                video_processor.mix_and_replace_audio(
                    video_path=video_to_use_for_chunks,
                    original_audio_path=full_original_audio,
                    dubbed_audio_path=full_dubbed_audio,
                    output_path=final_assembled_video_path,
                    original_volume=final_original_vol,
                    dubbed_volume=final_dubbed_vol
                )
                self._update_status(f"Final full video assembled: {final_assembled_video_path}", append=True)
            elif process_first_chunk_only_test_flag and self.processed_video_chunks:
                 self._update_status(f"--- Test Mode: First chunk processing finished. Use 'Save Processed Part' to get it. ---", append=True)
                 final_assembled_video_path = self.processed_video_chunks[0] 
                 success_overall = True
            current_overall_progress_value = base_progress_val_for_chunks + weight_for_chunk_loop
            current_overall_progress_value += dub_steps_weights.get("Final Assembly", 0)
            self._update_progress(current_overall_progress_value / total_weight_op_overall * 100, "Final Assembly")
        except Exception as e_main_loop:
            tb_str = traceback.format_exc(); failed_step = "Chunk Processing Loop or Unknown Step"
            error_message_detail = str(e_main_loop)
            error_message_full = f"\n❌❌❌ ERROR during '{failed_step}':\n{type(e_main_loop).__name__}: {error_message_detail}\n--- Traceback ---\n{tb_str}-----------------"
            print(error_message_full); self._update_status(error_message_full, append=True)
            self.root_tk.after(0, lambda f_step=failed_step, e_msg=error_message_detail: messagebox.showerror("Processing Error", f"Error during '{f_step}':\n{e_msg}"))
            success_overall = False
        finally:
            self.chunk_processing_active = False 
            self.root_tk.after(0, self.stop_and_hide_progressbar)
            self.root_tk.after(0, self.enable_process_button) 
            if hasattr(self, 'save_processed_button'): 
                self.root_tk.after(0, lambda: self.save_processed_button.config(state=tk.NORMAL if self.processed_video_chunks else tk.DISABLED))
            if success_overall and final_assembled_video_path and os.path.exists(final_assembled_video_path):
                total_time_seconds = time.time() - self.total_start_time; total_time_minutes = total_time_seconds / 60.0
                self._update_status(f"\n✅🎉 Total processing time: {total_time_minutes:.2f} minutes.", append=True)
                self._update_status(f"Output file:\n{final_assembled_video_path}", append=True)
                self.root_tk.after(0, lambda p=final_assembled_video_path, t=total_time_minutes: messagebox.showinfo("Success", f"Operation completed in {t:.2f} minutes!\n\nOutput Video: {p}"))
            elif success_overall and process_first_chunk_only_test_flag:
                self._update_status(f"\n✅ Test processing of the first chunk finished. Use 'Save Processed Part'.", append=True)
            elif not success_overall:
                self._update_status(f"\n❌ Operation '{op_key}' failed.", append=True)
            if self.current_temp_dir_for_operation: 
                cleanup_temp_dir(self.current_temp_dir_for_operation)
    def stop_and_hide_progressbar(self):
        # ... (без изменений)
        pass
    def enable_process_button(self):
        # ... (без изменений)
        pass

if __name__ == "__main__":
    # ... (без изменений)
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