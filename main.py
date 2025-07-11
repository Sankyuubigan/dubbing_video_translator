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
from utils import segment_utils # Импортируем новый модуль

if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
    problematic_classes_tts_list = []
    tts_classes_to_check = [
        ("TTS.tts.configs.xtts_config", "XttsConfig"), ("TTS.tts.models.xtts", "XttsAudioConfig"),
        ("TTS.config.shared_configs", "BaseDatasetConfig"), ("TTS.tts.models.xtts", "XttsArgs")]
    for module_path, class_name in tts_classes_to_check:
        module_obj = None; class_obj = None
        try:
            module_obj = __import__(module_path, fromlist=[class_name])
            if hasattr(module_obj, class_name): class_obj = getattr(module_obj, class_name); problematic_classes_tts_list.append(class_obj)
        except ImportError: print(f"WARNING: Could not import module {module_path} for torch safe globals.")
        except AttributeError: print(f"WARNING: AttributeError while trying to get {class_name} from {module_path}.")
    if problematic_classes_tts_list:
        try: torch.serialization.add_safe_globals(list(set(problematic_classes_tts_list)))
        except Exception as e_sg_add: print(f"WARNING: Error trying to add TTS classes to torch safe globals: {e_sg_add}")
os.environ["COQUI_AGREED_TO_CPML"] = "1"

import video_processor 
import transcriber # transcriber теперь использует segment_utils
import translator
import voice_cloner

ffmpeg_lib_available = False; yt_dlp_lib_available = False; srt_lib_available = False
langdetect_lib_available = False; detect_func = None; LangDetectException_cls = None
try: import ffmpeg; ffmpeg_lib_available = True
except ImportError: print("WARNING: ffmpeg-python library not found."); ffmpeg = None
try: import yt_dlp; yt_dlp_lib_available = True
except ImportError: print("WARNING: yt-dlp library not found."); yt_dlp = None
try: import srt; srt_lib_available = True # srt теперь используется в segment_utils
except ImportError: print("WARNING: srt library not found."); srt = None 
try:
    from langdetect import detect, LangDetectException
    langdetect_lib_available = True; detect_func = detect; LangDetectException_cls = LangDetectException
except ImportError: print("WARNING: langdetect library not found.")

def create_temp_dir_main(): return tempfile.mkdtemp(prefix="video_translator_")
def cleanup_temp_dir_main(temp_dir_to_clean):
    if temp_dir_to_clean and os.path.exists(temp_dir_to_clean) and os.path.isdir(temp_dir_to_clean):
        try: shutil.rmtree(temp_dir_to_clean)
        except OSError as e_rmtree_os: print(f"OSError cleaning temp dir {temp_dir_to_clean}: {e_rmtree_os}")
        except Exception as e_rmtree_generic: print(f"Generic error cleaning temp dir {temp_dir_to_clean}: {e_rmtree_generic}")

class App:
    def __init__(self, root_tk_app): 
        self.root_tk = root_tk_app
        self.app_display_version = datetime.date.today().strftime('%d.%m.%y')
        self.root_tk.title(f"Video Dubbing Tool - v{self.app_display_version}") 
        try: # Установка иконки
            icon_path = "icon.ico" 
            if os.path.exists(icon_path): self.root_tk.iconbitmap(default=icon_path)
            elif getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                alt_icon_path = os.path.join(sys._MEIPASS, icon_path)
                if os.path.exists(alt_icon_path): self.root_tk.iconbitmap(default=alt_icon_path)
                else: print(f"WARN: Icon not found in MEIPASS: {alt_icon_path}")
            else: print(f"WARN: Icon not found: {os.path.abspath(icon_path)}")
        except Exception as e_icon: print(f"WARN: Could not set app icon: {e_icon}")
        self.root_tk.geometry("700x720") # Немного увеличим для новой кнопки
        self.current_work_dir = config_manager.get_work_dir_from_config()
        if not self.current_work_dir:
            # ... (логика выбора рабочей директории, как раньше) ...
            default_dir_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
            chosen_dir_path = filedialog.askdirectory(title="Выберите или создайте рабочую директорию", initialdir=default_dir_path)
            if chosen_dir_path:
                self.current_work_dir = chosen_dir_path
                if not os.path.isdir(self.current_work_dir):
                    try: os.makedirs(self.current_work_dir, exist_ok=True)
                    except OSError as e_mkd: messagebox.showerror("Ошибка", f"Не удалось создать директорию: {e_mkd}"); self.root_tk.destroy(); return
                config_manager.save_work_dir_to_config(self.current_work_dir) 
            else: messagebox.showerror("Ошибка", "Рабочая директория не выбрана."); self.root_tk.destroy(); return
        
        # ... (инициализация переменных, как раньше, НО БЕЗ process_in_chunks_var, chunk_duration_var) ...
        self.work_dir_path_var = tk.StringVar(value=self.current_work_dir); self.ffmpeg_configured = False; self.ffprobe_configured = False
        self.video_source_text = tk.StringVar(); self.srt_path = tk.StringVar(); self.processing_times = {}; self.total_start_time = 0
        self.downloaded_video_path_session = None; self.downloaded_srt_path_session = None; self.downloaded_srt_lang_session = None
        self.target_chunk_duration_var = tk.StringVar(value="45"); self.max_chunk_duration_var = tk.StringVar(value="60")   
        self.start_time_offset_var = tk.StringVar(value="0:00") 
        self.processed_video_chunks_paths = []; self.processed_original_audio_chunks_paths = []; self.processed_dubbed_audio_chunks_paths = []
        self.all_phrase_segments_for_video = []; self.total_video_duration_seconds = 0.0
        self.chunk_processing_in_progress = False; self.fully_processed_duration_seconds = 0.0
        self.current_temp_dir_for_operation_path = None; self.current_final_output_dir_for_operation_path = None

        style = ttk.Style(); available_themes = style.theme_names()
        if 'clam' in available_themes: style.theme_use('clam')
        elif 'vista' in available_themes: style.theme_use('vista')
        style.configure("TButton", padding=6, relief="raised", background="#d9d9d9", foreground="black")
        style.map("TButton", background=[('pressed', '#c0c0c0'), ('active', '#e8e8e8')], relief=[('pressed', 'sunken')])
        style.configure("TLabel", padding=6, background="#f0f0f0"); style.configure("TEntry", padding=6)
        self.make_ui_layout()
        if self.current_work_dir: self.apply_and_check_work_dir_gui_wrapper(show_success_message=False, initial_setup=True)

    def make_ui_layout(self): 
        self.root_tk.configure(bg="#f0f0f0")
        self.entry_context_menu = tk.Menu(self.root_tk, tearoff=0) # ... (add commands) ...
        self.entry_context_menu.add_command(label="Вырезать", command=self.do_cut)
        self.entry_context_menu.add_command(label="Копировать", command=self.do_copy)
        self.entry_context_menu.add_command(label="Вставить", command=self.do_paste)
        self.entry_context_menu.add_separator()
        self.entry_context_menu.add_command(label="Выделить всё", command=self.do_select_all)

        work_dir_frame = ttk.LabelFrame(self.root_tk, text="Рабочая директория", padding=(10, 5)); work_dir_frame.pack(padx=10, pady=5, fill="x")
        # ... (элементы work_dir_frame)
        self.work_dir_entry = ttk.Entry(work_dir_frame, textvariable=self.work_dir_path_var, width=50); self.work_dir_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        self.work_dir_entry.bind("<Button-3>", self.show_context_menu); self.work_dir_entry.bind("<KeyPress>", self.handle_keypress_for_paste_and_log)
        self.work_dir_browse_button = ttk.Button(work_dir_frame, text="Выбрать...", command=self.browse_work_dir); self.work_dir_browse_button.pack(side=tk.LEFT, padx=(0,5))
        self.work_dir_apply_button = ttk.Button(work_dir_frame, text="Применить и Проверить", command=self.apply_and_check_work_dir_gui_wrapper); self.work_dir_apply_button.pack(side=tk.LEFT, padx=(0,5))
        self.download_tools_button = ttk.Button(work_dir_frame, text="Скачать FFmpeg", command=self.trigger_download_tools_gui_wrapper, state=tk.NORMAL if self.current_work_dir else tk.DISABLED); self.download_tools_button.pack(side=tk.LEFT)

        source_frame = ttk.LabelFrame(self.root_tk, text="Video Source (File Path or YouTube URL)", padding=(10, 5)); source_frame.pack(padx=10, pady=5, fill="x", after=work_dir_frame)
        # ... (элементы source_frame)
        self.video_source_entry = ttk.Entry(source_frame, textvariable=self.video_source_text, width=60); self.video_source_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        self.video_source_entry.bind("<Button-3>", self.show_context_menu); self.video_source_entry.bind("<KeyPress>", self.handle_keypress_for_paste_and_log)
        self.video_browse_button = ttk.Button(source_frame, text="Browse File...", command=self.browse_video_for_source_entry); self.video_browse_button.pack(side=tk.LEFT)
        self.video_clear_button = ttk.Button(source_frame, text="Clear", command=self.clear_video_source_text); self.video_clear_button.pack(side=tk.LEFT, padx=(5,0))
        # Новая кнопка "Сохранить оригинал"
        self.save_original_video_button = ttk.Button(source_frame, text="Save Original Video", command=self.save_original_video_gui_wrapper, state=tk.DISABLED)
        self.save_original_video_button.pack(side=tk.LEFT, padx=(5,0))
        ttk.Label(source_frame, text="For YouTube URLs, subtitles (RU then EN) are auto-downloaded if available (used if no local SRT).", font=("Segoe UI", 8)).pack(side=tk.BOTTOM, anchor=tk.W, padx=0, pady=(5,0))
        
        self.srt_outer_frame = ttk.LabelFrame(self.root_tk, text="Subtitles File (RU text, timings will be used & post-processed)", padding=(10, 5)); self.srt_outer_frame.pack(padx=10, pady=5, fill="x", after=source_frame)
        # ... (элементы srt_outer_frame)
        srt_input_line_frame = ttk.Frame(self.srt_outer_frame); srt_input_line_frame.pack(fill="x")
        self.srt_path_entry = ttk.Entry(srt_input_line_frame, textvariable=self.srt_path, width=60, state="readonly"); self.srt_path_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        self.srt_path_entry.bind("<Button-3>", self.show_context_menu) 
        self.srt_browse_button = ttk.Button(srt_input_line_frame, text="Browse...", command=self.browse_srt); self.srt_browse_button.pack(side=tk.LEFT)
        self.srt_clear_button = ttk.Button(srt_input_line_frame, text="Clear", command=self.clear_srt_path) ; self.srt_clear_button.pack(side=tk.LEFT, padx=(5,0))
        
        processing_options_frame = ttk.LabelFrame(self.root_tk, text="Processing Options (Smart Chunking by Phrases)", padding=(10, 5)); processing_options_frame.pack(padx=10, pady=5, fill="x", after=self.srt_outer_frame)
        chunk_params_frame = ttk.Frame(processing_options_frame); chunk_params_frame.pack(fill="x", pady=(0,5))
        ttk.Label(chunk_params_frame, text="Target chunk (s):").pack(side=tk.LEFT)
        self.target_chunk_duration_entry = ttk.Entry(chunk_params_frame, textvariable=self.target_chunk_duration_var, width=5); self.target_chunk_duration_entry.pack(side=tk.LEFT, padx=(0,10))
        self.target_chunk_duration_entry.bind("<Button-3>", self.show_context_menu); self.target_chunk_duration_entry.bind("<KeyPress>", self.handle_keypress_for_paste_and_log)
        ttk.Label(chunk_params_frame, text="Max chunk (s):").pack(side=tk.LEFT)
        self.max_chunk_duration_entry = ttk.Entry(chunk_params_frame, textvariable=self.max_chunk_duration_var, width=5); self.max_chunk_duration_entry.pack(side=tk.LEFT)
        self.max_chunk_duration_entry.bind("<Button-3>", self.show_context_menu); self.max_chunk_duration_entry.bind("<KeyPress>", self.handle_keypress_for_paste_and_log)
        start_time_line_frame = ttk.Frame(processing_options_frame); start_time_line_frame.pack(fill="x", pady=(5,0), after=chunk_params_frame) 
        ttk.Label(start_time_line_frame, text="Start processing from (mm:ss or seconds):").pack(side=tk.LEFT, padx=(0,5))
        self.start_time_offset_entry = ttk.Entry(start_time_line_frame, textvariable=self.start_time_offset_var, width=10); self.start_time_offset_entry.pack(side=tk.LEFT, padx=(0,10))
        self.start_time_offset_entry.bind("<Button-3>", self.show_context_menu); self.start_time_offset_entry.bind("<KeyPress>", self.handle_keypress_for_paste_and_log)

        self.process_button = ttk.Button(self.root_tk, text="Translate & Dub Video", command=self.start_processing_thread); self.process_button.pack(pady=(10,0), after=processing_options_frame)
        self.save_processed_button = ttk.Button(self.root_tk, text="Save Processed Part", command=self.save_processed_part_gui_wrapper, state=tk.DISABLED); self.save_processed_button.pack(pady=(5,10), after=self.process_button)
        self.progress_frame = ttk.Frame(self.root_tk, padding=(10,5)) 
        self.progress_label_text = tk.StringVar(value="Progress: 0%"); self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_label_text, width=30, anchor="w"); self.progress_label.pack(side=tk.LEFT, padx=(0,5))
        self.progressbar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', maximum=100); self.progressbar.pack(side=tk.LEFT, fill="x", expand=True)
        status_frame_outer = ttk.LabelFrame(self.root_tk, text="Status / Log", padding=(10,5)); status_frame_outer.pack(padx=10, pady=(0,5), fill="both", expand=True, after=self.save_processed_button) 
        self.status_text = scrolledtext.ScrolledText(status_frame_outer, height=10, wrap=tk.WORD, state="disabled", bg="#ffffff", relief="sunken", borderwidth=1, font=("Consolas", 9)); self.status_text.pack(side=tk.TOP, fill="both", expand=True)
        self.copy_log_button = ttk.Button(status_frame_outer, text="Copy Log to Clipboard", command=self.copy_log_to_clipboard); self.copy_log_button.pack(side=tk.TOP, pady=(5,0))

    def save_original_video_gui_wrapper(self):
        video_to_save = self.downloaded_video_path_session # Скачанное видео
        source_text = self.video_source_text.get().strip()
        is_yt = self.is_youtube_url(source_text)

        if not video_to_save and not is_yt: # Если это локальный файл, а не скачанный с YT
            video_to_save = source_text
        
        if not video_to_save or not os.path.exists(video_to_save):
            messagebox.showwarning("Save Original", "No original video file available to save. (Download or select a local file first).")
            return

        original_filename = os.path.basename(video_to_save)
        # Предлагаем имя файла с суффиксом _original
        base, ext = os.path.splitext(original_filename)
        suggested_filename = f"{base}_original{ext}"

        save_path = filedialog.asksaveasfilename(
            title="Save Original Video As...",
            initialfile=suggested_filename,
            defaultextension=ext,
            filetypes=[(f"{ext.upper()} files", f"*{ext}"), ("All files", "*.*")]
        )

        if save_path:
            try:
                shutil.copy2(video_to_save, save_path) # copy2 сохраняет метаданные
                self._update_status(f"INFO: Original video saved to: {save_path}", append=True)
                messagebox.showinfo("Save Original", f"Original video saved successfully to:\n{save_path}")
            except Exception as e_copy_orig:
                self._update_status(f"ERROR: Could not save original video to {save_path}: {e_copy_orig}", append=True)
                messagebox.showerror("Save Original Error", f"Failed to save original video:\n{e_copy_orig}")
    
    # ... (остальные методы GUI и обработки без изменений, кроме start_processing_thread и actual_processing) ...
    # handle_keypress_for_paste_and_log, do_cut, do_copy, do_paste, do_select_all, show_context_menu, 
    # browse_work_dir, apply_and_check_work_dir_gui_wrapper, trigger_download_tools_gui_wrapper, 
    # _download_tools_thread_with_feedback, copy_log_to_clipboard, is_youtube_url, log_time, 
    # _update_status, _update_progress, browse_video_for_source_entry, clear_video_source_text, 
    # browse_srt, clear_srt_path, save_processed_part_gui_wrapper, _save_processed_part_thread - остаются как в предыдущем ответе.

    def start_processing_thread(self): 
        # ... (проверки FFmpeg, источника видео, SRT - как в предыдущем ответе)
        if not (self.ffmpeg_configured and self.ffprobe_configured):
            if not self.apply_and_check_work_dir_gui_wrapper(show_success_message=True): messagebox.showerror("Ошибка конфигурации", "Инструменты FFmpeg/FFprobe не настроены..."); return 
        if not config_manager.CURRENT_FFMPEG_PATH or not config_manager.CURRENT_FFPROBE_PATH: messagebox.showerror("Ошибка FFmpeg/FFprobe", "Пути к FFmpeg/FFprobe не установлены..."); return
        source_in = self.video_source_text.get().strip()
        if not source_in: messagebox.showerror("Error", "Please provide a video source..."); return
        is_yt = self.is_youtube_url(source_in)
        if is_yt and not yt_dlp_lib_available: messagebox.showerror("Error", "yt-dlp library is not installed..."); return
        if not is_yt and not os.path.exists(source_in): messagebox.showerror("Error", f"Local video file not found: {source_in}"); return
        local_srt_in = self.srt_path.get().strip() 
        if local_srt_in and not srt_lib_available: messagebox.showerror("Error", "The 'srt' library is not installed..."); return
        if local_srt_in and not os.path.exists(local_srt_in): messagebox.showerror("Error", f"Local SRT/VTT file not found: {local_srt_in}"); return

        # Параметры "умной" нарезки (теперь единственный вариант)
        target_chunk_s = segment_utils.DEFAULT_TARGET_CHUNK_DURATION_SEC
        max_chunk_s = segment_utils.DEFAULT_MAX_CHUNK_DURATION_SEC
        try: target_chunk_s = float(self.target_chunk_duration_var.get())
        except ValueError: self._update_status(f"WARN: Invalid target chunk duration, using default {target_chunk_s}s.", append=True)
        try: max_chunk_s = float(self.max_chunk_duration_var.get())
        except ValueError: self._update_status(f"WARN: Invalid max chunk duration, using default {max_chunk_s}s.", append=True)
        if target_chunk_s <= 0: target_chunk_s = segment_utils.DEFAULT_TARGET_CHUNK_DURATION_SEC
        if max_chunk_s <= target_chunk_s : max_chunk_s = target_chunk_s + 15.0 
        user_offset_str = self.start_time_offset_var.get()

        op_key = "Translate & Dub Video"; self.disable_gui_elements_during_processing(); self._update_status(f"🚀 Starting: {op_key}...", append=False); self.show_progressbar()
        self.processing_times = {}; self.total_start_time = time.time()
        self.downloaded_video_path_session = None; self.downloaded_srt_path_session = None; self.downloaded_srt_lang_session = None
        self.processed_video_chunks_paths = []; self.processed_original_audio_chunks_paths = []; self.processed_dubbed_audio_chunks_paths = []
        self.all_phrase_segments_for_video = []; self.chunk_processing_in_progress = True; self.fully_processed_duration_seconds = 0.0
        
        if is_yt: self.current_final_output_dir_for_operation_path = os.path.join(self.current_work_dir, "Translated_YouTube_Output")
        elif os.path.isfile(source_in): self.current_final_output_dir_for_operation_path = os.path.join(os.path.dirname(source_in), "Translated_Output")
        else: self.current_final_output_dir_for_operation_path = os.path.join(self.current_work_dir, "Translated_Generic_Output")
        
        if not os.path.isdir(self.current_final_output_dir_for_operation_path):
            try:
                os.makedirs(self.current_final_output_dir_for_operation_path, exist_ok=True)
                if not os.path.isdir(self.current_final_output_dir_for_operation_path): raise OSError("Failed to create output dir.")
            except OSError as e_mk_out:
                self._update_status(f"ERR: Could not create output dir: {self.current_final_output_dir_for_operation_path}\n{e_mk_out}", append=True)
                self.enable_gui_elements_after_processing(); self.hide_progressbar()
                messagebox.showerror("Directory Error", f"Failed to create output dir:\n{self.current_final_output_dir_for_operation_path}\n{e_mk_out}"); return
        self._update_status(f"INFO: Output files will be saved in/near: {self.current_final_output_dir_for_operation_path}", append=True)
        
        threading.Thread(target=self.actual_processing,
            args=(source_in, local_srt_in, op_key, is_yt, source_in if is_yt else None, 
                  target_chunk_s, max_chunk_s, user_offset_str), daemon=True).start()

    def actual_processing(self, video_file_path_or_url_actual, local_srt_path_input_actual, op_key_actual,
                          is_youtube_flag_actual, youtube_url_str_actual,
                          target_chunk_duration_s_actual,
                          max_chunk_duration_s_actual,
                          user_start_offset_str_actual 
                          ):
        self.current_temp_dir_for_operation_path = create_temp_dir_main() 
        if not self.current_temp_dir_for_operation_path or not os.path.isdir(self.current_temp_dir_for_operation_path):
            self._update_status("CRITICAL ERROR: Failed to create temporary directory. Aborting.", append=True)
            self.root_tk.after(0, self.processing_finished_cleanup_and_feedback, False, "Temp dir creation failed"); return 
        self._update_status(f"Using temp dir: {self.current_temp_dir_for_operation_path}")
        
        dub_steps_weights_map = { 
            "YouTube Download & Subs": 10 if is_youtube_flag_actual else 0, 
            "Video Info & Initial SRT Load": 10, 
            "Full Audio Transcription (for smart chunks)": 25, 
            "Chunk Processing Loop": 50, "Final Assembly": 5 }
        total_weight_op_overall_val = sum(w for w in dub_steps_weights_map.values() if w > 0) 
        current_overall_progress_value_tracker = 0.0 
        video_to_use_for_processing = None; processing_error_message = ""

        if is_youtube_flag_actual:
            self._update_status(f"➡️ Downloading YouTube video and subs from {youtube_url_str_actual}..."); start_t_dl = time.time()
            download_output_dir_yt = os.path.join(self.current_temp_dir_for_operation_path, "youtube_download")
            if not os.path.isdir(download_output_dir_yt): os.makedirs(download_output_dir_yt, exist_ok=True)
            vid_path, srt_path_yt, srt_lang_yt, dl_error_msg = video_processor.download_youtube_video(
                url=youtube_url_str_actual, output_dir=download_output_dir_yt, preferred_sub_lang='ru', fallback_sub_lang='en')
            self.log_time("YouTube Download & Subs", start_t_dl)
            if dl_error_msg and "yt-dlp library not found" not in dl_error_msg : self._update_status(f"WARNING (YouTube Download): {dl_error_msg}", append=True)
            if vid_path and os.path.exists(vid_path):
                self.downloaded_video_path_session = vid_path; video_to_use_for_processing = vid_path
                if hasattr(self, 'save_original_video_button'): self.root_tk.after(0, lambda: self.save_original_video_button.config(state=tk.NORMAL)) # Активируем кнопку сохранения
                if not local_srt_path_input_actual and srt_path_yt and srt_lang_yt == 'ru':
                    self.downloaded_srt_path_session = srt_path_yt; local_srt_path_input_actual = srt_path_yt
                    self._update_status(f"INFO: Using downloaded Russian SRT from YouTube: {os.path.basename(srt_path_yt)}", append=True)
                elif srt_path_yt: self.downloaded_srt_path_session = srt_path_yt; self.downloaded_srt_lang_session = srt_lang_yt
            else: processing_error_message = f"Failed to download YouTube video: {dl_error_msg if dl_error_msg else 'Unknown error'}"
        else: 
            video_to_use_for_processing = video_file_path_or_url_actual
            self.downloaded_video_path_session = video_to_use_for_processing # Для локальных файлов тоже делаем доступным для сохранения
            if hasattr(self, 'save_original_video_button'): self.root_tk.after(0, lambda: self.save_original_video_button.config(state=tk.NORMAL))

        current_overall_progress_value_tracker += dub_steps_weights_map.get("YouTube Download & Subs", 0)
        self._update_progress((current_overall_progress_value_tracker / total_weight_op_overall_val) * 100, "Initial Download")
        if not video_to_use_for_processing or not os.path.exists(video_to_use_for_processing):
            final_err_msg = processing_error_message if processing_error_message else f"Video file for processing not found: {video_to_use_for_processing or 'None'}"
            self._update_status(f"CRITICAL ERROR: {final_err_msg}", append=True)
            self.root_tk.after(0, self.processing_finished_cleanup_and_feedback, False, final_err_msg); return

        # ... (остальная часть actual_processing как в предыдущем ответе, используя segment_utils)
        # ВАЖНО: Заменить вызовы transcriber.parse_srt_file, transcriber._postprocess_srt_segments, 
        # transcriber.get_chunk_definitions_from_phrases и т.д. на segment_utils.*
        
        self._update_status(f"➡️ Getting video info and loading SRT (if provided)..."); start_t_info_srt = time.time()
        self.total_video_duration_seconds = video_processor.get_video_duration(video_to_use_for_processing)
        if self.total_video_duration_seconds <= 0:
            processing_error_message = "Could not determine video duration or video is empty."
            self._update_status(f"CRITICAL ERROR: {processing_error_message}", append=True)
            self.root_tk.after(0, self.processing_finished_cleanup_and_feedback, False, processing_error_message); return
        processing_start_offset_seconds_val = 0.0 # ... (логика смещения)
        if ':' in user_start_offset_str_actual:
            mm_ss_parts = user_start_offset_str_actual.split(':')
            if len(mm_ss_parts) == 2:
                try: m_offset, s_offset = map(int, mm_ss_parts); processing_start_offset_seconds_val = m_offset * 60 + s_offset
                except ValueError: self._update_status(f"WARN: Invalid start time format '{user_start_offset_str_actual}'. Starting from 00:00.", append=True)
        elif user_start_offset_str_actual.strip():
            try: processing_start_offset_seconds_val = float(user_start_offset_str_actual)
            except ValueError: self._update_status(f"WARN: Invalid start time '{user_start_offset_str_actual}'. Starting from 0s.", append=True)
        min_dur_proc = 0.1
        processing_start_offset_seconds_val = max(0, min(processing_start_offset_seconds_val, self.total_video_duration_seconds - min_dur_proc if self.total_video_duration_seconds > min_dur_proc else 0))
        self._update_status(f"Video Total Duration: {self.total_video_duration_seconds:.2f}s. Processing will start from {processing_start_offset_seconds_val:.2f}s.", append=True)
        
        all_srt_segments_loaded_list = []; diarization_for_srt_segments = None; skip_translation_based_on_srt_lang = False; subs_source_tag_for_output_name = ""
        if local_srt_path_input_actual and os.path.exists(local_srt_path_input_actual):
            self._update_status(f"➡️ Processing External SRT: {os.path.basename(local_srt_path_input_actual)}...", append=True)
            parsed_segments, parse_srt_success = transcriber.parse_srt_file(local_srt_path_input_actual) # transcriber.parse_srt_file все еще существует, но использует segment_utils внутри
            if parse_srt_success and parsed_segments:
                all_srt_segments_loaded_list = segment_utils.postprocess_srt_segments(parsed_segments, is_external_srt=True)
                subs_source_tag_for_output_name = "_customsrt"
                texts_for_punct_srt = [s.get('text', '') for s in all_srt_segments_loaded_list]
                restored_texts_srt = transcriber._restore_punctuation(texts_for_punct_srt) # _restore_punctuation остается в transcriber
                if len(restored_texts_srt) == len(all_srt_segments_loaded_list):
                    for i_s_srt, s_data_srt in enumerate(all_srt_segments_loaded_list): s_data_srt['text'] = restored_texts_srt[i_s_srt]
                temp_audio_for_srt_diar_path = os.path.join(self.current_temp_dir_for_operation_path, "temp_audio_for_srt_diar.wav")
                diar_audio_duration_ref = min(300, self.total_video_duration_seconds) 
                extracted_audio_diar_srt = video_processor.extract_audio(video_to_use_for_processing, temp_audio_for_srt_diar_path, 16000, 0, diar_audio_duration_ref)
                if extracted_audio_diar_srt and os.path.exists(extracted_audio_diar_srt):
                    diarization_for_srt_segments = transcriber.perform_diarization_only(extracted_audio_diar_srt, device=("cuda" if torch.cuda.is_available() else "cpu"))
                    all_srt_segments_loaded_list = segment_utils.assign_srt_segments_to_speakers(all_srt_segments_loaded_list, diarization_for_srt_segments, trust_srt_speaker_field=True)
                else: self._update_status("WARN: Could not extract audio for SRT diarization.", append=True)
                sample_text_from_srt = " ".join([s.get('text','') for s in all_srt_segments_loaded_list[:20] if s.get('text','').strip()])
                if detect_func and sample_text_from_srt: 
                    try:
                        if detect_func(sample_text_from_srt) == 'ru': skip_translation_based_on_srt_lang = True
                    except LangDetectException_cls as e_lang_detect: self._update_status(f"WARN: langdetect error for SRT: {e_lang_detect}", append=True)
                    if skip_translation_based_on_srt_lang: self._update_status("INFO: External SRT seems Russian. Translation will be skipped.", append=True)
            elif not parse_srt_success: self._update_status(f"WARN: Failed to parse SRT {local_srt_path_input_actual}.", append=True); local_srt_path_input_actual = None 
            else: self._update_status(f"WARN: SRT file {local_srt_path_input_actual} parsed as empty.", append=True); local_srt_path_input_actual = None 
        self.log_time("Video Info & SRT Load", start_t_info_srt)
        current_overall_progress_value_tracker += dub_steps_weights_map.get("Video Info & Initial SRT Load", 0)
        self._update_progress((current_overall_progress_value_tracker / total_weight_op_overall_val) * 100, "Info/SRT")

        chunk_definitions_for_processing = [] 
        self._update_status(f"➡️ Performing full audio transcription for smart chunking..."); start_t_full_transcribe = time.time()
        full_audio_for_phrases_path = os.path.join(self.current_temp_dir_for_operation_path, "full_audio_for_phrases.wav")
        path_to_full_audio_for_transcribe = None
        # Проверяем, есть ли уже извлеченное аудио (например, для SRT диаризации)
        # 'temp_audio_for_srt_diar_path' и 'diar_audio_duration_ref' должны быть определены, если SRT использовался
        existing_audio_path_check = locals().get('temp_audio_for_srt_diar_path')
        existing_audio_duration_check = locals().get('diar_audio_duration_ref', 0)

        if existing_audio_path_check and os.path.exists(existing_audio_path_check) and \
           existing_audio_duration_check >= self.total_video_duration_seconds - 1.0: # -1.0 для небольшой погрешности
            if os.path.abspath(existing_audio_path_check) != os.path.abspath(full_audio_for_phrases_path):
                 shutil.copy(existing_audio_path_check, full_audio_for_phrases_path)
            path_to_full_audio_for_transcribe = full_audio_for_phrases_path
        else:
            path_to_full_audio_for_transcribe = video_processor.extract_audio(
                video_to_use_for_processing, full_audio_for_phrases_path, 16000, 
                start_time_seconds=0, duration_seconds=self.total_video_duration_seconds
            )
        if path_to_full_audio_for_transcribe and os.path.exists(path_to_full_audio_for_transcribe):
            self.all_phrase_segments_for_video, 인식_success = transcriber.transcribe_full_audio_for_phrases(path_to_full_audio_for_transcribe, language_for_stt='en')
            if 인식_success and self.all_phrase_segments_for_video:
                chunk_definitions_for_processing = segment_utils.get_chunk_definitions_from_phrases(
                    self.all_phrase_segments_for_video, target_chunk_duration_s_actual, max_chunk_duration_s_actual, processing_start_offset_seconds_val)
                self._update_status(f"Smart chunking: Found {len(self.all_phrase_segments_for_video)} phrases, defined {len(chunk_definitions_for_processing)} chunks.", append=True)
            else: processing_error_message = "Full audio transcription for smart chunks failed or returned no phrases."; 
        else: processing_error_message = "Could not extract full audio for smart chunking."; 
        self.log_time("Full Audio Transcription (smart chunks)", start_t_full_transcribe)
        current_overall_progress_value_tracker += dub_steps_weights_map.get("Full Audio Transcription (for smart chunks)", 0)
        self._update_progress((current_overall_progress_value_tracker / total_weight_op_overall_val) * 100, "Transcribe All")
        if not chunk_definitions_for_processing: 
            final_err_msg_chunks = processing_error_message if processing_error_message else "No chunks defined by smart chunking strategy."
            self._update_status(f"CRITICAL ERROR: {final_err_msg_chunks}", append=True)
            self.root_tk.after(0, self.processing_finished_cleanup_and_feedback, False, final_err_msg_chunks); return

        # ... (дальнейшая логика цикла по чанкам, как в предыдущем ответе, но с использованием segment_utils.*) ...
        # Заменяем transcriber.assign_srt_segments_to_speakers на segment_utils.assign_srt_segments_to_speakers
        # Заменяем transcriber._postprocess_srt_segments на segment_utils.postprocess_srt_segments
        
        self._update_status(f"➡️ Starting chunk processing loop ({len(chunk_definitions_for_processing)} chunks)...")
        base_prog_chunks_loop = current_overall_progress_value_tracker
        weight_chunks_loop = dub_steps_weights_map.get("Chunk Processing Loop", 50)
        chunk_loop_crit_err = False

        for chunk_idx, chunk_info in enumerate(chunk_definitions_for_processing):
            if chunk_loop_crit_err: self._update_status(f"Chunk loop aborted due to critical error.", append=True); break
            chunk_start_s = chunk_info['start']; chunk_end_s = chunk_info['end']; chunk_dur_s = chunk_end_s - chunk_start_s
            chunk_id_log = f"chunk_{chunk_info['id']}"
            self._update_status(f"\n--- Processing {chunk_id_log} (Time: {chunk_start_s:.2f}s - {chunk_end_s:.2f}s, Dur: {chunk_dur_s:.2f}s) ---", append=True)
            chunk_temp_dir = os.path.join(self.current_temp_dir_for_operation_path, chunk_id_log); os.makedirs(chunk_temp_dir, exist_ok=True)
            audio_stt_ref_path = video_processor.extract_audio(video_to_use_for_processing, os.path.join(chunk_temp_dir, "stt_ref_audio.wav"), 16000, chunk_start_s, chunk_dur_s)
            orig_mix_audio_path = video_processor.extract_audio(video_to_use_for_processing, os.path.join(chunk_temp_dir, "original_mix_audio.wav"), 44100, chunk_start_s, chunk_dur_s)
            if not audio_stt_ref_path or not os.path.exists(audio_stt_ref_path): self._update_status(f"ERR: Failed to extract STT/ref audio for {chunk_id_log}. Skipping.", append=True); continue
            segments_tts_chunk = []; diar_chunk_df = pd.DataFrame(); skip_trans_chunk = False
            if all_srt_segments_loaded_list:
                for srt_s in all_srt_segments_loaded_list:
                    overlap_s = max(srt_s['start'], chunk_start_s); overlap_e = min(srt_s['end'], chunk_end_s)
                    if overlap_e > overlap_s + segment_utils.MIN_SEGMENT_DURATION_FOR_POSTPROCESS: 
                        new_s_srt = srt_s.copy(); new_s_srt['start'] = overlap_s - chunk_start_s; new_s_srt['end'] = overlap_e - chunk_start_s
                        new_s_srt['original_abs_start'] = overlap_s; new_s_srt['original_abs_end'] = overlap_e
                        segments_tts_chunk.append(new_s_srt)
                if segments_tts_chunk:
                    if diarization_for_srt_segments is not None and not diarization_for_srt_segments.empty:
                        diar_chunk_df = diarization_for_srt_segments[(diarization_for_srt_segments['start'] < chunk_end_s) & (diarization_for_srt_segments['end'] > chunk_start_s)].copy()
                        if not diar_chunk_df.empty: diar_chunk_df['start'] -= chunk_start_s; diar_chunk_df['end'] -= chunk_start_s; diar_chunk_df[['start', 'end']] = diar_chunk_df[['start', 'end']].clip(lower=0)
                    else: diar_chunk_df = transcriber.perform_diarization_only(audio_stt_ref_path, device=("cuda" if torch.cuda.is_available() else "cpu"))
                    segments_tts_chunk = segment_utils.assign_srt_segments_to_speakers(segments_tts_chunk, diar_chunk_df, trust_srt_speaker_field=True)
                    segments_tts_chunk = segment_utils.postprocess_srt_segments(segments_tts_chunk, is_external_srt=True)
                    skip_trans_chunk = skip_translation_based_on_srt_lang
                    self._update_status(f"  Using {len(segments_tts_chunk)} segments from external SRT for {chunk_id_log}.", append=True)
                else: self._update_status(f"  No external SRT segments for {chunk_id_log}. Using smart chunk phrases.", append=True)
            if not segments_tts_chunk: 
                diar_chunk_df = transcriber.perform_diarization_only(audio_stt_ref_path, device=("cuda" if torch.cuda.is_available() else "cpu"))
                segments_tts_chunk = segment_utils.assign_srt_segments_to_speakers(list(chunk_info.get('phrases_relative', [])), diar_chunk_df, trust_srt_speaker_field=False)
                segments_tts_chunk = segment_utils.postprocess_srt_segments(segments_tts_chunk, is_external_srt=False)
                self._update_status(f"  Using {len(segments_tts_chunk)} pre-transcribed phrases for {chunk_id_log} (smart chunking).", append=True)
                skip_trans_chunk = False 
            dub_audio_chunk_path = None # ... (логика TTS и сборки чанка, как раньше) ...
            # ... (остальная часть цикла и финальная сборка, как в предыдущем ответе) ...
            # (Продолжение логики обработки чанка, TTS, сборки и т.д. без изменений по сравнению с предыдущим полным ответом,
            # кроме использования segment_utils для функций, которые были туда перенесены)
            if not segments_tts_chunk: 
                self._update_status(f"  No text segments for TTS in {chunk_id_log}. Creating silent audio.", append=True)
                dub_audio_chunk_path = os.path.join(chunk_temp_dir, "dubbed_audio_silent.wav"); ff_silent_ok = False
                try:
                    (ffmpeg.input('anullsrc', format='lavfi', r=24000, channel_layout='mono').output(dub_audio_chunk_path, t=chunk_dur_s, acodec='pcm_s16le').overwrite_output().run(capture_stdout=True, capture_stderr=True))
                    if os.path.exists(dub_audio_chunk_path): ff_silent_ok = True
                except Exception as e_ff_sl: self._update_status(f"    ERR creating silent audio for {chunk_id_log}: {e_ff_sl}", append=True)
                if not ff_silent_ok: self._update_status(f"  CRIT ERR: Failed to create silent audio for {chunk_id_log}. Skipping chunk.", append=True); chunk_loop_crit_err = True; continue 
            else: 
                trans_segments_chunk = list(segments_tts_chunk) 
                if not skip_trans_chunk:
                    trans_segments_chunk, trans_ok = translator.translate_segments(segments_tts_chunk)
                    if not trans_ok: self._update_status(f"  ERR: Translation failed for segments in {chunk_id_log}. Using original text.", append=True)
                else: 
                    for s_no_tr in trans_segments_chunk: s_no_tr['translated_text'] = s_no_tr.get('text', '')
                dub_path, _, _, synth_ok = voice_cloner.synthesize_speech_segments(trans_segments_chunk, audio_stt_ref_path, chunk_temp_dir, diar_chunk_df, 'ru', None, chunk_start_s)
                if synth_ok and dub_path and os.path.exists(dub_path): dub_audio_chunk_path = dub_path
                else: self._update_status(f"  CRIT ERR: Failed to synthesize dubbed audio for {chunk_id_log}. Skipping chunk.", append=True); chunk_loop_crit_err = True; continue 
            
            vid_chunk_proc_path = os.path.join(chunk_temp_dir, "video_chunk_processed.mp4")
            mix_ok_chunk = video_processor.mix_and_replace_audio(video_to_use_for_processing, orig_mix_audio_path, dub_audio_chunk_path, vid_chunk_proc_path, video_start_time=chunk_start_s, video_duration=chunk_dur_s)
            if not mix_ok_chunk or not os.path.exists(vid_chunk_proc_path): self._update_status(f"WARN: Failed to assemble video for {chunk_id_log}. Part might be missing.", append=True)
            else: self.processed_video_chunks_paths.append(vid_chunk_proc_path)
            if orig_mix_audio_path and os.path.exists(orig_mix_audio_path): self.processed_original_audio_chunks_paths.append(orig_mix_audio_path)
            if dub_audio_chunk_path and os.path.exists(dub_audio_chunk_path): self.processed_dubbed_audio_chunks_paths.append(dub_audio_chunk_path)
            self.fully_processed_duration_seconds += chunk_dur_s
            self._update_status(f"{chunk_id_log} processed. Total: {self.fully_processed_duration_seconds:.2f}s / {self.total_video_duration_seconds:.2f}s", append=True)
            prog_chunk_loop = ((chunk_idx + 1) / len(chunk_definitions_for_processing) ) * weight_chunks_loop
            self._update_progress((base_prog_chunks_loop + prog_chunk_loop) / total_weight_op_overall_val * 100, f"Chunk {chunk_idx+1}/{len(chunk_definitions_for_processing)}")
            if self.processed_video_chunks_paths and hasattr(self, 'save_processed_button') and self.save_processed_button.cget('state') == tk.DISABLED:
                 if hasattr(self, 'root_tk') and self.root_tk.winfo_exists(): self.root_tk.after(0, lambda: self.save_processed_button.config(state=tk.NORMAL))
        
        final_vid_path_out = None; op_overall_ok = False
        if chunk_loop_crit_err: processing_error_message = "Critical error during chunk processing."
        elif not self.processed_video_chunks_paths: processing_error_message = "No video chunks were successfully processed."
        else: 
            self._update_status("\n--- Assembling Final Full Video ---", append=True)
            vid_base_name_final = os.path.splitext(os.path.basename(video_to_use_for_processing))[0]
            final_vid_path_out = os.path.join(self.current_final_output_dir_for_operation_path, f"{vid_base_name_final}_dubbed_ru{subs_source_tag_for_output_name}_FULL.mp4")
            full_dub_audio_path = video_processor.merge_audio_segments(self.processed_dubbed_audio_chunks_paths, os.path.join(self.current_temp_dir_for_operation_path, "FULL_dubbed_audio.wav"), log_prefix=None)
            full_orig_audio_path = None
            if self.processed_original_audio_chunks_paths: full_orig_audio_path = video_processor.merge_audio_segments(self.processed_original_audio_chunks_paths, os.path.join(self.current_temp_dir_for_operation_path, "FULL_original_audio.wav"), log_prefix=None)
            if not full_dub_audio_path or not os.path.exists(full_dub_audio_path): processing_error_message = "Failed to assemble full dubbed audio track."
            else:
                final_mix_ok = video_processor.mix_and_replace_audio(video_to_use_for_processing, full_orig_audio_path, full_dub_audio_path, final_vid_path_out, 0.1, 0.95)
                if final_mix_ok: self._update_status(f"Final full video assembled: {final_vid_path_out}", append=True); op_overall_ok = True
                else: processing_error_message = f"Failed to assemble final video to {final_vid_path_out}."
        current_overall_progress_value_tracker = base_prog_chunks_loop + weight_chunks_loop
        current_overall_progress_value_tracker += dub_steps_weights_map.get("Final Assembly", 0)
        self._update_progress((current_overall_progress_value_tracker / total_weight_op_overall_val) * 100, "Final Assembly")
        self.root_tk.after(0, self.processing_finished_cleanup_and_feedback, op_overall_ok, processing_error_message, final_vid_path_out)

    def processing_finished_cleanup_and_feedback(self, success_flag, error_msg_str, final_video_path_str=None): # Без изменений
        self.chunk_processing_in_progress = False; self.hide_progressbar(); self.enable_gui_elements_after_processing() 
        if success_flag and final_video_path_str and os.path.exists(final_video_path_str):
            total_t_s = time.time() - self.total_start_time; total_t_m = total_t_s / 60.0
            self._update_status(f"\n✅🎉 Total processing time: {total_t_m:.2f} minutes.", append=True)
            self._update_status(f"Output file:\n{final_video_path_str}", append=True)
            if hasattr(self, 'root_tk') and self.root_tk.winfo_exists(): messagebox.showinfo("Success", f"Operation completed in {total_t_m:.2f} minutes!\n\nOutput Video: {final_video_path_str}")
        elif success_flag: self._update_status(f"\n✅ Operation finished. Check logs. Use 'Save Processed Part' if applicable.", append=True)
        else: 
            self._update_status(f"\n❌ Operation FAILED. Error: {error_msg_str if error_msg_str else 'Unknown error'}", append=True)
            if hasattr(self, 'root_tk') and self.root_tk.winfo_exists(): messagebox.showerror("Processing Error", f"Operation failed:\n{error_msg_str if error_msg_str else 'Unknown error. Check log.'}")
        if self.current_temp_dir_for_operation_path: cleanup_temp_dir_main(self.current_temp_dir_for_operation_path); self.current_temp_dir_for_operation_path = None

    def disable_gui_elements_during_processing(self, disable_save=True): # Без изменений
        if hasattr(self, 'process_button'): self.process_button.config(state=tk.DISABLED)
        if hasattr(self, 'save_processed_button') and disable_save: self.save_processed_button.config(state=tk.DISABLED)
        if hasattr(self, 'save_original_video_button'): self.save_original_video_button.config(state=tk.DISABLED) # Блокируем и ее
        if hasattr(self, 'work_dir_apply_button'): self.work_dir_apply_button.config(state=tk.DISABLED)
        if hasattr(self, 'download_tools_button'): self.download_tools_button.config(state=tk.DISABLED)
        if hasattr(self, 'video_browse_button'): self.video_browse_button.config(state=tk.DISABLED)
        if hasattr(self, 'srt_browse_button'): self.srt_browse_button.config(state=tk.DISABLED)
        if hasattr(self, 'work_dir_browse_button'): self.work_dir_browse_button.config(state=tk.DISABLED)

    def enable_gui_elements_after_processing(self): # Без изменений
        if hasattr(self, 'process_button'): self.process_button.config(state=tk.NORMAL)
        if hasattr(self, 'save_processed_button'): self.save_processed_button.config(state=tk.NORMAL if self.processed_video_chunks_paths else tk.DISABLED)
        # Кнопка сохранения оригинала активируется, если есть self.downloaded_video_path_session
        if hasattr(self, 'save_original_video_button'): 
            self.save_original_video_button.config(state=tk.NORMAL if self.downloaded_video_path_session and os.path.exists(self.downloaded_video_path_session) else tk.DISABLED)
        if hasattr(self, 'work_dir_apply_button'): self.work_dir_apply_button.config(state=tk.NORMAL)
        if hasattr(self, 'download_tools_button'): self.download_tools_button.config(state=tk.NORMAL)
        if hasattr(self, 'video_browse_button'): self.video_browse_button.config(state=tk.NORMAL)
        if hasattr(self, 'srt_browse_button'): self.srt_browse_button.config(state=tk.NORMAL)
        if hasattr(self, 'work_dir_browse_button'): self.work_dir_browse_button.config(state=tk.NORMAL)

    def show_progressbar(self): # Без изменений
        if hasattr(self, 'progress_frame') and not self.progress_frame.winfo_ismapped():
            self.progress_frame.pack(padx=10, pady=(5, 5), fill="x", after=self.save_processed_button)
            self._update_progress(0, "Initializing"); 
            if hasattr(self, 'root_tk'): self.root_tk.update_idletasks()
    def hide_progressbar(self): # Без изменений
        if hasattr(self, 'progress_frame') and self.progress_frame.winfo_ismapped(): self.progress_frame.pack_forget()
        if hasattr(self, 'progress_label_text'): self.progress_label_text.set("Progress: 0%")

if __name__ == "__main__": # Без изменений
    initial_work_dir_path = config_manager.get_work_dir_from_config()
    ffmpeg_ready_startup, ffprobe_ready_startup, _ = config_manager.initialize_paths_from_work_dir(initial_work_dir_path) 
    if not pyperclip: messagebox.showwarning("Dependency Warning", "The 'pyperclip' library is not installed. Log copying will not work. Please install it using: pip install pyperclip")
    if not langdetect_lib_available: messagebox.showwarning("Dependency Warning", "The 'langdetect' library not installed. Language detection of custom SRTs will rely on filename only. Install with: pip install langdetect")
    if transcriber.PunctuationModel is None: messagebox.showwarning("Dependency Warning", "The 'deepmultilingualpunctuation' library not found or failed to load. Punctuation restoration will be skipped. Install with 'pip install deepmultilingualpunctuation'")
    pytorch_warn_msg = ""; show_pyt_warn = False; pyt_msg_type = "info"
    try: 
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            if "+cpu" in torch.__version__: show_pyt_warn=True; pyt_msg_type="warning"; pytorch_warn_msg="CUDA detected, but PyTorch is CPU-only. Install PyTorch with CUDA for faster processing."
        else: 
            if "+cpu" not in torch.__version__ and ("cuda" in torch.__version__ or "gpu" in torch.__version__): show_pyt_warn=True; pyt_msg_type="warning"; pytorch_warn_msg="PyTorch is GPU-enabled, but CUDA not detected. ML models will use CPU."
            else: show_pyt_warn=True; pyt_msg_type="info"; pytorch_warn_msg="PyTorch is using CPU. For faster processing on NVIDIA GPUs, ensure CUDA-enabled PyTorch and compatible hardware/drivers."
    except ImportError: messagebox.showerror("Fatal Dependency Error", "PyTorch not found. Please install it."); sys.exit(1)
    except Exception as e_torch_c: messagebox.showwarning("Dependency Check Warning", f"Could not check PyTorch/CUDA status:\n{e_torch_c}"); show_pyt_warn = False
    if not (ffmpeg_ready_startup and ffprobe_ready_startup):
        ff_sys_ok, ff_sys_msg = video_processor.check_command_availability('ffmpeg') 
        ffp_sys_ok, ffp_sys_msg = video_processor.check_command_availability('ffprobe')
        if not (ff_sys_ok and ffp_sys_ok): messagebox.showerror("Fatal Dependency Error", f"FFmpeg/ffprobe not found or not executable.\nFFmpeg: {ff_sys_msg}\nFFprobe: {ffp_sys_msg}\nEnsure they are in PATH or configure work directory.")
    if not srt_lib_available: messagebox.showwarning("Dependency Warning", "The 'srt' library not installed. SRT parsing/generation unavailable. Install with: pip install srt")
    if not yt_dlp_lib_available: print("WARNING: yt-dlp not installed. YouTube download disabled.")
    root_tk_main_app = tk.Tk(); app_instance_main = None
    try:
        app_instance_main = App(root_tk_main_app)
        if not root_tk_main_app.winfo_exists(): sys.exit("Приложение закрыто во время инициализации.")
    except Exception as e_app_init_main:
        init_err_full = f"Критическая ошибка инициализации окна приложения:\n{e_app_init_main}\n\nTraceback:\n{traceback.format_exc()}"; print(init_err_full)
        if root_tk_main_app.winfo_exists(): messagebox.showerror("Ошибка инициализации приложения", init_err_full)
        sys.exit(f"Критическая ошибка инициализации App: {e_app_init_main}")
    if app_instance_main:
        if initial_work_dir_path: 
            if not (ffmpeg_ready_startup and ffprobe_ready_startup): app_instance_main._update_status("WARN: Work dir loaded, but FFmpeg/FFprobe not found there or system-wide. Check settings or use 'Download FFmpeg'.", append=True)
        elif not (ffmpeg_ready_startup and ffprobe_ready_startup): app_instance_main._update_status("WARN: Work dir not set, and system FFmpeg/FFprobe not found. Configure work dir and use 'Download FFmpeg'.", append=True)
        if show_pyt_warn and pytorch_warn_msg:
            if pyt_msg_type == "info": root_tk_main_app.after(200, lambda: messagebox.showinfo("System Info (PyTorch)", pytorch_warn_msg))
            elif pyt_msg_type == "warning": root_tk_main_app.after(200, lambda: messagebox.showwarning("System Configuration Warning (PyTorch)", pytorch_warn_msg))
    try: root_tk_main_app.mainloop()
    except Exception as e_gui_ml: 
        gui_err_full = f"Критическая ошибка в главном цикле приложения:\n\n{e_gui_ml}\n\nTraceback:\n{traceback.format_exc()}"; print(f"\n--- GUI Error ---"); print(gui_err_full)
        if root_tk_main_app.winfo_exists(): messagebox.showerror("Application Error", gui_err_full)