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

# --- Попытка решить проблему с загрузкой модели Coqui TTS и PyTorch 2.1+ ---
# ... (без изменений) ...
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

class App: # ... (GUI __init__ и методы без изменений, кроме версии и удаления cookie-полей) ...
    def __init__(self, root):
        self.root = root
        self.root.title("Video Dubbing Tool (WhisperX/SRT + XTTS) - v0.7.2") 
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
        try: self.video_source_entry.unbind("<<Paste>>") 
        except tk.TclError: pass 
        self.video_source_entry.bind("<Control-v>", self._force_paste_event); self.video_source_entry.bind("<Control-V>", self._force_paste_event)
        if self.root.tk.call('tk', 'windowingsystem') == 'aqua': self.video_source_entry.bind("<Command-v>", self._force_paste_event); self.video_source_entry.bind("<Command-V>", self._force_paste_event)
        self.video_source_entry.bind("<Shift-Insert>", self._force_paste_event)
        self.video_browse_button = ttk.Button(source_frame, text="Browse File...", command=self.browse_video_for_source_entry); self.video_browse_button.pack(side=tk.LEFT)
        self.video_clear_button = ttk.Button(source_frame, text="Clear", command=self.clear_video_source_text); self.video_clear_button.pack(side=tk.LEFT, padx=(5,0))
        yt_subs_info_label = ttk.Label(source_frame, text="For YouTube URLs, subtitles (RU then EN) are auto-downloaded if available.", font=("Segoe UI", 8)); yt_subs_info_label.pack(side=tk.BOTTOM, anchor=tk.W, padx=0, pady=(5,0))
        self.srt_outer_frame = ttk.LabelFrame(root, text="Subtitles File (Optional: if provided, used instead of WhisperX/YT subs)", padding=(10, 5)); self.srt_outer_frame.pack(padx=10, pady=5, fill="x")
        srt_input_line_frame = ttk.Frame(self.srt_outer_frame); srt_input_line_frame.pack(fill="x") 
        self.srt_path_entry = ttk.Entry(srt_input_line_frame, textvariable=self.srt_path, width=60, state="readonly"); self.srt_path_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        self.srt_browse_button = ttk.Button(srt_input_line_frame, text="Browse...", command=self.browse_srt) ; self.srt_browse_button.pack(side=tk.LEFT)
        self.srt_clear_button = ttk.Button(srt_input_line_frame, text="Clear", command=self.clear_srt_path) ; self.srt_clear_button.pack(side=tk.LEFT, padx=(5,0))
        output_dir_frame = ttk.LabelFrame(root, text="Output Directory (Optional: defaults to subfolder in CWD or near original video)", padding=(10,5)); output_dir_frame.pack(padx=10, pady=5, fill="x")
        self.output_dir_entry = ttk.Entry(output_dir_frame, textvariable=self.output_dir_path, width=60, state="readonly"); self.output_dir_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(output_dir_frame, text="Browse...", command=self.browse_output_dir).pack(side=tk.LEFT)
        ttk.Button(output_dir_frame, text="Clear", command=self.clear_output_dir).pack(side=tk.LEFT, padx=(5,0))
        self.process_button = ttk.Button(root, text="Translate & Dub Video", command=self.start_processing_thread); self.process_button.pack(pady=10) 
        self.progress_frame = ttk.Frame(root, padding=(10,5))
        self.progress_label_text = tk.StringVar(value="Progress: 0%"); self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_label_text, width=30, anchor="w"); self.progress_label.pack(side=tk.LEFT, padx=(0,5))
        self.progressbar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', maximum=100); self.progressbar.pack(side=tk.LEFT, fill="x", expand=True)
        status_frame_outer = ttk.LabelFrame(root, text="Status / Log", padding=(10,5)); status_frame_outer.pack(padx=10, pady=(0,5), fill="both", expand=True)
        self.status_text = scrolledtext.ScrolledText(status_frame_outer, height=10, wrap=tk.WORD, state="disabled", bg="#ffffff", relief="sunken", borderwidth=1, font=("Consolas", 9)); self.status_text.pack(side=tk.TOP, fill="both", expand=True)
        self.copy_log_button = ttk.Button(status_frame_outer, text="Copy Log to Clipboard", command=self.copy_log_to_clipboard); self.copy_log_button.pack(side=tk.TOP, pady=(5,0))

    def copy_log_to_clipboard(self): # ... (без изменений) ...
        try: log_content = self.status_text.get("1.0", tk.END); pyperclip.copy(log_content); self._update_status("Log copied to clipboard.", append=True); messagebox.showinfo("Log Copied", "The log content has been copied to your clipboard.")
        except pyperclip.PyperclipException as e: self._update_status(f"Error copying log: {e}. Pyperclip might not be configured correctly.", append=True); messagebox.showerror("Copy Error", f"Could not copy log: {e}\nInstall xclip or xsel (Linux).")
        except Exception as e_general: self._update_status(f"Unexpected error copying log: {e_general}", append=True); messagebox.showerror("Copy Error", f"Unexpected error: {e_general}")
    def _force_paste_event(self, event): # ... (без изменений) ...
        widget = event.widget; log_msg = f"DEBUG: Hotkey event: keysym='{event.keysym}', keycode={event.keycode}, state={event.state}, char='{event.char}' on widget {widget}"; self._update_status(log_msg, append=True)
        if widget != self.video_source_entry: self._update_status(f"DEBUG: Hotkey event ignored, widget is not video_source_entry.", append=True); return
        if widget.cget("state") == 'readonly': self._update_status(f"DEBUG: video_source_entry is readonly, paste ignored.", append=True); return "break" 
        try:
            clipboard_content = ""; 
            try: clipboard_content = widget.clipboard_get(); self._update_status(f"DEBUG: Clipboard content (first 50 chars): '{clipboard_content[:50]}'", append=True)
            except tk.TclError: self._update_status(f"DEBUG: TclError getting clipboard content (clipboard empty or not text?).", append=True)
            current_text_before_paste = widget.get(); self._update_status(f"DEBUG: Text in widget BEFORE <<Paste>>: '{current_text_before_paste}'", append=True)
            widget.event_generate("<<Paste>>"); self._update_status(f"DEBUG: <<Paste>> event generated for {widget} via hotkey '{event.keysym}'.", append=True)
            widget.after(20, lambda w=widget, clip_text=clipboard_content: self._log_text_after_paste(w, clip_text))
            return "break" 
        except tk.TclError as e: self._update_status(f"DEBUG: TclError during <<Paste>> event generation: {e}", append=True); return "break" 
        except Exception as e_paste: self._update_status(f"DEBUG: Generic error during hotkey paste handling: {e_paste}\n{traceback.format_exc()}", append=True); return "break"
    def _log_text_after_paste(self, widget, expected_clipboard_content): # ... (без изменений) ...
        current_text_after_paste = widget.get(); self._update_status(f"DEBUG: Text in widget AFTER <<Paste>>: '{current_text_after_paste}'", append=True)
        if expected_clipboard_content and expected_clipboard_content not in current_text_after_paste: self._update_status(f"WARNING: Expected clipboard content ('{expected_clipboard_content[:30]}...') not found in widget after paste. Something might be wrong.", append=True)
    def make_context_menu_for_entry_fields(self): # ... (без изменений) ...
        self.entry_context_menu = tk.Menu(self.root, tearoff=0)
        self.entry_context_menu.add_command(label="Cut", command=lambda: self.root.focus_get().event_generate("<<Cut>>"))
        self.entry_context_menu.add_command(label="Copy", command=lambda: self.root.focus_get().event_generate("<<Copy>>"))
        self.entry_context_menu.add_command(label="Paste", command=lambda: self.root.focus_get().event_generate("<<Paste>>"))
        self.entry_context_menu.add_separator()
        self.entry_context_menu.add_command(label="Select All", command=lambda: self.root.focus_get().event_generate("<<SelectAll>>"))
    def show_entry_context_menu(self, event): # ... (без изменений) ...
        widget = event.widget 
        if isinstance(widget, (ttk.Entry, tk.Entry, scrolledtext.ScrolledText)) and widget.cget("state") != 'readonly':
            self.entry_context_menu.tk_popup(event.x_root, event.y_root)
    def is_youtube_url(self, url_string): # ... (без изменений) ...
        if not url_string: return False
        youtube_regex = re.compile(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
        return bool(youtube_regex.match(url_string))
    def log_time(self, step_name, start_time): # ... (без изменений) ...
        duration = time.time() - start_time; self.processing_times[step_name] = duration; self._update_status(f"⏱️ {step_name} took {duration:.2f} seconds.", append=True)
    def _update_status(self, message, append=True): # ... (без изменений) ...
        def update_gui():
            if not self.status_text.winfo_exists(): return
            current_state = self.status_text.cget("state")
            self.status_text.config(state="normal")
            if append: self.status_text.insert(tk.END, str(message) + "\n")
            else: self.status_text.delete("1.0", tk.END); self.status_text.insert("1.0", str(message) + "\n")
            self.status_text.config(state=current_state); self.status_text.see(tk.END)
        if self.root.winfo_exists(): self.root.after_idle(update_gui)
    def _update_progress(self, value, step_name=""): # ... (без изменений) ...
        clamped_value = max(0, min(100, int(value)))
        def update_gui():
            if not (self.root.winfo_exists() and self.progressbar.winfo_exists() and self.progress_label.winfo_exists()): return
            self.progressbar['value'] = clamped_value
            progress_text = f"{step_name}: {clamped_value}%" if step_name else f"Progress: {clamped_value}%"
            if clamped_value == 100 and step_name == "Completed": progress_text = f"Process: Done!"
            elif clamped_value == 100 and step_name and step_name != "Completed": progress_text = f"{step_name}: Done!"
            elif not step_name and clamped_value == 0: progress_text = "Progress: 0%"
            self.progress_label_text.set(progress_text)
        if self.root.winfo_exists(): self.root.after_idle(update_gui)
    def browse_video_for_source_entry(self): # ... (без изменений) ...
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm *.flv")])
        if path: self.video_source_text.set(path); self._update_status(f"Video file selected: {path}", append=False)
    def clear_video_source_text(self): # ... (без изменений) ...
        self.video_source_text.set(""); self._update_status("Video source cleared.", append=False)
    def browse_srt(self): # ... (без изменений) ...
        path = filedialog.askopenfilename(filetypes=[("SRT/VTT Files", "*.srt *.vtt")]) 
        if path: self.srt_path.set(path); self._update_status(f"Subtitles file selected: {path}. If provided, this will be used instead of WhisperX/YT subs.", append=True)
    def clear_srt_path(self): # ... (без изменений) ...
        self.srt_path.set(""); self._update_status("SRT/VTT file selection cleared. WhisperX or downloaded YouTube subs will be used.", append=True)
    def browse_output_dir(self): # ... (без изменений) ...
        path = filedialog.askdirectory(); 
        if path: self.output_dir_path.set(path); self._update_status(f"Output directory set to: {path}", append=True)
    def clear_output_dir(self): # ... (без изменений) ...
        self.output_dir_path.set(""); self._update_status("Output directory cleared. Will use default.", append=True)
    
    def start_processing_thread(self): # ... (без изменений в вызове actual_processing) ...
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
        # ... (начало функции actual_processing без изменений) ...
        current_temp_dir = None; output_path = None; success = False
        dub_steps = { 
            "YouTube Download & Subs": 8, "Audio Extraction": 5, "SRT Parsing": 2, "Diarization for SRT": 8,
            "Transcription & Diarization": 32, "Translation": 10, "Voice Synthesis": 30, "Video Assembly": 5
        }
        video_to_use = video_file_path; srt_to_use_for_transcription = local_srt_path 
        subs_source_tag = ""; diarization_data_for_cloning = None; skip_translation_step = False
        # final_translated_srt_path = None # Убрали, так как не сохраняем больше SRT
        total_raw_tts_duration = 0.0
        total_adjusted_segments_duration = 0.0
        try:
            current_temp_dir = create_temp_dir(); self._update_status(f"Using temp dir: {current_temp_dir}")
            device = "cuda" if torch.cuda.is_available() else "cpu"; current_progress = 0 
            total_weight_op = sum(dub_steps.values())
            download_step_name = "YouTube Download & Subs"
            if download_step_name in dub_steps:
                if is_youtube:
                    self._update_status(f"➡️ {download_step_name} from {youtube_url_str}...");
                    self._update_progress(current_progress / total_weight_op * 100, download_step_name)
                    start_t_dl = time.time(); download_output_dir_vid = os.path.join(current_temp_dir, "youtube_download"); os.makedirs(download_output_dir_vid, exist_ok=True)
                    self.downloaded_video_path, self.downloaded_srt_path, self.downloaded_srt_lang = video_processor.download_youtube_video(
                        url=youtube_url_str, output_dir=download_output_dir_vid, 
                        preferred_sub_lang='ru', fallback_sub_lang='en')
                    video_to_use = self.downloaded_video_path 
                    self.log_time(download_step_name, start_t_dl)
                    if self.downloaded_srt_path and os.path.exists(self.downloaded_srt_path):
                        self._update_status(f"✅ YouTube subtitles ({self.downloaded_srt_lang or 'unknown'}) downloaded: {os.path.basename(self.downloaded_srt_path)}", append=True)
                        if not srt_to_use_for_transcription: 
                            srt_to_use_for_transcription = self.downloaded_srt_path 
                            subs_source_tag = f"_ytbsubs-{self.downloaded_srt_lang}" if self.downloaded_srt_lang else "_ytbsubs"
                            if self.downloaded_srt_lang and 'ru' in self.downloaded_srt_lang.lower(): 
                                skip_translation_step = True; self._update_status("INFO: Downloaded Russian YouTube subtitles. Translation step will be skipped.", append=True)
                    else: self._update_status(f"⚠️ YouTube subtitles for specified languages were not found/downloaded.", append=True)
                current_progress += dub_steps[download_step_name]
            if not video_to_use or not os.path.exists(video_to_use): raise FileNotFoundError(f"Video file for processing not found: {video_to_use or 'None'}")
            aligned_segments = None; stt_audio_path_for_cloning = None; original_audio_for_mixing = None; whisperx_srt_output_path = None 
            if user_output_dir and os.path.isdir(user_output_dir): final_output_dir = user_output_dir
            elif video_file_path and not is_youtube : final_output_dir = os.path.dirname(video_file_path)
            else: final_output_dir = os.path.join(os.getcwd(), "Translated_Videos"); os.makedirs(final_output_dir, exist_ok=True)
            self._update_status(f"Final outputs will be saved to: {final_output_dir}", append=True)
            video_name_base_for_outputs = os.path.splitext(os.path.basename(video_to_use))[0]
            step_name_ae = "Audio Extraction"
            if step_name_ae in dub_steps:
                self._update_status(f"➡️ {step_name_ae} from {os.path.basename(video_to_use)}..."); self._update_progress(current_progress / total_weight_op * 100, step_name_ae); start_t_ae = time.time()
                original_audio_for_mixing = video_processor.extract_audio(video_to_use, os.path.join(current_temp_dir, "original_for_mix.wav"), sample_rate=44100)
                stt_audio_path_for_cloning = video_processor.extract_audio(video_to_use, os.path.join(current_temp_dir, "audio_for_diar_clone.wav"), sample_rate=16000) 
                self.log_time(step_name_ae, start_t_ae); current_progress += dub_steps[step_name_ae]
            if stt_audio_path_for_cloning is None or original_audio_for_mixing is None: raise RuntimeError("Audio for cloning or mixing was not prepared.")
            step_name_parse = "SRT Parsing"; step_name_diar_srt = "Diarization for SRT"
            if srt_to_use_for_transcription: 
                if not os.path.exists(srt_to_use_for_transcription):
                     self._update_status(f"⚠️ Specified/downloaded subtitle file not found: {srt_to_use_for_transcription}. Proceeding with WhisperX transcription.", append=True)
                     srt_to_use_for_transcription = None 
                else:
                    if not subs_source_tag: subs_source_tag = "_customsubs"
                    if step_name_parse in dub_steps:
                        self._update_status(f"➡️ {step_name_parse} from {os.path.basename(srt_to_use_for_transcription)}...");
                        self._update_progress(current_progress / total_weight_op * 100, step_name_parse); start_t = time.time()
                        parsed_srt_segments = transcriber.parse_srt_file(srt_to_use_for_transcription)
                        if not parsed_srt_segments: 
                            self._update_status(f"⚠️ SRT parsing failed for {os.path.basename(srt_to_use_for_transcription)}. Will attempt WhisperX transcription.", append=True)
                            srt_to_use_for_transcription = None 
                        else:
                            self.log_time(step_name_parse, start_t)
                            if local_srt_path and not subs_source_tag.startswith("_ytbsubs-"): 
                                if detect: 
                                    try:
                                        sample_text = " ".join([seg['text'] for seg in parsed_srt_segments[:10] if seg.get('text','').strip()])
                                        if len(sample_text) > 50: 
                                            detected_lang = detect(sample_text)
                                            self._update_status(f"Langdetect: Detected language for custom SRT: '{detected_lang}'", append=True)
                                            if detected_lang == 'ru':
                                                skip_translation_step = True
                                                self._update_status("INFO: Custom subtitles detected as Russian by content. Translation step will be skipped.", append=True)
                                        else: 
                                            self._update_status("Langdetect: Not enough text in custom SRT to reliably detect language by content. Checking filename...", append=True)
                                            if ('.ru.' in os.path.basename(local_srt_path).lower() or \
                                                os.path.basename(local_srt_path).lower().endswith(('.ru.srt', '.ru.vtt'))):
                                                skip_translation_step = True
                                                self._update_status("INFO: Custom subtitles seem to be Russian (based on filename). Translation step will be skipped.", append=True)
                                    except LangDetectException as e_lang: self._update_status(f"Langdetect: Could not detect language for custom SRT: {e_lang}. Assuming non-Russian.", append=True)
                                    except Exception as e_gen_lang: self._update_status(f"Langdetect: Error during language detection: {e_gen_lang}. Assuming non-Russian.", append=True)
                                else: 
                                    if ('.ru.' in os.path.basename(local_srt_path).lower() or \
                                        os.path.basename(local_srt_path).lower().endswith(('.ru.srt', '.ru.vtt'))):
                                        skip_translation_step = True
                                        self._update_status("INFO: Custom subtitles seem to be Russian (based on filename, langdetect not available). Translation step will be skipped.", append=True)
                            elif subs_source_tag.startswith("_ytbsubs-") and 'ru' in subs_source_tag.lower(): 
                                skip_translation_step = True; self._update_status("INFO: Using parsed Russian YouTube subtitles. Translation step will be skipped.", append=True)
                            if step_name_diar_srt in dub_steps:
                                self._update_status(f"➡️ {step_name_diar_srt} using audio: {os.path.basename(stt_audio_path_for_cloning)}..."); 
                                self._update_progress((current_progress + dub_steps.get(step_name_parse,0)) / total_weight_op * 100, step_name_diar_srt); start_t_ds = time.time()
                                diarization_data_for_cloning = transcriber.perform_diarization_only(stt_audio_path_for_cloning, device=device) 
                                if diarization_data_for_cloning is not None and not diarization_data_for_cloning.empty:
                                    aligned_segments = transcriber.assign_srt_segments_to_speakers(parsed_srt_segments, diarization_data_for_cloning)
                                    self._update_status(f"Assigned speakers to {len(aligned_segments)} SRT segments.", append=True)
                                else: 
                                    self._update_status("⚠️ Diarization for SRT failed or no speakers found. Using default speaker for all SRT segments.", append=True)
                                    aligned_segments = parsed_srt_segments 
                                self.log_time(step_name_diar_srt, start_t_ds)
            if step_name_parse in dub_steps: current_progress += dub_steps[step_name_parse]
            if step_name_diar_srt in dub_steps: current_progress += dub_steps[step_name_diar_srt]
            step_name_td = "Transcription & Diarization" # ... (остальной код до Voice Synthesis без изменений) ...
            if step_name_td in dub_steps:
                if not aligned_segments: 
                    subs_source_tag = "_whspsubs"
                    self._update_status(f"➡️ {step_name_td} (WhisperX) using audio: {os.path.basename(stt_audio_path_for_cloning)}..."); 
                    self._update_progress(current_progress / total_weight_op * 100, step_name_td); start_t_td = time.time()
                    whisperx_srt_output_path = os.path.join(final_output_dir, f"{video_name_base_for_outputs}{subs_source_tag}_transcription.srt") 
                    aligned_segments, diarization_data_for_cloning = transcriber.transcribe_and_diarize(
                        stt_audio_path_for_cloning, output_srt_path=whisperx_srt_output_path, return_diarization_df=True )
                    if not aligned_segments: raise ValueError("Transcription/Diarization failed or returned no segments.")
                    if os.path.exists(whisperx_srt_output_path): self._update_status(f"WhisperX transcription saved to: {whisperx_srt_output_path}", append=True)
                    self.log_time(step_name_td, start_t_td)
                current_progress += dub_steps[step_name_td]
            if aligned_segments is None: raise RuntimeError("Failed to obtain segments for translation/synthesis.")
            for seg_idx, seg_val in enumerate(aligned_segments):
                cleaned_text = seg_val.get('text', ''); cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text).strip(); aligned_segments[seg_idx]['text'] = cleaned_text
            translated_segments = list(aligned_segments)
            step_name_trans = "Translation"
            if step_name_trans in dub_steps:
                if not skip_translation_step:
                    self._update_status(f"➡️ {step_name_trans} (Helsinki-NLP)..."); self._update_progress(current_progress / total_weight_op * 100, step_name_trans); start_t_tr = time.time()
                    translator.load_translator_model(device=device); translated_segments = translator.translate_segments(aligned_segments) 
                    self.log_time(step_name_trans, start_t_tr)
                else: 
                    self._update_status("➡️ Translation step skipped (using pre-translated/Russian text).", append=True); 
                    self._update_progress(current_progress / total_weight_op * 100, f"{step_name_trans} (Skipped)")
                    for seg_idx, seg_val in enumerate(translated_segments): translated_segments[seg_idx]['translated_text'] = seg_val.get('text', '')
                current_progress += dub_steps[step_name_trans]
            step_name_vs = "Voice Synthesis"
            if step_name_vs in dub_steps:
                self._update_status(f"➡️ {step_name_vs} (Coqui XTTS-v2)...");
                progress_before_synthesis_step = current_progress 
                def synthesis_progress_callback(fraction_done): 
                    base_prog_val = (progress_before_synthesis_step / total_weight_op * 100)
                    step_prog_val = (dub_steps.get(step_name_vs, 0) / total_weight_op * 100) * fraction_done
                    self._update_progress(int(base_prog_val + step_prog_val), f"{step_name_vs} ({int(fraction_done*100)}%)")
                self._update_progress(current_progress / total_weight_op * 100, step_name_vs); start_t_vs = time.time()
                voice_cloner.load_tts_model(device=device)
                final_dubbed_audio_path, total_raw_tts_duration, total_adjusted_segments_duration = voice_cloner.synthesize_speech_segments(
                    translated_segments, stt_audio_path_for_cloning, current_temp_dir, 
                    diarization_result_df=diarization_data_for_cloning, progress_callback=synthesis_progress_callback )
                self.log_time(step_name_vs, start_t_vs)
                self._update_status(f"📊 Voice Synthesis Stats: Total raw TTS duration: {total_raw_tts_duration:.2f}s, Total final segments duration (incl. silence): {total_adjusted_segments_duration:.2f}s", append=True)
                current_progress += dub_steps[step_name_vs]
            
            # Убрали сохранение SRT файла здесь
            # if translated_segments and srt:
            #    ... 

            step_name_va = "Video Assembly" # ... (остальной код до конца без изменений, кроме сообщения об успехе) ...
            if step_name_va in dub_steps:
                self._update_status(f"➡️ {step_name_va} (FFmpeg)..."); self._update_progress(current_progress / total_weight_op * 100, step_name_va); start_t_va = time.time()
                output_path = os.path.join(final_output_dir, f"{video_name_base_for_outputs}_dubbed_ru{subs_source_tag}.mp4") 
                if not os.path.exists(original_audio_for_mixing): raise FileNotFoundError(f"Original audio for mixing not found: {original_audio_for_mixing}")
                if not final_dubbed_audio_path or not os.path.exists(final_dubbed_audio_path): raise FileNotFoundError(f"Final dubbed audio not found or not generated: {final_dubbed_audio_path}")
                video_processor.mix_and_replace_audio(video_to_use, original_audio_for_mixing, final_dubbed_audio_path, output_path, original_volume=0.10, dubbed_volume=0.95)
                self.log_time(step_name_va, start_t_va); 
            self._update_progress(100, "Completed")
            if output_path and os.path.exists(output_path): 
                total_time_seconds = time.time() - self.total_start_time; total_time_minutes = total_time_seconds / 60.0 
                self._update_status(f"\n✅🎉 Total processing time: {total_time_minutes:.2f} minutes.") 
                self._update_status(f"📊 Final Audio Durations: Raw TTS Sum: {total_raw_tts_duration:.2f}s, Final Segments Sum (incl. silence): {total_adjusted_segments_duration:.2f}s", append=True)
                try: 
                    final_video_info = ffmpeg.probe(output_path); final_video_duration = float(final_video_info['format']['duration'])
                    self._update_status(f"  Output video duration: {final_video_duration:.2f}s", append=True)
                except Exception as e_probe: self._update_status(f"  Could not probe output video duration: {e_probe}", append=True)
                self._update_status(f"✅ Operation '{op_key}' finished successfully!", append=True); self._update_status(f"Output file:\n{output_path}", append=True)
                if whisperx_srt_output_path and os.path.exists(whisperx_srt_output_path): self._update_status(f"WhisperX transcription (source) saved to:\n{whisperx_srt_output_path}", append=True)
                success_message = f"Operation '{op_key}' completed in {total_time_minutes:.2f} minutes!\n\nOutput Video: {output_path}"
                # Убрали final_translated_srt_path из сообщения
                if whisperx_srt_output_path and os.path.exists(whisperx_srt_output_path): success_message += f"\nSource Transcription SRT: {whisperx_srt_output_path}"
                self.root.after(0, lambda msg=success_message: messagebox.showinfo("Success", msg)); success = True
            else: 
                err_msg_out = f"⚠️ Operation '{op_key}' finished, but the expected output file was not found"
                if output_path: err_msg_out += f":\n{output_path}"
                else: err_msg_out += ", as no output path was generated."
                self._update_status(err_msg_out, append=True)
                if output_path: self.root.after(0, lambda: messagebox.showwarning("Warning", f"Operation '{op_key}' finished, but the output file seems to be missing."))
        except Exception as e: 
            tb_str = traceback.format_exc(); failed_step = "Initialization or Unknown Step" 
            temp_cumulative_weight = 0
            for name, weight in dub_steps.items():
                if current_progress < temp_cumulative_weight + weight: failed_step = name; break
                temp_cumulative_weight += weight
            if failed_step == "Initialization or Unknown Step" and current_progress == 0 : 
                if is_youtube: failed_step = "YouTube Download & Subs" 
                else: failed_step = "Audio Extraction"
            if failed_step == "Voice Synthesis" and isinstance(e, NameError) and 'shutil' in str(e): failed_step = "Voice Synthesis (shutil error)"
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
    def stop_and_hide_progressbar(self): # ... (без изменений) ...
        if self.root.winfo_exists():
            try:
                if self.progress_frame.winfo_ismapped(): self.progress_frame.pack_forget() 
                self.progressbar['value'] = 0; self.progress_label_text.set("Progress: 0%")
            except tk.TclError as e: print(f"Warning: Could not reset/hide progress bar: {e}")
    def enable_process_button(self): # ... (без изменений) ...
        if self.root.winfo_exists():
             try: self.process_button.config(state="normal")
             except tk.TclError as e: print(f"Warning: Could not enable process button: {e}")

if __name__ == "__main__": # ... (без изменений) ...
    if not pyperclip: messagebox.showwarning("Dependency Warning", "The 'pyperclip' library is not installed. Log copying will not work. Please install it using: pip install pyperclip")
    if detect is None: messagebox.showwarning("Dependency Warning", "The 'langdetect' library is not installed. Language detection of custom SRTs will rely on filename only. Install with: pip install langdetect")
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
    ffmpeg_ok, ffmpeg_msg = video_processor.check_command_availability('ffmpeg'); ffprobe_ok, ffprobe_msg = video_processor.check_command_availability('ffprobe')
    if not (ffmpeg_ok and ffprobe_ok): messagebox.showerror("Fatal Dependency Error", f"FFmpeg and/or ffprobe not found or not executable.\nFFmpeg: {ffmpeg_msg}\nFFprobe: {ffprobe_msg}\nPlease ensure they are installed and in your system's PATH."); exit(1)
    if srt is None: messagebox.showwarning("Dependency Warning", "The 'srt' library is not installed. SRT file parsing and generation will be unavailable. Please install it using: pip install srt")
    if yt_dlp is None: print("WARNING: yt-dlp is not installed. YouTube download functionality will be disabled.")
    root = tk.Tk(); app = App(root)
    if show_pytorch_warning:
        if pytorch_msg_type == "info": root.after(200, lambda: messagebox.showinfo("System Info", pytorch_warning_msg))
        elif pytorch_msg_type == "warning": root.after(200, lambda: messagebox.showwarning("System Configuration Warning", pytorch_warning_msg))
    try: root.mainloop()
    except Exception as e_gui: print(f"\n--- GUI Error ---"); traceback.print_exc(); messagebox.showerror("Application Error", f"A critical error occurred in the application's main loop:\n\n{e_gui}")