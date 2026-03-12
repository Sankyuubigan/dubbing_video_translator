import os
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import threading
import traceback
from utils import config_manager
import video_processor
import transcriber
import translator
import voice_cloner

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("No-Torch Video Dubbing (ONNX)")
        self.root.geometry("600x500")
        
        self.work_dir = config_manager.get_work_dir_from_config()
        self.models_dir = os.path.join(self.work_dir, "models_onnx")
        
        self.setup_ui()
        
        # Check models on startup
        threading.Thread(target=self.check_models, daemon=True).start()

    def setup_ui(self):
        p = ttk.Frame(self.root, padding=10)
        p.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(p, text="Video File:").pack(anchor=tk.W)
        self.vid_var = tk.StringVar()
        hbox = ttk.Frame(p)
        hbox.pack(fill=tk.X)
        ttk.Entry(hbox, textvariable=self.vid_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(hbox, text="...", command=lambda: self.vid_var.set(filedialog.askopenfilename())).pack(side=tk.LEFT)
        
        ttk.Button(p, text="Start Processing", command=self.on_start).pack(pady=10)
        
        self.log = scrolledtext.ScrolledText(p, height=15)
        self.log.pack(fill=tk.BOTH, expand=True)
        
        # --- Context Menu for Logs ---
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self.copy_selection)
        self.context_menu.add_command(label="Copy All", command=self.copy_all)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Clear", command=self.clear_log)
        
        self.log.bind("<Button-3>", self.show_context_menu) # Right click

    def show_context_menu(self, event):
        self.context_menu.tk_popup(event.x_root, event.y_root)

    def copy_selection(self):
        try:
            sel = self.log.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(sel)
        except tk.TclError:
            pass # No selection

    def copy_all(self):
        text = self.log.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def clear_log(self):
        self.log.delete("1.0", tk.END)

    def log_msg(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def check_models(self):
        self.log_msg("Checking/Downloading ONNX models...")
        try:
            config_manager.check_and_download_models(self.work_dir)
            self.log_msg("Models ready.")
        except Exception as e:
            self.log_msg(f"Error checking models: {e}")

    def on_start(self):
        threading.Thread(target=self.process, daemon=True).start()

    def process(self):
        vid = self.vid_var.get()
        if not os.path.exists(vid):
            self.log_msg("Video not found")
            return
            
        temp_dir = tempfile.mkdtemp()
        try:
            self.log_msg("1. Extracting audio...")
            wav_path = os.path.join(temp_dir, "source_16k.wav")
            video_processor.extract_audio(vid, wav_path, 16000) # For STT
            
            self.log_msg("2. Transcribing (Sherpa-ONNX)...")
            segments = transcriber.transcribe_with_sherpa(wav_path, self.models_dir)
            self.log_msg(f"   Found {len(segments)} segments.")
            
            self.log_msg("3. Diarizing (Speaker ID)...")
            segments, _ = transcriber.diarize_segments(wav_path, segments, self.models_dir)
            
            self.log_msg("4. Translating (NLLB CTranslate2)...")
            segments, _ = translator.translate_segments(segments, self.models_dir)
            
            self.log_msg("5. Synthesizing (VITS ONNX)...")
            dub_wav = voice_cloner.synthesize_segments(segments, self.models_dir, temp_dir)
            
            self.log_msg("6. Mixing...")
            out_vid = os.path.join(self.work_dir, "result_dubbed.mp4")
            
            # Original audio (background)
            orig_bg = os.path.join(temp_dir, "bg.wav")
            video_processor.extract_audio(vid, orig_bg, 44100)
            
            video_processor.mix_and_replace_audio(vid, orig_bg, dub_wav, out_vid)
            
            self.log_msg(f"Done! Saved to {out_vid}")
            
        except Exception as e:
            self.log_msg(f"Error: {e}")
            traceback.print_exc()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()