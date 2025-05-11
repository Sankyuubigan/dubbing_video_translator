import os
import tempfile
import shutil
from deep_translator import GoogleTranslator
import pyttsx3
# import gradio as gr # УДАЛЕНО
import speech_recognition as sr
import moviepy.editor as mp
import pysubs2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# Создание директории для хранения видео на рабочем столе
OUTPUT_VIDEO_DIR = os.path.expanduser("~/Desktop/VideoTranslator")
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# TEMP_DIR_GRADIO = "C:\\Users\\De1pl\\AppData\\Local\\Temp\\gradio" # УДАЛЕНО

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


def split_text(text, max_length=4500): # Уменьшил немного для GoogleTranslator, он иногда капризничает
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def extract_audio_text(videoclip, temp_audio_file_path):
    try:
        videoclip.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", logger=None) # logger=None чтобы не было вывода в консоль
        r = sr.Recognizer()
        with sr.AudioFile(temp_audio_file_path) as source:
            audio = r.record(source)
        # Попробуем Google, если Sphinx не настроен или вызывает проблемы.
        # Для recognize_google нужен интернет.
        try:
            text = r.recognize_google(audio, language="en-US")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio, trying Sphinx")
            try:
                text = r.recognize_sphinx(audio) # Убедитесь, что Sphinx настроен!
            except sr.UnknownValueError:
                raise ValueError("Sphinx could not understand audio")
            except sr.RequestError as e:
                raise RuntimeError(f"Sphinx error; {e}")
        except sr.RequestError as e:
            raise RuntimeError(f"Could not request results from Google Speech Recognition service; {e}")
        return text
    except Exception as e: # Более общее исключение для отладки
        raise RuntimeError(f"Ошибка при извлечении/распознавании текста: {e}")


def translate_text(text, max_chars=4500):
    try:
        translator_obj = GoogleTranslator(source="en", target="ru")
        text_parts = split_text(text, max_chars)
        translated_parts = []
        for part in text_parts:
            if part.strip(): # Переводим только непустые части
                 translated_parts.append(translator_obj.translate(part))
        return ' '.join(filter(None, translated_parts)) # filter(None,) уберет None если переводчик вернул его
    except Exception as e:
        raise RuntimeError(f"Ошибка при переводе текста: {e}")


def synthesize_speech(text, temp_speech_file_path):
    try:
        engine_obj = pyttsx3.init()
        voices = engine_obj.getProperty('voices')
        # Попытка найти русский голос. На разных системах ID могут отличаться.
        ru_voice_id = None
        for voice in voices:
            if "russian" in voice.name.lower() or "ru-ru" in voice.id.lower():
                 ru_voice_id = voice.id
                 break
        if ru_voice_id:
            engine_obj.setProperty('voice', ru_voice_id)
        else:
            print("Русский голос не найден, будет использован голос по умолчанию.")
            # Вы можете установить 'ru' если pyttsx3 его поддерживает на вашей системе
            # engine_obj.setProperty('voice', 'ru') # Это может не сработать если нет русского пакета

        engine_obj.save_to_file(text, temp_speech_file_path)
        engine_obj.runAndWait()
        return temp_speech_file_path
    except Exception as e:
        raise RuntimeError(f"Ошибка при синтезе речи: {e}")


def create_text_clip(text, start_time, end_time, video_size, font_size=28, color='white', bg_color='rgba(0,0,0,0.5)',
                     font='Arial', position=('center', 'bottom')): # Изменены bg_color и position
    """
    Создает текстовый клип для видео с помощью библиотеки MoviePy.
    video_size: (width, height) видео для корректного позиционирования.
    """
    # Убедимся, что есть какой-то текст, иначе TextClip может выдать ошибку
    if not text.strip():
        return None

    txt_clip = mp.TextClip(text, fontsize=font_size, color=color, bg_color=bg_color, font=font, size=(video_size[0]*0.9, None), method='caption')
    
    # Рассчитываем позицию
    # 'center' для x, 'bottom' для y с небольшим отступом
    pos_x = 'center'
    if position[1] == 'bottom':
        pos_y = video_size[1] - txt_clip.size[1] - video_size[1]*0.05 # 5% отступ снизу
    elif position[1] == 'top':
        pos_y = video_size[1]*0.05 # 5% отступ сверху
    elif position[1] == 'center':
        pos_y = 'center'
    else: # числовое значение
        pos_y = position[1]

    return txt_clip.set_position((pos_x, pos_y)) \
                   .set_duration(end_time - start_time) \
                   .set_start(start_time)


def add_subtitles_to_video_with_pysubs2(video_path, subtitles_path, output_video_path):
    videoclip = None
    composite_clip = None
    try:
        videoclip = mp.VideoFileClip(video_path)
        subs = pysubs2.load(subtitles_path, encoding='utf-8') # Добавил encoding

        text_clips = []
        for line in subs:
            start_time = line.start / 1000
            end_time = line.end / 1000
            # Уберем теги SSA/ASS из текста для простоты, MoviePy их не поймет
            plain_text = pysubs2.ssaevent.EVENTS_TAG_MATCHER.sub('', line.text)
            
            text_clip = create_text_clip(plain_text, start_time, end_time, videoclip.size)
            if text_clip:
                text_clips.append(text_clip)

        if not text_clips: # Если субтитров нет или все пустые
            print("Нет субтитров для добавления, копируем оригинальное видео.")
            shutil.copy(video_path, output_video_path)
            return

        composite_clip = mp.CompositeVideoClip([videoclip] + text_clips, size=videoclip.size)
        composite_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', logger=None)

    except Exception as e:
        raise RuntimeError(f"Ошибка при добавлении субтитров: {e}")
    finally:
        if videoclip:
            videoclip.close()
        if composite_clip and hasattr(composite_clip, 'close'): # Не все клипы имеют close
             # CompositeVideoClip сам по себе не имеет метода close, закрываются его компоненты
             # MoviePy должен сам управлять закрытием временных файлов при write_videofile
             pass


def translate_video_logic(file_video, temp_dir):
    videoclip = None
    audioclip = None
    try:
        videoclip = mp.VideoFileClip(file_video)
        temp_audio_file = os.path.join(temp_dir, "extracted_audio.wav")
        extracted_text = extract_audio_text(videoclip, temp_audio_file)
        if not extracted_text.strip():
            raise ValueError("Не удалось извлечь текст из аудио.")

        translated_text = translate_text(extracted_text)
        if not translated_text.strip():
            raise ValueError("Переведенный текст пуст.")

        temp_speech_path = os.path.join(temp_dir, "translated_speech.wav")
        synthesize_speech(translated_text, temp_speech_path)

        audioclip = mp.AudioFileClip(temp_speech_path)
        final_video = videoclip.set_audio(audioclip)

        video_name = os.path.splitext(os.path.basename(file_video))[0]
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_translated.mp4")
        final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac', logger=None)
        
        return output_video_path
    finally:
        if audioclip:
            audioclip.close()
        if videoclip: # videoclip может быть final_video если set_audio возвращает новый объект
            videoclip.close()
        # if final_video and final_video is not videoclip: # если final_video это другой объект
        #     final_video.close() # MoviePy должен сам закрывать ресурсы при write_videofile

def add_subtitles_to_video_logic(file_video, file_srt, temp_dir): # temp_dir пока не используется здесь, но для консистентности
    try:
        video_name = os.path.splitext(os.path.basename(file_video))[0]
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}_with_subs.mp4")
        add_subtitles_to_video_with_pysubs2(file_video, file_srt, output_video_path)
        return output_video_path
    except Exception as e: # Более общее
        raise RuntimeError(f"Ошибка при добавлении субтитров (логика): {e}")


def translate_video_with_subtitles_logic(file_video, file_srt, temp_dir):
    # Сначала переводим видео
    translated_video_path = translate_video_logic(file_video, temp_dir) # Передаем temp_dir
    # Затем добавляем субтитры к переведенному видео
    # Для субтитров может понадобиться своя временная папка, если add_subtitles_to_video_logic ее использует
    # но сейчас он не использует temp_dir
    final_output_path = add_subtitles_to_video_logic(translated_video_path, file_srt, temp_dir)
    return final_output_path


# --- GUI Part ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor")
        self.root.geometry("600x450") # Увеличил размер окна

        self.video_path = tk.StringVar()
        self.srt_path = tk.StringVar()
        self.operation = tk.StringVar(value="Translate") # Значение по умолчанию

        # Стили для ttk виджетов
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", padding=6)
        style.configure("TEntry", padding=6)
        style.configure("TRadiobutton", padding=6)

        # Frame for video input
        video_frame = ttk.LabelFrame(root, text="Video File", padding=(10, 5))
        video_frame.pack(padx=10, pady=5, fill="x")
        ttk.Entry(video_frame, textvariable=self.video_path, width=60, state="readonly").pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(video_frame, text="Browse", command=self.browse_video).pack(side=tk.LEFT)

        # Frame for SRT input
        srt_frame = ttk.LabelFrame(root, text="SRT File (Optional)", padding=(10, 5))
        srt_frame.pack(padx=10, pady=5, fill="x")
        ttk.Entry(srt_frame, textvariable=self.srt_path, width=60, state="readonly").pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
        ttk.Button(srt_frame, text="Browse", command=self.browse_srt).pack(side=tk.LEFT)

        # Frame for operations
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

        # Status area (используем Text для многострочного вывода)
        status_frame = ttk.LabelFrame(root, text="Status", padding=(10,5))
        status_frame.pack(padx=10, pady=5, fill="both", expand=True)
        self.status_text = tk.Text(status_frame, height=8, wrap=tk.WORD, state="disabled")
        self.status_text.pack(fill="both", expand=True)
        
        # Добавим скроллбар к текстовому полю статуса
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
        self.status_text.see(tk.END) # Автопрокрутка вниз
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
        self._update_status("Processing started...", append=False) # Очищаем предыдущий статус

        # Запускаем обработку в отдельном потоке
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
            # Выведем полный трейсбек в консоль для детальной отладки
            import traceback
            print("--- ERROR TRACEBACK ---")
            traceback.print_exc()
            print("-----------------------")
        finally:
            cleanup_temp_dir(current_temp_dir)
            # Важно обновить GUI из главного потока
            self.root.after(0, self.enable_process_button)

    def enable_process_button(self):
        self.process_button.config(state="normal")


if __name__ == "__main__":
    import traceback # Для вывода полного трейсбека в actual_processing
    root = tk.Tk()
    app = App(root)
    root.mainloop()