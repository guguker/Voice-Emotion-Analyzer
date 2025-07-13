import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageSequence
import threading
import logging
import os
from tkinter import filedialog
from audio.recorder import AudioRecorder
from audio.audio_utils import AudioProcessor
from model.predict import EmotionPredictor
from analysis.timeline import EmotionTimeline

logger = logging.getLogger(__name__)

class VoiceAnalyzeGUI:
    # Словарь переводов эмоций
    EMOTION_TRANSLATIONS = {
        'anger': 'Злость',
        'happy': 'Радость',
        'sad': 'Грусть',
        'neutral': 'Нейтральность',
        'surprise': 'Удивление',
        'fear': 'Страх',
        'disgust': 'Отвращение'
    }
    
    def __init__(self):
        # Базовые компоненты
        self.recorder = AudioRecorder()
        self.processor = AudioProcessor()
        self.predictor = EmotionPredictor()
        self.timeline = EmotionTimeline()
        
        # Состояние приложения
        self.is_recording = False
        self.is_animating = False
        self.current_frame = 0
        self.gif_frames = []
        self.selected_language = "English"
        self.animation_running = True  # Флаг для контроля анимации
        self.animation_widgets = {'wave_label': None, 'animation_label': None}  # Отслеживание виджетов
        
        # Настройка окна
        self.window = ctk.CTk()
        self.window.title("Voice Emotion Analyzer")
        self.window.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Загрузка ресурсов
        self.load_resources()
        
        # Инициализация matplotlib
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.grid(True, color='gray', alpha=0.3)
        
        # Создание интерфейса (включая начальный экран)
        self.setup_ui()
    
    def load_resources(self):
        """Загрузка изображений и анимаций"""
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        
        # Загрузка логотипа
        logo_path = os.path.join(assets_dir, "image.png")
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            self.logo_image = ctk.CTkImage(light_image=logo, dark_image=logo, size=(800, 800))
            
        # Загрузка анимации
        animation_path = os.path.join(assets_dir, "wave.gif")
        if os.path.exists(animation_path):
            gif = Image.open(animation_path)
            frames = []
            for frame in ImageSequence.Iterator(gif):
                frame = frame.convert('RGBA')
                ctk_frame = ctk.CTkImage(light_image=frame, dark_image=frame, size=(600, 400))
                frames.append(ctk_frame)
            self.gif_frames = frames
    
    def setup_ui(self):
        """Настройка базового интерфейса"""
        # Главный контейнер
        self.main_container = ctk.CTkFrame(self.window, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Верхняя панель (всегда видима)
        self.top_panel = ctk.CTkFrame(self.main_container, fg_color="transparent", height=50)
        self.top_panel.pack(fill="x", pady=(0, 20))
        self.top_panel.pack_propagate(False)
        
        # Кнопки и меню
        self.record_button = ctk.CTkButton(
            self.top_panel,
            text="Начать запись",
            command=self.toggle_recording,
            width=150,
            height=35
        )
        self.record_button.pack(side="left", padx=10)
        
        self.file_button = ctk.CTkButton(
            self.top_panel,
            text="Загрузить файл",
            command=self.load_audio_file,
            width=150,
            height=35
        )
        self.file_button.pack(side="left", padx=10)
        
        languages = ["English", "Русский"]
        self.language_menu = ctk.CTkOptionMenu(
            self.top_panel,
            values=languages,
            command=self.change_language,
            width=150,
            height=35
        )
        self.language_menu.pack(side="right", padx=10)
        self.language_menu.set(self.selected_language)
        
        # Контейнер для контента (меняется в зависимости от состояния)
        self.content_container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_container.pack(fill="both", expand=True)
        
        # Инициализация matplotlib
        self.setup_plot()
        
        # Показываем приветственный экран
        self.show_welcome_screen()
        
    def setup_plot(self):
        """Инициализация графика"""
        plt.style.use('dark_background')
        
        # Создаем фигуру и оси
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        
        # Настраиваем стиль
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.grid(True, color='gray', alpha=0.3)
        
        # Добавляем отступы
        self.fig.tight_layout(pad=1.5)
        
        # Canvas будет создаваться в show_results_screen
    
    def show_wave_animation(self, label_name='wave_label', interval=50):
        """Показать анимацию звуковой волны
        
        Args:
            label_name (str): Имя атрибута с меткой для анимации ('wave_label' или 'animation_label')
            interval (int): Интервал обновления в миллисекундах
        """
        # Проверяем, что анимация должна продолжаться
        if not self.is_animating or not self.animation_running:
            return
            
        # Проверяем наличие кадров
        if not self.gif_frames:
            return
            
        # Получаем виджет из словаря
        label = self.animation_widgets.get(label_name)
        
        # Проверяем, что виджет существует и не уничтожен
        if label is None or not label.winfo_exists():
            return
            
        try:
            # Обновляем кадр
            label.configure(image=self.gif_frames[self.current_frame])
            self.current_frame = (self.current_frame + 1) % len(self.gif_frames)
            
            # Планируем следующее обновление
            if self.animation_running and label.winfo_exists():
                self.window.after(interval, lambda: self.show_wave_animation(label_name, interval))
                
        except Exception as e:
            logger.error(f"Ошибка при обновлении анимации: {e}")
            self.stop_animation()
            

            
    def set_buttons_state(self, state="normal"):
        """Управление состоянием кнопок интерфейса"""
        self.record_button.configure(state=state)
        self.file_button.configure(state=state)
        self.language_menu.configure(state=state)
    
    def toggle_recording(self):
        """Управление записью"""
        if not self.is_recording:
            # Начало записи
            self.is_recording = True
            self.record_button.configure(text="Остановить запись")
            
            # Блокируем все кнопки кроме кнопки записи
            self.file_button.configure(state="disabled")
            self.language_menu.configure(state="disabled")
            
            # Показываем экран записи
            self.show_recording_screen()
            
            # Запускаем запись в отдельном потоке
            thread = threading.Thread(target=self.record_audio)
            thread.daemon = True
            thread.start()
        else:
            # Остановка записи
            self.is_recording = False
            self.record_button.configure(text="Начать запись")
            
            # Блокируем все кнопки на время анализа
            self.set_buttons_state("disabled")
            
            # Показываем экран анализа
            self.show_analysis_screen()
            
            # Останавливаем запись и получаем данные
            audio_data, sample_rate = self.recorder.stop_recording()
            
            if audio_data is not None:
                # Запускаем обработку в отдельном потоке
                thread = threading.Thread(
                    target=self.process_audio,
                    args=(audio_data, sample_rate)
                )
                thread.daemon = True
                thread.start()
            else:
                # Возвращаемся к начальному экрану
                self.show_welcome_screen()
                self.set_buttons_state("normal")
    
    def record_audio(self):
        """Запись аудио"""
        try:
            self.recorder.start_recording()
        except Exception as e:
            logger.error(f"Ошибка при записи: {e}")
            self.window.after(0, lambda: self.show_error(f"Ошибка при записи:\n{str(e)}"))
            self.is_recording = False
            self.record_button.configure(text="Начать запись")
            self.file_button.configure(state="normal")
            self.show_welcome_screen()
            
    def load_audio_file(self):
        """Загрузка аудио файла"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3")]
        )
        if file_path:
            # Блокируем все кнопки на время анализа
            self.set_buttons_state("disabled")
            
            # Показываем экран анализа с анимацией
            self.clear_content()
            
            # Контейнер для центрирования
            center_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
            center_frame.pack(expand=True)
            
            # Анимация
            animation_label = ctk.CTkLabel(
                center_frame,
                text="",
                image=self.gif_frames[0]
            )
            animation_label.pack(expand=True, pady=(0, 20))
            
            # Сохраняем ссылку на виджет
            self.animation_widgets['animation_label'] = animation_label
            
            # Текст анализа
            analysis_text = ctk.CTkLabel(
                center_frame,
                text="Пожалуйста, подождите\nАудио анализируется...",
                font=("Arial", 30)
            )
            analysis_text.pack(pady=(0, 40))
            
            # Запускаем анимацию
            self.animation_running = True
            self.is_animating = True
            self.show_wave_animation('animation_label', 50)
            
            # Запускаем обработку в отдельном потоке
            thread = threading.Thread(
                target=self.process_audio_file,
                args=(file_path,)
            )
            thread.daemon = True
            thread.start()
    
    def process_audio_file(self, file_path):
        """Обработка загруженного файла"""
        try:
            audio_data, sample_rate = self.processor.load_audio(file_path)
            self.process_audio(audio_data, sample_rate)
        except Exception as e:
            print(f"Ошибка при обработке файла: {e}")
            
    def process_audio(self, audio_data, sample_rate):
        """Обработка аудио и отображение результатов"""
        try:
            # Получаем временную шкалу эмоций
            timeline_data = self.predictor.get_emotion_timeline(audio_data, sample_rate)
            
            # Показываем результаты в главном потоке
            self.window.after(0, lambda: self.show_results_screen(timeline_data))
                
        except Exception as e:
            logger.error(f"Ошибка при анализе: {e}")
            self.window.after(0, lambda: self.show_welcome_screen())
    
    def format_results(self, timeline_data):
        """Форматирование результатов анализа временной шкалы"""
        if not timeline_data:
            return "Нет данных для анализа"
            
        translations = self.EMOTION_TRANSLATIONS
        
        # Получаем результаты анализа от timeline
        _, emotion_averages = self.timeline.plot_timeline(timeline_data, self.ax)
        
        # Сортируем эмоции по убыванию средних значений
        sorted_emotions = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
        
        # Формируем текст результатов
        text = "Результаты анализа:\n\n"
        text += "Средние значения эмоций:\n"
        
        # Добавляем все эмоции с их значениями
        for emotion, avg in sorted_emotions:
            translated = translations.get(emotion, emotion)
            text += f"{translated}: {avg:.1%}\n"
        
        # Добавляем интерпретацию
        text += "\nИнтерпретация:\n"
        
        # Находим преобладающие эмоции (с ненулевыми значениями)
        significant_emotions = [(emotion, avg) for emotion, avg in sorted_emotions if avg > 0.01]
        
        if significant_emotions:
            main_emotion, main_score = significant_emotions[0]
            main_emotion_ru = translations.get(main_emotion, main_emotion)
            text += f"Преобладающая эмоция: {main_emotion_ru} ({main_score:.1%})\n"
            
            if len(significant_emotions) > 1:
                secondary_emotions = [translations.get(emotion, emotion) 
                                   for emotion, _ in significant_emotions[1:3]]
                text += f"Также присутствуют: {', '.join(secondary_emotions)}"
        else:
            text += "В записи не обнаружено явно выраженных эмоций"
            
        return text
    
    def format_analysis_results(self, predictions):
        """Форматирование результатов анализа"""
        if not predictions:
            return "Нет данных для анализа"
            
        emotion_translations = self.EMOTION_TRANSLATIONS
        
        # Сортируем предсказания по уверенности
        sorted_predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        # Формируем текст с процентами для каждой эмоции
        percentages = []
        for pred in sorted_predictions:
            percentage = pred['score'] * 100
            emotion = pred['label'].lower()
            translated_emotion = emotion_translations.get(emotion, emotion)
            percentages.append(f"{translated_emotion}: {percentage:.1f}%")
            
        # Делаем текстовую интерпретацию
        main_emotion = sorted_predictions[0]['label'].lower()
        secondary_emotion = sorted_predictions[1]['label'].lower()
        confidence = sorted_predictions[0]['score'] * 100
        
        main_emotion_ru = emotion_translations.get(main_emotion, main_emotion)
        secondary_emotion_ru = emotion_translations.get(secondary_emotion, secondary_emotion)
        
        interpretation = f"Голос преимущественно выражает {main_emotion_ru} "
        if confidence < 50:
            interpretation += f"с некоторой неуверенностью. "
        else:
            interpretation += f"({confidence:.1f}%). "
            
        if sorted_predictions[1]['score'] > 0.2:  # Если вторая эмоция значима
            interpretation += f"Также присутствуют нотки {secondary_emotion_ru}."
            
        return "\n".join([
            "Анализ эмоций в голосе:",
            f"Доминирующая эмоция: {main_emotion} ({confidence:.1f}%)",
            "",
            "Все обнаруженные эмоции:",
            "\n".join(percentages),
            "",
            "Интерпретация:",
            interpretation
        ])
    
    def show_welcome_screen(self):
        """Отображение приветственного экрана"""
        self.clear_content()
        
        # Контейнер для приветственного экрана
        welcome_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        welcome_frame.pack(fill="both", expand=True)
        
        # Левая часть с логотипом (75%)
        logo_frame = ctk.CTkFrame(welcome_frame, fg_color="transparent")
        logo_frame.pack(side="left", fill="both", expand=True)
        
        logo_label = ctk.CTkLabel(logo_frame, text="", image=self.logo_image)
        logo_label.pack(expand=True)
        
        # Правая часть с текстом (25%)
        text_frame = ctk.CTkFrame(welcome_frame, fg_color="transparent", width=600)
        text_frame.pack(side="right", fill="y", padx=(20, 0))
        text_frame.pack_propagate(False)
        
        welcome_text = ctk.CTkLabel(
            text_frame,
            text="👋 Привет! Я — Voice Analyzer. \n " \
            "Загрузи 🎙️ голосовую запись или просто поговори со мной — и я помогу определить, какие эмоции звучат в твоём голосе. \n " \
            " Я понимаю русский и английский, работаю как с файлами .wav и .mp3, так и с живым звуком. \n" \
            "🌍 Нажми на кнопку в правом верхнем углу, чтобы сменить язык. \n " \
            "Готов? Давай послушаем 👾",
            font=("Arial", 24),
            wraplength=400
        )
        welcome_text.pack(expand=True)
        
    def show_recording_screen(self):
        """Отображение экрана записи"""
        self.clear_content()
        
        # Контейнер для записи
        recording_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        recording_frame.pack(fill="both", expand=True)
        
        # Контейнер для центрирования
        center_frame = ctk.CTkFrame(recording_frame, fg_color="transparent")
        center_frame.pack(expand=True)
        
        # Анимация волны (большая, в центре)
        wave_label = ctk.CTkLabel(
            center_frame,
            text="",
            image=self.gif_frames[0]
        )
        wave_label.pack(expand=True, pady=(0, 20))
        
        # Сохраняем ссылку на виджет
        self.animation_widgets['wave_label'] = wave_label
        
        # Текст записи (внизу)
        self.recording_text = ctk.CTkLabel(
            center_frame,
            text="Производится запись звука...\nПродолжайте говорить",
            font=("Arial", 30)
        )
        self.recording_text.pack(pady=(0, 40))
        
        # Запуск анимации
        self.animation_running = True
        self.is_animating = True
        self.show_wave_animation('wave_label', 50)
        
    def show_analysis_screen(self):
        """Отображение экрана анализа"""
        self.clear_content()
        
        # Контейнер для центрирования
        center_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        center_frame.pack(expand=True)
        
        # Анимация волны (большая, в центре)
        animation_label = ctk.CTkLabel(
            center_frame,
            text="",
            image=self.gif_frames[0]
        )
        animation_label.pack(expand=True, pady=(0, 20))
        
        # Сохраняем ссылку на виджет
        self.animation_widgets['animation_label'] = animation_label
        
        # Текст анализа (внизу)
        self.analysis_text = ctk.CTkLabel(
            center_frame,
            text="Пожалуйста, подождите\nЗапись анализируется...",
            font=("Arial", 30)
        )
        self.analysis_text.pack(pady=(0, 40))
        
        # Запуск анимации
        self.animation_running = True
        self.is_animating = True
        self.show_wave_animation('animation_label', 50)
        
    def show_results_screen(self, timeline_data):
        """Отображение экрана результатов"""
        self.clear_content()
        
        try:
            # Левая часть с графиком (75%)
            graph_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
            graph_frame.pack(side="left", fill="both", expand=True, padx=(0, 20))
            
            # Очищаем график
            self.ax.clear()
            
            # Отрисовка графика (без легенды)
            figure, _ = self.timeline.plot_timeline(timeline_data, self.ax)
            
            # Создаем новый canvas для графика
            canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Правая часть с легендой и текстом (25%)
            text_frame = ctk.CTkFrame(self.content_container, fg_color="transparent", width=500)
            text_frame.pack(side="right", fill="y")
            text_frame.pack_propagate(False)
            
            # --- Легенда ---
            try:
                legend_buf = self.timeline.get_legend_figure()
                legend_img = Image.open(legend_buf)
                legend_img_resized = legend_img.copy()
                legend_img_resized = legend_img_resized.resize((350, 400), Image.LANCZOS)
                legend_ctk = ctk.CTkImage(light_image=legend_img_resized, dark_image=legend_img_resized, size=(350, 400))
                legend_label = ctk.CTkLabel(text_frame, image=legend_ctk, text="")
                legend_label.pack(pady=(10, 5), anchor="n")
            finally:
                # Закрываем ресурсы
                if 'legend_buf' in locals():
                    legend_buf.close()
                if 'legend_img' in locals():
                    legend_img.close()
                if 'legend_img_resized' in locals() and legend_img_resized != legend_img:
                    legend_img_resized.close()
            
            # --- Текстовый анализ ---
            text_widget = ctk.CTkTextbox(
                text_frame,
                wrap="word",
                font=("Arial", 12),
                height=333
            )
            text_widget.pack(fill="x", expand=False, padx=10, pady=(0, 10), side="bottom")
            results_text = self.format_results(timeline_data)
            text_widget.insert("1.0", results_text)
            text_widget.configure(state="disabled")
            
            # Разблокируем все кнопки после отображения результатов
            self.set_buttons_state("normal")
            
        except Exception as e:
            logger.error(f"Ошибка при отображении результатов: {str(e)}")
            self.show_error("Произошла ошибка при отображении результатов")
            # Разблокируем кнопки в случае ошибки
            self.set_buttons_state("normal")
    
    def clear_content(self):
        """Очистка контейнера с контентом"""
        # Останавливаем анимацию перед уничтожением виджетов
        self.stop_animation()
        
        # Очищаем все виджеты
        for widget in self.content_container.winfo_children():
            try:
                widget.destroy()
            except Exception as e:
                logger.error(f"Ошибка при уничтожении виджета: {e}")
                
        # Сбрасываем все ссылки на виджеты
        self.animation_widgets = {'wave_label': None, 'animation_label': None}
            
    def change_language(self, language):
        """Смена языка"""
        self.selected_language = language
        self.predictor.update_model_for_language(language)
        
    def run(self):
        """Запуск приложения"""
        self.window.mainloop()
    
    def show_error(self, message):
        """Показать сообщение об ошибке"""
        # Очищаем контент
        self.clear_content()
        
        # Показываем текст ошибки
        error_label = ctk.CTkLabel(
            self.content_container,
            text=message,
            wraplength=250,
            text_color="red"
        )
        error_label.pack(expand=True)
        
        # Возвращаемся к начальному экрану через 3 секунды
        self.window.after(3000, self.show_welcome_screen)
    
    def stop_animation(self):
        """Безопасная остановка анимации"""
        self.is_animating = False
        self.animation_running = False
        # Очищаем ссылки на виджеты анимации
        self.animation_widgets = {'wave_label': None, 'animation_label': None}

def launch_gui():
    app = VoiceAnalyzeGUI()
    app.run()
