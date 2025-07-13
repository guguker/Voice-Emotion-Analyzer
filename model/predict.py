import logging
from .model_loader import EmotionModelLoader
import numpy as np

logger = logging.getLogger(__name__)

class EmotionPredictor:
    def __init__(self):
        self.model_loader = EmotionModelLoader()
        self.model = None
        logger.debug("EmotionPredictor initialized")
        
    def initialize(self):
        """Инициализация предиктора"""
        try:
            self.model = self.model_loader.get_model()
            logger.info("Предиктор успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка при инициализации предиктора: {e}")
            raise
        
    def update_model_for_language(self, language):
        """Обновление модели для выбранного языка"""
        try:
            self.model = self.model_loader.get_model(language)
            logger.info(f"Модель обновлена для языка: {language}")
        except Exception as e:
            logger.error(f"Ошибка при обновлении модели для языка {language}: {e}")
            raise
        
    def normalize_emotion_label(self, label):
        """Нормализация метки эмоции"""
        label = label.lower().strip()
        # Словарь для нормализации названий эмоций
        emotion_map = {
            # Метки модели superb/wav2vec2-base-superb-er
            'neu': 'neutral',
            'hap': 'happy',
            'ang': 'anger',
            'sad': 'sad',
            # Альтернативные написания
            'angry': 'anger',
            'joy': 'happy',
            'happiness': 'happy',
            'sadness': 'sad',
            'neutral': 'neutral',
            'surprise': 'surprise',
            'surprised': 'surprise',
            'fear': 'fear',
            'scared': 'fear',
            'disgust': 'disgust',
            'disgusted': 'disgust'
        }
        return emotion_map.get(label, label)

    def predict_emotion(self, audio_data, sample_rate):
        """Предсказание эмоций из аудио"""
        if self.model is None:
            logger.debug("Модель не инициализирована, выполняю инициализацию")
            self.initialize()
            
        try:
            if len(audio_data) == 0:
                raise ValueError("Получены пустые аудио данные")
                
            # Получаем предсказания модели
            predictions = self.model(audio_data)
            logger.debug(f"Сырые предсказания от модели: {predictions}")
            
            # Нормализуем метки эмоций и объединяем одинаковые
            normalized_predictions = {}
            for pred in predictions:
                norm_label = self.normalize_emotion_label(pred['label'])
                logger.debug(f"Нормализация метки: {pred['label']} -> {norm_label}")
                if norm_label not in normalized_predictions or pred['score'] > normalized_predictions[norm_label]['score']:
                    normalized_predictions[norm_label] = {
                        'label': norm_label,
                        'score': pred['score']
                    }
            
            # Сортируем предсказания по уверенности
            sorted_predictions = sorted(normalized_predictions.values(), key=lambda x: x['score'], reverse=True)
            
            logger.debug(f"Нормализованные предсказания: {sorted_predictions}")
            logger.info(f"Успешно определены эмоции: {sorted_predictions[0]['label']} ({sorted_predictions[0]['score']:.2f})")
            return sorted_predictions
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании эмоций: {e}")
            return None

    def get_emotion_timeline(self, audio_data, sample_rate, window_size=2.0, step=0.5):
        """Получение временной шкалы эмоций"""
        try:
            timeline = []
            audio_length = len(audio_data) / sample_rate
            window_samples = int(window_size * sample_rate)
            step_samples = int(step * sample_rate)
            
            logger.debug(f"Анализ аудио длительностью {audio_length:.1f} сек")
            logger.debug(f"Размер окна: {window_size} сек, шаг: {step} сек")
            
            total_steps = (len(audio_data) - window_samples) // step_samples
            processed_steps = 0
            
            for start in range(0, len(audio_data) - window_samples, step_samples):
                end = start + window_samples
                audio_segment = audio_data[start:end]
                
                predictions = self.predict_emotion(audio_segment, sample_rate)
                if predictions:
                    time_point = start / sample_rate
                    timeline.append({
                        'time': time_point,
                        'emotions': predictions
                    })
                
                processed_steps += 1
                if processed_steps % 10 == 0:  # Логируем каждый 10-й шаг
                    logger.debug(f"Прогресс анализа: {processed_steps}/{total_steps}")
                    
            logger.info(f"Временная шкала эмоций создана успешно: {len(timeline)} точек")
            return timeline
            
        except Exception as e:
            logger.error(f"Ошибка при создании временной шкалы: {e}")
            return []