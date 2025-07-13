from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
import torch
import os
import shutil
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class EmotionModelLoader:
    def __init__(self, models_dir=None):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_language = "English"
        logger.info(f"Device set to use {self.device}")
        
        # Устанавливаем каталог для моделей
        self.models_dir = models_dir if models_dir else os.path.join(os.getcwd(), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Словарь моделей для разных языков
        self.language_models = {
            "English": "superb/wav2vec2-base-superb-er",
            "Русский": "superb/wav2vec2-base-superb-er"
        }
        
        # Создаем подпапки для каждого языка
        for lang in self.language_models:
            os.makedirs(os.path.join(self.models_dir, lang), exist_ok=True)
    
    def get_model_path(self, language):
        """Получить локальный путь для модели конкретного языка"""
        return os.path.join(self.models_dir, language)
    
    def is_model_downloaded(self, language):
        """Проверить, скачана ли модель для данного языка"""
        model_path = self.get_model_path(language)
        return os.path.exists(os.path.join(model_path, CONFIG_NAME))
    
    def download_model(self, language):
        """Скачать модель для конкретного языка"""
        try:
            model_name = self.language_models[language]
            model_path = self.get_model_path(language)
            
            logger.info(f"Загрузка модели {model_name} для языка {language}...")
            
            # Загружаем модель и токенизатор
            model = AutoModelForAudioClassification.from_pretrained(model_name)
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            
            # Сохраняем локально
            model.save_pretrained(model_path)
            feature_extractor.save_pretrained(model_path)
            
            logger.info(f"Модель для языка {language} успешно загружена в {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели для языка {language}: {e}")
            return False
    
    def load_model(self, language=None):
        """Загрузка модели для определения эмоций"""
        try:
            if language:
                self.current_language = language
            
            # Проверяем, скачана ли модель
            if not self.is_model_downloaded(self.current_language):
                if not self.download_model(self.current_language):
                    raise Exception(f"Не удалось загрузить модель для языка {self.current_language}")
            
            # Загружаем модель из локального пути
            model_path = self.get_model_path(self.current_language)
            self.model = pipeline(
                "audio-classification",
                model=model_path,
                device=self.device
            )
            logger.info(f"Модель успешно загружена из {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            return False
    
    def get_model(self, language=None):
        """Получить загруженную модель"""
        if language and language != self.current_language:
            self.load_model(language)
        elif self.model is None:
            self.load_model()
        return self.model
    
    def get_cache_info(self):
        """Получить информацию о локальных моделях"""
        total_size = 0
        model_count = 0
        
        for lang in self.language_models:
            model_path = self.get_model_path(lang)
            if os.path.exists(model_path):
                size = sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, _, files in os.walk(model_path)
                    for file in files
                )
                total_size += size
                model_count += 1
        
        return {
            'cache_dir': self.models_dir,
            'size_gb': total_size / (1024 * 1024 * 1024),
            'model_count': model_count
        }