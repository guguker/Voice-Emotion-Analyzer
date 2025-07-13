import sys
import logging
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # Явно задаем backend
matplotlib.rcParams['backend'] = 'TkAgg'
import os

from gui.interface import launch_gui
from model.model_loader import EmotionModelLoader

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,  # Изменили уровень на DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_analyze.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Проверка наличия необходимых зависимостей"""
    try:
        import torch
        import transformers
        import sounddevice
        import numpy
        import matplotlib
        import customtkinter
        logger.info("Все зависимости успешно загружены")
        return True
    except ImportError as e:
        logger.error(f"Ошибка при загрузке зависимостей: {e}")
        print("Пожалуйста, установите необходимые зависимости:")
        print("pip install -r requirements.txt")
        return False

def init_model(models_dir):
    """Инициализация модели при запуске"""
    try:
        model_loader = EmotionModelLoader(models_dir)
        if model_loader.load_model():
            logger.info("Модель успешно загружена")
            cache_info = model_loader.get_cache_info()
            logger.info(f"Модели хранятся в: {cache_info['cache_dir']}")
            logger.info(f"Размер кэша моделей: {cache_info['size_gb']:.2f} GB")
            return True
        else:
            logger.error("Ошибка при загрузке модели")
            return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка при инициализации модели: {e}")
        return False

def main():
    try:
        # Проверка зависимостей
        if not check_dependencies():
            sys.exit(1)

        # Запуск GUI
        launch_gui()

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
