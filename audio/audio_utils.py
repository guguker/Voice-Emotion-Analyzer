import numpy as np
import soundfile as sf
import librosa
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    @staticmethod
    def load_audio(file_path):
        """Загрузка аудиофайла и преобразование в нужный формат"""
        try:
            # Загружаем как моно
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            logger.debug(f"Аудиофайл загружен: {file_path}, длительность: {len(audio)/sr:.1f} сек")
            return audio, sr
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудиофайла {file_path}: {e}")
            raise

    @staticmethod
    def process_audio(audio_data, sr):
        """Предобработка аудио для модели"""
        try:
            # Убеждаемся, что аудио одноканальное
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                logger.debug("Преобразование в моно")
            
            # Нормализация
            audio_data = audio_data / np.max(np.abs(audio_data))
            logger.debug("Аудио нормализовано")
            
            return audio_data
        except Exception as e:
            logger.error(f"Ошибка при обработке аудио: {e}")
            raise

    @staticmethod
    def save_audio(audio_data, sr, file_path):
        """Сохранение аудио в файл"""
        try:
            # Убеждаемся, что аудио одноканальное
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                logger.debug("Преобразование в моно перед сохранением")
                
            sf.write(file_path, audio_data, sr)
            logger.debug(f"Аудио сохранено в {file_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении аудио в {file_path}: {e}")
            raise