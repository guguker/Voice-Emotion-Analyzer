import sounddevice as sd
import numpy as np
import threading
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.check_audio_device()
        
    def check_audio_device(self):
        """Проверка наличия устройства записи"""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                raise RuntimeError("Устройство записи не найдено")
            # Выбираем первое доступное устройство ввода
            self.device = input_devices[0]['index']
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации устройства записи: {e}")
        
    def start_recording(self):
        """Начать запись аудио"""
        if self.recording:
            return
            
        self.recording = True
        self.audio_data = []
        
        def record():
            try:
                with sd.InputStream(samplerate=self.sample_rate, 
                                 channels=1, 
                                 dtype=np.float32,
                                 device=self.device,
                                 blocksize=2048) as stream:  # Увеличиваем размер блока
                    while self.recording:
                        try:
                            audio_chunk, _ = stream.read(2048)  # Читаем больший блок
                            if len(audio_chunk) > 0:
                                # Нормализуем данные
                                normalized_chunk = audio_chunk.flatten()
                                if np.max(np.abs(normalized_chunk)) > 0.001:  # Проверяем, есть ли звук
                                    self.audio_data.append(normalized_chunk)
                                    if len(self.audio_data) % 50 == 0:  # Логируем чаще
                                        logger.debug(f"Записано {len(self.audio_data)} блоков аудио")
                            else:
                                logger.warning("Получен пустой блок аудио данных")
                        except Exception as e:
                            logger.error(f"Ошибка при чтении аудио данных: {e}")
                            self.recording = False
                            break
            except Exception as e:
                logger.error(f"Ошибка при инициализации потока записи: {e}")
                self.recording = False
                
        self.record_thread = threading.Thread(target=record)
        self.record_thread.start()
        
    def stop_recording(self):
        """Остановить запись и вернуть записанные данные"""
        if not self.recording:
            return None, self.sample_rate
            
        self.recording = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        
        if not self.audio_data:
            return None, self.sample_rate
            
        try:
            if not self.audio_data:
                logger.warning("Нет записанных данных")
                return None, self.sample_rate
                
            # Объединяем все чанки в один массив
            audio_data = np.concatenate(self.audio_data)
            
            # Проверяем качество записи
            if len(audio_data) < self.sample_rate:  # Меньше секунды
                logger.warning("Запись слишком короткая")
                return None, self.sample_rate
                
            if np.max(np.abs(audio_data)) < 0.001:  # Слишком тихо
                logger.warning("Запись слишком тихая")
                return None, self.sample_rate
                
            # Нормализуем финальные данные
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            logger.info(f"Запись остановлена, получено {len(audio_data)} сэмплов ({len(audio_data)/self.sample_rate:.1f} сек)")
            return audio_data, self.sample_rate
        except Exception as e:
            logger.error(f"Ошибка при обработке записанных данных: {e}")
            return None, self.sample_rate

    def save_recording(self, file_path=None):
        """Сохранить запись в файл"""
        if not file_path:
            file_path = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        try:
            audio_data, sr = self.stop_recording()
            if audio_data is not None:
                import soundfile as sf
                sf.write(file_path, audio_data, sr)
                return file_path
        except Exception as e:
            print(f"Ошибка при сохранении записи: {e}")
        return None

    def record(self, duration=5):
        """Записать аудио и вернуть результат
        
        Args:
            duration (int): Длительность записи в секундах
        """
        try:
            self.start_recording()
            # Записываем указанное количество секунд
            time.sleep(duration)
            return self.stop_recording()
        except Exception as e:
            print(f"Ошибка при записи: {e}")
            return None, self.sample_rate