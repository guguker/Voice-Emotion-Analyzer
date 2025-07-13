import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmotionTimeline:
    def __init__(self):
        self.all_emotions = {
            'anger': 'Злость',
            'happy': 'Радость',
            'sad': 'Грусть',
            'neutral': 'Нейтральность',
            'surprise': 'Удивление',
            'fear': 'Страх',
            'disgust': 'Отвращение'
        }
        self.colors = {
            'anger': '#FF4444',     # Яркий красный
            'happy': '#50C878',     # Изумрудный зеленый
            'sad': '#4169E1',       # Синий
            'neutral': '#808080',    # Серый
            'surprise': '#9370DB',   # Средний пурпурный
            'fear': '#FFA500',      # Оранжевый
            'disgust': '#8B4513'    # Коричневый
        }
        logger.debug("EmotionTimeline инициализирован с эмоциями: %s", self.all_emotions)
        
    def plot_timeline(self, timeline_data, ax=None):
        """Создание графика изменения эмоций во времени (без легенды)"""
        try:
            if not timeline_data:
                logger.warning("Получены пустые данные для построения графика")
                return None, {}
                
            logger.debug("Начало построения графика для %d точек данных", len(timeline_data))
            times = [point['time'] for point in timeline_data]
            
            # Создаем словари для хранения данных и средних значений
            emotion_scores = {emotion: [] for emotion in self.all_emotions.keys()}
            emotion_averages = {}
            
            # Собираем данные для каждой эмоции и применяем сглаживание
            for point in timeline_data:
                emotions_dict = {pred['label'].lower(): pred['score'] for pred in point['emotions']}
                logger.debug("Данные точки: %s", emotions_dict)
                for emotion in self.all_emotions.keys():
                    score = emotions_dict.get(emotion, 0.0)
                    # Применяем порог для уменьшения шума
                    score = score if score > 0.05 else 0.0
                    emotion_scores[emotion].append(score)
            
            logger.debug("Собранные данные по эмоциям: %s", {k: len(v) for k, v in emotion_scores.items()})
            
            # Сглаживание данных с помощью скользящего среднего
            window_size = 5  # Размер окна сглаживания
            for emotion in emotion_scores:
                scores = emotion_scores[emotion]
                if any(score > 0 for score in scores):
                    logger.debug("Эмоция %s имеет ненулевые значения", emotion)
                smoothed = []
                for i in range(len(scores)):
                    start = max(0, i - window_size // 2)
                    end = min(len(scores), i + window_size // 2 + 1)
                    smoothed.append(sum(scores[start:end]) / (end - start))
                emotion_scores[emotion] = smoothed
            
            logger.debug("Данные после сглаживания: %s", {k: v[:5] for k, v in emotion_scores.items()})
            
            # Вычисляем средние значения
            for emotion in self.all_emotions.keys():
                if emotion_scores[emotion]:
                    emotion_averages[emotion] = sum(emotion_scores[emotion]) / len(emotion_scores[emotion])
                else:
                    emotion_averages[emotion] = 0.0
            
            logger.debug("Средние значения эмоций: %s", emotion_averages)
            
            # Если оси не переданы, создаем новую фигуру
            if ax is None:
                plt.figure(figsize=(12, 6))
                ax = plt.gca()
            
            # Очищаем текущие линии
            ax.clear()
            
            # Словарь для хранения линий
            self.lines = {}
            
            # Рисуем все эмоции, даже если их значения незначительны
            for emotion in self.all_emotions.keys():
                has_significant_values = any(score > 0.05 for score in emotion_scores[emotion])
                
                line, = ax.plot(times, emotion_scores[emotion],
                              color=self.colors[emotion],
                              linewidth=2 if has_significant_values else 1,
                              alpha=0.8 if has_significant_values else 0.3,
                              linestyle='-' if has_significant_values else '--')
                self.lines[emotion] = line
            
            # Настройка внешнего вида
            ax.set_xlabel('Время (секунды)', color='white')
            ax.set_ylabel('Уверенность', color='white')
            ax.set_title('Временная шкала эмоций', color='white')
            
            ax.grid(True, color='gray', alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
            ax.tick_params(colors='white')
            ax.set_facecolor('#2b2b2b')
            
            logger.debug(f"График создан успешно, {len(times)} точек данных")
            return ax.figure, emotion_averages
            
        except Exception as e:
            logger.error(f"Ошибка при построении графика: {str(e)}")
            return None, {}
    
    def get_summary(self, timeline_data):
        """Создание текстового описания анализа эмоций"""
        try:
            if not timeline_data:
                logger.warning("Получены пустые данные для анализа")
                return "Недостаточно данных для анализа"
                
            # Подсчет средних значений для каждой эмоции
            total_emotions = {emotion: [] for emotion in self.all_emotions.keys()}
            for point in timeline_data:
                for pred in point['emotions']:
                    total_emotions[pred['label']].append(pred['score'])
            
            # Вычисляем средние значения
            avg_emotions = {
                self.all_emotions[emotion]: np.mean(scores) if scores else 0 
                for emotion, scores in total_emotions.items()
            }
            
            # Находим доминирующие эмоции
            dominant_emotions = sorted(
                avg_emotions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:2]
            
            # Формируем описание
            summary = "Анализ эмоций в голосе:\n\n"
            summary += f"Доминирующие эмоции:\n"
            has_dominant = False
            for emotion, score in dominant_emotions:
                if score > 0.1:  # Порог уверенности
                    summary += f"- {emotion}: {score:.1%}\n"
                    has_dominant = True
            
            if not has_dominant:
                summary += "- Не удалось определить явные эмоции\n"
            
            logger.debug(f"Анализ выполнен, обнаружено {len([e for e in dominant_emotions if e[1] > 0.1])} доминирующих эмоций")
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка при создании описания: {e}")
            return "Ошибка при анализе эмоций"
    
    def get_legend_figure(self):
        """Создаёт отдельную фигуру только с легендой эмоций"""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import io
        
        fig = None
        buf = None
        try:
            fig, ax = plt.subplots(figsize=(3, 4))
            handles = []
            
            # Создаем только один набор линий для всех эмоций
            for emotion, russian_name in self.all_emotions.items():
                line = Line2D([0], [0], 
                            color=self.colors[emotion],
                            lw=3,
                            label=russian_name)
                handles.append(line)
            
            legend = ax.legend(handles=handles, 
                             loc='center',
                             frameon=True,
                             fontsize=12)
            
            # Настройка внешнего вида
            ax.axis('off')
            fig.patch.set_alpha(0)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Сохраняем в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format='png', 
                       bbox_inches='tight',
                       transparent=True,
                       dpi=100)
            buf.seek(0)
            return buf
            
        finally:
            # Очищаем ресурсы
            if fig is not None:
                plt.close(fig)
            if buf is not None and buf.closed:
                buf.close()