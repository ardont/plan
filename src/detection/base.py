from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """
    Базовый класс для всех алгоритмов обнаружения объектов (УГО) на планах.
    """
    def __init__(self, config):
        """
        Инициализация детектора.
        config: словарь конфигурации (содержимое config.yaml)
        """
        self.config = config

    @abstractmethod
    def detect(self, image, templates, exclude_region=None):
        """
        Выполнить обнаружение символов на чертеже.
        
        Параметры:
            image: numpy array (BGR) чертежа
            templates: словарь эталонов {class_name: template_image}
            exclude_region: кортеж (x_min, y_min, x_max, y_max) области, 
                            которую нужно исключить из поиска (например, легенда)
                            
        Возвращает:
            list of dict: список детекций в формате:
                [
                    {
                        'box': [x_1, y_1, x_2, y_2],
                        'class_name': 'Название класса',
                        'score': float (уверенность от 0.0 до 1.0)
                    },
                    ...
                ]
        """
        pass
