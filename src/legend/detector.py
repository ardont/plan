import cv2
import numpy as np

class LegendDetector:
    """
    Класс для обнаружения и выделения области легенды на чертеже.
    """
    def __init__(self, config):
        self.config = config

    def detect(self, image):
        """
        Найти координаты легенды на чертеже.
        image: numpy array (BGR) чертежа
        Возвращает:
            tuple: (x_min, y_min, x_max, y_max) в пикселях
        """
        legend_config = self.config.get('legend', {})
        auto_detect = legend_config.get('auto_detect', True)
        
        h, w = image.shape[:2]
        
        if auto_detect:
            print("Попытка автоматического поиска легенды...")
            coords = self._auto_detect_legend(image)
            if coords is not None:
                print(f"Легенда успешно обнаружена автоматически: {coords}")
                return coords
            print("Автоматическое обнаружение не удалось. Откат к ручным координатам.")
            
        # Откат к ручным координатам из конфига
        manual_coords = legend_config.get('manual_coords', [0.7, 0.5, 1.0, 1.0])
        x_min = int(manual_coords[0] * w)
        y_min = int(manual_coords[1] * h)
        x_max = int(manual_coords[2] * w)
        y_max = int(manual_coords[3] * h)
        
        # Ограничиваем в рамках изображения
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(0, min(x_max, w))
        y_max = max(0, min(y_max, h))
        
        print(f"Используются ручные координаты легенды: {(x_min, y_min, x_max, y_max)}")
        return (x_min, y_min, x_max, y_max)

    def _auto_detect_legend(self, image):
        """
        Автоматический поиск легенды через морфологическое выделение линий таблицы.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Бинаризация (инвертируем, чтобы линии были белыми на черном фоне)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        
        # Длина структурного элемента для поиска линий (примерно 2% от размера изображения)
        line_length_h = int(w * 0.02)
        line_length_v = int(h * 0.02)
        
        # Поиск горизонтальных линий
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length_h, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Поиск вертикальных линий
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length_v))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Объединение линий для получения сетки таблиц
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Нахождение контуров на маске сетки
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_box = None
        max_area = 0
        
        # Легенда - это крупная таблица, обычно расположенная по краям чертежа
        # (чаще всего в нижнем правом углу)
        for cnt in contours:
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            area = box_w * box_h
            
            # Фильтруем слишком маленькие таблицы (менее 2% площади листа) 
            # и слишком большие (весь чертеж)
            if area < (w * h * 0.01) or area > (w * h * 0.3):
                continue
                
            # Проверяем, находится ли таблица с краю чертежа
            # (легенды чертежей по ГОСТ располагаются на периферии)
            is_at_border = (x + box_w > w * 0.7) or (y + box_h > h * 0.7) or (x < w * 0.3) or (y < h * 0.3)
            
            if is_at_border and area > max_area:
                max_area = area
                best_box = (x, y, x + box_w, y + box_h)
                
        return best_box
