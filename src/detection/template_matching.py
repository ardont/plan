import cv2
import numpy as np
from src.detection.base import BaseDetector
from src.postprocessing.nms import non_max_suppression, filter_by_area, filter_by_aspect_ratio

class TemplateMatchingDetector(BaseDetector):
    """
    Детектор на основе OpenCV Template Matching (сопоставление шаблонов) 
    с поддержкой дискретных углов поворота (0, 90, 180, 270 градусов).
    """
    
    def rotate_image(self, image, angle):
        """Вспомогательная функция вращения изображения шаблона."""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def detect(self, image, templates, exclude_region=None):
        """
        Поиск шаблонов на чертеже с помощью Template Matching.
        """
        # Читаем конфигурацию
        tm_config = self.config['detectors']['template_matching']
        threshold = tm_config.get('threshold', 0.65)
        rotations = tm_config.get('rotations', [0, 90, 180, 270])
        nms_iou = tm_config.get('nms_iou_threshold', 0.3)
        min_area = tm_config.get('min_area', 50)
        
        # Создаем копию изображения для работы
        search_img = image.copy()
        
        # Если есть область легенды, закрашиваем её белым цветом, чтобы избежать ложных детекций
        if exclude_region is not None:
            x_min, y_min, x_max, y_max = map(int, exclude_region)
            cv2.rectangle(search_img, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
            
        # Для matchTemplate лучше использовать grayscale-изображения
        gray_search = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for class_name, template_img in templates.items():
            if template_img is None or template_img.size == 0:
                continue
                
            # Переводим шаблон в grayscale
            if len(template_img.shape) == 3:
                gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            else:
                gray_template = template_img.copy()
                
            # Перебираем углы поворота шаблона
            for angle in rotations:
                rotated_temp = self.rotate_image(gray_template, angle)
                h, w = rotated_temp.shape[:2]
                
                # Запускаем шаблонное сопоставление
                res = cv2.matchTemplate(gray_search, rotated_temp, cv2.TM_CCOEFF_NORMED)
                
                # Находим все позиции, где схожесть больше порога
                loc = np.where(res >= threshold)
                
                for pt in zip(*loc[::-1]):  # pt - это (x, y) верхнего левого угла
                    score = float(res[pt[1], pt[0]])
                    box = [pt[0], pt[1], pt[0] + w, pt[1] + h]
                    
                    all_boxes.append(box)
                    all_scores.append(score)
                    all_classes.append(class_name)
                    
        # Фильтрация по минимальной площади
        all_boxes, all_scores, all_classes = filter_by_area(
            all_boxes, all_scores, all_classes, min_area=min_area
        )
        
        # Фильтрация по соотношению сторон (aspect ratio)
        all_boxes, all_scores, all_classes = filter_by_aspect_ratio(
            all_boxes, all_scores, all_classes, templates, max_diff=0.25
        )
        
        if len(all_boxes) == 0:
            return []
            
        # Применяем NMS для удаления дубликатов
        keep_indices = non_max_suppression(all_boxes, all_scores, iou_threshold=nms_iou)
        
        # Формируем итоговый список детекций
        detections = []
        for idx in keep_indices:
            detections.append({
                'box': all_boxes[idx],
                'class_name': all_classes[idx],
                'score': all_scores[idx]
            })
            
        return detections
