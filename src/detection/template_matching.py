import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.detection.base import BaseDetector
from src.postprocessing.nms import non_max_suppression, filter_by_area, filter_by_aspect_ratio

class TemplateMatchingDetector(BaseDetector):
    """
    Детектор на основе OpenCV Template Matching (сопоставление шаблонов) 
    с поддержкой дискретных углов поворота (0, 90, 180, 270 градусов),
    пирамидальным Coarse-to-Fine поиском и многопоточностью ThreadPoolExecutor.
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

    def _detect_single_template(self, class_name, template_img, gray_search, coarse_gray_search, 
                                 coarse_scale_factor, coarse_threshold, threshold, 
                                 rotations, scales):
        """
        Поиск одного шаблона на чертеже по двухэтапной схеме: coarse-to-fine.
        Выполняется в отдельном потоке.
        """
        if template_img is None or template_img.size == 0:
            return []
            
        # Переводим шаблон в grayscale
        if len(template_img.shape) == 3:
            gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template_img.copy()

        h_orig_t, w_orig_t = gray_template.shape[:2]
        
        # 1. Шаг Coarse (грубый проход)
        # Вычисляем динамический scale factor для грубого прохода, чтобы шаблон не сжимался меньше 12 пикселей
        min_coarse_size = 12
        min_dim = min(h_orig_t, w_orig_t)
        
        t_coarse_scale = coarse_scale_factor
        if min_dim * coarse_scale_factor < min_coarse_size:
            if min_dim > min_coarse_size:
                t_coarse_scale = min_coarse_size / min_dim
            else:
                t_coarse_scale = 1.0 # не сжимаем вообще, если он уже меньше min_coarse_size

        if abs(t_coarse_scale - coarse_scale_factor) < 1e-4:
            local_coarse_search = coarse_gray_search
            actual_coarse_scale = coarse_scale_factor
        else:
            local_coarse_search = cv2.resize(gray_search, (0, 0), fx=t_coarse_scale, fy=t_coarse_scale, interpolation=cv2.INTER_AREA)
            actual_coarse_scale = t_coarse_scale

        coarse_candidates = []
        
        for angle in rotations:
            rotated_temp = self.rotate_image(gray_template, angle)
            
            # Масштабируем повернутый шаблон для грубого прохода
            if actual_coarse_scale != 1.0:
                w_new = int(rotated_temp.shape[1] * actual_coarse_scale)
                h_new = int(rotated_temp.shape[0] * actual_coarse_scale)
                if w_new < 5 or h_new < 5:
                    continue
                coarse_temp = cv2.resize(rotated_temp, (w_new, h_new), interpolation=cv2.INTER_AREA)
            else:
                coarse_temp = rotated_temp
                
            h_c, w_c = coarse_temp.shape[:2]
            if h_c > local_coarse_search.shape[0] or w_c > local_coarse_search.shape[1]:
                continue
                
            res = cv2.matchTemplate(local_coarse_search, coarse_temp, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= coarse_threshold)
            pts = list(zip(*loc[::-1]))
            
            if len(pts) > 2000:
                # Защита от переполнения при шуме
                scores_pts = [float(res[pt[1], pt[0]]) for pt in pts]
                sorted_idx = np.argsort(scores_pts)[::-1][:2000]
                pts = [pts[i] for i in sorted_idx]
            
            for pt in pts:
                score = float(res[pt[1], pt[0]])
                box = [pt[0], pt[1], pt[0] + w_c, pt[1] + h_c]
                coarse_candidates.append({
                    'box': box,
                    'score': score,
                    'angle': angle
                })

        # Если на грубом проходе ничего не найдено - выходим досрочно (Early Exit)
        if not coarse_candidates:
            return []

        # Фильтруем дубликаты кандидатов на грубом масштабе с помощью NMS
        coarse_boxes = [c['box'] for c in coarse_candidates]
        coarse_scores = [c['score'] for c in coarse_candidates]
        keep_coarse_idx = non_max_suppression(coarse_boxes, coarse_scores, iou_threshold=0.5)
        
        fine_detections = []
        h_img, w_img = gray_search.shape[:2]
        
        for idx in keep_coarse_idx:
            cand = coarse_candidates[idx]
            c_box = cand['box']
            
            # Маппинг координат обратно в оригинальный масштаб
            x_center_coarse = (c_box[0] + c_box[2]) / 2.0
            y_center_coarse = (c_box[1] + c_box[3]) / 2.0
            
            x_center_orig = x_center_coarse / actual_coarse_scale
            y_center_orig = y_center_coarse / actual_coarse_scale
            
            # Вычисляем максимальные габариты шаблона с запасом
            max_scale = max(scales)
            max_w_orig = w_orig_t * max_scale
            max_h_orig = h_orig_t * max_scale
            
            # Добавим margin (запас)
            margin = 25
            
            x1 = int(x_center_orig - max_w_orig / 2.0 - margin)
            y1 = int(y_center_orig - max_h_orig / 2.0 - margin)
            x2 = int(x_center_orig + max_w_orig / 2.0 + margin)
            y2 = int(y_center_orig + max_h_orig / 2.0 + margin)
            
            # Защита границ ROI (Clipping)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            
            # Проверим, что ROI получился валидным
            if (x2 - x1) < min_coarse_size or (y2 - y1) < min_coarse_size:
                continue
                
            roi_img = gray_search[y1:y2, x1:x2]
            
            # Внутри ROI запускаем точный поиск
            for angle in rotations:
                rotated_temp = self.rotate_image(gray_template, angle)
                
                for scale in scales:
                    if scale != 1.0:
                        w_fine = int(rotated_temp.shape[1] * scale)
                        h_fine = int(rotated_temp.shape[0] * scale)
                        if w_fine < 5 or h_fine < 5:
                            continue
                        scaled_temp = cv2.resize(rotated_temp, (w_fine, h_fine), interpolation=cv2.INTER_CUBIC)
                    else:
                        scaled_temp = rotated_temp
                        
                    h_f, w_f = scaled_temp.shape[:2]
                    
                    if h_f > roi_img.shape[0] or w_f > roi_img.shape[1]:
                        continue
                        
                    res_fine = cv2.matchTemplate(roi_img, scaled_temp, cv2.TM_CCOEFF_NORMED)
                    loc_fine = np.where(res_fine >= threshold)
                    pts_f = list(zip(*loc_fine[::-1]))
                    
                    for pt_f in pts_f:
                        score_f = float(res_fine[pt_f[1], pt_f[0]])
                        
                        # Координаты внутри ROI -> Координаты глобальные
                        x_global = x1 + pt_f[0]
                        y_global = y1 + pt_f[1]
                        
                        box_global = [x_global, y_global, x_global + w_f, y_global + h_f]
                        
                        fine_detections.append({
                            'box': box_global,
                            'score': score_f,
                            'class_name': class_name
                        })
                        
        return fine_detections

    def detect(self, image, templates, exclude_region=None):
        """
        Поиск шаблонов на чертеже с помощью двухэтапного пирамидального Template Matching.
        """
        # Читаем конфигурацию
        tm_config = self.config['detectors']['template_matching']
        threshold = tm_config.get('threshold', 0.65)
        rotations = tm_config.get('rotations', [0, 90, 180, 270])
        scales = tm_config.get('scales', [1.0])
        coarse_scale_factor = tm_config.get('coarse_scale_factor', 0.25)
        coarse_threshold = tm_config.get('coarse_threshold', 0.5)
        num_workers = tm_config.get('num_workers', 4)
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
        
        # Глобальный грубый чертеж
        coarse_gray_search = cv2.resize(
            gray_search, (0, 0), fx=coarse_scale_factor, fy=coarse_scale_factor, interpolation=cv2.INTER_AREA
        )
        
        all_results = []
        
        # Запускаем параллельный поиск по шаблонам в пуле потоков
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for class_name, template_img in templates.items():
                futures.append(
                    executor.submit(
                        self._detect_single_template,
                        class_name=class_name,
                        template_img=template_img,
                        gray_search=gray_search,
                        coarse_gray_search=coarse_gray_search,
                        coarse_scale_factor=coarse_scale_factor,
                        coarse_threshold=coarse_threshold,
                        threshold=threshold,
                        rotations=rotations,
                        scales=scales
                    )
                )
            
            for future in futures:
                try:
                    res = future.result()
                    all_results.extend(res)
                except Exception as e:
                    print(f"Ошибка в потоке сопоставления шаблонов: {e}")
                    
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for det in all_results:
            all_boxes.append(det['box'])
            all_scores.append(det['score'])
            all_classes.append(det['class_name'])
                    
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
