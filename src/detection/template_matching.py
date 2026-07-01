import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.detection.base import BaseDetector

# Отключаем внутреннюю многопоточность OpenCV для предотвращения конфликта потоков (Thread Thrashing)
# с многопоточным пулом ThreadPoolExecutor в Python.
cv2.setNumThreads(1)
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
        tm_config = self.config.get('detectors', {}).get('template_matching', {})
        use_morphology = tm_config.get('use_morphology', True)
        
        # Утолщаем темные линии (применяем эрозию на белом фоне) перед сжатием, если это включено.
        if use_morphology:
            kernel = np.ones((2, 2), np.uint8)
            eroded_search = cv2.erode(gray_search, kernel, iterations=1)
        else:
            eroded_search = gray_search.copy()
        
        # Вычисляем динамический scale factor для грубого прохода
        min_coarse_size = 12
        min_dim = min(h_orig_t, w_orig_t)
        
        t_coarse_scale = coarse_scale_factor
        if min_dim * coarse_scale_factor < min_coarse_size:
            if min_dim > min_coarse_size:
                t_coarse_scale = min_coarse_size / min_dim
            else:
                t_coarse_scale = 1.0 # не сжимаем вообще, если он уже меньше min_coarse_size
 
        if abs(t_coarse_scale - coarse_scale_factor) < 1e-4:
            # Предотрендеренная coarse_gray_search уже должна быть обработана (сделаем это в методе detect ниже)
            local_coarse_search = coarse_gray_search
            actual_coarse_scale = coarse_scale_factor
        else:
            local_coarse_search = cv2.resize(eroded_search, (0, 0), fx=t_coarse_scale, fy=t_coarse_scale, interpolation=cv2.INTER_AREA)
            actual_coarse_scale = t_coarse_scale
 
        coarse_candidates = []
        # Составляем репрезентативные масштабы для грубого прохода на основе переданного списка scales
        if len(scales) <= 4:
            coarse_scales_to_test = list(scales)
        else:
            sorted_scales = sorted(list(scales))
            coarse_scales_to_test = [sorted_scales[0], sorted_scales[len(sorted_scales)//2], sorted_scales[-1]]
            if 1.0 not in coarse_scales_to_test:
                coarse_scales_to_test.append(1.0)
            coarse_scales_to_test = sorted(list(set(coarse_scales_to_test)))
        
        for angle in rotations:
            rotated_temp = self.rotate_image(gray_template, angle)
            # Также утолщаем темные линии на шаблоне, если это включено
            if use_morphology:
                kernel = np.ones((2, 2), np.uint8)
                eroded_temp = cv2.erode(rotated_temp, kernel, iterations=1)
            else:
                eroded_temp = rotated_temp
            
            for c_scale in coarse_scales_to_test:
                combined_scale = actual_coarse_scale * c_scale
                
                # Масштабируем повернутый шаблон для грубого прохода
                if combined_scale != 1.0:
                    w_new = int(eroded_temp.shape[1] * combined_scale)
                    h_new = int(eroded_temp.shape[0] * combined_scale)
                    if w_new < 5 or h_new < 5:
                        continue
                    coarse_temp = cv2.resize(eroded_temp, (w_new, h_new), interpolation=cv2.INTER_AREA)
                else:
                    coarse_temp = eroded_temp
                    
                h_c, w_c = coarse_temp.shape[:2]
                if h_c > local_coarse_search.shape[0] or w_c > local_coarse_search.shape[1]:
                    continue
                
                # Сохраняем отладочное изображение шаблона (только для первого масштаба и угла, чтобы не перегружать диск)
                if angle == rotations[0] and c_scale == coarse_scales_to_test[0]:
                    try:
                        output_dir = self.config.get('paths', {}).get('output_dir', 'output')
                        clean_class_name = "".join([c if c.isalnum() or c in ' _-' else '_' for c in class_name])
                        os.makedirs(output_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(output_dir, f"debug_coarse_template_{clean_class_name}.jpg"), coarse_temp)
                    except Exception as e:
                        pass
                    
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
        
        # Гарантируем, что порог грубого прохода согласован со значением финального порога
        coarse_threshold = min(coarse_threshold, threshold - 0.1)
        coarse_threshold = max(0.1, coarse_threshold)
        
        num_workers = tm_config.get('num_workers', 4)
        nms_iou = tm_config.get('nms_iou_threshold', 0.3)
        min_area = tm_config.get('min_area', 50)
        use_morphology = tm_config.get('use_morphology', True)
        
        # Создаем копию изображения для работы
        search_img = image.copy()
        
        # Если есть область легенды, закрашиваем её белым цветом, чтобы избежать ложных детекций
        if exclude_region is not None:
            x_min, y_min, x_max, y_max = map(int, exclude_region)
            cv2.rectangle(search_img, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
            
        # Для matchTemplate лучше использовать grayscale-изображения
        gray_search = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        
        # Утолщаем темные линии (применяем эрозию на белом фоне) перед сжатием, если это включено.
        if use_morphology:
            kernel = np.ones((2, 2), np.uint8)
            eroded_search = cv2.erode(gray_search, kernel, iterations=1)
        else:
            eroded_search = gray_search.copy()
        
        # Глобальный грубый чертеж
        coarse_gray_search = cv2.resize(
            eroded_search, (0, 0), fx=coarse_scale_factor, fy=coarse_scale_factor, interpolation=cv2.INTER_AREA
        )
        
        # Отладочное сохранение сжатого чертежа
        try:
            output_dir = self.config.get('paths', {}).get('output_dir', 'output')
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "debug_coarse_image.jpg"), coarse_gray_search)
            print(f"[DEBUG] Сохранено отладочное изображение сжатого чертежа: {os.path.join(output_dir, 'debug_coarse_image.jpg')}")
        except Exception as e:
            print(f"[WARNING] Не удалось сохранить отладочное изображение сжатого чертежа: {e}")
            
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
            
        # Применяем внутриклассовый NMS (intra-class NMS), чтобы не затирать близко расположенные значки разных классов.
        # Группируем кандидатов по их классам:
        class_groups = {}
        for idx, class_name in enumerate(all_classes):
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append({
                'box': all_boxes[idx],
                'score': all_scores[idx],
                'idx': idx
            })
            
        detections = []
        for class_name, group in class_groups.items():
            # Сортируем внутри класса по уверенности (уже сделано в NMS, но полезно для ясности)
            group.sort(key=lambda x: x['score'], reverse=True)
            
            boxes_cls = [x['box'] for x in group]
            scores_cls = [x['score'] for x in group]
            
            keep_indices_cls = non_max_suppression(boxes_cls, scores_cls, iou_threshold=nms_iou)
            for k in keep_indices_cls:
                item = group[k]
                detections.append({
                    'box': item['box'],
                    'class_name': class_name,
                    'score': item['score']
                })
            
        return detections
