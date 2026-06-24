import numpy as np

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    """
    Алгоритм Non-Maximum Suppression (NMS) для удаления дублирующихся рамок.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        union = np.maximum(union, 1e-6)
        ovr = inter / union

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def filter_by_area(boxes, scores, class_names, min_area=50):
    """
    Фильтрация рамок по минимальной площади для удаления мелкого шума.
    """
    filtered_boxes = []
    filtered_scores = []
    filtered_class_names = []
    
    for box, score, name in zip(boxes, scores, class_names):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_class_names.append(name)
            
    return filtered_boxes, filtered_scores, filtered_class_names

def filter_by_aspect_ratio(boxes, scores, class_names, templates, max_diff=0.25):
    """
    Фильтрация детекций по соотношению сторон (aspect ratio).
    Сравнивает отношение сторон найденной рамки с отношением сторон исходного шаблона.
    Если разница больше max_diff, детекция отбрасывается.
    
    templates: словарь {class_name: template_image}
    """
    filtered_boxes = []
    filtered_scores = []
    filtered_class_names = []
    
    # Кэшируем соотношение сторон для шаблонов, чтобы не считать каждый раз
    template_ratios = {}
    for name, img in templates.items():
        if img is not None and img.size > 0:
            th, tw = img.shape[:2]
            # Записываем соотношение сторон (ширина / высота)
            template_ratios[name] = tw / th
            
    for box, score, name in zip(boxes, scores, class_names):
        if name not in template_ratios:
            # Если шаблона нет в базе, пропускаем фильтрацию для этого бокса
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_class_names.append(name)
            continue
            
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            continue
            
        det_ratio = w / h
        t_ratio = template_ratios[name]
        
        # Проверяем также обратные соотношения (на случай поворотов на 90/270 градусов!)
        # Если шаблон повернут, то его ширина и высота меняются местами.
        t_ratio_rotated = 1.0 / t_ratio
        
        diff_normal = abs(det_ratio - t_ratio) / t_ratio
        diff_rotated = abs(det_ratio - t_ratio_rotated) / t_ratio_rotated
        
        # Если разница как с обычным, так и с повернутым шаблоном больше max_diff, отбрасываем
        if diff_normal > max_diff and diff_rotated > max_diff:
            continue
            
        filtered_boxes.append(box)
        filtered_scores.append(score)
        filtered_class_names.append(name)
        
    return filtered_boxes, filtered_scores, filtered_class_names
