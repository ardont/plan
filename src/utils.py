import yaml
import cv2
import numpy as np
import os

def load_config(config_path="config/config.yaml"):
    """Загрузка конфигурационного файла YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Чтение изображения из пути с поддержкой кириллицы и юникода на Windows."""
    try:
        with open(path, 'rb') as f:
            nparr = np.frombuffer(f.read(), np.uint8)
            return cv2.imdecode(nparr, flags)
    except Exception as e:
        print(f"Ошибка чтения изображения с юникод-путем {path}: {e}")
        return None

def imwrite_unicode(path, img, params=None):
    """Запись изображения по пути с поддержкой кириллицы и юникода на Windows."""
    try:
        ext = os.path.splitext(path)[1]
        result, nparr = cv2.imencode(ext, img, params)
        if result:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(nparr.tobytes())
            return True
        return False
    except Exception as e:
        print(f"Ошибка записи изображения с юникод-путем {path}: {e}")
        return False

def draw_detections(image, detections):
    """
    Отрисовка рамок и названий условных обозначений на чертеже.
    detections: список словарей [{'box': [x1, y1, x2, y2], 'class_name': '...', 'score': 0.8}]
    """
    vis_img = image.copy()
    colors = {}
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        class_name = det['class_name']
        score = det.get('score', 1.0)
        
        if class_name not in colors:
            # Генерация уникального цвета для каждого класса
            colors[class_name] = [int(c) for c in np.random.randint(50, 200, size=3)]
            
        color = colors[class_name]
        
        # Отрисовка рамки
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Текст с названием и уверенностью
        label = f"{class_name} ({score:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Подложка под текст
        cv2.rectangle(vis_img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        # Белый текст поверх подложки
        cv2.putText(vis_img, label, (x1, y1 - 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return vis_img
