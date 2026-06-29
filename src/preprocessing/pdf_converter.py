import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from src.utils import imread_unicode, imwrite_unicode

def pdf_to_images(pdf_path, dpi=300):
    """
    Рендеринг PDF-файла в список изображений OpenCV (numpy arrays).
    """
    images = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Ошибка открытия PDF-файла {pdf_path}: {e}")
        return []
        
    zoom = dpi / 72  # 72 points per inch in PDF
    mat = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        
        # Конвертация в numpy array (RGB)
        img_data = pix.samples
        img = np.frombuffer(img_data, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
        
        # Перевод из RGB/RGBA в BGR для OpenCV
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        images.append(img)
    return images

def deskew_image(image):
    """
    Автоматическое выравнивание чертежа (устранение перекоса скана).
    Находит прямые линии с помощью преобразования Хафа, вычисляет медианный угол 
    и поворачивает изображение в обратную сторону.
    """
    if image is None or image.size == 0:
        return image
        
    try:
        # Переводим в Grayscale для детекции границ
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Бинаризация по Otsu для контрастности линий
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Детекция краев по методу Кэнни
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        
        # Нахождение линий с помощью вероятностного преобразования Хафа
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return image
            
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Вычисляем угол наклона линии в градусах
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            # Приводим к отклонению от ближайшего ортогонального направления (0, 90, 180, 270)
            angle = (angle + 45) % 90 - 45
            angles.append(angle)
            
        # Медианный угол является устойчивой оценкой перекоса всего чертежа
        median_angle = np.median(angles)
        
        # Если перекос незначительный (< 0.05 град.) или неправдоподобно большой (> 15 град.), не выравниваем
        if abs(median_angle) < 0.05 or abs(median_angle) > 15.0:
            return image
            
        print(f"[INFO] Обнаружен перекос чертежа: {median_angle:.2f} градусов. Выравнивание...")
        
        # Поворачиваем чертеж относительно центра
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Вращаем и заливаем освободившиеся углы белым цветом (255, 255, 255)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotated
    except Exception as e:
        print(f"[WARNING] Ошибка при автовыравнивании чертежа (deskew): {e}")
        return image

def get_cached_pdf_image(pdf_path, page_num=0, dpi=300, cache_dir="data/processed", deskew=True):
    """
    Возвращает изображение страницы PDF. Если файл уже отрендерен и лежит в кэше,
    загружает его. Иначе рендерит, при необходимости выравнивает перекос, и сохраняет в кэш.
    """
    os.makedirs(cache_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_suffix = "_deskewed" if deskew else ""
    cache_filename = f"{basename}_page{page_num}_dpi{dpi}{cache_suffix}.png"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_path):
        print(f"Загрузка страницы {page_num} из кэша: {cache_path}")
        img = imread_unicode(cache_path)
        if img is not None:
            return img
            
    print(f"Рендеринг страницы {page_num} из PDF с DPI={dpi}...")
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        raise ValueError(f"Страница {page_num} отсутствует в файле {pdf_path} (всего страниц: {len(doc)})")
        
    page = doc.load_page(page_num)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    img_data = pix.samples
    img = np.frombuffer(img_data, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
    
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    # Выравнивание перекоса скана
    if deskew:
        img = deskew_image(img)
        
    # Сохраняем в кэш
    imwrite_unicode(cache_path, img)
    print(f"Изображение сохранено в кэш: {cache_path}")
    return img
