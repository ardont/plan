import os
import cv2
import numpy as np
import fitz  # PyMuPDF

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

def get_cached_pdf_image(pdf_path, page_num=0, dpi=300, cache_dir="data/processed"):
    """
    Возвращает изображение страницы PDF. Если файл уже отрендерен и лежит в кэше,
    загружает его. Иначе рендерит и сохраняет в кэш.
    """
    os.makedirs(cache_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_filename = f"{basename}_page{page_num}_dpi{dpi}.png"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_path):
        print(f"Загрузка страницы {page_num} из кэша: {cache_path}")
        img = cv2.imread(cache_path)
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
        
    # Сохраняем в кэш
    cv2.imwrite(cache_path, img)
    print(f"Изображение сохранено в кэш: {cache_path}")
    return img
