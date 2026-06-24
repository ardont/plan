import easyocr
import numpy as np
import cv2

class OCREngine:
    """
    Класс для распознавания текстовых описаний символов в легенде 
    с использованием библиотеки EasyOCR.
    """
    def __init__(self, config):
        self.config = config
        ocr_config = config.get('ocr', {})
        languages = ocr_config.get('languages', ['ru'])
        gpu = ocr_config.get('gpu', True)
        
        print(f"Инициализация EasyOCR для языков: {languages} (GPU={gpu})...")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract_text(self, image):
        """
        Распознать весь текст на изображении.
        image: numpy array (BGR или Grayscale)
        Возвращает строку с распознанным текстом.
        """
        if image is None or image.size == 0:
            return ""
            
        # Препроцессинг: увеличиваем картинку в 2 раза с бикубической интерполяцией для лучшего качества OCR
        try:
            image_resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(f"Ошибка при изменении размера изображения для OCR: {e}")
            image_resized = image.copy()
            
        # EasyOCR принимает RGB-изображение или путь к файлу
        # Переводим BGR в RGB, если изображение цветное
        if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
            img_rgb = image_resized[:, :, ::-1]
        else:
            img_rgb = image_resized
            
        try:
            results = self.reader.readtext(img_rgb)
            # Извлекаем распознанный текст (каждая запись имеет формат: (bbox, text, confidence))
            text_parts = [res[1] for res in results if res[2] > 0.2]  # фильтр по уверенности
            return " ".join(text_parts).strip()
        except Exception as e:
            print(f"Ошибка при работе EasyOCR: {e}")
            return ""
