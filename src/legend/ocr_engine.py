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
        # Белый список символов для фильтрации шума при распознавании ГОСТ шрифтов
        self.allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 -.,()/'

    def extract_text(self, image):
        """
        Распознать весь текст на изображении с препроцессингом.
        image: numpy array (BGR или Grayscale)
        Возвращает строку с распознанным текстом.
        """
        if image is None or image.size == 0:
            return ""
            
        # Препроцессинг:
        # 1. Увеличиваем в 2 раза для четкости
        # 2. Бинаризация по методу Оцу для контраста
        # 3. Морфологическое расширение (Dilation) для утолщения линий букв
        try:
            image_resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Переводим в grayscale
            if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
                gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_resized.copy()
                
            # Бинаризация (белый текст на черном фоне)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Утолщаем линии с помощью Dilation (ядро 2x2, 1 итерация)
            kernel = np.ones((2, 2), np.uint8)
            dilate = cv2.dilate(thresh, kernel, iterations=1)
            
            # Возвращаем обратно: черный текст на белом фоне
            processed_gray = cv2.bitwise_not(dilate)
            
            # EasyOCR лучше работает с RGB
            img_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            print(f"Ошибка при предобработке изображения для OCR: {e}")
            if len(image.shape) == 3 and image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
            
        try:
            results = self.reader.readtext(img_rgb, allowlist=self.allowlist)
            # Извлекаем распознанный текст (каждая запись имеет формат: (bbox, text, confidence))
            text_parts = [res[1] for res in results if res[2] > 0.2]  # фильтр по уверенности
            return " ".join(text_parts).strip()
        except Exception as e:
            print(f"Ошибка при работе EasyOCR: {e}")
            return ""
