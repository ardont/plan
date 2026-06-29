import easyocr
import numpy as np
import cv2

class OCREngine:
    """
    Класс для распознавания текстовых описаний символов в легенде 
    с использованием библиотеки EasyOCR или Tesseract OCR.
    """
    def __init__(self, config):
        self.config = config
        self.ocr_config = config.get('ocr', {})
        self.engine_type = self.ocr_config.get('engine', 'easyocr')
        self.tesseract_cmd = self.ocr_config.get('tesseract_cmd', '')
        
        self.easyocr_reader = None
        # Белый список символов для фильтрации шума при распознавании ГОСТ шрифтов
        self.allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 -.,()/'

    def _get_easyocr_reader(self):
        """Ленивая инициализация EasyOCR."""
        if self.easyocr_reader is None:
            import easyocr
            languages = self.ocr_config.get('languages', ['ru'])
            gpu = self.ocr_config.get('gpu', True)
            print(f"Инициализация EasyOCR для языков: {languages} (GPU={gpu})...")
            self.easyocr_reader = easyocr.Reader(languages, gpu=gpu)
        return self.easyocr_reader

    def _init_tesseract(self):
        """Настройка пути к исполняемому файлу Tesseract."""
        import pytesseract
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def read_text_blocks(self, img_rgb):
        """
        Распознает текстовые сегменты на изображении легенды.
        Возвращает список блоков вида [[bbox, text, confidence], ...]
        где bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]].
        """
        if self.engine_type == 'tesseract':
            try:
                import pytesseract
                from pytesseract import Output
                self._init_tesseract()
                
                # Запускаем распознавание с разметкой
                # psm 11 (Sparse text. Find as much text as possible in no particular order.)
                d = pytesseract.image_to_data(img_rgb, output_type=Output.DICT, lang='rus+eng', config='--psm 11')
                
                results = []
                n_boxes = len(d['level'])
                for i in range(n_boxes):
                    text = d['text'][i].strip()
                    conf = float(d['conf'][i])
                    # Игнорируем пустые распознавания и явный мусор с низкой уверенностью
                    if text and conf > 15:
                        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                        # Формируем bbox в формате EasyOCR (четыре точки по часовой стрелке)
                        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                        results.append((bbox, text, conf / 100.0))
                return results
            except Exception as e:
                print(f"[WARNING] Ошибка pytesseract: {e}. Переключение на EasyOCR...")
                self.engine_type = 'easyocr'
                
        # По умолчанию EasyOCR
        reader = self._get_easyocr_reader()
        return reader.readtext(img_rgb, allowlist=self.allowlist)

    def extract_text(self, image):
        """
        Распознать весь текст на изображении с препроцессингом.
        image: numpy array (BGR или Grayscale)
        Возвращает строку с распознанным текстом.
        """
        if image is None or image.size == 0:
            return ""
            
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
            
            # EasyOCR / Tesseract лучше работают с RGB
            img_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            print(f"Ошибка при предобработке изображения для OCR: {e}")
            if len(image.shape) == 3 and image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
            
        if self.engine_type == 'tesseract':
            try:
                import pytesseract
                self._init_tesseract()
                # psm 6 (Assume a single uniform block of text)
                text = pytesseract.image_to_string(img_rgb, lang='rus+eng', config='--psm 6')
                return text.strip()
            except Exception as e:
                print(f"[WARNING] Ошибка pytesseract в extract_text: {e}. Переключение на EasyOCR...")
                self.engine_type = 'easyocr'
                
        # По умолчанию EasyOCR
        try:
            reader = self._get_easyocr_reader()
            results = reader.readtext(img_rgb, allowlist=self.allowlist)
            text_parts = [res[1] for res in results if res[2] > 0.2]  # фильтр по уверенности
            return " ".join(text_parts).strip()
        except Exception as e:
            print(f"Ошибка при работе EasyOCR в extract_text: {e}")
            return ""
