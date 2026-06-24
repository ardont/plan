import cv2
import numpy as np

class LegendExtractor:
    """
    Класс для сегментации области легенды на отдельные строки,
    извлечения чистых изображений символов (шаблонов) и их описаний.
    Использует интеллектуальное сопоставление по координатам.
    """
    def __init__(self, config, ocr_engine):
        self.config = config
        self.ocr_engine = ocr_engine

    def extract_templates(self, legend_image):
        """
        Извлечь шаблоны и текстовые описания из изображения легенды.
        legend_image: numpy array (BGR) области легенды
        Возвращает:
            dict: {class_name: template_image}
        """
        if legend_image is None or legend_image.size == 0:
            return {}

        h, w = legend_image.shape[:2]
        print(f"Размер области легенды для разбора: {w}x{h}")
        
        # 1. Сначала запускаем OCR на всей области легенды, чтобы получить координаты текста
        # EasyOCR reader.readtext возвращает список: [([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], text, conf), ...]
        try:
            raw_ocr_results = self.ocr_engine.reader.readtext(legend_image)
        except Exception as e:
            print(f"Ошибка EasyOCR при разборе всей легенды: {e}")
            raw_ocr_results = []
            
        print(f"OCR обнаружил текстовых блоков: {len(raw_ocr_results)}")
        
        # Фильтруем пустые результаты и сортируем
        ocr_blocks = []
        for bbox, text, conf in raw_ocr_results:
            if conf < 0.25 or not text.strip():
                continue
            # Преобразуем bbox в [x_min, y_min, x_max, y_max]
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            ocr_blocks.append({
                'box': [x_min, y_min, x_max, y_max],
                'text': text.strip(),
                'center_y': (y_min + y_max) / 2,
                'center_x': (x_min + x_max) / 2
            })
            
        # 2. Создаем маску для поиска графических символов
        # Переводим в grayscale и инвертируем бинаризацию (объекты белые, фон черный)
        gray = cv2.cvtColor(legend_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        
        # Закрашиваем текстовые области белым цветом (черным на инвертированной маске),
        # чтобы текст не детектировался как графический контур значка
        clean_thresh = thresh.copy()
        for block in ocr_blocks:
            x1, y1, x2, y2 = block['box']
            # Расширяем область закраски на 2 пикселя во избежание краевых эффектов
            cv2.rectangle(clean_thresh, (max(0, x1-2), max(0, y1-2)), (min(w, x2+2), min(h, y2+2)), 0, -1)
            
        # Удаляем горизонтальные и вертикальные линии таблицы из маски
        # Горизонтальные линии
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.15), 1))
        hor_lines = cv2.morphologyEx(clean_thresh, cv2.MORPH_OPEN, hor_kernel)
        clean_thresh = cv2.subtract(clean_thresh, hor_lines)
        
        # Вертикальные линии
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.05)))
        vert_lines = cv2.morphologyEx(clean_thresh, cv2.MORPH_OPEN, vert_kernel)
        clean_thresh = cv2.subtract(clean_thresh, vert_lines)
        
        # 3. Находим графические контуры (значки)
        contours, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        icons = []
        for cnt in contours:
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            
            # Отсекаем слишком мелкий шум и слишком огромные объекты
            if box_w < 5 or box_h < 5 or box_w > w * 0.25 or box_h > h * 0.2:
                continue
                
            # Проверяем, что значок лежит в левой половине легенды (УГО обычно слева)
            if x > w * 0.4:
                continue
                
            icons.append({
                'box': [x, y, x + box_w, y + box_h],
                'center_y': y + box_h / 2,
                'center_x': x + box_w / 2
            })
            
        print(f"Обнаружено графических символов-кандидатов: {len(icons)}")
        
        templates = {}
        
        # 4. Сопоставляем каждый значок с текстовыми блоками
        # Для каждого значка ищем текстовые блоки, лежащие примерно на той же высоте (по Y) и справа от него
        for icon in icons:
            ix1, iy1, ix2, iy2 = icon['box']
            icon_cy = icon['center_y']
            icon_height = iy2 - iy1
            
            # Находим текстовые блоки на той же горизонтальной линии
            matched_text_blocks = []
            for block in ocr_blocks:
                tx1, ty1, tx2, ty2 = block['box']
                block_cy = block['center_y']
                
                # Критерий 1: Текст находится справа от значка
                if tx1 < ix1:
                    continue
                    
                # Критерий 2: Текст находится на близкой высоте по Y
                # Разница по Y не должна превышать высоту значка или высоту текстового блока
                y_dist = abs(icon_cy - block_cy)
                max_allowed_dist = max(icon_height * 1.2, (ty2 - ty1) * 1.2, 20)
                
                if y_dist <= max_allowed_dist:
                    matched_text_blocks.append(block)
                    
            if not matched_text_blocks:
                continue
                
            # Сортируем текстовые блоки слева направо (по X), чтобы правильно склеить слова в строку
            matched_text_blocks.sort(key=lambda b: b['box'][0])
            
            # Склеиваем слова
            description = " ".join([b['text'] for b in matched_text_blocks])
            description = self._clean_text(description)
            
            if not description:
                continue
                
            # 5. Вырезаем чистый значок из исходного цветного изображения
            # Добавляем отступ в 1 пиксель, если возможно
            pad = 1
            x_min = max(0, ix1 - pad)
            y_min = max(0, iy1 - pad)
            x_max = min(w, ix2 + pad)
            y_max = min(h, iy2 + pad)
            
            clean_template = legend_image[y_min:y_max, x_min:x_max]
            
            # Дополнительно очищаем шаблон от белых краев
            clean_template = self._crop_to_exact_bounds(clean_template)
            
            if clean_template is not None and clean_template.size > 0:
                templates[description] = clean_template
                print(f"Успешно сопоставлено: '{description}' -> Шаблон {clean_template.shape[1]}x{clean_template.shape[0]}")
                
        # Если продвинутый метод сопоставления свободных контуров не нашел ничего, 
        # откатываемся к простому табличному методу по сетке горизонтальных линий
        if not templates:
            print("Предупреждение: Метод сопоставления контуров не дал результатов. Откат к табличному методу...")
            return self._extract_templates_grid(legend_image, thresh)
            
        return templates

    def _extract_templates_grid(self, legend_image, thresh):
        """Резервный табличный метод нарезки легенды по горизонтальной сетке."""
        h, w = legend_image.shape[:2]
        
        # Поиск горизонтальных линий
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.4), 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        y_indices = np.where(horizontal_lines > 0)[0]
        
        # Группировка Y-координат
        row_y = self._group_coords(y_indices, min_dist=15)
        
        if 0 not in row_y:
            row_y.insert(0, 0)
        if h not in row_y:
            row_y.append(h)
            
        row_y = sorted(row_y)
        templates = {}
        
        for i in range(len(row_y) - 1):
            y_start = row_y[i]
            y_end = row_y[i+1]
            
            if y_end - y_start < 15:
                continue
                
            row_img = legend_image[y_start:y_end, 0:w]
            
            # Берем первые 25% ширины под значок
            split_x = int(w * 0.25)
            symbol_area = row_img[:, 0:split_x]
            text_area = row_img[:, split_x:w]
            
            # Очищаем шаблон
            template_img = self._crop_to_exact_bounds(symbol_area)
            if template_img is None:
                continue
                
            # Распознаем текст
            text_desc = self.ocr_engine.extract_text(text_area)
            text_desc = self._clean_text(text_desc)
            
            if not text_desc:
                text_desc = f"Символ_Строка_{i+1}"
                
            templates[text_desc] = template_img
            
        return templates

    def _crop_to_exact_bounds(self, img):
        """Обрезает белые поля вокруг значка, оставляя только значащие пиксели."""
        if img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        # Отступаем от краев рамки во избежание линий таблицы
        margin = 3
        if h <= margin*2 or w <= margin*2:
            return img
            
        inner = img[margin:h-margin, margin:w-margin]
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return inner
            
        x_min, y_min = inner.shape[1], inner.shape[0]
        x_max, y_max = 0, 0
        has_contour = False
        for cnt in contours:
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            if box_w < 3 or box_h < 3:
                continue
            has_contour = True
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + box_w)
            y_max = max(y_max, y + box_h)
            
        if not has_contour:
            return inner
            
        # Возвращаем обрезанный значок
        return inner[y_min:y_max, x_min:x_max]

    def _group_coords(self, coords, min_dist=10):
        if len(coords) == 0:
            return []
        sorted_coords = sorted(coords)
        groups = [[sorted_coords[0]]]
        for c in sorted_coords[1:]:
            if c - groups[-1][-1] <= min_dist:
                groups[-1].append(c)
            else:
                groups.append([c])
        return [int(np.mean(g)) for g in groups]

    def _clean_text(self, text):
        if not text:
            return ""
        text = text.replace('\n', ' ')
        text = " ".join(text.split())
        text = text.lstrip(".-:•* ")
        return text.strip()
