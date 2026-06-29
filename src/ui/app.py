import sys
import os
# Добавляем корень проекта в sys.path для корректных импортов при запуске через Streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import cv2
import pandas as pd
import numpy as np
import io
import json

# Импорты модулей системы
from src.utils import load_config, draw_detections, imread_unicode, imwrite_unicode
from src.preprocessing.pdf_converter import get_cached_pdf_image
from src.legend.detector import LegendDetector
from src.legend.ocr_engine import OCREngine
from src.legend.extractor import LegendExtractor
from src.detection.template_matching import TemplateMatchingDetector

# Настройки оформления страницы Streamlit
st.set_page_config(
    page_title="Анализ и подсчет УГО на чертежах",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS для создания премиального вида
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 10px;
    }
    .css-1y4p8pa {
        padding: 1.5rem 1rem;
    }
    h1 {
        color: #1e3d59;
        font-family: 'Inter', sans-serif;
    }
    h2, h3 {
        color: #17b978;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #17b978;
    }
</style>
""", unsafe_allow_html=True)

# Функция загрузки конфигурации (кэшированная)
@st.cache_data
def get_config():
    return load_config("config/config.yaml")

# Функция рендеринга и кэширования страницы PDF в памяти Streamlit
@st.cache_data
def get_cached_pdf_image_streamlit(pdf_path, page_num, dpi, cache_dir):
    return get_cached_pdf_image(pdf_path, page_num=page_num, dpi=dpi, cache_dir=cache_dir)

# Функция извлечения легенды (кэшированная)
@st.cache_data
def extract_legend_data(pdf_path, dpi, auto_legend, manual_coords):
    config = get_config()
    image = get_cached_pdf_image_streamlit(pdf_path, page_num=0, dpi=dpi, cache_dir=config['paths']['processed_dir'])
    
    # Клонируем конфиг и переопределяем параметры локализации легенды
    config_run = config.copy()
    config_run['legend'] = config_run.get('legend', {}).copy()
    config_run['legend']['auto_detect'] = auto_legend
    config_run['legend']['manual_coords'] = manual_coords
    
    legend_det = LegendDetector(config_run)
    legend_coords = legend_det.detect(image)
    
    x_min, y_min, x_max, y_max = legend_coords
    legend_image = image[y_min:y_max, x_min:x_max]
    
    ocr_engine = OCREngine(config_run)
    extractor = LegendExtractor(config_run, ocr_engine)
    
    templates = extractor.extract_templates(legend_image)
    return legend_coords, templates

# Функция параллельного поиска всех значков на чертеже
def detect_all_templates(pdf_path, dpi, templates_dict, legend_coords, tm_threshold):
    config = get_config()
    image = get_cached_pdf_image_streamlit(pdf_path, page_num=0, dpi=dpi, cache_dir=config['paths']['processed_dir'])
    
    config_run = config.copy()
    config_run['detectors'] = config_run.get('detectors', {}).copy()
    config_run['detectors']['template_matching'] = config_run['detectors'].get('template_matching', {}).copy()
    config_run['detectors']['template_matching']['threshold'] = tm_threshold
    
    detector = TemplateMatchingDetector(config_run)
    detections = detector.detect(image, templates_dict, exclude_region=legend_coords)
    return detections


def list_available_presets(config):
    """
    Сканирует templates_dir на наличие папок с preset.json.
    """
    templates_dir = config['paths'].get('templates_dir', 'data/templates')
    os.makedirs(templates_dir, exist_ok=True)
    presets = []
    try:
        for entry in os.listdir(templates_dir):
            entry_path = os.path.join(templates_dir, entry)
            if os.path.isdir(entry_path):
                manifest_path = os.path.join(entry_path, "preset.json")
                if os.path.isfile(manifest_path):
                    presets.append(entry)
    except Exception as e:
        print(f"Ошибка при сканировании шаблонов: {e}")
    return sorted(presets)

def load_template_preset(preset_name, config):
    """
    Загружает пресет из папки templates и возвращает список legend_items.
    """
    templates_dir = config['paths'].get('templates_dir', 'data/templates')
    preset_dir = os.path.join(templates_dir, preset_name)
    manifest_path = os.path.join(preset_dir, "preset.json")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    legend_items = []
    for item_data in manifest.get("items", []):
        filename = item_data["filename"]
        img_path = os.path.join(preset_dir, filename)
        
        # cv2.imread загружает изображение в BGR
        image_np = imread_unicode(img_path)
        if image_np is not None:
            legend_items.append({
                'id': item_data['id'],
                'image_np': image_np,
                'raw_text': item_data.get('raw_text', item_data['label']),
                'label': item_data['label'],
                'enabled': True
            })
    return legend_items

def save_template_preset(preset_name, legend_items, config):
    """
    Сохраняет активный набор символов как пресет в папке templates.
    """
    templates_dir = config['paths'].get('templates_dir', 'data/templates')
    preset_dir = os.path.join(templates_dir, preset_name)
    os.makedirs(preset_dir, exist_ok=True)
    
    preset_items = []
    for item in legend_items:
        if item['enabled']:
            filename = f"symbol_{item['id']}.png"
            img_path = os.path.join(preset_dir, filename)
            
            # Сохраняем BGR-изображение напрямую через opencv
            imwrite_unicode(img_path, item['image_np'])
            
            preset_items.append({
                "id": item['id'],
                "label": item['label'],
                "filename": filename,
                "raw_text": item['raw_text']
            })
            
    manifest = {
        "preset_name": preset_name,
        "items": preset_items
    }
    
    manifest_path = os.path.join(preset_dir, "preset.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        # Важно: ensure_ascii=False для кириллицы
        json.dump(manifest, f, ensure_ascii=False, indent=4)
        
    return len(preset_items)

def draw_preview_legend(image, auto_legend, manual_coords, config):
    """
    Отрисовка оранжевой рамки и направляющих (crosshairs) на изображении предпросмотра
    для точной ручной/автоматической локализации легенды.
    """
    preview_vis = image.copy()
    h, w = preview_vis.shape[:2]
    
    if auto_legend:
        legend_det = LegendDetector(config)
        legend_coords = legend_det.detect(preview_vis)
        if legend_coords is not None:
            x_min, y_min, x_max, y_max = legend_coords
            color = (46, 204, 113) # BGR Sleek Green
            cv2.rectangle(preview_vis, (x_min, y_min), (x_max, y_max), color, 3)
            cv2.putText(preview_vis, "AUTO-DETECTED LEGEND AREA", (x_min + 10, y_min + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    else:
        # Преобразуем относительные координаты в пиксельные
        x_min = int(manual_coords[0] * w)
        y_min = int(manual_coords[1] * h)
        x_max = int(manual_coords[2] * w)
        y_max = int(manual_coords[3] * h)
        
        # Ограничиваем в рамках изображения
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(0, min(x_max, w))
        y_max = max(0, min(y_max, h))
        
        color = (0, 165, 255) # BGR Sleek Orange
        
        # Отрисовка основной рамки
        cv2.rectangle(preview_vis, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Отрисовка тонких вспомогательных направляющих линий (перекрестия) во весь чертеж
        cv2.line(preview_vis, (0, y_min), (w, y_min), color, 1, cv2.LINE_AA)
        cv2.line(preview_vis, (x_min, 0), (x_min, h), color, 1, cv2.LINE_AA)
        cv2.line(preview_vis, (0, y_max), (w, y_max), color, 1, cv2.LINE_AA)
        cv2.line(preview_vis, (x_max, 0), (x_max, h), color, 1, cv2.LINE_AA)
        
        cv2.putText(preview_vis, "MANUAL CROP AREA", (x_min + 10, y_min + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    
    return preview_vis

def sync_legend_items():
    """Синхронизирует введенные пользователем названия и чекбоксы с сессией."""
    if 'legend_items' in st.session_state and st.session_state['legend_items'] is not None:
        for item in st.session_state['legend_items']:
            label_key = f"label_input_{item['id']}"
            enabled_key = f"enabled_check_{item['id']}"
            if label_key in st.session_state:
                item['label'] = st.session_state[label_key]
            if enabled_key in st.session_state:
                item['enabled'] = st.session_state[enabled_key]

def main():
    st.title("📐 Интеллектуальный подсчет условных обозначений (УГО)")
    st.markdown("Система автоматического распознавания графических символов из легенды ГОСТ на планах помещений.")
    st.write("---")
    
    # Синхронизируем изменения данных пользователя в начале
    sync_legend_items()
    
    config = get_config()
    raw_dir = config['paths']['raw_dir']
    
    # Поиск файлов в папке raw
    os.makedirs(raw_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.warning(f"В папке `{raw_dir}` не найдено PDF-файлов планов. Пожалуйста, загрузите чертежи в папку.")
        uploaded_file = st.file_uploader("Загрузить чертеж PDF", type=["pdf"])
        if uploaded_file is not None:
            pdf_path = os.path.join(raw_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Файл {uploaded_file.name} успешно загружен!")
            st.rerun()
        return

    # Боковая панель управления (Sidebar)
    st.sidebar.header("⚙️ Параметры обработки")
    
    selected_pdf = st.sidebar.selectbox("Выберите чертеж для анализа", pdf_files)
    pdf_path = os.path.join(raw_dir, selected_pdf)
    
    # Предварительно загружаем превью чертежа для получения размеров (150 DPI)
    try:
        preview_img = get_cached_pdf_image_streamlit(pdf_path, page_num=0, dpi=150, cache_dir=config['paths']['processed_dir'])
        img_h, img_w = preview_img.shape[:2]
    except Exception as e:
        st.error(f"Не удалось загрузить предварительный просмотр чертежа: {e}")
        return
    
    st.sidebar.write("---")
    dpi = st.sidebar.slider("DPI (разрешение рендеринга)", min_value=150, max_value=600, value=300, step=50, 
                            help="Выше DPI - точнее распознавание мелких значков, но больше времени на обработку.")
    
    st.sidebar.write("---")
    st.sidebar.subheader("📍 Локализация легенды")
    auto_legend = st.sidebar.checkbox("Автоопределение таблицы легенды", value=True)
    
    manual_coords = config['legend']['manual_coords']
    if not auto_legend:
        st.sidebar.markdown("### Настройка рамки (в пикселях)")
        st.sidebar.caption(f"📏 Разрешение превью: **{img_w}x{img_h}** пикселей")
        
        # Дефолтные координаты переводятся из долей в пиксели
        def_x_min = int(manual_coords[0] * img_w)
        def_y_min = int(manual_coords[1] * img_h)
        def_x_max = int(manual_coords[2] * img_w)
        def_y_max = int(manual_coords[3] * img_h)
        
        col1, col2 = st.sidebar.columns(2)
        x_min = col1.number_input("X min (лево)", min_value=0, max_value=img_w, value=min(def_x_min, img_w), step=10)
        x_max = col2.number_input("X max (право)", min_value=0, max_value=img_w, value=min(def_x_max, img_w), step=10)
        y_min = col1.number_input("Y min (верх)", min_value=0, max_value=img_h, value=min(def_y_min, img_h), step=10)
        y_max = col2.number_input("Y max (низ)", min_value=0, max_value=img_h, value=min(def_y_max, img_h), step=10)
        
        if x_min >= x_max:
            st.sidebar.error("X min должен быть больше или равен X max!")
        if y_min >= y_max:
            st.sidebar.error("Y min должен быть больше или равен Y max!")
            
        # Нормализуем координаты обратно для LegendDetector
        manual_coords = [x_min / img_w, y_min / img_h, x_max / img_w, y_max / img_h]
        
    st.sidebar.write("---")
    st.sidebar.subheader("🔍 Настройки детектора")
    detector_type = st.sidebar.selectbox("Алгоритм поиска", ["Template Matching (MVP)", "ORB Keypoints (В разработке)", "YOLO v8 (В разработке)"])
    
    tm_threshold = st.sidebar.slider("Порог схожести шаблона", min_value=0.4, max_value=0.9, value=0.65, step=0.05,
                                     help="Снизьте порог, если значки пропускаются. Повысьте порог, если много ложных рамок.")

    # Проверка на изменение параметров кадрирования/файла для сброса кэша легенды
    current_source = (selected_pdf, dpi, auto_legend, tuple(manual_coords) if not auto_legend else None)
    if st.session_state.get('last_source') != current_source:
        st.session_state['legend_items'] = None
        st.session_state['detections'] = None
        st.session_state['last_source'] = current_source

    # ШАГ 1: ВЫДЕЛЕНИЕ И ИЗВЛЕЧЕНИЕ ЛЕГЕНДЫ
    if st.session_state.get('legend_items') is None:
        st.subheader("📍 Шаг 1: Разметка области легенды")
        
        # Сканируем доступные пресеты шаблонов
        available_presets = list_available_presets(config)
        use_preset = False
        
        if available_presets:
            st.write("Вы можете загрузить готовый сохраненный шаблон УГО (чтобы пропустить кадрирование и OCR) или разметить область легенды вручную.")
            use_preset = st.checkbox("⚙️ Загрузить готовый шаблон УГО из файла", value=False)
            
        if use_preset:
            selected_preset = st.selectbox("Выберите сохраненный шаблон", available_presets)
            if st.button("📥 Подгрузить выбранный шаблон (Перейти к Шагу 2)", type="primary", use_container_width=True):
                try:
                    loaded_items = load_template_preset(selected_preset, config)
                    st.session_state['legend_items'] = loaded_items
                    st.session_state['legend_coords'] = (0, 0, 0, 0)
                    st.success(f"Шаблон '{selected_preset}' успешно загружен! Вы перешли к Шагу 2.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Не удалось загрузить шаблон: {e}")
        else:
            st.write("Скорректируйте координаты легенды в боковой панели. Оранжевые направляющие линии на предпросмотре помогут точно выставить границы.")
            
            # Отрендерим превью страницы с направляющими
            try:
                with st.spinner("Загрузка предварительного просмотра..."):
                    preview_img = get_cached_pdf_image_streamlit(pdf_path, page_num=0, dpi=150, cache_dir=config['paths']['processed_dir'])
                    preview_vis = draw_preview_legend(preview_img, auto_legend, manual_coords, config)
                    preview_rgb = cv2.cvtColor(preview_vis, cv2.COLOR_BGR2RGB)
                    st.image(preview_rgb, use_container_width=True, caption=f"План помещения с оверлеем разметки легенды ({selected_pdf})")
            except Exception as e:
                st.error(f"Не удалось загрузить чертеж: {e}")
                return
                
            if st.button("🔍 Разобрать и проверить легенду", type="primary", use_container_width=True):
                with st.spinner("Выполняется разбор и сегментация легенды чертежа..."):
                    try:
                        legend_coords, templates = extract_legend_data(pdf_path, dpi, auto_legend, manual_coords)
                        
                        if not templates:
                            st.error("Не удалось найти графические значки в указанной области легенды. Проверьте координаты кадрирования.")
                            return
                            
                        # Сохраняем элементы в session_state для инспектирования
                        legend_items = []
                        for idx, (name, t_img) in enumerate(templates.items()):
                            # Автоопределение шума в OCR: слишком длинная строка или обилие мусорных символов
                            is_noise = False
                            if len(name) > 40 or any(char in name for char in ['#', '{', '}', '[', ']', '%', '*', '~', '@', '$', '|', ':', ';', '`']):
                                is_noise = True
                                
                            legend_items.append({
                                'id': idx + 1,
                                'image_np': t_img,
                                'raw_text': name,
                                'label': name,
                                'enabled': not is_noise
                            })
                            
                        st.session_state['legend_items'] = legend_items
                        st.session_state['legend_coords'] = legend_coords
                        st.success(f"Легенда разобрана. Успешно выделено {len(legend_items)} элементов!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка при разборе легенды: {e}")
                        
    # ШАГ 2: ВАЛИДАЦИЯ И РЕДАКТИРОВАНИЕ ШАБЛОНОВ
    else:
        st.subheader("📋 Шаг 2: Валидация и редактирование элементов легенды")
        st.write("Ниже представлена таблица распознанных символов. Вы можете отредактировать подписи (если OCR ошибся) или исключить строки с мусором, сняв галочку.")
        
        # Отображение таблицы с помощью Streamlit-колонок
        col_hdr_id, col_hdr_img, col_hdr_raw, col_hdr_label, col_hdr_act = st.columns([1, 2, 4, 6, 2])
        with col_hdr_id:
            st.markdown("**№**")
        with col_hdr_img:
            st.markdown("**Символ**")
        with col_hdr_raw:
            st.markdown("**Текст OCR**")
        with col_hdr_label:
            st.markdown("**Название символа (для отчетов)**")
        with col_hdr_act:
            st.markdown("**Искать?**")
            
        st.write("---")
        
        for item in st.session_state['legend_items']:
            col_id, col_img, col_raw, col_label, col_act = st.columns([1, 2, 4, 6, 2])
            with col_id:
                st.write(f"#{item['id']}")
            with col_img:
                # Обязательно указываем BGR цветовой канал, так как OpenCV считывает в BGR!
                st.image(item['image_np'], channels="BGR", width=45)
            with col_raw:
                st.caption(item['raw_text'])
            with col_label:
                st.text_input("Название", value=item['label'], key=f"label_input_{item['id']}", label_visibility="collapsed")
            with col_act:
                st.checkbox("Использовать", value=item['enabled'], key=f"enabled_check_{item['id']}", label_visibility="collapsed")
                
        st.write("---")
        
        # Раздел сохранения шаблона
        st.subheader("💾 Сохранить этот набор символов как шаблон")
        st.write("Вы можете сохранить текущие активные значки УГО с их названиями как шаблон для будущих чертежей.")
        
        c_save_name, c_save_btn = st.columns([3, 1])
        basename = os.path.splitext(selected_pdf)[0]
        with c_save_name:
            preset_name_input = st.text_input("Название нового шаблона УГО", value=basename, key="new_preset_name_input", label_visibility="collapsed")
        with c_save_btn:
            save_preset = st.button("💾 Сохранить шаблон", use_container_width=True)
            
        if save_preset:
            clean_preset_name = "".join([c if c.isalnum() or c in ' _-' else '_' for c in preset_name_input.strip()])
            if not clean_preset_name:
                st.error("Пожалуйста, введите корректное название шаблона.")
            else:
                saved_count = save_template_preset(clean_preset_name, st.session_state['legend_items'], config)
                if saved_count > 0:
                    st.success(f"Шаблон '{clean_preset_name}' успешно сохранен! Записано символов: {saved_count}.")
                else:
                    st.error("Нет активных символов для сохранения. Убедитесь, что хотя бы у одного символа стоит галочка 'Искать?'.")
        
        st.write("---")
        
        c1, c2 = st.columns(2)
        with c1:
            run_search = st.button("🚀 Запустить поиск по чертежу", type="primary", use_container_width=True)
        with c2:
            reset_legend = st.button("↩️ Сбросить выделение легенды", use_container_width=True)
            if reset_legend:
                st.session_state['legend_items'] = None
                st.session_state['detections'] = None
                st.rerun()
                
        if run_search:
            templates_to_search = {}
            for item in st.session_state['legend_items']:
                if item['enabled']:
                    clean_name = item['label'].strip()
                    if not clean_name:
                        clean_name = f"Символ_{item['id']}"
                    templates_to_search[clean_name] = item['image_np']
                    
            if not templates_to_search:
                st.error("Пожалуйста, выберите хотя бы один активный символ (поставив галочку 'Искать?').")
            else:
                # Запуск параллельного поиска шаблонов
                with st.status("Поиск условных обозначений на чертеже...", expanded=True) as status:
                    status.write("Инициализация параллельного поиска...")
                    progress_bar = st.progress(0.1)
                    
                    try:
                        status.update(label="Поиск всех УГО на чертеже в параллельном режиме...", state="running")
                        progress_bar.progress(0.3)
                        
                        detections = detect_all_templates(
                            pdf_path, dpi, templates_to_search, st.session_state['legend_coords'], tm_threshold
                        )
                        
                        progress_bar.progress(0.8)
                            
                        # Склейка результатов и глобальный NMS
                        status.update(label="Фильтрация пересекающихся рамок (NMS)...", state="running")
                        progress_bar.progress(1.0)
                        
                        from src.postprocessing.nms import non_max_suppression
                        if detections:
                            boxes = [d['box'] for d in detections]
                            scores = [d['score'] for d in detections]
                            nms_iou = config['detectors']['template_matching'].get('nms_iou_threshold', 0.3)
                            
                            keep_indices = non_max_suppression(boxes, scores, iou_threshold=nms_iou)
                            detections = [detections[i] for i in keep_indices]
                            
                        st.session_state['detections'] = detections
                        status.update(label="Поиск успешно завершен!", state="complete", expanded=False)
                        st.success(f"Поиск завершен! Распознано объектов на чертеже: {len(detections)}")
                        st.rerun()
                    except Exception as e:
                        status.update(label="Произошла ошибка при поиске", state="error", expanded=True)
                        st.error(f"Ошибка во время поиска символов: {e}")

        # ШАГ 3: ВЫВОД РЕЗУЛЬТАТОВ ПОИСКА
        if st.session_state.get('detections') is not None:
            st.write("---")
            st.subheader("📊 Шаг 3: Результаты распознавания и выгрузка")
            
            detections = st.session_state['detections']
            legend_coords = st.session_state['legend_coords']
            
            # Подсчитываем количество по классам
            active_classes = [item['label'].strip() or f"Символ_{item['id']}" for item in st.session_state['legend_items'] if item['enabled']]
            counts = {name: 0 for name in active_classes}
            
            for det in detections:
                name = det['class_name']
                counts[name] = counts.get(name, 0) + 1
                
            # Сохранение результатов на диск в папку output (как в main.py)
            output_dir = config['paths'].get('output_dir', 'output')
            os.makedirs(output_dir, exist_ok=True)
            basename = os.path.splitext(selected_pdf)[0]
            
            # Экспорт в Excel (на диск)
            excel_path = os.path.join(output_dir, f"{basename}_report.xlsx")
            df_results = pd.DataFrame(list(counts.items()), columns=["Название символа", "Количество на плане"])
            df_results.to_excel(excel_path, index=False)
            
            # Экспорт в JSON (на диск)
            json_path = os.path.join(output_dir, f"{basename}_report.json")
            report_data = {
                "filename": selected_pdf,
                "total_detected": len(detections),
                "counts": counts,
                "detections": [
                    {
                        "class_name": d["class_name"],
                        "box": [int(coord) for coord in d["box"]],
                        "score": float(d["score"])
                    }
                    for d in detections
                ]
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=4)
                
            col_res1, col_res2 = st.columns([2, 3])
            
            with col_res1:
                st.subheader("📋 Таблица подсчета")
                
                # Общее количество
                st.markdown(f"""
                <div class="metric-card">
                    <small>Всего распознано УГО на плане</small>
                    <h1>{len(detections)} шт.</h1>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
                
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Кнопка скачивания Excel
                output_excel = io.BytesIO()
                with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                    df_results.to_excel(writer, index=False, sheet_name='Отчет')
                excel_data = output_excel.getvalue()
                
                st.download_button(
                    label="📥 Скачать отчет Excel",
                    data=excel_data,
                    file_name=f"report_{basename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                # Кнопка скачивания CSV (с экранированием)
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать отчет CSV",
                    data=csv_data,
                    file_name=f"report_{basename}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            with col_res2:
                st.subheader("🖼️ Размеченный план чертежа")
                
                # Отрисовываем результаты на изображении высокого разрешения
                image = get_cached_pdf_image_streamlit(pdf_path, page_num=0, dpi=dpi, cache_dir=config['paths']['processed_dir'])
                vis_img = draw_detections(image, detections)
                
                # Подсвечиваем область легенды желтой рамкой
                lx1, ly1, lx2, ly2 = map(int, legend_coords)
                cv2.rectangle(vis_img, (lx1, ly1), (lx2, ly2), (0, 255, 255), 3)
                cv2.putText(vis_img, "OBLAST' LEGENDY", (lx1 + 10, ly1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                st.image(vis_img_rgb, use_container_width=True, caption=f"План с распознанными объектами (область легенды выделена желтым)")
                
                # Сохранение на диск размеченной картинки
                vis_path = os.path.join(output_dir, f"{basename}_detected.png")
                imwrite_unicode(vis_path, vis_img)
                
                # Скачивание размеченной картинки
                _, img_encoded = cv2.imencode('.png', vis_img)
                st.download_button(
                    label="💾 Скачать изображение чертежа с рамками",
                    data=img_encoded.tobytes(),
                    file_name=f"detected_{basename}.png",
                    mime="image/png",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
