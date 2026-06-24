import os
import streamlit as st
import cv2
import pandas as pd
import numpy as np
import io

# Импорты модулей системы
from src.utils import load_config, draw_detections
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
""", unsafe_allowed_html=True)

# Функция загрузки конфигурации (кэшированная)
@st.cache_data
def get_config():
    return load_config("config/config.yaml")

# Единая функция обработки чертежа (кэшированная)
@st.cache_data
def process_drawing(pdf_path, dpi, auto_legend, manual_coords, tm_threshold):
    config = get_config()
    
    # 1. Загрузка/рендеринг страницы чертежа с дисковым кэшированием
    image = get_cached_pdf_image(pdf_path, page_num=0, dpi=dpi, cache_dir=config['paths']['processed_dir'])
    
    # 2. Локализация легенды
    # Переопределяем параметры в соответствии с UI
    config['legend']['auto_detect'] = auto_legend
    config['legend']['manual_coords'] = manual_coords
    
    legend_det = LegendDetector(config)
    legend_coords = legend_det.detect(image)
    
    x_min, y_min, x_max, y_max = legend_coords
    legend_image = image[y_min:y_max, x_min:x_max]
    
    # 3. Инициализация OCR и извлечение шаблонов
    ocr_engine = OCREngine(config)
    extractor = LegendExtractor(config, ocr_engine)
    
    templates = extractor.extract_templates(legend_image)
    
    if not templates:
        return image, legend_coords, {}, []
        
    # 4. Детекция значков на чертеже
    config['detectors']['template_matching']['threshold'] = tm_threshold
    detector = TemplateMatchingDetector(config)
    detections = detector.detect(image, templates, exclude_region=legend_coords)
    
    # Сериализуем шаблоны (переводим в список, чтобы st.cache_data нормально работал с numpy массивами)
    # На самом деле st.cache_data отлично кэширует словари с numpy, но сохраним как есть.
    return image, legend_coords, templates, detections

def main():
    st.title("📐 Интеллектуальный подсчет условных обозначений (УГО)")
    st.markdown("Система автоматического распознавания графических символов из легенды ГОСТ на планах помещений.")
    st.write("---")
    
    config = get_config()
    raw_dir = config['paths']['raw_dir']
    
    # Поиск файлов в папке raw
    os.makedirs(raw_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.warning(f"В папке `{raw_dir}` не найдено PDF-файлов планов. Пожалуйста, загрузите чертежи в папку.")
        # Возможность загрузки файла через веб-интерфейс
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
    
    st.sidebar.write("---")
    dpi = st.sidebar.slider("DPI (разрешение рендеринга)", min_value=150, max_value=600, value=300, step=50, 
                            help="Выше DPI - точнее распознавание мелких значков, но больше времени на обработку.")
    
    st.sidebar.write("---")
    st.sidebar.subheader("📍 Локализация легенды")
    auto_legend = st.sidebar.checkbox("Автоопределение таблицы легенды", value=True)
    
    manual_coords = config['legend']['manual_coords']
    if not auto_legend:
        st.sidebar.info("Задайте координаты области легенды (в долях от 0.0 до 1.0):")
        x_min_f = st.sidebar.slider("X min (лево)", 0.0, 1.0, 0.7, step=0.05)
        y_min_f = st.sidebar.slider("Y min (верх)", 0.0, 1.0, 0.5, step=0.05)
        x_max_f = st.sidebar.slider("X max (право)", 0.0, 1.0, 1.0, step=0.05)
        y_max_f = st.sidebar.slider("Y max (низ)", 0.0, 1.0, 1.0, step=0.05)
        manual_coords = [x_min_f, y_min_f, x_max_f, y_max_f]
        
    st.sidebar.write("---")
    st.sidebar.subheader("🔍 Настройки детектора")
    detector_type = st.sidebar.selectbox("Алгоритм поиска", ["Template Matching (MVP)", "ORB Keypoints (В разработке)", "YOLO v8 (В разработке)"])
    
    tm_threshold = st.sidebar.slider("Порог схожести шаблона", min_value=0.4, max_value=0.9, value=0.65, step=0.05,
                                     help="Снизьте порог, если значки пропускаются. Повысьте порог, если много ложных рамок.")

    if st.sidebar.button("🚀 Запустить распознавание", type="primary"):
        with st.spinner("Выполняется обработка плана..."):
            image, legend_coords, templates, detections = process_drawing(
                pdf_path, dpi, auto_legend, manual_coords, tm_threshold
            )
            
        if not templates:
            st.error("Не удалось разобрать легенду чертежа. Проверьте правильность координат легенды в боковой панели.")
            return
            
        # Подсчет детекций
        counts = {name: 0 for name in templates.keys()}
        for det in detections:
            name = det['class_name']
            counts[name] = counts.get(name, 0) + 1
            
        # Вывод результатов в две колонки
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("📋 Результаты подсчета")
            
            # Общее количество
            st.markdown(f"""
            <div class="metric-card">
                <small>Всего распознано УГО на плане</small>
                <h1>{len(detections)} шт.</h1>
            </div>
            """, unsafe_allowed_html=True)
            st.write("")
            
            # Таблица
            df_results = pd.DataFrame(list(counts.items()), columns=["Название символа", "Количество"])
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            # Кнопка скачивания Excel отчета
            output_excel = io.BytesIO()
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name='Отчет')
            excel_data = output_excel.getvalue()
            
            st.download_button(
                label="📥 Скачать отчет Excel",
                data=excel_data,
                file_name=f"report_{selected_pdf.replace('.pdf', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Вывод вырезанных значков (шаблонов)
            st.write("---")
            st.subheader("🎨 Шаблоны из легенды")
            for name, t_img in templates.items():
                col_i1, col_i2 = st.columns([1, 4])
                with col_i1:
                    # Показываем картинку значка (черный на белом)
                    st.image(t_img, width=40)
                with col_i2:
                    st.markdown(f"**{name}**")
                    
        with col2:
            st.subheader("🖼️ Размеченный план чертежа")
            
            # Отрисовываем результаты на изображении
            vis_img = draw_detections(image, detections)
            
            # Опционально: подсвечиваем область легенды желтой рамкой
            lx1, ly1, lx2, ly2 = map(int, legend_coords)
            cv2.rectangle(vis_img, (lx1, ly1), (lx2, ly2), (0, 255, 255), 3)
            cv2.putText(vis_img, "ОБЛАСТЬ ЛЕГЕНДЫ", (lx1 + 10, ly1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Streamlit работает с RGB
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            st.image(vis_img_rgb, use_container_width=True, caption=f"План с распознанными объектами (область легенды выделена желтым)")
            
            # Сохранение изображения на диск для скачивания
            _, img_encoded = cv2.imencode('.png', vis_img)
            st.download_button(
                label="💾 Скачать изображение чертежа с рамками",
                data=img_encoded.tobytes(),
                file_name=f"detected_{selected_pdf.replace('.pdf', '.png')}",
                mime="image/png",
                use_container_width=True
            )

    else:
        st.info("Выберите чертеж и нажмите кнопку **🚀 Запустить распознавание** в боковой панели, чтобы начать анализ.")
        
        # Показываем исходный PDF-файл в виде первой страницы
        try:
            with st.spinner("Загрузка предварительного просмотра..."):
                preview_img = get_cached_pdf_image(pdf_path, page_num=0, dpi=150, cache_dir=config['paths']['processed_dir'])
                preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                st.subheader("Предварительный просмотр чертежа")
                st.image(preview_rgb, use_container_width=True, caption=selected_pdf)
        except Exception as e:
            st.error(f"Не удалось загрузить чертеж: {e}")

if __name__ == "__main__":
    main()
