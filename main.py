import os
import json
import cv2
import pandas as pd
from src.preprocessing.pdf_converter import pdf_to_images
from src.legend.detector import LegendDetector
from src.legend.ocr_engine import OCREngine
from src.legend.extractor import LegendExtractor
from src.detection.template_matching import TemplateMatchingDetector
from src.postprocessing.nms import filter_by_area
from src.utils import load_config, draw_detections

def run_pipeline(pdf_path, config_path="config/config.yaml"):
    """
    Запуск полного пайплайна обработки чертежа.
    """
    print(f"\n==========================================")
    print(f"Запуск пайплайна для чертежа: {os.path.basename(pdf_path)}")
    print(f"==========================================\n")
    
    # 1. Загрузка конфигурации
    config = load_config(config_path)
    dpi = config['preprocessing'].get('dpi', 300)
    output_dir = config['paths'].get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Рендеринг PDF в изображение
    print(f"Шаг 1: Рендеринг PDF (DPI={dpi})...")
    images = pdf_to_images(pdf_path, dpi=dpi)
    if not images:
        print("Ошибка: Не удалось отрендерить PDF.")
        return
        
    print(f"Успешно отрендерено страниц: {len(images)}")
    
    # Обрабатываем первую страницу чертежа (как основную)
    image = images[0]
    
    # 3. Обнаружение легенды
    print("Шаг 2: Локализация легенды...")
    legend_det = LegendDetector(config)
    legend_coords = legend_det.detect(image)
    
    x_min, y_min, x_max, y_max = legend_coords
    legend_image = image[y_min:y_max, x_min:x_max]
    
    # Сохраняем вырезанную легенду для отладки
    cv2.imwrite(os.path.join(output_dir, "debug_legend.png"), legend_image)
    
    # 4. Инициализация OCR и извлечение шаблонов из легенды
    print("Шаг 3: Инициализация OCR и разбор легенды...")
    ocr_engine = OCREngine(config)
    extractor = LegendExtractor(config, ocr_engine)
    
    templates = extractor.extract_templates(legend_image)
    if not templates:
        print("Ошибка: Не удалось извлечь ни одного шаблона из легенды.")
        return
        
    # Сохраняем извлеченные шаблоны для отладки
    debug_templates_dir = os.path.join(output_dir, "debug_templates")
    os.makedirs(debug_templates_dir, exist_ok=True)
    for name, t_img in templates.items():
        clean_name = "".join([c if c.isalnum() or c in ' _-' else '_' for c in name])
        cv2.imwrite(os.path.join(debug_templates_dir, f"{clean_name}.png"), t_img)
        
    # 5. Детекция условных обозначений на чертеже
    print("Шаг 4: Поиск условных обозначений на чертеже...")
    detector = TemplateMatchingDetector(config)
    detections = detector.detect(image, templates, exclude_region=legend_coords)
    
    print(f"Всего обнаружено объектов на чертеже (после NMS): {len(detections)}")
    
    # 6. Подсчет результатов по классам
    counts = {}
    for name in templates.keys():
        counts[name] = 0
        
    for det in detections:
        class_name = det['class_name']
        counts[class_name] = counts.get(class_name, 0) + 1
        
    # Вывод результатов в консоль
    print("\nРезультаты подсчета:")
    print("-" * 50)
    for name, count in counts.items():
        print(f"| {name:<35} | {count:>5} |")
    print("-" * 50)
    
    # 7. Сохранение отчетов и визуализации
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Визуализация с рамками
    vis_image = draw_detections(image, detections)
    vis_path = os.path.join(output_dir, f"{basename}_detected.png")
    cv2.imwrite(vis_path, vis_image)
    print(f"Изображение с разметкой сохранено: {vis_path}")
    
    # Экспорт в JSON
    json_path = os.path.join(output_dir, f"{basename}_report.json")
    report_data = {
        "filename": os.path.basename(pdf_path),
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
    print(f"JSON отчет сохранен: {json_path}")
    
    # Экспорт в Excel
    excel_path = os.path.join(output_dir, f"{basename}_report.xlsx")
    df = pd.DataFrame(list(counts.items()), columns=["Название символа", "Количество на плане"])
    df.to_excel(excel_path, index=False)
    print(f"Excel отчет сохранен: {excel_path}")
    
    return report_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Подсчет условных обозначений на планах чертежей.")
    parser.add_argument("--pdf", type=str, default="data/raw/план 1.pdf", help="Путь к PDF чертежу")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Путь к файлу конфигурации")
    args = parser.parse_args()
    
    run_pipeline(args.pdf, args.config)
