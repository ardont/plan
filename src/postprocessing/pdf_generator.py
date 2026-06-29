import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def register_cyrillic_font():
    """
    Регистрация кириллического шрифта. Ищет стандартный Arial на Windows,
    DejaVuSans на Linux/Docker или использует резервный шрифт.
    """
    possible_paths = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf"
    ]
    
    font_name = "Helvetica" # Стандартный резервный (без поддержки кириллицы)
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont('CyrillicFont', path))
                font_name = 'CyrillicFont'
                print(f"Успешно зарегистрирован шрифт для кириллицы: {path}")
                break
            except Exception as e:
                print(f"Не удалось зарегистрировать шрифт {path}: {e}")
                
    if font_name == "Helvetica":
        print("ВНИМАНИЕ: Кириллическая гарнитура не найдена. Текст в PDF может отображаться некорректно.")
        
    return font_name

def generate_pdf_report(pdf_name, report_data, output_pdf_path):
    """
    Генерирует PDF-отчет со спецификацией оборудования.
    pdf_name: имя исходного файла
    report_data: словарь с ключом 'counts', содержащим {'Название класса': количество}
    output_pdf_path: путь для сохранения PDF
    """
    # Регистрируем шрифт
    font_name = register_cyrillic_font()
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Создаем кастомные стили на основе зарегистрированного шрифта
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1e3d59'),
        alignment=1, # По центру
        spaceAfter=15
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#555555')
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=12,
        textColor=colors.white,
        fontStyle='Bold'
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        leading=11
    )
    
    story = []
    
    # 1. Заголовок
    story.append(Paragraph("СПЕЦИФИКАЦИЯ ОБОРУДОВАНИЯ И МАТЕРИАЛОВ", title_style))
    story.append(Spacer(1, 10))
    
    # 2. Метаданные отчета
    date_str = datetime.now().strftime("%d.%m.%Y %H:%M")
    story.append(Paragraph(f"<b>Исходный чертеж:</b> {pdf_name}", meta_style))
    story.append(Paragraph(f"<b>Дата формирования отчета:</b> {date_str}", meta_style))
    story.append(Paragraph(f"<b>Всего обнаружено элементов:</b> {report_data.get('total_detected', 0)} шт.", meta_style))
    story.append(Spacer(1, 20))
    
    # 3. Таблица спецификации
    # Заголовок таблицы
    table_data = [
        [
            Paragraph("<b>№</b>", table_header_style),
            Paragraph("<b>Наименование условного графического обозначения (УГО)</b>", table_header_style),
            Paragraph("<b>Кол-во (шт.)</b>", table_header_style)
        ]
    ]
    
    # Заполняем данными
    counts = report_data.get("counts", {})
    for idx, (name, count) in enumerate(counts.items(), 1):
        table_data.append([
            Paragraph(str(idx), table_cell_style),
            Paragraph(name, table_cell_style),
            Paragraph(str(count), table_cell_style)
        ])
        
    # Размеры колонок таблицы (сумма должна соответствовать ширине A4 за вычетом отступов: 595 - 72 = 523)
    col_widths = [40, 403, 80]
    
    spec_table = Table(table_data, colWidths=col_widths)
    spec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3d59')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    
    story.append(spec_table)
    
    # Строим документ
    doc.build(story)
    print(f"PDF-отчет успешно создан: {output_pdf_path}")
