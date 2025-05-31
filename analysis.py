# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import matplotlib.gridspec as gridspec
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import pandas as pd

def run_analysis(results_df):
    """
    Основная функция для запуска полного анализа результатов
    Выводит сводные таблицы и графики в папку analysis
    """
    print("\nЗапуск комплексного анализа результатов...")

    # Создаем папку для анализа с временной меткой
    analysis_dir = "analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Результаты анализа будут сохранены в папке: {analysis_dir}")
    analysis_df = results_df.copy(deep=True)
    print("Столбцы в results_df:", analysis_df.columns.tolist())

    # 1. Определение лучших моделей для каждого ряда
    best_models = find_best_models(analysis_df, analysis_dir)

    # 2. Сводный отчет по всем моделям
    generate_summary_report(analysis_df, best_models, analysis_dir)

    # 3. Детальный анализ для каждого ряда
    for series in ['A', 'B', 'C']:
        analyze_series(analysis_df, series, best_models, analysis_dir)

    # 4. Сравнение времени обучения
    analyze_training_time(analysis_df, analysis_dir)

    # 5. Создание PDF отчета
    create_pdf_report(analysis_dir)

    print("\nАнализ успешно завершен!")
    print(f"Все результаты сохранены в папке: {analysis_dir}")


def find_best_models(results_df, analysis_dir):
    """Определяет лучшие модели для каждого ряда по sMAPE"""
    best_models = {}

    for series in ['A', 'B', 'C']:
        # Фильтруем результаты для текущего ряда
        series_results = results_df[results_df['series'] == series]

        # Исключаем модели с ошибками
        valid_results = series_results[series_results['error_occurred'] == False]

        if not valid_results.empty:
            # Находим модель с минимальным sMAPE на тесте
            best_idx = valid_results['test_sMAPE'].idxmin()
            best_model = valid_results.loc[best_idx]
            best_models[series] = best_model

            # Сохраняем информацию о лучшей модели
            best_model.to_csv(
                f"{analysis_dir}/best_model_{series}.csv",
                index=False,
                sep=';',  # Используем точку с запятой как разделитель столбцов
                decimal=',',  # Используем запятую как десятичный разделитель
                encoding='cp1251'  # Добавляем BOM для правильного отображения кириллицы
            )

            print(f"Лучшая модель для ряда {series}: {best_model['model']} (sMAPE={best_model['test_sMAPE']:.2f}%)")

    return best_models


def generate_summary_report(results_df, best_models, analysis_dir):
    """Генерирует сводный отчет по всем моделям"""
    # Создаем сводную таблицу
    summary = results_df.groupby(['series', 'model']).agg({
        'test_MAE': 'mean',
        'test_RMSE': 'mean',
        'test_sMAPE': 'mean',
        'train_time': 'mean'
    }).reset_index()

    # Добавляем флаг лучшей модели
    summary['is_best'] = False
    for series, model in best_models.items():
        summary.loc[
            (summary['series'] == series) &
            (summary['model'] == model['model']),
            'is_best'
        ] = True

    # Сохраняем сводную таблицу
    summary.to_csv(
        f"{analysis_dir}/summary.csv",
        index=False,
        float_format='%.4f',
        sep=';',  # Используем точку с запятой как разделитель столбцов
        decimal=',',  # Используем запятую как десятичный разделитель
        encoding='cp1251'  # Добавляем BOM для правильного отображения кириллицы
    )

    # Визуализация: тепловая карта сравнения моделей
    plt.figure(figsize=(12, 8))
    pivot = summary.pivot_table(index='model', columns='series', values='test_sMAPE')
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Сравнение моделей по sMAPE (%)")
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/models_comparison_heatmap.png")
    plt.close()

    # Визуализация: топ-3 модели для каждого ряда
    for series in ['A', 'B', 'C']:
        series_data = summary[summary.series == series].sort_values('test_sMAPE').head(3)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='model', y='test_sMAPE', data=series_data)
        plt.title(f'Топ-3 модели для ряда {series} (sMAPE)')
        plt.ylabel('sMAPE (%)')

        # Добавляем значения на столбцы
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}%",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points'
            )

        plt.tight_layout()
        plt.savefig(f"{analysis_dir}/top3_models_{series}.png")
        plt.close()


def analyze_series(results_df, series, best_models, analysis_dir):
    """Выполняет детальный анализ для конкретного ряда"""
    print(f"\nАнализ ряда {series}...")


    # Фильтруем результаты для текущего ряда
    series_results = results_df[results_df['series'] == series]

    # Создаем папку для ряда
    series_dir = f"{analysis_dir}/series_{series}"
    os.makedirs(series_dir, exist_ok=True)

    # Сохраняем результаты для ряда
    series_results.to_csv(
        f"{series_dir}/results.csv",
        index=False,
        float_format='%.4f',
        sep=';',  # Используем точку с запятой как разделитель столбцов
        decimal=',',  # Используем запятую как десятичный разделитель
        encoding='cp1251'  # Добавляем BOM для правильного отображения кириллицы
    )

    # Визуализация: сравнение моделей по метрикам
    plt.figure(figsize=(12, 8))
    metrics = ['test_MAE', 'test_RMSE', 'test_sMAPE']

    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i + 1)
        sns.barplot(x='model', y=metric, data=series_results)
        plt.title(f'{metric} для ряда {series}')
        plt.ylabel(metric.split('_')[-1])
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{series_dir}/metrics_comparison.png")
    plt.close()

    # Анализ лучшей модели
    if series in best_models:
        best_model = best_models[series]
        print(f"  - Лучшая модель: {best_model['model']}")

        # Создаем отчет по лучшей модели
        with open(f"{series_dir}/best_model_report.txt", "w", encoding='cp1251') as f:
            f.write(f"Лучшая модель для ряда {series}: {best_model['model']}\n")
            f.write(f"Тестовые метрики:\n")
            f.write(f"  MAE:   {best_model['test_MAE']:.4f}\n")
            f.write(f"  RMSE:  {best_model['test_RMSE']:.4f}\n")
            f.write(f"  sMAPE: {best_model['test_sMAPE']:.4f}%\n")
            f.write(f"Время обучения: {best_model['train_time']:.2f} сек\n")

            if best_model.get('stat_Ljung-Box_pvalue'):
                f.write("\nСтатистические тесты остатков:\n")
                f.write(f"  Люнг-Бокс (p-value): {best_model['stat_Ljung-Box_pvalue']:.4f}\n")
                f.write(f"  Дарбин-Уотсон: {best_model['stat_Durbin-Watson']:.4f}\n")


def analyze_training_time(results_df, analysis_dir):
    """Анализирует время обучения моделей"""
    if 'series' not in results_df.columns:
        raise KeyError("Столбец 'series' отсутствует в DataFrame")
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=results_df,
        x='model',
        y='train_time',
        hue='series',
        estimator=np.mean
    )
    plt.title("Среднее время обучения моделей")
    plt.ylabel("Время (сек)")
    plt.xticks(rotation=45)
    plt.yscale('log')  # Логарифмическая шкала для наглядности
    plt.legend(title='Ряд')
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/training_time_comparison.png")
    plt.close()


def create_pdf_report(analysis_dir):
    """Создает PDF отчет со всеми результатами анализа с поддержкой кириллицы"""
    # Регистрация кириллического шрифта
    try:
        # Стандартный путь к Arial в Windows
        font_path = "C:\\Windows\\Fonts\\arial.ttf"
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('Arial', font_path))
            cyrillic_font = 'Arial'
        else:
            # Попробуем найти шрифт в текущей директории
            pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
            cyrillic_font = 'Arial'
    except:
        try:
            # Альтернативный шрифт для Linux
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            cyrillic_font = 'DejaVuSans'
        except:
            print("Предупреждение: Кириллический шрифт не найден. Используется Helvetica.")
            cyrillic_font = 'Helvetica'

    pdf_path = os.path.join(analysis_dir, "full_analysis_report.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        encoding='utf-8'
    )
    styles = getSampleStyleSheet()
    elements = []

    # Создаем стили с кириллическим шрифтом
    title_style = ParagraphStyle(
        name='Title',
        parent=styles['Heading1'],
        fontName=cyrillic_font,
        alignment=1,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        name='Subtitle',
        parent=styles['Heading2'],
        fontName=cyrillic_font,
        spaceBefore=12,
        spaceAfter=6
    )

    heading3_style = ParagraphStyle(
        name='Heading3',
        parent=styles['Heading3'],
        fontName=cyrillic_font,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        name='BodyText',
        parent=styles['BodyText'],
        fontName=cyrillic_font
    )

    # Заголовок отчета
    elements.append(Paragraph("Полный отчет анализа моделей", title_style))
    elements.append(Spacer(1, 0.25 * inch))

    # 1. Сводные результаты
    elements.append(Paragraph("Сводные результаты", subtitle_style))

    # Таблица summary.csv - ДОБАВЛЕНА ПРОВЕРКА И КОДИРОВКА
    summary_file = os.path.join(analysis_dir, "summary.csv")
    if os.path.exists(summary_file):
        try:
            summary_df = pd.read_csv(summary_file, sep=';', encoding='cp1251', decimal=',')
        except Exception as e:
            print(f"Ошибка чтения summary.csv: {e}")
            summary_df = pd.DataFrame()

        if not summary_df.empty:
            # Заменяем True/False на понятные значения
            if 'is_best' in summary_df.columns:
                summary_df['is_best'] = summary_df['is_best'].map({True: '✓', False: ''})

            # Форматируем заголовки
            headers = [
                "Серия", "Модель", "Средняя MAE", "Средняя RMSE",
                "Средняя sMAPE", "Среднее время обучения", "Лучшая модель"
            ]

            # Форматируем данные с проверкой столбцов
            formatted_data = [headers]
            for _, row in summary_df.iterrows():
                formatted_row = [
                    row.get('series', ''),
                    row.get('model', ''),
                    f"{row.get('test_MAE', 0):.4f}",
                    f"{row.get('test_RMSE', 0):.4f}",
                    f"{row.get('test_sMAPE', 0):.4f}%",
                    f"{row.get('train_time', 0):.2f} сек",
                    row.get('is_best', '')
                ]
                formatted_data.append(formatted_row)

            summary_table = Table(formatted_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 0), (-1, -1), cyrillic_font),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 0.2 * inch))

    # Графики сравнения - ДОБАВЛЕНА ПРОВЕРКА ФАЙЛОВ
    comparison_files = [
        "models_comparison_heatmap.png",
        "training_time_comparison.png"
    ]

    for img_file in comparison_files:
        img_path = os.path.join(analysis_dir, img_file)
        if os.path.exists(img_path):
            try:
                elements.append(Image(img_path, width=6 * inch, height=4 * inch))
                elements.append(Spacer(1, 0.1 * inch))
            except Exception as e:
                print(f"Ошибка загрузки изображения {img_path}: {e}")

    # 2. Результаты по каждому ряду - ДОБАВЛЕНА ОБРАБОТКА ОШИБОК
    for series in ['A', 'B', 'C']:
        try:
            elements.append(PageBreak())
            elements.append(Paragraph(f"Анализ ряда {series}", subtitle_style))
            series_dir = os.path.join(analysis_dir, f"series_{series}")

            # Лучшая модель - ДОБАВЛЕНА ПРОВЕРКА И КОДИРОВКА
            best_model_file = os.path.join(analysis_dir, f"best_model_{series}.csv")
            if os.path.exists(best_model_file):
                try:
                    best_df = pd.read_csv(best_model_file, sep=';', encoding='cp1251', decimal=',')
                except Exception as e:
                    print(f"Ошибка чтения best_model_{series}.csv: {e}")
                    best_df = pd.DataFrame()

                if not best_df.empty:
                    # Форматируем данные для таблицы
                    headers = best_df.columns.tolist()
                    header_translation = {
                        'series': 'Серия',
                        'model': 'Модель',
                        'test_MAE': 'MAE (тест)',
                        'test_RMSE': 'RMSE (тест)',
                        'test_sMAPE': 'sMAPE (тест)',
                        'train_time': 'Время обучения'
                    }
                    translated_headers = [header_translation.get(h, h) for h in headers]

                    # Преобразуем данные в строки
                    data = [translated_headers]
                    for _, row in best_df.iterrows():
                        row_data = []
                        for col in best_df.columns:
                            value = row[col]
                            # Форматируем числа
                            if isinstance(value, float):
                                if col.startswith('test_'):
                                    row_data.append(f"{value:.4f}")
                                elif col == 'train_time':
                                    row_data.append(f"{value:.2f} сек")
                                else:
                                    row_data.append(str(value))
                            else:
                                row_data.append(str(value))
                        data.append(row_data)

                    best_table = Table(data)
                    best_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 0), (-1, -1), cyrillic_font),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ]))
                    elements.append(Paragraph("Лучшая модель:", heading3_style))
                    elements.append(best_table)
                    elements.append(Spacer(1, 0.2 * inch))

            # Текстовый отчет - ДОБАВЛЕНА ПРОВЕРКА ФАЙЛА
            report_file = os.path.join(series_dir, "best_model_report.txt")
            if os.path.exists(report_file):
                try:
                    with open(report_file, 'r', encoding='cp1251') as f:
                        report_text = f.read()
                    elements.append(Paragraph("Детали модели:", heading3_style))
                    elements.append(Paragraph(report_text.replace('\n', '<br/>'), body_style))
                    elements.append(Spacer(1, 0.2 * inch))
                except Exception as e:
                    print(f"Ошибка чтения отчета {report_file}: {e}")

            # Графики ряда - ДОБАВЛЕНА ПРОВЕРКА ФАЙЛОВ
            series_plots = [
                f"top3_models_{series}.png",
                "metrics_comparison.png"
            ]

            for plot in series_plots:
                plot_path = os.path.join(series_dir, plot)
                if os.path.exists(plot_path):
                    try:
                        elements.append(Image(plot_path, width=6 * inch, height=4 * inch))
                        elements.append(Spacer(1, 0.1 * inch))
                    except Exception as e:
                        print(f"Ошибка загрузки изображения {plot_path}: {e}")
        except Exception as e:
            print(f"Ошибка при обработке ряда {series}: {e}")

    # Генерация PDF с обработкой ошибок
    try:
        doc.build(elements)
        print(f"Полный отчет сохранен: {pdf_path}")
    except Exception as e:
        print(f"Критическая ошибка при создании PDF: {e}")



def expand_metrics(df):
    """Разворачивает метрики из словарей в отдельные колонки"""
    # Разворачиваем валидационные метрики
    val_metrics = pd.json_normalize(df['val_metrics'])
    val_metrics.columns = ['val_' + col for col in val_metrics.columns]

    # Разворачиваем тестовые метрики
    test_metrics = pd.json_normalize(df['test_metrics'])
    test_metrics.columns = ['test_' + col for col in test_metrics.columns]

    # Разворачиваем статистические тесты (с проверкой наличия)
    stat_tests = pd.DataFrame()
    if 'stat_tests' in df.columns:
        stat_tests = pd.json_normalize(df['stat_tests'])
        if not stat_tests.empty:
            stat_tests.columns = ['stat_' + col for col in stat_tests.columns]

    # Собираем все вместе
    expanded_df = pd.concat([
        df[['series', 'model', 'train_time']],
        val_metrics,
        test_metrics,
        stat_tests
    ], axis=1)

    # Добавляем флаг ошибки с проверкой существования столбца
    expanded_df['error_occurred'] = df.get('error', pd.Series(None)).notna()

    return expanded_df
