# main.py
import time
import os
import pandas as pd
from config import DATA_PATHS
from utils import load_data
from preprocessing import preprocess_pipeline
from models import ModelFactory
from evaluation import Evaluator
from analysis import run_analysis, expand_metrics


os.makedirs("results", exist_ok=True)


print("Загрузка данных...")
series_a = load_data(DATA_PATHS['A'])
series_b = load_data(DATA_PATHS['B'])
series_c = load_data(DATA_PATHS['C'])


print("\nПредобработка данных...")
processed_a = preprocess_pipeline(series_a, 'A')
processed_b = preprocess_pipeline(series_b, 'B')
processed_c = preprocess_pipeline(series_c, 'C')


preprocessed_data = {
    'A': processed_a,
    'B': processed_b,
    'C': processed_c
}


model_factory = ModelFactory()


all_results = []


for series_name in ['A', 'B', 'C']:
    print(f"\n=== Обработка ряда {series_name} ===")
    data = preprocessed_data[series_name]
    models = model_factory.create_models_for_series(series_name)


    evaluator = Evaluator(scaler=data['scaler'])

    for model in models:
        try:
            print(f"\n--- Обучение модели: {model.name} ---")
            start_time = time.time()


            model.fit(data['train'])


            val_steps = len(data['val'])
            val_pred = model.predict(val_steps)


            test_steps = len(data['test'])
            test_pred = model.predict(test_steps)


            elapsed = time.time() - start_time


            val_metrics = evaluator.calculate_metrics(data['val'].values, val_pred)


            test_metrics = evaluator.calculate_metrics(data['test'].values, test_pred)


            residuals = None
            stat_tests = {}


            if hasattr(model, 'resid') and model.resid is not None and not model.resid.empty:
                residuals = model.resid
                stat_tests = evaluator.run_statistical_tests(residuals)


            result = {
                'series': series_name,
                'model': model.name,
                'train_time': elapsed,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'stat_tests': stat_tests,
                'error': None
            }
            all_results.append(result)


            plot_dir = f"results/{series_name}"
            os.makedirs(plot_dir, exist_ok=True)


            evaluator.plot_results(
                data['val'].values,
                val_pred,
                f"{model.name} (валидация)",
                f"{plot_dir}/{model.name}_val.png"
            )


            evaluator.plot_results(
                data['test'].values,
                test_pred,
                f"{model.name} (тест)",
                f"{plot_dir}/{model.name}_test.png"
            )


            if residuals is not None and not residuals.empty:
                evaluator.plot_residuals(
                    residuals,
                    model.name,
                    f"{plot_dir}/{model.name}_residuals.png"
                )


            if hasattr(model, 'trend') and hasattr(model, 'seasonal') and hasattr(model, 'resid'):
                evaluator.plot_components(
                    model.trend,
                    model.seasonal,
                    model.resid,
                    model.name,
                    f"{plot_dir}/{model.name}_components.png"
                )

            print(f"  - Готово! Время: {elapsed:.2f} сек")
            print(
                f"  - Метрики на валидации: MAE={val_metrics['MAE']:.2f}, RMSE={val_metrics['RMSE']:.2f}, sMAPE={val_metrics['sMAPE']:.2f}%")
            print(
                f"  - Метрики на тесте: MAE={test_metrics['MAE']:.2f}, RMSE={test_metrics['RMSE']:.2f}, sMAPE={test_metrics['sMAPE']:.2f}%")
            error = None
        except Exception as e:
            print(f"  Ошибка при работе с моделью {model.name}: {str(e)}")

            all_results.append({
                'series': series_name,
                'model': model.name,
                'error': str(e),
                'train_time': 0,
                'val_metrics': {},
                'test_metrics': {},
                'stat_tests': {},
                'error': str(e)
            })
            continue



results_df = pd.DataFrame(all_results)

expanded_results = expand_metrics(results_df)


expanded_results.to_csv(
    "results/all_metrics.csv",
    index=False,
    sep=';',
    decimal=',',
    encoding='cp1251'
)
print("\nВсе результаты сохранены в results/all_metrics.csv")



print("\nСводная таблица результатов:")
summary = results_df[['series', 'model', 'test_metrics']]
print(summary)


print("\nЗапуск анализа результатов...")
try:
    run_analysis(expanded_results)
    print("Анализ результатов успешно завершен!")
except Exception as e:
    print(f"Ошибка при анализе результатов: {str(e)}")

# utils.py
import pandas as pd
import os


def load_data(file_path: str) -> pd.Series:
    """Загружает данные из CSV-файла и возвращает временной ряд"""
    print(f"Попытка загрузить файл: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    if "GROUB_A" in file_path:
        return _load_group_a(file_path)
    elif "GROUB_B" in file_path:
        return _load_group_b(file_path)
    elif "GROUB_C" in file_path:
        return _load_group_c(file_path)
    else:
        raise ValueError("Unknown file type")


def _load_group_a(file_path: str) -> pd.Series:
    """Загрузка данных для розничной торговли (группа A)"""
    print("Загрузка данных группы A...")

    # Определение количества строк для пропуска
    with open(file_path, 'r', encoding='cp1251') as f:
        lines = f.readlines()
        skip_rows = 0
        for i, line in enumerate(lines):
            if any(char.isdigit() for char in line):
                skip_rows = i
                break
        print(f"Пропуск первых {skip_rows} строк")


    df = pd.read_csv(
        file_path,
        sep=';',           # Изменено с '\t' на ';'
        skiprows=skip_rows,
        header=None,
        usecols=[0, 1, 2],  # Берем только первые 3 колонки
        names=['year', 'month', 'value'],
        thousands=' ',
        decimal=',',
        encoding='cp1251',
        engine='python'
    )


    df['date'] = df['month'].apply(_ru_month_to_number) + '-' + df['year'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%m-%Y')


    ts = df.set_index('date')['value']
    ts.index.freq = 'MS'
    return ts


def _load_group_b(file_path: str) -> pd.Series:
    """Загрузка данных для добычи полезных ископаемых (группа B)"""
    print("Загрузка данных группы B...")

    with open(file_path, 'r', encoding='cp1251') as f:
        lines = f.readlines()
        skip_rows = 0
        for i, line in enumerate(lines):
            if any(char.isdigit() for char in line):
                skip_rows = i
                break

    df = pd.read_csv(
        file_path,
        sep=';',
        skiprows=skip_rows,
        header=None,
        usecols=[0, 1, 2],
        names=['year', 'month', 'value'],
        decimal=',',
        encoding='cp1251',
        engine='python'
    )


    df['date'] = df['month'].apply(_ru_month_to_number) + '-' + df['year'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%m-%Y')

    ts = df.set_index('date')['value']
    ts.index.freq = 'MS'
    return ts


def _load_group_c(file_path: str) -> pd.Series:
    """Загрузка данных для индекса потребительских цен (группа C)"""
    print("Загрузка данных группы C...")

    with open(file_path, 'r', encoding='cp1251') as f:
        lines = f.readlines()
        skip_rows = 0
        for i, line in enumerate(lines):
            if any(char.isdigit() for char in line):
                skip_rows = i
                break

    df = pd.read_csv(
        file_path,
        sep=';',
        skiprows=skip_rows,
        header=None,
        usecols=[0, 1, 3],
        names=['year', 'month', 'value'],
        decimal=',',
        encoding='cp1251',
        engine='python'
    )

    df['date'] = df['month'].apply(_ru_month_to_number) + '-' + df['year'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%m-%Y')

    ts = df.set_index('date')['value']
    ts = ts.sort_index()

    ts = ts[~ts.index.duplicated(keep='first')]

    if ts.index.inferred_freq == 'MS':
        ts.index.freq = pd.tseries.offsets.MonthBegin()
    return ts

def _ru_month_to_number(month: str) -> str:
    """Конвертирует русское название месяца в номер"""
    months = {
        'январь': '01',
        'февраль': '02',
        'март': '03',
        'апрель': '04',
        'май': '05',
        'июнь': '06',
        'июль': '07',
        'август': '08',
        'сентябрь': '09',
        'октябрь': '10',
        'ноябрь': '11',
        'декабрь': '12'
    }
    return months.get(month.lower(), '01')

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
import math
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

    analysis_dir = "analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Результаты анализа будут сохранены в папке: {analysis_dir}")
    analysis_df = results_df.copy(deep=True)
    print("Столбцы в results_df:", analysis_df.columns.tolist())

    best_models = find_best_models(analysis_df, analysis_dir)

    generate_summary_report(analysis_df, best_models, analysis_dir)
    for series in ['A', 'B', 'C']:
        analyze_series(analysis_df, series, best_models, analysis_dir)

    analyze_training_time(analysis_df, analysis_dir)

    create_pdf_report(analysis_dir)

    print("\nАнализ успешно завершен!")
    print(f"Все результаты сохранены в папке: {analysis_dir}")


def find_best_models(results_df, analysis_dir):
    """Определяет лучшие модели для каждого ряда по sMAPE"""
    best_models = {}

    for series in ['A', 'B', 'C']:
        series_results = results_df[results_df['series'] == series]

        valid_results = series_results[series_results['error_occurred'] == False]

        if not valid_results.empty:
            best_idx = valid_results['test_sMAPE'].idxmin()
            best_model = valid_results.loc[best_idx]
            best_models[series] = best_model

            best_model.to_csv(
                f"{analysis_dir}/best_model_{series}.csv",
                index=False,
                sep=';',
                decimal=',',
                encoding='cp1251'
            )

            print(f"Лучшая модель для ряда {series}: {best_model['model']} (sMAPE={best_model['test_sMAPE']:.2f}%)")

    return best_models


def generate_summary_report(results_df, best_models, analysis_dir):
    """Генерирует сводный отчет по всем моделям"""
    summary = results_df.groupby(['series', 'model']).agg({
        'test_MAE': 'mean',
        'test_RMSE': 'mean',
        'test_sMAPE': 'mean',
        'train_time': 'mean'
    }).reset_index()

    summary['is_best'] = False
    for series, model in best_models.items():
        summary.loc[
            (summary['series'] == series) &
            (summary['model'] == model['model']),
            'is_best'
        ] = True

    summary.to_csv(
        f"{analysis_dir}/summary.csv",
        index=False,
        float_format='%.4f',
        sep=';',
        decimal=',',
        encoding='cp1251'
    )

    plt.figure(figsize=(12, 8))
    pivot = summary.pivot_table(index='model', columns='series', values='test_sMAPE')
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Сравнение моделей по sMAPE (%)")
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/models_comparison_heatmap.png")
    plt.close()

    for series in ['A', 'B', 'C']:
        series_data = summary[summary.series == series].sort_values('test_sMAPE').head(3)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='model', y='test_sMAPE', data=series_data)
        plt.title(f'Топ-3 модели для ряда {series} (sMAPE)')
        plt.ylabel('sMAPE (%)')

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


    series_results = results_df[results_df['series'] == series]

    series_dir = f"{analysis_dir}/series_{series}"
    os.makedirs(series_dir, exist_ok=True)

    series_results.to_csv(
        f"{series_dir}/results.csv",
        index=False,
        float_format='%.4f',
        sep=';',
        decimal=',',
        encoding='cp1251'
    )

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

    if series in best_models:
        best_model = best_models[series]
        print(f"  - Лучшая модель: {best_model['model']}")

        with open(f"{series_dir}/best_model_report.txt", "w", encoding='cp1251') as f:
            f.write(f"Лучшая модель для ряда {series}: {best_model['model']}\n")
            f.write(f"Тестовые метрики:\n")
            f.write(f"  MAE:   {best_model['test_MAE']:.4f}\n")
            f.write(f"  RMSE:  {best_model['test_RMSE']:.4f}\n")
            f.write(f"  sMAPE: {best_model['test_sMAPE']:.4f}%\n")
            f.write(f"Время обучения: {best_model['train_time']:.2f} сек\n")
            lb_value = best_model.get('stat_Ljung-Box_pvalue')
            dw_value = best_model.get('stat_Durbin-Watson')
            if lb_value is not None and dw_value is not None \
                    and not math.isnan(lb_value) and not math.isnan(dw_value):
                f.write("\nСтатистические тесты остатков:\n")
                f.write(f"  Люнг-Бокс (p-value): {lb_value:.4f}\n")
                f.write(f"  Дарбин-Уотсон: {dw_value:.4f}\n")


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
    plt.yscale('log')
    plt.legend(title='Ряд')
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/training_time_comparison.png")
    plt.close()


def create_pdf_report(analysis_dir):
    """Создает PDF отчет со всеми результатами анализа с поддержкой кириллицы"""
    try:
        font_path = "C:\\Windows\\Fonts\\arial.ttf"
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('Arial', font_path))
            cyrillic_font = 'Arial'
        else:
            pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
            cyrillic_font = 'Arial'
    except:
        try:
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

    elements.append(Paragraph("Полный отчет анализа моделей", title_style))
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("Сводные результаты", subtitle_style))

    summary_file = os.path.join(analysis_dir, "summary.csv")
    if os.path.exists(summary_file):
        try:
            summary_df = pd.read_csv(summary_file, sep=';', encoding='cp1251', decimal=',')
        except Exception as e:
            print(f"Ошибка чтения summary.csv: {e}")
            summary_df = pd.DataFrame()

        if not summary_df.empty:
            if 'is_best' in summary_df.columns:
                summary_df['is_best'] = summary_df['is_best'].map({True: '+', False: ''})

            headers = [
                "Серия", "Модель", "Средняя MAE", "Средняя RMSE",
                "Средняя sMAPE", "Среднее время обучения", "Лучшая модель"
            ]

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

    for series in ['A', 'B', 'C']:
        try:
            elements.append(PageBreak())
            elements.append(Paragraph(f"Анализ ряда {series}", subtitle_style))
            series_dir = os.path.join(analysis_dir, f"series_{series}")

            best_model_file = os.path.join(analysis_dir, f"best_model_{series}.csv")
            if os.path.exists(best_model_file):
                try:
                    best_df = pd.read_csv(best_model_file, sep=';', encoding='cp1251', decimal=',')
                except Exception as e:
                    print(f"Ошибка чтения best_model_{series}.csv: {e}")
                    best_df = pd.DataFrame()

                if not best_df.empty:
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

                    data = [translated_headers]
                    for _, row in best_df.iterrows():
                        row_data = []
                        for col in best_df.columns:
                            value = row[col]
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

    try:
        doc.build(elements)
        print(f"Полный отчет сохранен: {pdf_path}")
    except Exception as e:
        print(f"Критическая ошибка при создании PDF: {e}")



def expand_metrics(df):
    """Разворачивает метрики из словарей в отдельные колонки"""
    val_metrics = pd.json_normalize(df['val_metrics'])
    val_metrics.columns = ['val_' + col for col in val_metrics.columns]

    test_metrics = pd.json_normalize(df['test_metrics'])
    test_metrics.columns = ['test_' + col for col in test_metrics.columns]

    stat_tests = pd.DataFrame()
    if 'stat_tests' in df.columns:
        stat_tests = pd.json_normalize(df['stat_tests'])
        if not stat_tests.empty:
            stat_tests.columns = ['stat_' + col for col in stat_tests.columns]

    expanded_df = pd.concat([
        df[['series', 'model', 'train_time']],
        val_metrics,
        test_metrics,
        stat_tests
    ], axis=1)

    expanded_df['error_occurred'] = df.get('error', pd.Series(None)).notna()

    return expanded_df

# models.py (исправления и дополнения)
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from statsmodels.tsa.api import ExponentialSmoothing, ARIMA, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.layers import SimpleRNN
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Отключаем интерактивный режим matplotlib
import matplotlib.pyplot as plt

class BaseModel(ABC):
    """Абстрактный базовый класс для всех моделей"""

    def __init__(self, name):
        self.name = name
        self.is_fitted = False
        self.resid = None
        self.trend = None
        self.seasonal = None

    @abstractmethod
    def fit(self, train_data):
        pass

    @abstractmethod
    def predict(self, steps):
        pass


class AdditiveComponentModel(BaseModel):
    """Аддитивная компонентная модель"""

    def __init__(self):
        super().__init__("Additive Component")

    def fit(self, train_data):
        decomposition = seasonal_decompose(
            train_data,
            model='additive',
            period=12
        )

        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.resid = decomposition.resid
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            raise Exception("Model not fitted yet")

        last_trend = self.trend.dropna().iloc[-1]
        seasonal_component = self.seasonal.iloc[-12:].values

        forecast = [last_trend + seasonal_component[i % 12] for i in range(steps)]

        return np.array(forecast)


class MultiplicativeComponentModel(BaseModel):
    """Мультипликативная компонентная модель"""

    def __init__(self):
        super().__init__("Multiplicative Component")

    def fit(self, train_data):
        if (train_data <= 0).any():
            train_data = train_data - train_data.min() + 1e-6

        decomposition = seasonal_decompose(
            train_data,
            model='multiplicative',
            period=12
        )

        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.resid = decomposition.resid
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            raise Exception("Model not fitted yet")

        last_trend = self.trend.dropna().iloc[-1]
        seasonal_component = self.seasonal.iloc[-12:].values

        forecast = [last_trend * seasonal_component[i % 12] for i in range(steps)]
        return np.array(forecast)


class SARIMAModel(BaseModel):
    """SARIMA модель для ряда A"""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), name="SARIMA"):
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, train_data):
        train_data = train_data.dropna().values
        try:
            self.model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            self.is_fitted = True
        except Exception as e:
            print(f"  - Ошибка при обучении SARIMA: {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)
        return self.model.forecast(steps)


class HoltWintersAdditive(BaseModel):
    """Аддитивная модель Холта-Винтерса для ряда A"""

    def __init__(self):
        super().__init__("Holt-Winters Additive")

    def fit(self, train_data):
        try:
            self.model = ExponentialSmoothing(
                train_data.values,
                trend='add',
                seasonal='add',
                seasonal_periods=12
            ).fit()
            self.is_fitted = True
        except Exception as e:
            print(f"  - Ошибка при обучении Holt-Winters: {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)
        return self.model.forecast(steps)


class LSTMModel(BaseModel):
    """LSTM модель для ряда A"""

    def __init__(self, units=64, epochs=100, window_size=12, batch_size=32, name="LSTM"):
        super().__init__(name)
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(self.window_size, 1)),
            LSTM(units),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.last_window = None

    def fit(self, train_data):
        train_values = train_data.values
        X, y = [], []
        for i in range(len(train_values) - self.window_size):
            X.append(train_values[i:i + self.window_size])
            y.append(train_values[i + self.window_size])

        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.last_window = train_values[-self.window_size:]
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        current_input = self.last_window.copy().reshape(1, self.window_size, 1)
        predictions = []

        for _ in range(steps):
            next_pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1, 0] = next_pred

        return np.array(predictions)


class MixedComponentModel(BaseModel):
    """Смешанная компонентная модель (тренд * сезонность + остатки)"""

    def __init__(self):
        super().__init__("Mixed Component")

    def fit(self, train_data):
        if (train_data <= 0).any():
            train_data = train_data - train_data.min() + 1e-6

        decomposition = seasonal_decompose(
            train_data,
            model='multiplicative',
            period=12
        )

        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.resid = decomposition.resid
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            raise Exception("Model not fitted yet")

        last_trend = self.trend.dropna().iloc[-1]
        seasonal_component = self.seasonal.iloc[-12:].values

        forecast = [
            (last_trend * seasonal_component[i % 12]) + self.resid.mean()
            for i in range(steps)
        ]
        return np.array(forecast)


class TheilSenModel(BaseModel):
    """Процесс Тейла-Вейджа с адаптацией для временных рядов"""

    def __init__(self, window_size=24, name="Theil-Sen"):
        super().__init__(name)
        self.window_size = window_size

    def fit(self, train_data):
        self.last_values = train_data.values[-self.window_size:]
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        X = np.arange(len(self.last_values)).reshape(-1, 1)
        y = self.last_values

        model = LinearRegression()
        model.fit(X, y)

        last_idx = len(self.last_values) - 1
        return np.array([
            model.predict([[last_idx + i]])[0]
            for i in range(1, steps + 1)
        ])


class NaiveModel(BaseModel):
    """Наивная модель (прогноз = последнее значение)"""

    def __init__(self):
        super().__init__("Naive Forecast")
        self.last_value = None

    def fit(self, train_data):
        self.last_value = train_data.iloc[-1]
        self.is_fitted = True

    def predict(self, steps):
        return np.array([self.last_value] * steps)


class ARIMAModel(BaseModel):
    """ARIMA модель для ряда B"""

    def __init__(self, order=(2,1,1), name="ARIMA"):
        super().__init__(name)
        self.order=order

    def fit(self, train_data):
        try:
            train_values = train_data.dropna().values
            self.model = ARIMA(train_values, order=self.order).fit()
            self.is_fitted = True
        except Exception as e:
            print(f"  - Ошибка при обучении ARIMA: {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)
        return self.model.forecast(steps)


class HoltModel(BaseModel):
    """Модель Хольта с затуханием для ряда B"""

    def __init__(self, damped=True):
        name = "Holt Damped" if damped else "Holt"
        super().__init__(name)
        self.damped = damped

    def fit(self, train_data):
        try:
            self.model = ExponentialSmoothing(
                train_data.values,
                trend='add',
                damped_trend=self.damped
            ).fit()
            self.is_fitted = True
        except Exception as e:
            print(f"  - Ошибка при обучении Holt: {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)
        return self.model.forecast(steps)


class AdaptiveSESModel(BaseModel):
    """SES с адаптивным параметром сглаживания"""

    def __init__(self, initial_smoothing=0.3, alpha=0.5, name="Adaptive SES"):
        super().__init__(name)
        self.initial_alpha = alpha
        self.initial_smoothing = initial_smoothing
        self.alpha = alpha

    def fit(self, train_data):
        self.train = train_data.values
        self.is_fitted = True

        self.alpha = (self.initial_alpha + self.initial_smoothing) / 2

        if len(self.train) > 12:
            errors = []
            for i in range(1, len(self.train)):
                pred = self.train[i - 1]
                error = self.train[i] - pred
                errors.append(abs(error))

            error_ratio = np.mean(errors) / np.max(errors) if np.max(errors) != 0 else 0.5
            adaptive_alpha = 1 - error_ratio
            self.alpha = min(0.99, max(0.01, (self.initial_smoothing * 0.3 + adaptive_alpha * 0.7)))

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        predictions = []
        last = self.train[-1]
        for _ in range(steps):
            predictions.append(last)
            last = self.alpha * self.train[-1] + (1 - self.alpha) * last
        return np.array(predictions)


class RecursiveLeastSquares(BaseModel):
    """Рекуррентный МНК с ограничением сложности"""

    def __init__(self, degree=2, forgetting_factor=0.99, name="Recursive Least Squares"):
        super().__init__(name)
        self.degree = degree
        self.forgetting_factor = forgetting_factor

    def fit(self, train_data):
        n = len(train_data)
        X = np.column_stack([
            np.power(np.arange(n), d)
            for d in range(self.degree + 1)
        ])
        y = train_data.values

        self.coef_ = np.zeros(self.degree + 1)
        P = np.eye(self.degree + 1) * 100

        for i in range(n):
            x_i = X[i]
            K = P @ x_i / (self.forgetting_factor + x_i.T @ P @ x_i)
            error = y[i] - x_i.T @ self.coef_
            self.coef_ += K * error
            P = (P - np.outer(K, x_i.T @ P)) / self.forgetting_factor

        self.is_fitted = True
        self.last_index = n - 1

    def predict(self, steps):
        return np.array([
            np.sum([coef * np.power(self.last_index + i + 1, d)
                    for d, coef in enumerate(self.coef_)])
            for i in range(steps)
        ])



class ARMAModel(BaseModel):
    """ARMA модель для ряда C"""

    def __init__(self, order=(1,0,1), name="ARMA"):
        super().__init__(name)
        self.order=order

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=self.order).fit()
        self.resid = self.model.resid
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps).values


class SESModel(BaseModel):
    """Простое экспоненциальное сглаживание для ряда C"""

    def __init__(self, smoothing_level=0.5, name="Simple Exponential Smoothing"):
        super().__init__(name)
        self.smoothing_level = smoothing_level
        self.model = None
        self.is_fitted = False

    def fit(self, train_data):
        try:
            self.model = SimpleExpSmoothing(train_data.values).fit(
                smoothing_level=self.smoothing_level,
                optimized=False
            )
            self.is_fitted = True
        except ValueError as e:
            if "must have at least one observation" in str(e):
                self.model = None
                self.is_fitted = False
                print(f"  - Предупреждение: ряд слишком короткий для SES. Используйте минимум 1 точку данных")
            else:
                print(f"  - Ошибка при обучении SES: {str(e)}")
                self.is_fitted = False
        except Exception as e:
            print(f"  - Неожиданная ошибка при обучении SES: {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted or self.model is None:
            return np.zeros(steps)

        try:
            return self.model.forecast(steps)
        except Exception as e:
            print(f"  - Ошибка при прогнозировании SES: {str(e)}")
            return np.zeros(steps)


class GRUModel(BaseModel):
    """GRU модель для ряда C"""

    def __init__(self, units=64, epochs=100, window_size=12, batch_size=32, name="GRU"):
        super().__init__(name)
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential([
            GRU(units, input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.last_window = None

    def fit(self, train_data):
        train_values = train_data.values
        X, y = [], []
        for i in range(len(train_values) - self.window_size):
            X.append(train_values[i:i + self.window_size])
            y.append(train_values[i + self.window_size])

        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                       verbose=0)
        self.last_window = train_values[-self.window_size:]
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        current_input = self.last_window.copy().reshape(1, self.window_size, 1)
        predictions = []

        for _ in range(steps):
            next_pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1, 0] = next_pred

        return np.array(predictions)

class ARModel(BaseModel):
    """AR(p) модель для стационарных рядов"""

    def __init__(self, p=1, name="AR"):
        super().__init__(name)
        self.p = p

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=(self.p, 0, 0)).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps).values


class MAModel(BaseModel):
    """MA(q) модель для стационарных рядов"""

    def __init__(self, q=1, name="MA"):
        super().__init__(name)
        self.q = q

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=(0, 0, self.q)).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps).values


class VanillaRNN(BaseModel):
    """Простая RNN модель для ряда C"""

    def __init__(self, units=64, epochs=100, window_size=12, batch_size=32, name="Vanilla RNN"):
        super().__init__(name)
        self.window_size = window_size
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential([
            SimpleRNN(units, input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.last_window = None
        self.is_fitted = False

    def fit(self, train_data):
        train_values = train_data.values
        X, y = [], []
        for i in range(len(train_values) - self.window_size):
            X.append(train_values[i:i + self.window_size])
            y.append(train_values[i + self.window_size])

        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.last_window = train_values[-self.window_size:]
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        current_input = self.last_window.copy().reshape(1, self.window_size, 1)
        predictions = []

        for _ in range(steps):
            next_pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1, 0] = next_pred

        return np.array(predictions)


class ModelFactory:
    @staticmethod
    def create_models_for_series(series_name):
        if series_name == 'A':
            return [
                AdditiveComponentModel(),
                MultiplicativeComponentModel(),
                MixedComponentModel(),  #
                SARIMAModel(order=(1, 1, 2), seasonal_order=(1, 0, 1, 12)),
                HoltWintersAdditive(),
                TheilSenModel(window_size=60),
                LSTMModel(units=128, epochs=24, window_size=150, batch_size=32),
                GRUModel(units=16, epochs=24, window_size=150, batch_size=16),
                NaiveModel()
            ]

        elif series_name == 'B':
            return [
                ARIMAModel(order=(1, 1, 0), name="ARIMA"),
                HoltModel(damped=True),
                HoltModel(damped=False),
                AdaptiveSESModel(initial_smoothing=0.1, alpha=0.05),
                RecursiveLeastSquares(degree=1, forgetting_factor=0.95),
                LSTMModel(units=128, epochs=24, window_size=50, batch_size=16),
                GRUModel(units=48, epochs=16, window_size=100, batch_size=64),
                NaiveModel()
            ]

        elif series_name == 'C':
            return [
                ARMAModel(order=(4, 0, 3)),
                ARModel(p=1),
                MAModel(q=4),
                SESModel(smoothing_level=0.2),
                VanillaRNN(units=64, epochs=50, window_size=18, batch_size=16),
                GRUModel(units=64, epochs=12, window_size=24, batch_size=16),
                LSTMModel(units=16, epochs=200, window_size=12, batch_size=32),
                NaiveModel()
            ]
        else:
            raise ValueError(f"Unknown series name: {series_name}")

# preprocessing.py
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> tuple:
    """Проверяет стационарность ряда с помощью ADF-теста

    Возвращает:
        (is_stationary, p_value, adf_statistic)
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    adf_stat = result[0]
    return (p_value < alpha, p_value, adf_stat)


def transform_data(series: pd.Series, method: str = None) -> pd.Series:
    """Применяет преобразование к ряду"""
    if method == 'log':
        return np.log(series)
    elif method == 'sqrt':
        return np.sqrt(series)
    elif method == 'boxcox':
        from scipy import stats
        transformed, _ = stats.boxcox(series[series > 0] + 1e-6)
        return pd.Series(transformed, index=series.index)
    return series


def normalize_data(series: pd.Series) -> tuple:
    """Нормализует ряд с помощью MinMaxScaler

    Возвращает:
        (нормализованный_ряд, scaler)
    """
    scaler = MinMaxScaler()
    values = series.values.reshape(-1, 1)
    normalized = scaler.fit_transform(values)
    return pd.Series(normalized.flatten(), index=series.index), scaler


def split_data(series: pd.Series, ratios: tuple = (0.7, 0.15, 0.15)) -> tuple:
    """Разделяет ряд на train, validation и test"""
    n = len(series)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]

    return train, val, test


def preprocess_pipeline(series: pd.Series, name: str) -> dict:
    """Полный пайплайн предобработки для временного ряда"""
    print(f"\nПредобработка ряда {name}")


    stationary, p_value, adf_stat = check_stationarity(series)
    print(f"  - Стационарен: {'Да' if stationary else 'Нет'} (p-value={p_value:.4f}, ADF-stat={adf_stat:.2f})")


    if name == 'A':
        transformed = transform_data(series, 'log')
        transformed = transformed - transformed.min() + 1e-6
        print("  - Применено логарифмическое преобразование со сдвигом")
    else:
        transformed = series.copy()

    normalized, scaler = normalize_data(transformed)

    train, val, test = split_data(normalized)
    print(f"  - Разделение данных: train={len(train)} ({len(train) / len(series):.0%}), "
          f"val={len(val)} ({len(val) / len(series):.0%}), "
          f"test={len(test)} ({len(test) / len(series):.0%})")

    return {
        'original': series,
        'transformed': transformed,
        'normalized': normalized,
        'train': train,
        'val': val,
        'test': test,
        'scaler': scaler,
        'stationary': stationary,
        'p_value': p_value,
        'adf_stat': adf_stat
    }

# evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


class Evaluator:
    def __init__(self, scaler=None):
        self.scaler = scaler

    def calculate_metrics(self, y_true, y_pred):
        """Вычисляет метрики качества прогноза"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        if self.scaler:
            y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

        return {
            'MAE': mae,
            'RMSE': rmse,
            'sMAPE': smape
        }

    def run_statistical_tests(self, residuals):
        """Выполняет статистические тесты на остатках"""
        residuals = np.asarray(residuals)
        if len(residuals) == 0:
            return {}

        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].values[0]

        dw_stat = durbin_watson(residuals)

        return {
            'Ljung-Box_pvalue': lb_pvalue,
            'Durbin-Watson': dw_stat
        }

    def plot_results(self, y_true, y_pred, model_name, save_path=None):
        """Визуализирует фактические и прогнозные значения"""
        try:
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            # Проверка данных
            if y_true.size == 0 or y_pred.size == 0:
                print(f"  - Ошибка визуализации для {model_name}: нет данных для построения графика")
                return

            if len(y_true) != len(y_pred):
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

            plt.figure(figsize=(12, 6))

            x_vals = np.arange(len(y_true))

            plt.plot(x_vals, y_true, label='Фактические значения', color='blue')
            plt.plot(x_vals, y_pred, label='Прогноз', color='red', linestyle='--')

            plt.title(f'Факт vs Прогноз: {model_name}')
            plt.xlabel('Период')
            plt.ylabel('Значение')
            plt.legend()
            plt.grid(True)

            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"  - Ошибка при построении графика для {model_name}: {str(e)}")

    def plot_residuals(self, residuals, model_name, save_path=None):
        """Визуализирует остатки модели"""
        try:
            residuals = np.asarray(residuals)
            if residuals.size == 0:
                return

            plt.figure(figsize=(12, 6))
            plt.plot(residuals)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title(f'Остатки модели: {model_name}')
            plt.xlabel('Период')
            plt.ylabel('Ошибка')
            plt.grid(True)

            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"  - Ошибка при построении остатков для {model_name}: {str(e)}")

    def plot_components(self, trend, seasonal, residual, model_name, save_path=None):
        """Визуализирует компоненты временного ряда (если они доступны)"""
        try:
            if trend is None or seasonal is None or residual is None:
                return

            trend = np.asarray(trend)
            seasonal = np.asarray(seasonal)
            residual = np.asarray(residual)

            if len(trend) == 0 or len(seasonal) == 0 or len(residual) == 0:
                return

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))

            ax1.plot(trend)
            ax1.set_title('Тренд')
            ax1.grid(True)

            ax2.plot(seasonal)
            ax2.set_title('Сезонность')
            ax2.grid(True)

            ax3.plot(residual)
            ax3.axhline(y=0, color='r', linestyle='-')
            ax3.set_title('Остатки')
            ax3.grid(True)

            ax4.plot(trend + seasonal + residual)
            ax4.set_title('Тренд + Сезонность + Остатки')
            ax4.grid(True)

            plt.suptitle(f'Декомпозиция ряда: {model_name}')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"  - Ошибка при построении компонентов для {model_name}: {str(e)}")

# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATHS = {
    'A': os.path.join(BASE_DIR, 'data', 'GROUB_A.csv'),
    'B': os.path.join(BASE_DIR, 'data', 'GROUB_B.csv'),
    'C': os.path.join(BASE_DIR, 'data', 'GROUB_C.csv')
}


PREPROCESSING = {
    'split_ratios': (0.7, 0.15, 0.15),
    'stationarity_alpha': 0.05,
    'random_seed': 42
}