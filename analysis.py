# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import matplotlib.gridspec as gridspec


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

    # 1. Определение лучших моделей для каждого ряда
    best_models = find_best_models(results_df, analysis_dir)

    # 2. Сводный отчет по всем моделям
    generate_summary_report(results_df, best_models, analysis_dir)

    # 3. Детальный анализ для каждого ряда
    for series in ['A', 'B', 'C']:
        analyze_series(results_df, series, best_models, analysis_dir)

    # 4. Сравнение времени обучения
    analyze_training_time(results_df, analysis_dir)

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
            best_model.to_csv(f"{analysis_dir}/best_model_{series}.csv", index=False)

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
    summary.to_csv(f"{analysis_dir}/summary.csv", index=False, float_format='%.4f')

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
    series_results.to_csv(f"{series_dir}/results.csv", index=False, float_format='%.4f')

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
        with open(f"{series_dir}/best_model_report.txt", "w") as f:
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
    """Создает PDF отчет со всеми результатами (заглушка)"""
    # В реальной реализации здесь можно использовать ReportLab или PyPDF2
    # Для простоты создаем текстовый файл с инструкциями
    with open(f"{analysis_dir}/README.txt", "w") as f:
        f.write("ОТЧЕТ ОБ АНАЛИЗЕ РЕЗУЛЬТАТОВ\n")
        f.write("=============================\n\n")
        f.write("Этот каталог содержит полные результаты анализа моделей.\n\n")
        f.write("Ключевые файлы:\n")
        f.write("- summary.csv: Сводная таблица по всем моделям\n")
        f.write("- best_model_*.csv: Информация о лучших моделях для каждого ряда\n")
        f.write("- models_comparison_heatmap.png: Тепловая карта сравнения моделей\n")
        f.write("- top3_models_*.png: Топ-3 модели для каждого ряда\n")
        f.write("- training_time_comparison.png: Сравнение времени обучения\n\n")
        f.write("Для каждого ряда создана отдельная папка (series_*) с детальными результатами.\n")


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
