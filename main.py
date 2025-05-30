# main.py
import time
import os
import pandas as pd
from config import DATA_PATHS, MODEL_CONFIG
from utils import load_data
from preprocessing import preprocess_pipeline
from models import ModelFactory
from evaluation import Evaluator

# Создаем директорию для результатов
os.makedirs("results", exist_ok=True)

# Загрузка данных
print("Загрузка данных...")
series_a = load_data(DATA_PATHS['A'])
series_b = load_data(DATA_PATHS['B'])
series_c = load_data(DATA_PATHS['C'])

# Предобработка данных
print("\nПредобработка данных...")
processed_a = preprocess_pipeline(series_a, 'A')
processed_b = preprocess_pipeline(series_b, 'B')
processed_c = preprocess_pipeline(series_c, 'C')

# Словарь с предобработанными данными
preprocessed_data = {
    'A': processed_a,
    'B': processed_b,
    'C': processed_c
}

# Создаем фабрику моделей
model_factory = ModelFactory()

# Для хранения результатов
all_results = []

# Для каждого ряда обучаем и оцениваем модели
for series_name in ['A', 'B', 'C']:
    print(f"\n=== Обработка ряда {series_name} ===")
    data = preprocessed_data[series_name]
    models = model_factory.create_models_for_series(series_name)

    # Создаем оценщик с scaler для обратного преобразования
    evaluator = Evaluator(scaler=data['scaler'])

    for model in models:
        print(f"\n--- Обучение модели: {model.name} ---")
        start_time = time.time()

        # Обучение модели
        model.fit(data['train'])

        # Прогноз на валидационной выборке
        val_steps = len(data['val'])
        val_pred = model.predict(val_steps)

        # Прогноз на тестовой выборке
        test_steps = len(data['test'])
        test_pred = model.predict(test_steps)

        # Время выполнения
        elapsed = time.time() - start_time

        # Оценка на валидационной выборке
        val_metrics = evaluator.calculate_metrics(data['val'].values, val_pred)

        # Оценка на тестовой выборке
        test_metrics = evaluator.calculate_metrics(data['test'].values, test_pred)

        # Статистические тесты на остатках (если доступны)
        residuals = []
        if hasattr(model, 'resid'):
            residuals = model.resid
            stat_tests = evaluator.run_statistical_tests(residuals)
        else:
            stat_tests = {}

        # Сохранение результатов
        result = {
            'series': series_name,
            'model': model.name,
            'train_time': elapsed,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'stat_tests': stat_tests
        }
        all_results.append(result)

        # Визуализация результатов
        plot_dir = f"results/{series_name}"
        os.makedirs(plot_dir, exist_ok=True)

        # График прогноза на валидации
        evaluator.plot_results(
            data['val'].values,
            val_pred,
            f"{model.name} (валидация)",
            f"{plot_dir}/{model.name}_val.png"
        )

        # График прогноза на тесте
        evaluator.plot_results(
            data['test'].values,
            test_pred,
            f"{model.name} (тест)",
            f"{plot_dir}/{model.name}_test.png"
        )

        # График остатков
        if residuals:
            evaluator.plot_residuals(
                residuals,
                model.name,
                f"{plot_dir}/{model.name}_residuals.png"
            )

        # График компонентов (если доступны)
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

# Сохранение всех результатов в CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("results/all_metrics.csv", index=False)
print("\nВсе результаты сохранены в results/all_metrics.csv")

# Вывод сводной таблицы
print("\nСводная таблица результатов:")
summary = results_df[['series', 'model', 'test_metrics']]
print(summary)