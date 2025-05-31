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
        # Для Box-Cox требуется положительные значения
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

    # 1. Проверка стационарности
    stationary, p_value, adf_stat = check_stationarity(series)
    print(f"  - Стационарен: {'Да' if stationary else 'Нет'} (p-value={p_value:.4f}, ADF-stat={adf_stat:.2f})")

    # 2. Преобразования
    if name == 'A':
        # Для ряда A применяем логарифмирование
        transformed = transform_data(series, 'log')
        # Добавляем сдвиг, чтобы избежать отрицательных значений
        transformed = transformed - transformed.min() + 1e-6
        print("  - Применено логарифмическое преобразование со сдвигом")
    else:
        transformed = series.copy()

    # 3. Нормализация
    normalized, scaler = normalize_data(transformed)

    # 4. Разделение данных
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