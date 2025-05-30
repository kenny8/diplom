# main.py
from config import DATA_PATHS
from utils import load_data
from preprocessing import preprocess_pipeline
import pandas as pd

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

# Сохранение результатов предобработки для дальнейшего использования
preprocessed_data = {
    'A': processed_a,
    'B': processed_b,
    'C': processed_c
}

# Вывод сводки
print("\nСводка по предобработанным данным:")
for name, data in preprocessed_data.items():
    print(f"Ряд {name}:")
    print(f"  - Оригинальный ряд: {len(data['original'])} записей")
    print(f"  - Train: {len(data['train'])} записей")
    print(f"  - Validation: {len(data['val'])} записей")
    print(f"  - Test: {len(data['test'])} записей")
    print(f"  - Стационарен: {'Да' if data['stationary'] else 'Нет'}")