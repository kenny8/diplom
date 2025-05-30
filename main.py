# main.py
from config import DATA_PATHS
from utils import load_data

# Загрузка данных
series_a = load_data(DATA_PATHS['A'])
series_b = load_data(DATA_PATHS['B'])
series_c = load_data(DATA_PATHS['C'])

# Проверка
print(f"Ряд A: {len(series_a)} записей, пример:\n{series_a.head()}")
print(f"Ряд B: {len(series_b)} записей, пример:\n{series_b.head()}")
print(f"Ряд C: {len(series_c)} записей, пример:\n{series_c.head()}")