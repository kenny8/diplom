# config.py
import os

# Пути к данным
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATHS = {
    'A': os.path.join(BASE_DIR, 'data', 'GROUB_A.csv'),
    'B': os.path.join(BASE_DIR, 'data', 'GROUB_B.csv'),
    'C': os.path.join(BASE_DIR, 'data', 'GROUB_C.csv')
}



# Параметры предобработки
PREPROCESSING = {
    'split_ratios': (0.7, 0.15, 0.15),  # train, val, test
    'stationarity_alpha': 0.05,           # уровень значимости для ADF-теста
    'random_seed': 42                     # для воспроизводимости
}