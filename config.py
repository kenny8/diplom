# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATHS = {
    'A': os.path.join(BASE_DIR, 'data', 'GROUB_A.csv'),
    'B': os.path.join(BASE_DIR, 'data', 'GROUB_B.csv'),
    'C': os.path.join(BASE_DIR, 'data', 'GROUB_C.csv')
}