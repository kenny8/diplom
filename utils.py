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

    # Загрузка данных
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

    # Отладочный вывод
    #print(f"Загружено строк: {len(df)}")
    #print(f"Первые строки:\n{df.head()}")

    # Конвертация даты
    df['date'] = df['month'].apply(_ru_month_to_number) + '-' + df['year'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%m-%Y')

    # Создание временного ряда
    ts = df.set_index('date')['value']
    ts.index.freq = 'MS'  # Monthly Start
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
        sep=';',           # Изменено с '\t' на ';'
        skiprows=skip_rows,
        header=None,
        usecols=[0, 1, 2],  # Берем только первые 3 колонки
        names=['year', 'month', 'value'],
        decimal=',',
        encoding='cp1251',
        engine='python'
    )

    #print(f"Загружено строк: {len(df)}")
    #print(f"Первые строки:\n{df.head()}")

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
        sep=';',           # Изменено с '\t' на ';'
        skiprows=skip_rows,
        header=None,
        usecols=[0, 1, 3],  # Берем первые 4 колонки
        names=['year', 'month', 'value'],
        decimal=',',
        encoding='cp1251',
        engine='python'
    )

    #print(f"Загружено строк: {len(df)}")
    #print(f"Первые строки:\n{df.head()}")


    df['date'] = df['month'].apply(_ru_month_to_number) + '-' + df['year'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%m-%Y')

    ts = df.set_index('date')['value']
    ts = ts.sort_index()

    # Удаление возможных дубликатов
    ts = ts[~ts.index.duplicated(keep='first')]

    # Установка частоты только если данные регулярные
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