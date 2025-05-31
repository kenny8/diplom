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
import warnings

# Игнорируем предупреждения
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class BaseModel(ABC):
    """Абстрактный базовый класс для всех моделей"""

    def __init__(self, name):
        self.name = name
        self.is_fitted = False
        self.resid = None  # остатки модели
        self.trend = None  # трендовая компонента
        self.seasonal = None  # сезонная компонента

    @abstractmethod
    def fit(self, train_data):
        pass

    @abstractmethod
    def predict(self, steps):
        pass


# === РЯД A (Сезонность + Тренд) ===

class AdditiveComponentModel(BaseModel):
    """Аддитивная компонентная модель"""

    def __init__(self):
        super().__init__("Additive Component")

    def fit(self, train_data):
        # Декомпозиция ряда
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

        # Исправлено: использование iloc вместо индексации
        last_trend = self.trend.dropna().iloc[-1]
        seasonal_component = self.seasonal.iloc[-12:].values

        # Прогноз на steps периодов
        forecast = [last_trend + seasonal_component[i % 12] for i in range(steps)]

        return np.array(forecast)


class MultiplicativeComponentModel(BaseModel):
    """Мультипликативная компонентная модель"""

    def __init__(self):
        super().__init__("Multiplicative Component")

    def fit(self, train_data):
        # Проверяем на отрицательные значения
        if (train_data <= 0).any():
            train_data = train_data - train_data.min() + 1e-6

        # Декомпозиция ряда
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

    def __init__(self):
        super().__init__("SARIMA")

    def fit(self, train_data):
        # Преобразуем в numpy array и убедимся в отсутствии NaN
        train_data = train_data.dropna().values

        try:
            self.model = SARIMAX(
                train_data,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
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
                train_data.values,  # Используем values вместо Series
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

    def __init__(self):
        super().__init__("LSTM")
        self.window_size = 12
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.window_size, 1)),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.last_window = None

    def fit(self, train_data):
        # Используем iloc для безопасного доступа
        train_values = train_data.values
        X, y = [], []
        for i in range(len(train_values) - self.window_size):
            X.append(train_values[i:i + self.window_size])
            y.append(train_values[i + self.window_size])

        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)

        self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
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
        # Декомпозиция ряда
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

        # Смешанная модель: (тренд * сезонность) + остаток
        forecast = [
            (last_trend * seasonal_component[i % 12]) + self.resid.mean()
            for i in range(steps)
        ]
        return np.array(forecast)


class TheilSenModel(BaseModel):
    """Процесс Тейла-Вейджа с адаптацией для временных рядов"""

    def __init__(self, window_size=24):
        super().__init__("Theil-Sen")
        self.window_size = window_size

    def fit(self, train_data):
        self.last_values = train_data.values[-self.window_size:]
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        # Используем только последние N точек для прогноза
        X = np.arange(len(self.last_values)).reshape(-1, 1)
        y = self.last_values

        # Модель линейной регрессии с робастной оценкой
        model = LinearRegression()
        model.fit(X, y)

        # Прогноз на следующие steps шагов
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

# === РЯД B (Тренд без сезонности) ===

class ARIMAModel(BaseModel):
    """ARIMA модель для ряда B"""

    def __init__(self):
        super().__init__("ARIMA")

    def fit(self, train_data):
        try:
            # Преобразуем в numpy array
            train_values = train_data.dropna().values
            self.model = ARIMA(train_values, order=(2, 1, 1)).fit()
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
                train_data.values,  # Используем values вместо Series
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

    def __init__(self):
        super().__init__("Adaptive SES")
        self.alpha = 0.5

    def fit(self, train_data):
        self.train = train_data.values
        self.is_fitted = True

        # Адаптация alpha на основе последних ошибок
        if len(self.train) > 12:
            errors = []
            for i in range(1, len(self.train)):
                pred = self.train[i - 1]
                error = self.train[i] - pred
                errors.append(abs(error))
            self.alpha = min(0.99, max(0.01, 1 - np.mean(errors) / np.max(errors)))

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        predictions = []
        last = self.train[-1]
        for _ in range(steps):
            predictions.append(last)
            last = self.alpha * last + (1 - self.alpha) * last
        return np.array(predictions)


class RecursiveLeastSquares(BaseModel):
    """Рекуррентный МНК с ограничением сложности"""

    def __init__(self, degree=2, forgetting_factor=0.99):
        super().__init__("Recursive Least Squares")
        self.degree = degree
        self.forgetting_factor = forgetting_factor

    def fit(self, train_data):
        n = len(train_data)
        X = np.column_stack([
            np.power(np.arange(n), d)
            for d in range(self.degree + 1)
        ])
        y = train_data.values

        # Инициализация рекуррентной оценки
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


# === РЯД C (Стационарный ряд) ===

class ARMAModel(BaseModel):
    """ARMA модель для ряда C"""

    def __init__(self):
        super().__init__("ARMA")

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=(1, 0, 1)).fit()
        self.resid = self.model.resid
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps).values


class SESModel(BaseModel):
    """Простое экспоненциальное сглаживание для ряда C"""

    def __init__(self):
        super().__init__("Simple Exponential Smoothing")

    def fit(self, train_data):
        try:
            self.model = SimpleExpSmoothing(train_data.values).fit()  # Используем values
            self.is_fitted = True
        except Exception as e:
            print(f"  - Ошибка при обучении SES: {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)
        return self.model.forecast(steps)


class GRUModel(BaseModel):
    """GRU модель для ряда C"""

    def __init__(self):
        super().__init__("GRU")
        self.window_size = 12
        self.model = Sequential([
            GRU(32, input_shape=(self.window_size, 1)),
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

        self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
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

    def __init__(self, p=1):
        super().__init__(f"AR({p})")
        self.p = p

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=(self.p, 0, 0)).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps).values


class MAModel(BaseModel):
    """MA(q) модель для стационарных рядов"""

    def __init__(self, q=1):
        super().__init__(f"MA({q})")
        self.q = q

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=(0, 0, self.q)).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps).values


class VanillaRNN(BaseModel):
    """Простая RNN модель для ряда C"""

    def __init__(self, units=32):
        super().__init__(f"Vanilla RNN")
        self.window_size = 12
        self.units = units
        self.model = Sequential([
            SimpleRNN(units, input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, train_data):
        train_values = train_data.values
        X, y = [], []
        for i in range(len(train_values) - self.window_size):
            X.append(train_values[i:i + self.window_size])
            y.append(train_values[i + self.window_size])

        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)

        self.model.fit(X, y, epochs=30, batch_size=16, verbose=0)
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


# Фабрика моделей для удобного создания
class ModelFactory:
    @staticmethod
    def create_models_for_series(series_name):
        if series_name == 'A':
            return [
                AdditiveComponentModel(),
                MultiplicativeComponentModel(),
                MixedComponentModel(),  # Новая
                SARIMAModel(),
                HoltWintersAdditive(),
                TheilSenModel(),  # Новая
                LSTMModel(),
                GRUModel(),  # Добавляем GRU для ряда A
                NaiveModel()  # Базовая модель для сравнения
            ]
        elif series_name == 'B':
            return [
                ARIMAModel(),
                HoltModel(damped=True),
                HoltModel(damped=False),
                AdaptiveSESModel(),  # Новая
                RecursiveLeastSquares(),  # Новая
                LSTMModel()  # Добавляем LSTM для ряда B
            ]
        elif series_name == 'C':
            return [
                ARMAModel(),
                ARModel(p=1),  # Новая
                MAModel(q=1),  # Новая
                SESModel(),
                VanillaRNN(),  # Новая
                GRUModel(),
                LSTMModel()  # Добавляем LSTM для ряда C
            ]
        else:
            raise ValueError(f"Unknown series name: {series_name}")