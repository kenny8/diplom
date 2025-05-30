import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from statsmodels.tsa.api import ExponentialSmoothing, ARIMA, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from statsmodels.tsa.seasonal import seasonal_decompose


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



class ComponentModel(BaseModel):
    """Базовый класс для компонентных моделей"""

    def __init__(self, name, model_type='additive'):
        super().__init__(name)
        self.model_type = model_type  # 'additive', 'multiplicative', 'mixed'

    def fit(self, train_data):
        # Здесь будет реализация декомпозиции и обучения
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            raise Exception("Model not fitted yet")
        # Реализация прогнозирования


class ClassicalModel(BaseModel):
    """Базовый класс для классических моделей"""

    def __init__(self, name, model_params=None):
        super().__init__(name)
        self.model_params = model_params or {}

    def fit(self, train_data):
        # Общая логика обучения классических моделей
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            raise Exception("Model not fitted yet")


class SmoothingModel(BaseModel):
    """Базовый класс для моделей сглаживания"""

    def __init__(self, name, trend_type, damped=False, seasonal=None):
        super().__init__(name)
        self.trend_type = trend_type
        self.damped = damped
        self.seasonal = seasonal

    def fit(self, train_data):
        # Логика инициализации и обучения
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            raise Exception("Model not fitted yet")


class RecursiveModel(BaseModel):
    """Базовый класс для рекуррентных моделей"""

    def __init__(self, name, model_type):
        super().__init__(name)
        self.model_type = model_type  # 'linear', 'quadratic', 'lstm', 'gru'
        self.window_size = 12  # размер окна для нейросетевых моделей

    def fit(self, train_data):
        # Подготовка данных для рекуррентных моделей
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            raise Exception("Model not fitted yet")


# Реализация конкретных моделей для каждого ряда

# === РЯД A (Сезонность + Тренд) ===
class AdditiveComponentModel(ComponentModel):
    """Аддитивная компонентная модель для ряда A"""

    def __init__(self):
        super().__init__("Additive Component", model_type='additive')


class MultiplicativeComponentModel(ComponentModel):
    """Мультипликативная компонентная модель для ряда A"""

    def __init__(self):
        super().__init__("Multiplicative Component", model_type='multiplicative')


class SARIMAModel(ClassicalModel):
    """SARIMA модель для ряда A"""

    def __init__(self):
        super().__init__("SARIMA", model_params={'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)})

    def fit(self, train_data):
        self.model = SARIMAX(
            train_data,
            order=self.model_params['order'],
            seasonal_order=self.model_params['seasonal_order']
        ).fit(disp=False)
        self.resid = self.model.resid
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps)


class HoltWintersAdditive(SmoothingModel):
    """Аддитивная модель Холта-Винтерса для ряда A"""

    def __init__(self):
        super().__init__("Holt-Winters Additive", trend_type='add', seasonal='add')

    def fit(self, train_data):
        self.model = ExponentialSmoothing(
            train_data,
            trend=self.trend_type,
            seasonal=self.seasonal,
            seasonal_periods=12
        ).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps)


class LSTMModel(RecursiveModel):
    """LSTM модель для ряда A"""

    def __init__(self):
        super().__init__("LSTM", model_type='lstm')
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.window_size, 1)),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, train_data):
        # Преобразование данных в формат для LSTM
        X, y = [], []
        for i in range(len(train_data) - self.window_size):
            X.append(train_data[i:i + self.window_size])
            y.append(train_data[i + self.window_size])

        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)

        self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        self.is_fitted = True

    def predict(self, steps):
        # Реализация рекуррентного прогнозирования
        pass


# === РЯД B (Тренд без сезонности) ===
class ARIMAModel(ClassicalModel):
    """ARIMA модель для ряда B"""

    def __init__(self):
        super().__init__("ARIMA", model_params={'order': (2, 1, 1)})

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=self.model_params['order']).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps)


class HoltModel(SmoothingModel):
    """Модель Хольта с затуханием для ряда B"""

    def __init__(self, damped=True):
        super().__init__("Holt Damped", trend_type='add', damped=damped)

    def fit(self, train_data):
        self.model = ExponentialSmoothing(
            train_data,
            trend=self.trend_type,
            damped_trend=self.damped
        ).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps)


# === РЯД C (Стационарный ряд) ===
class ARMAModel(ClassicalModel):
    """ARMA модель для ряда C"""

    def __init__(self):
        super().__init__("ARMA", model_params={'order': (1, 0, 1)})

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=self.model_params['order']).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps)


class SESModel(ClassicalModel):
    """Простое экспоненциальное сглаживание для ряда C"""

    def __init__(self):
        super().__init__("Simple Exponential Smoothing")

    def fit(self, train_data):
        self.model = SimpleExpSmoothing(train_data).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps)


class GRUModel(RecursiveModel):
    """GRU модель для ряда C"""

    def __init__(self):
        super().__init__("GRU", model_type='gru')
        self.model = Sequential([
            GRU(32, input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, train_data):
        # Аналогично LSTM
        pass

class AdditiveComponentModel(ComponentModel):
    """Аддитивная компонентная модель"""

    def __init__(self):
        super().__init__("Additive Component", model_type='additive')

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

        # Простейший прогноз: последний тренд + сезонность
        last_trend = self.trend.dropna()[-1]
        seasonal_component = self.seasonal[-12:].values

        # Прогноз на steps периодов
        forecast = [last_trend + seasonal_component[i % 12] for i in range(steps)]

        return np.array(forecast)

class MultiplicativeComponentModel(ComponentModel):
    """Мультипликативная компонентная модель"""

    def __init__(self):
        super().__init__("Multiplicative Component", model_type='multiplicative')

    def fit(self, train_data):
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

        # Простейший прогноз: последний тренд * сезонность
        last_trend = self.trend.dropna()[-1]
        seasonal_component = self.seasonal[-12:].values

        # Прогноз на steps периодов
        forecast = [last_trend * seasonal_component[i % 12] for i in range(steps)]

        return np.array(forecast)


# Фабрика моделей для удобного создания
class ModelFactory:
    @staticmethod
    def create_models_for_series(series_name):
        """Создает список моделей для указанного ряда"""
        if series_name == 'A':
            return [
                AdditiveComponentModel(),
                MultiplicativeComponentModel(),
                SARIMAModel(),
                HoltWintersAdditive(),
                LSTMModel()
            ]
        elif series_name == 'B':
            return [
                ARIMAModel(),
                HoltModel(damped=True),
                HoltModel(damped=False)
            ]
        elif series_name == 'C':
            return [
                ARMAModel(),
                SESModel(),
                GRUModel()
            ]
        else:
            raise ValueError(f"Unknown series name: {series_name}")