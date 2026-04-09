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

class BaseModel(ABC):
    """Абстрактный базовый класс для всех моделей"""

    def __init__(self, name):
        self.name = name
        self.is_fitted = False
        self.resid = None
        self.trend = None
        self.seasonal = None

    @abstractmethod
    def fit(self, train_data):
        pass

    @abstractmethod
    def predict(self, steps):
        pass


class AdditiveComponentModel(BaseModel):
    """Аддитивная компонентная модель"""

    def __init__(self):
        super().__init__("Additive Component")

    def fit(self, train_data):
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

        last_trend = self.trend.dropna().iloc[-1]
        seasonal_component = self.seasonal.iloc[-12:].values

        forecast = [last_trend + seasonal_component[i % 12] for i in range(steps)]

        return np.array(forecast)


class MultiplicativeComponentModel(BaseModel):
    """Мультипликативная компонентная модель"""

    def __init__(self):
        super().__init__("Multiplicative Component")

    def fit(self, train_data):
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

        forecast = [last_trend * seasonal_component[i % 12] for i in range(steps)]
        return np.array(forecast)


class SARIMAModel(BaseModel):
    """SARIMA модель для ряда A"""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), name="SARIMA"):
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, train_data):
        train_data = train_data.dropna().values
        try:
            self.model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
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
                train_data.values,
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

    def __init__(self, units=64, epochs=100, window_size=12, batch_size=32, name="LSTM"):
        super().__init__(name)
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(self.window_size, 1)),
            LSTM(units),
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

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
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

        forecast = [
            (last_trend * seasonal_component[i % 12]) + self.resid.mean()
            for i in range(steps)
        ]
        return np.array(forecast)


class TheilSenModel(BaseModel):
    """Процесс Тейла-Вейджа с адаптацией для временных рядов"""

    def __init__(self, window_size=24, name="Theil-Sen"):
        super().__init__(name)
        self.window_size = window_size

    def fit(self, train_data):
        self.last_values = train_data.values[-self.window_size:]
        self.is_fitted = True

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        X = np.arange(len(self.last_values)).reshape(-1, 1)
        y = self.last_values

        model = LinearRegression()
        model.fit(X, y)

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


class ARIMAModel(BaseModel):
    """ARIMA модель для ряда B"""

    def __init__(self, order=(2,1,1), name="ARIMA"):
        super().__init__(name)
        self.order=order

    def fit(self, train_data):
        try:
            train_values = train_data.dropna().values
            self.model = ARIMA(train_values, order=self.order).fit()
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
                train_data.values,
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

    def __init__(self, initial_smoothing=0.3, alpha=0.5, name="Adaptive SES"):
        super().__init__(name)
        self.initial_alpha = alpha
        self.initial_smoothing = initial_smoothing
        self.alpha = alpha

    def fit(self, train_data):
        self.train = train_data.values
        self.is_fitted = True

        self.alpha = (self.initial_alpha + self.initial_smoothing) / 2

        if len(self.train) > 12:
            errors = []
            for i in range(1, len(self.train)):
                pred = self.train[i - 1]
                error = self.train[i] - pred
                errors.append(abs(error))

            error_ratio = np.mean(errors) / np.max(errors) if np.max(errors) != 0 else 0.5
            adaptive_alpha = 1 - error_ratio
            self.alpha = min(0.99, max(0.01, (self.initial_smoothing * 0.3 + adaptive_alpha * 0.7)))

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        predictions = []
        last = self.train[-1]
        for _ in range(steps):
            predictions.append(last)
            last = self.alpha * self.train[-1] + (1 - self.alpha) * last
        return np.array(predictions)


class RecursiveLeastSquares(BaseModel):
    """Рекуррентный МНК с ограничением сложности"""

    def __init__(self, degree=2, forgetting_factor=0.99, name="Recursive Least Squares"):
        super().__init__(name)
        self.degree = degree
        self.forgetting_factor = forgetting_factor

    def fit(self, train_data):
        n = len(train_data)
        X = np.column_stack([
            np.power(np.arange(n), d)
            for d in range(self.degree + 1)
        ])
        y = train_data.values

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



class ARMAModel(BaseModel):
    """ARMA модель для ряда C"""

    def __init__(self, order=(1,0,1), name="ARMA"):
        super().__init__(name)
        self.order=order

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=self.order).fit()
        self.resid = self.model.resid
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps).values


class SESModel(BaseModel):
    """Простое экспоненциальное сглаживание для ряда C"""

    def __init__(self, smoothing_level=0.5, name="Simple Exponential Smoothing"):
        super().__init__(name)
        self.smoothing_level = smoothing_level
        self.model = None
        self.is_fitted = False

    def fit(self, train_data):
        try:
            self.model = SimpleExpSmoothing(train_data.values).fit(
                smoothing_level=self.smoothing_level,
                optimized=False
            )
            self.is_fitted = True
        except ValueError as e:
            if "must have at least one observation" in str(e):
                self.model = None
                self.is_fitted = False
                print(f"  - Предупреждение: ряд слишком короткий для SES. Используйте минимум 1 точку данных")
            else:
                print(f"  - Ошибка при обучении SES: {str(e)}")
                self.is_fitted = False
        except Exception as e:
            print(f"  - Неожиданная ошибка при обучении SES: {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted or self.model is None:
            return np.zeros(steps)

        try:
            return self.model.forecast(steps)
        except Exception as e:
            print(f"  - Ошибка при прогнозировании SES: {str(e)}")
            return np.zeros(steps)


class GRUModel(BaseModel):
    """GRU модель для ряда C"""

    def __init__(self, units=64, epochs=100, window_size=12, batch_size=32, name="GRU"):
        super().__init__(name)
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential([
            GRU(units, input_shape=(self.window_size, 1)),
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

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                       verbose=0)
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

    def __init__(self, p=1, name="AR"):
        super().__init__(name)
        self.p = p

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=(self.p, 0, 0)).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps).values


class MAModel(BaseModel):
    """MA(q) модель для стационарных рядов"""

    def __init__(self, q=1, name="MA"):
        super().__init__(name)
        self.q = q

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=(0, 0, self.q)).fit()
        self.is_fitted = True

    def predict(self, steps):
        return self.model.forecast(steps).values


class VanillaRNN(BaseModel):
    """Простая RNN модель для ряда C"""

    def __init__(self, units=64, epochs=100, window_size=12, batch_size=32, name="Vanilla RNN"):
        super().__init__(name)
        self.window_size = window_size
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential([
            SimpleRNN(units, input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.last_window = None
        self.is_fitted = False

    def fit(self, train_data):
        train_values = train_data.values
        X, y = [], []
        for i in range(len(train_values) - self.window_size):
            X.append(train_values[i:i + self.window_size])
            y.append(train_values[i + self.window_size])

        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
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


class ModelFactory:
    @staticmethod
    def create_models_for_series(series_name):
        if series_name == 'A':
            return [
                AdditiveComponentModel(),
                MultiplicativeComponentModel(),
                MixedComponentModel(),  #
                SARIMAModel(order=(1, 1, 2), seasonal_order=(1, 0, 1, 12)),
                HoltWintersAdditive(),
                TheilSenModel(window_size=60),
                LSTMModel(units=128, epochs=24, window_size=150, batch_size=32),
                GRUModel(units=16, epochs=24, window_size=150, batch_size=16),
                NaiveModel()
            ]

        elif series_name == 'B':
            return [
                ARIMAModel(order=(1, 1, 0), name="ARIMA"),
                HoltModel(damped=True),
                HoltModel(damped=False),
                AdaptiveSESModel(initial_smoothing=0.1, alpha=0.05),
                RecursiveLeastSquares(degree=1, forgetting_factor=0.95),
                LSTMModel(units=128, epochs=24, window_size=50, batch_size=16),
                GRUModel(units=48, epochs=16, window_size=100, batch_size=64),
                NaiveModel()
            ]

        elif series_name == 'C':
            return [
                ARMAModel(order=(4, 0, 3)),
                ARModel(p=1),
                MAModel(q=4),
                SESModel(smoothing_level=0.2),
                VanillaRNN(units=64, epochs=50, window_size=18, batch_size=16),
                GRUModel(units=64, epochs=12, window_size=24, batch_size=16),
                LSTMModel(units=16, epochs=200, window_size=12, batch_size=32),
                NaiveModel()
            ]
        else:
            raise ValueError(f"Unknown series name: {series_name}")