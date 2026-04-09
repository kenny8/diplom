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
# models.py (исправления и дополнения)
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from statsmodels.tsa.api import ExponentialSmoothing, ARIMA, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler  # Добавлено для StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout  # Добавлен Dropout
from keras.optimizers import Adam  # Добавлено для Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Добавлено для callback'ов
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.layers import SimpleRNN
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Отключаем интерактивный режим matplotlib
import matplotlib.pyplot as plt
import warnings
from keras.layers import BatchNormalization
import tensorflow as tf
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
        self.period = 12

    def fit(self, train_data):
        try:
            # Преобразуем в pandas Series если это numpy array
            if isinstance(train_data, np.ndarray):
                train_data = pd.Series(train_data)

            decomposition = seasonal_decompose(
                train_data,
                model='additive',
                period=self.period
            )

            self.trend = decomposition.trend
            self.seasonal = decomposition.seasonal
            self.resid = decomposition.resid
            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении AdditiveComponentModel: {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        try:
            # Проверяем и преобразуем компоненты в pandas Series если нужно
            if isinstance(self.trend, np.ndarray):
                self.trend = pd.Series(self.trend)
            if isinstance(self.seasonal, np.ndarray):
                self.seasonal = pd.Series(self.seasonal)

            last_trend = self.trend.dropna().iloc[-1]
            seasonal_component = self.seasonal.iloc[-12:].values

            forecast = [last_trend + seasonal_component[i % 12] for i in range(steps)]
            return np.array(forecast)

        except Exception as e:
            print(f"  - Ошибка прогнозирования AdditiveComponentModel: {str(e)}")
            return np.zeros(steps)


class MultiplicativeComponentModel(BaseModel):
    """Мультипликативная компонентная модель"""

    def __init__(self):
        super().__init__("Multiplicative Component")
        self.period = 12

    def fit(self, train_data):
        try:
            # Преобразуем в pandas Series если это numpy array
            if isinstance(train_data, np.ndarray):
                train_data = pd.Series(train_data)

            # Корректируем данные если есть отрицательные значения
            if (train_data <= 0).any():
                self.min_value = train_data.min() - 1e-6
                train_data = train_data - self.min_value + 1e-6
            else:
                self.min_value = 0

            decomposition = seasonal_decompose(
                train_data,
                model='multiplicative',
                period=self.period
            )

            self.trend = decomposition.trend
            self.seasonal = decomposition.seasonal
            self.resid = decomposition.resid
            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении MultiplicativeComponentModel: {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        try:
            # Проверяем и преобразуем компоненты в pandas Series если нужно
            if isinstance(self.trend, np.ndarray):
                self.trend = pd.Series(self.trend)
            if isinstance(self.seasonal, np.ndarray):
                self.seasonal = pd.Series(self.seasonal)

            last_trend = self.trend.dropna().iloc[-1]
            seasonal_component = self.seasonal.iloc[-12:].values

            forecast = [last_trend * seasonal_component[i % 12] for i in range(steps)]

            # Обратное преобразование если были отрицательные значения
            if hasattr(self, 'min_value'):
                forecast = np.array(forecast) + self.min_value - 1e-6

            return np.array(forecast)

        except Exception as e:
            print(f"  - Ошибка прогнозирования MultiplicativeComponentModel: {str(e)}")
            return np.zeros(steps)


class SARIMAModel(BaseModel):
    """SARIMA модель"""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), name="SARIMA"):
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, train_data):
        try:
            # Преобразуем в pandas Series если это numpy array
            if isinstance(train_data, np.ndarray):
                train_data = pd.Series(train_data)

            train_data = train_data.dropna()
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

        try:
            forecast = self.model.forecast(steps)
            return forecast.values if hasattr(forecast, 'values') else forecast
        except Exception as e:
            print(f"  - Ошибка прогнозирования SARIMA: {str(e)}")
            return np.zeros(steps)


class HoltWintersAdditive(BaseModel):
    """Улучшенная аддитивная модель Холта-Винтерса с автоматической оптимизацией параметров"""

    def __init__(self):
        super().__init__("Holt-Winters Additive")
        self._last_train_values = None  # Для хранения последних значений ряда
        self._seasonal_periods = 12  # Фиксированный период сезонности

    def _optimize_parameters(self, train_values):
        """Автоматический подбор оптимальных параметров сглаживания"""
        from scipy.optimize import differential_evolution

        def model_error(params):
            alpha, beta, gamma = params
            try:
                model = ExponentialSmoothing(
                    train_values,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=self._seasonal_periods
                ).fit(smoothing_level=alpha,
                      smoothing_trend=beta,
                      smoothing_seasonal=gamma,
                      optimized=False)
                return model.sse
            except:
                return np.inf

        bounds = [(0, 1), (0, 1), (0, 1)]
        result = differential_evolution(model_error, bounds, maxiter=20, popsize=10)
        return result.x

    def fit(self, train_data):
        """Обучение модели с автоматической оптимизацией и улучшенной обработкой ошибок"""
        self._last_train_values = train_data.values

        try:
            # 1. Автоматическая оптимизация параметров сглаживания
            optimal_params = self._optimize_parameters(self._last_train_values)

            # 2. Обучение модели с оптимальными параметрами
            self.model = ExponentialSmoothing(
                self._last_train_values,
                trend='add',
                seasonal='add',
                seasonal_periods=self._seasonal_periods,
                initialization_method='estimated'  # Улучшенная инициализация
            ).fit(smoothing_level=optimal_params[0],
                  smoothing_trend=optimal_params[1],
                  smoothing_seasonal=optimal_params[2],
                  remove_bias=True)  # Удаление смещения

            # 3. Проверка адекватности модели
            if np.isinf(self.model.sse) or self.model.sse <= 0:
                raise ValueError("Invalid model SSE")

            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении Holt-Winters: {str(e)}")
            self.is_fitted = False

            # Fallback: простая модель с параметрами по умолчанию
            try:
                self.model = ExponentialSmoothing(
                    self._last_train_values,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=self._seasonal_periods
                ).fit()
                self.is_fitted = True
            except:
                self.is_fitted = False

    def predict(self, steps):
        """Прогнозирование с улучшенной обработкой краевых эффектов"""
        if not self.is_fitted:
            if hasattr(self, '_last_train_values'):
                # Возвращаем последнее известное значение при ошибке
                return np.full(steps, np.nanmean(self._last_train_values))
            return np.zeros(steps)

        try:
            # Получаем базовый прогноз
            forecast = self.model.forecast(steps)

            # Корректировка на основе последних наблюдаемых значений
            if len(self._last_train_values) >= self._seasonal_periods:
                last_season = self._last_train_values[-self._seasonal_periods:]
                forecast_season = forecast[:self._seasonal_periods]

                # Корректируем амплитуду сезонности
                if np.mean(forecast_season) > 1e-6:
                    adjustment_factor = np.mean(last_season) / np.mean(forecast_season)
                    forecast = forecast * adjustment_factor

            return forecast

        except Exception as e:
            print(f"  - Ошибка прогнозирования Holt-Winters: {str(e)}")
            if hasattr(self, '_last_train_values'):
                return np.full(steps, np.nanmean(self._last_train_values))
            return np.zeros(steps)


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
    """Смешанная компонентная модель"""

    def __init__(self):
        super().__init__("Mixed Component")
        self.period = 12

    def fit(self, train_data):
        try:
            # Преобразуем в pandas Series если это numpy array
            if isinstance(train_data, np.ndarray):
                train_data = pd.Series(train_data)

            # Корректируем данные если есть отрицательные значения
            if (train_data <= 0).any():
                self.min_value = train_data.min() - 1e-6
                train_data = train_data - self.min_value + 1e-6
            else:
                self.min_value = 0

            decomposition = seasonal_decompose(
                train_data,
                model='multiplicative',
                period=self.period
            )

            self.trend = decomposition.trend
            self.seasonal = decomposition.seasonal
            self.resid = decomposition.resid
            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении MixedComponentModel: {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        try:
            # Проверяем и преобразуем компоненты в pandas Series если нужно
            if isinstance(self.trend, np.ndarray):
                self.trend = pd.Series(self.trend)
            if isinstance(self.seasonal, np.ndarray):
                self.seasonal = pd.Series(self.seasonal)
            if isinstance(self.resid, np.ndarray):
                self.resid = pd.Series(self.resid)

            last_trend = self.trend.dropna().iloc[-1]
            seasonal_component = self.seasonal.iloc[-12:].values
            resid_mean = self.resid.mean() if not self.resid.empty else 0

            forecast = [(last_trend * seasonal_component[i % 12]) + resid_mean
                        for i in range(steps)]

            # Обратное преобразование если были отрицательные значения
            if hasattr(self, 'min_value'):
                forecast = np.array(forecast) + self.min_value - 1e-6

            return np.array(forecast)

        except Exception as e:
            print(f"  - Ошибка прогнозирования MixedComponentModel: {str(e)}")
            return np.zeros(steps)


class TheilSenModel(BaseModel):
    """Улучшенная модель Тейла-Сена с робастной регрессией и обработкой тренда"""

    def __init__(self, window_size=24, name="Theil-Sen"):
        super().__init__(name)
        self.window_size = window_size
        self._last_values = None
        self._trend_model = None
        self._data_stats = {}

    def _fit_robust_trend(self, X, y):
        """Обучение робастной модели тренда с использованием Theil-Sen"""
        from sklearn.linear_model import TheilSenRegressor

        try:
            # Пробуем Theil-Sen с 5 итерациями для скорости
            model = TheilSenRegressor(max_iter=5, random_state=42)
            model.fit(X, y)
            return model
        except:
            # Fallback на обычную линейную регрессию
            return LinearRegression().fit(X, y)

    def fit(self, train_data):
        """Обучение модели с расширенной предобработкой"""
        self._last_values = train_data.values[-self.window_size:]

        # Сохраняем статистики данных
        self._data_stats = {
            'min': np.min(self._last_values),
            'max': np.max(self._last_values),
            'mean': np.mean(self._last_values)
        }

        # Подготовка данных для регрессии
        X = np.arange(len(self._last_values)).reshape(-1, 1)
        y = self._last_values

        # Обучение робастной модели тренда
        self._trend_model = self._fit_robust_trend(X, y)

        self.is_fitted = True

    def predict(self, steps):
        """Прогнозирование с обработкой граничных условий"""
        if not self.is_fitted:
            return np.zeros(steps) + self._data_stats.get('mean', 0)

        try:
            # Генерация прогноза
            last_idx = len(self._last_values) - 1
            X_future = np.arange(last_idx + 1, last_idx + 1 + steps).reshape(-1, 1)
            forecast = self._trend_model.predict(X_future)

            # Ограничение прогноза разумными пределами
            forecast = np.clip(forecast,
                               self._data_stats['min'] * 0.9,
                               self._data_stats['max'] * 1.1)

            return forecast

        except Exception as e:
            print(f"  - Ошибка прогнозирования Theil-Sen: {str(e)}")
            # Возвращаем среднее значение при ошибке
            return np.zeros(steps) + self._data_stats.get('mean', 0)

    def get_trend_coef(self):
        """Дополнительный метод для получения коэффициента тренда"""
        if hasattr(self._trend_model, 'coef_'):
            return self._trend_model.coef_[0]
        return 0


class NaiveModel(BaseModel):
    """Улучшенная наивная модель с сезонной адаптацией"""

    def __init__(self):
        super().__init__("Naive Forecast")
        self.last_value = None
        self.seasonal_last_values = None
        self.seasonal_period = 12  # Можно настроить для разных данных
        self.model_type = "simple"  # simple или seasonal

    def _detect_seasonality(self, data):
        """Автоматическое определение сезонности"""
        if len(data) < 2 * self.seasonal_period:
            return False

        # Простая проверка на сезонность (можно заменить более сложным алгоритмом)
        seasonal_part = data[-self.seasonal_period:]
        mean_val = np.mean(data)
        return np.any(np.abs(seasonal_part - mean_val) > 0.5 * np.std(data))

    def fit(self, train_data):
        """Обучение модели с автоматическим определением типа"""
        self.last_value = train_data.iloc[-1]

        # Проверяем наличие сезонности
        if self._detect_seasonality(train_data.values):
            self.model_type = "seasonal"
            self.seasonal_last_values = train_data.values[-self.seasonal_period:]

        self.is_fitted = True

    def predict(self, steps):
        """Прогнозирование с учетом сезонности при наличии"""
        if not self.is_fitted:
            return np.zeros(steps)

        if self.model_type == "seasonal" and self.seasonal_last_values is not None:
            # Сезонный прогноз - циклическое повторение последних значений
            repeats = int(np.ceil(steps / len(self.seasonal_last_values)))
            return np.tile(self.seasonal_last_values, repeats)[:steps]
        else:
            # Простой прогноз - повторение последнего значения
            return np.full(steps, self.last_value)

    def get_model_type(self):
        """Возвращает тип используемой модели (simple/seasonal)"""
        return self.model_type

# === РЯД B (Тренд без сезонности) ===

class ARIMAModel(BaseModel):
    """ARIMA модель для ряда B"""

    def __init__(self, order=(2, 1, 1), name="ARIMA"):
        super().__init__(name)
        self.order = order
        self._last_train_values = None

    def fit(self, train_data):
        try:
            # Сохраняем исходные данные для fallback
            self._last_train_values = train_data.copy()

            # Преобразуем в numpy array если это pandas Series/DataFrame
            if hasattr(train_data, 'values'):
                train_values = train_data.values
            else:
                train_values = np.array(train_data)

            # Удаляем NaN значения
            train_values = train_values[~np.isnan(train_values)]

            # Основная попытка обучения
            self.model = ARIMA(train_values, order=self.order).fit()
            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении ARIMA: {str(e)}")
            # Fallback: пробуем более простую модель
            try:
                self.model = ARIMA(train_values, order=(1, 1, 1)).fit()
                self.is_fitted = True
            except:
                self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            # Fallback: возвращаем среднее значение если модель не обучилась
            if self._last_train_values is not None:
                fallback_value = np.nanmean(self._last_train_values)
                return np.zeros(steps) + fallback_value
            return np.zeros(steps)

        try:
            forecast = self.model.forecast(steps)
            # Возвращаем как numpy array
            return forecast if isinstance(forecast, np.ndarray) else forecast.values
        except Exception as e:
            print(f"  - Ошибка прогнозирования ARIMA: {str(e)}")
            if self._last_train_values is not None:
                fallback_value = np.nanmean(self._last_train_values)
                return np.zeros(steps) + fallback_value
            return np.zeros(steps)


class HoltModel(BaseModel):
    """Улучшенная модель Хольта с затуханием для ряда B"""

    def __init__(self, damped=True):
        name = "Holt Damped" if damped else "Holt"
        super().__init__(name)
        self.damped = damped
        # Параметры сглаживания (оставляем по умолчанию, но делаем доступными)
        self.smoothing_level = None  # None означает автоматический подбор
        self.smoothing_trend = None
        self.optimized = True  # Флаг оптимизации параметров
        self.initialization_method = 'estimated'  # Метод инициализации

    def fit(self, train_data):
        try:
            # Проверка входных данных
            if len(train_data) < 5:
                raise ValueError("Слишком мало данных для обучения (минимум 5 точек)")

            # Улучшенная инициализация модели
            self.model = ExponentialSmoothing(
                train_data.values,
                trend='add',
                damped_trend=self.damped,
                initialization_method=self.initialization_method
            ).fit(
                smoothing_level=self.smoothing_level,
                smoothing_trend=self.smoothing_trend,
                optimized=self.optimized,
                remove_bias=False  # Улучшает точность для некоторых рядов
            )

            # Проверка качества подгонки
            if np.any(self.model.fittedvalues < 0):
                print("  - Предупреждение: модель дает отрицательные значения")

            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении Holt: {str(e)}")
            self.is_fitted = False
            # Возвращаем простую модель как запасной вариант
            self._fallback_fit(train_data)

    def _fallback_fit(self, train_data):
        """Простая модель как запасной вариант"""
        try:
            self.model = ExponentialSmoothing(
                train_data.values,
                trend='add',
                damped_trend=False,  # Упрощаем модель
                initialization_method='heuristic'
            ).fit(optimized=False)
            self.is_fitted = True
            print("  - Использована упрощенная модель как запасной вариант")
        except:
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            return np.zeros(steps)

        try:
            # Добавляем проверку на разумность прогноза
            forecast = self.model.forecast(steps)
            if np.any(np.isnan(forecast)):
                raise ValueError("Прогноз содержит NaN значения")
            return forecast
        except Exception as e:
            print(f"  - Ошибка прогнозирования: {str(e)}")
            # Возвращаем последнее значение как простой прогноз
            return np.full(steps, self.model.fittedvalues[-1])


class AdaptiveSESModel(BaseModel):
    """Улучшенный SES с адаптивным параметром сглаживания и защитными механизмами"""

    def __init__(self, initial_smoothing=0.3, alpha=0.5, name="Adaptive SES"):
        super().__init__(name)
        self.initial_alpha = min(0.99, max(0.01, alpha))  # Ограничиваем диапазон
        self.initial_smoothing = min(0.99, max(0.01, initial_smoothing))  # Ограничение 0.01-0.99
        self.alpha = (self.initial_alpha + self.initial_smoothing) / 2  # Начальное среднее

        # Параметры адаптации
        self.error_window = 12  # Окно для расчета ошибок
        self.min_alpha = 0.05  # Минимальное значение alpha
        self.max_alpha = 0.95  # Максимальное значение alpha

    def fit(self, train_data):
        try:
            self.train = train_data.values
            if len(self.train) < 2:
                raise ValueError("Недостаточно данных для обучения (минимум 2 точки)")

            # Базовый расчет alpha
            self.alpha = (self.initial_alpha + self.initial_smoothing) / 2

            # Улучшенная адаптация alpha с экспоненциальным взвешиванием ошибок
            if len(self.train) >= self.error_window:
                errors = []
                weights = np.exp(np.linspace(-1, 0, len(self.train) - 1))  # Экспоненциальные веса

                for i in range(1, len(self.train)):
                    error = abs(self.train[i] - self.train[i - 1])
                    weighted_error = error * weights[i - 1]
                    errors.append(weighted_error)

                if np.sum(errors) > 0:
                    # Нормализованная взвешенная ошибка
                    norm_error = np.sum(errors) / (np.sum(weights) * np.max(self.train))
                    # Нелинейная адаптация alpha (логистическая функция)
                    self.alpha = 1 / (1 + np.exp(-10 * (norm_error - 0.5)))
                    # Смешиваем с начальным значением и ограничиваем диапазон
                    self.alpha = min(self.max_alpha, max(self.min_alpha,
                                                         0.3 * self.initial_smoothing + 0.7 * self.alpha))

            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении AdaptiveSES: {str(e)}")
            # Запасной вариант - простой SES
            self.alpha = self.initial_smoothing
            self.is_fitted = len(self.train) > 0

    def predict(self, steps):
        if not self.is_fitted or len(self.train) == 0:
            return np.zeros(steps)

        try:
            predictions = []
            last = self.train[-1]

            # Улучшенная формула прогнозирования с защитой от крайних значений
            for _ in range(steps):
                predictions.append(last)
                last = self.alpha * self.train[-1] + (1 - self.alpha) * last
                # Защита от аномальных скачков
                if len(self.train) > 1:
                    diff = np.diff(self.train[-2:])[0]
                    last = np.clip(last,
                                   min(self.train[-1] - 2 * abs(diff), self.train[-1] + 2 * abs(diff)),
                                   max(self.train[-1] - 2 * abs(diff), self.train[-1] + 2 * abs(diff)))

            return np.array(predictions)

        except Exception as e:
            print(f"  - Ошибка прогнозирования: {str(e)}")
            return np.full(steps, self.train[-1])  # Возвращаем последнее значение


class RecursiveLeastSquares(BaseModel):
    """Улучшенный рекуррентный МНК с экспоненциальным забыванием и регуляризацией"""

    def __init__(self, degree=2, forgetting_factor=0.99, name="Recursive Least Squares"):
        super().__init__(name)
        self.degree = min(max(1, degree), 5)  # Ограничиваем степень 1-5
        self.forgetting_factor = min(max(0.9, forgetting_factor), 1.0)  # Ограничиваем 0.9-1.0
        self.regularization = 1e-4  # Коэффициент регуляризации
        self.coef_ = None
        self.P = None
        self.last_index = 0

    def fit(self, train_data):
        try:
            n = len(train_data)
            if n < self.degree + 1:
                raise ValueError(
                    f"Недостаточно данных для степени {self.degree} (нужно минимум {self.degree + 1} точек)")

            y = train_data.values

            # Инициализация с регуляризацией
            self.coef_ = np.zeros(self.degree + 1)
            self.P = np.eye(self.degree + 1) * (1.0 / self.regularization)

            # Предварительное вычисление матрицы степеней
            indices = np.arange(n)
            X = np.column_stack([np.power(indices, d) for d in range(self.degree + 1)])

            # Векторизованная реализация RLS
            for i in range(n):
                x_i = X[i]
                # Улучшенное вычисление коэффициента усиления
                P_x = self.P @ x_i
                denom = self.forgetting_factor + x_i @ P_x
                K = P_x / denom

                error = y[i] - x_i @ self.coef_
                self.coef_ += K * error

                # Стабилизированное обновление матрицы ковариации
                self.P = (self.P - np.outer(K, P_x)) / self.forgetting_factor

                # Добавляем небольшую регуляризацию для устойчивости
                if i % 10 == 0:
                    self.P += np.eye(self.degree + 1) * self.regularization

            self.last_index = n - 1
            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении RLS: {str(e)}")
            # Запасной вариант - линейная регрессия
            self._fallback_fit(train_data)

    def _fallback_fit(self, train_data):
        """Простая линейная регрессия как запасной вариант"""
        try:
            n = len(train_data)
            X = np.column_stack([np.arange(n), np.ones(n)])
            y = train_data.values
            self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
            self.degree = 1
            self.last_index = n - 1
            self.is_fitted = True
            print("  - Использована упрощенная линейная модель как запасной вариант")
        except:
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted or self.coef_ is None:
            return np.zeros(steps)

        try:
            # Векторизованное вычисление прогноза
            future_indices = np.arange(self.last_index + 1, self.last_index + steps + 1)
            X_future = np.column_stack([np.power(future_indices, d) for d in range(self.degree + 1)])
            predictions = X_future @ self.coef_

            # Ограничение прогноза на основе исторических изменений
            if hasattr(self, 'train'):
                last_value = self.train[-1]
                max_change = np.max(np.abs(np.diff(self.train[-min(10, len(self.train)):])))
                predictions = np.clip(predictions,
                                      last_value - 3 * max_change,
                                      last_value + 3 * max_change)

            return predictions

        except Exception as e:
            print(f"  - Ошибка прогнозирования RLS: {str(e)}")
            if hasattr(self, 'train'):
                return np.full(steps, self.train[-1])
            return np.zeros(steps)


# === РЯД C (Стационарный ряд) ===

class ARMAModel(BaseModel):
    """Улучшенная ARMA модель с обработкой ошибок и проверками стабильности"""

    def __init__(self, order=(1, 0, 1), name="ARMA"):
        super().__init__(name)
        self.order = order
        self._last_train_values = None
        # Fix this line - calculate min length properly
        self._min_train_length = max(order[0], order[2]) + 1

    def fit(self, train_data):
        try:
            # Проверка минимальной длины ряда
            if len(train_data) < self._min_train_length:
                raise ValueError(
                    f"Недостаточно данных для ARMA{self.order}. Нужно минимум {self._min_train_length} точек"
                )

            # Сохраняем последние значения для резервного прогноза
            self._last_train_values = train_data.values[-3:]  # Последние 3 значения

            # Обучение модели с обработкой возможных ошибок
            self.model = ARIMA(
                train_data,
                order=self.order,
                enforce_stationarity=False,  # Улучшает сходимость
                enforce_invertibility=False  # Улучшает сходимость
            ).fit()

            # Проверка адекватности модели
            if any(np.isnan(self.model.params)):
                raise ValueError("Получены NaN параметры в модели")

            self.resid = self.model.resid
            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении ARMA{self.order}: {str(e)}")
            self.is_fitted = False
            # Сохраняем данные для простого прогноза
            self._last_train_values = train_data.values[-3:]

    def predict(self, steps):
        if not self.is_fitted:
            # Простой прогноз на основе последних значений
            if self._last_train_values is not None and len(self._last_train_values) > 0:
                last_value = self._last_train_values[-1]
                # Линейная экстраполяция на основе последних 2 точек
                if len(self._last_train_values) >= 2:
                    trend = self._last_train_values[-1] - self._last_train_values[-2]
                    return np.array([last_value + i * trend for i in range(1, steps + 1)])
                return np.full(steps, last_value)
            return np.zeros(steps)

        try:
            # Получаем прогноз от ARMA модели
            forecast = self.model.forecast(steps).values

            # Проверка на разумность прогноза
            if any(np.isnan(forecast)):
                raise ValueError("Прогноз содержит NaN значения")

            # Ограничение экстремальных значений
            if self._last_train_values is not None:
                last_value = self._last_train_values[-1]
                historical_range = np.ptp(self._last_train_values)
                forecast = np.clip(
                    forecast,
                    last_value - 2 * historical_range,
                    last_value + 2 * historical_range
                )

            return forecast

        except Exception as e:
            print(f"  - Ошибка прогнозирования ARMA: {str(e)}")
            # Возвращаем простое продолжение тренда
            if self._last_train_values is not None and len(self._last_train_values) > 0:
                last_value = self._last_train_values[-1]
                if len(self._last_train_values) >= 2:
                    trend = self._last_train_values[-1] - self._last_train_values[-2]
                    return np.array([last_value + i * trend for i in range(1, steps + 1)])
                return np.full(steps, last_value)
            return np.zeros(steps)


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
    """Улучшенная GRU модель с обработкой ошибок и дополнительными проверками"""

    def __init__(self, units=64, epochs=100, window_size=12, batch_size=32, name="GRU"):
        super().__init__(name)
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units

        # Инициализация модели с улучшенными параметрами
        self.model = Sequential([
            GRU(self.units, input_shape=(self.window_size, 1), return_sequences=False),
            Dense(1, activation='linear')
        ])

        # Улучшенный оптимизатор с настройками по умолчанию
        optimizer = Adam(
            learning_rate=0.001,
            clipvalue=1.0  # Защита от взрыва градиентов
        )

        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        self.last_window = None
        self.scaler = None
        self.is_trained = False

    def fit(self, train_data):
        try:
            train_values = train_data.values

            # Проверка достаточности данных
            if len(train_values) < self.window_size + 1:
                raise ValueError(
                    f"Недостаточно данных. Нужно минимум {self.window_size + 1} точек, получено {len(train_values)}"
                )

            # Нормализация данных
            self.scaler = lambda x: x  # identity function если нормализация не используется
            if np.std(train_values) > 1e-6:  # Проверка на постоянный ряд
                self.scaler = StandardScaler().fit(train_values.reshape(-1, 1))
                scaled_values = self.scaler.transform(train_values.reshape(-1, 1)).flatten()
            else:
                scaled_values = train_values

            # Подготовка оконных данных
            X, y = [], []
            for i in range(len(scaled_values) - self.window_size):
                X.append(scaled_values[i:i + self.window_size])
                y.append(scaled_values[i + self.window_size])

            X = np.array(X).reshape(-1, self.window_size, 1)
            y = np.array(y)

            # Добавление callback'ов
            callbacks = [
                EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
            ]

            # Обучение модели
            self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=0
            )

            # Сохранение последнего окна (нормализованного)
            self.last_window = scaled_values[-self.window_size:]
            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении GRU: {str(e)}")
            self.is_fitted = False
            # Сохраняем последние значения для простого прогноза
            self.last_window = train_values[-min(self.window_size, len(train_values)):] if len(
                train_values) > 0 else None

    def predict(self, steps):
        if not self.is_fitted or self.last_window is None:
            return np.zeros(steps)

        try:
            # Прогнозирование с использованием последнего окна
            current_input = self.last_window.copy().reshape(1, self.window_size, 1)
            predictions = []

            for _ in range(steps):
                next_pred = self.model.predict(current_input, verbose=0)[0, 0]
                predictions.append(next_pred)
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = next_pred

            # Обратное преобразование если использовалась нормализация
            if hasattr(self.scaler, 'inverse_transform'):
                predictions = self.scaler.inverse_transform(
                    np.array(predictions).reshape(-1, 1)
                ).flatten()

            # Проверка на NaN и бесконечности
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                raise ValueError("Прогноз содержит NaN или бесконечные значения")

            return predictions

        except Exception as e:
            print(f"  - Ошибка при прогнозировании GRU: {str(e)}")
            # Возвращаем последнее значение как простой прогноз
            if self.last_window is not None:
                last_value = self.last_window[-1] if hasattr(self.scaler, 'inverse_transform') else \
                    self.scaler.inverse_transform([[self.last_window[-1]]])[0][0]
                return np.full(steps, last_value)
            return np.zeros(steps)


class ARModel(BaseModel):
    """Улучшенная AR(p) модель с обработкой ошибок и дополнительными проверками"""

    def __init__(self, p=1, name="AR"):
        super().__init__(name)
        self.p = p  # Сохраняем исходный порядок модели без изменений
        self._last_values = None  # Для резервного прогнозирования
        self._min_train_length = max(p + 1, 5)  # Минимальная длина для обучения

    def fit(self, train_data):
        try:
            # Сохраняем последние значения для резервного прогноза
            self._last_values = train_data.values[-self.p:] if len(train_data) >= self.p else train_data.values

            # Проверка минимальной длины ряда
            if len(train_data) < self._min_train_length:
                raise ValueError(
                    f"Недостаточно данных для AR({self.p}). Нужно минимум {self._min_train_length} точек"
                )

            # Проверка стационарности (простая проверка разности стандартных отклонений)
            if len(train_data) > 10:
                split = len(train_data) // 2
                std1, std2 = np.std(train_data[:split]), np.std(train_data[split:])
                if abs(std1 - std2) / min(std1, std2) > 0.5:
                    print(f"  - Предупреждение: ряд может быть нестационарным (std1={std1:.2f}, std2={std2:.2f})")

            # Обучение модели с обработкой возможных ошибок
            self.model = ARIMA(
                train_data,
                order=(self.p, 0, 0),
                enforce_stationarity=False,  # Улучшает сходимость
                enforce_invertibility=False  # Улучшает сходимость
            ).fit()

            # Проверка адекватности модели
            if any(np.isnan(self.model.params)):
                raise ValueError("Получены NaN параметры в модели")

            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении AR({self.p}): {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            # Простой прогноз на основе последних значений
            if self._last_values is not None and len(self._last_values) > 0:
                if len(self._last_values) >= 2:
                    # Используем среднее последних значений если p > 1
                    avg_trend = np.mean(np.diff(self._last_values))
                    return np.array([self._last_values[-1] + i * avg_trend for i in range(1, steps + 1)])
                return np.full(steps, self._last_values[-1])
            return np.zeros(steps)

        try:
            # Получаем прогноз от AR модели
            forecast = self.model.forecast(steps).values

            # Проверка на разумность прогноза
            if any(np.isnan(forecast)):
                raise ValueError("Прогноз содержит NaN значения")

            # Ограничение экстремальных значений
            if self._last_values is not None:
                last_value = self._last_values[-1]
                historical_range = np.ptp(self._last_values)
                forecast = np.clip(
                    forecast,
                    last_value - 2 * historical_range,
                    last_value + 2 * historical_range
                )

            return forecast

        except Exception as e:
            print(f"  - Ошибка прогнозирования AR: {str(e)}")
            # Возвращаем простое продолжение тренда
            if self._last_values is not None and len(self._last_values) > 0:
                if len(self._last_values) >= 2:
                    avg_trend = np.mean(np.diff(self._last_values))
                    return np.array([self._last_values[-1] + i * avg_trend for i in range(1, steps + 1)])
                return np.full(steps, self._last_values[-1])
            return np.zeros(steps)


class MAModel(BaseModel):
    """Улучшенная MA(q) модель с обработкой ошибок и дополнительными проверками"""

    def __init__(self, q=1, name="MA"):
        super().__init__(name)
        self.q = q  # Сохраняем исходный порядок модели без изменений
        self._last_values = None  # Для резервного прогнозирования
        self._min_train_length = max(2 * q + 1, 10)  # Минимальная длина для обучения MA(q)

    def fit(self, train_data):
        try:
            # Сохраняем последние значения для резервного прогноза
            self._last_values = train_data.values[-min(5, len(train_data)):]

            # Проверка минимальной длины ряда
            if len(train_data) < self._min_train_length:
                raise ValueError(
                    f"Недостаточно данных для MA({self.q}). Нужно минимум {self._min_train_length} точек"
                )

            # Проверка на постоянный ряд (дисперсия близка к нулю)
            if np.var(train_data) < 1e-6:
                print("  - Предупреждение: ряд практически постоянный, MA модель может быть неэффективна")

            # Обучение модели с обработкой возможных ошибок
            self.model = ARIMA(
                train_data,
                order=(0, 0, self.q),
                enforce_stationarity=True,
                enforce_invertibility=True
            ).fit()

            # Проверка адекватности модели
            if any(np.isnan(self.model.params)):
                raise ValueError("Получены NaN параметры в модели")

            # Проверка значимости параметров
            if any(np.abs(self.model.params) < 1e-6):
                print("  - Предупреждение: некоторые параметры MA модели близки к нулю")

            self.is_fitted = True

        except Exception as e:
            print(f"  - Ошибка при обучении MA({self.q}): {str(e)}")
            self.is_fitted = False

    def predict(self, steps):
        if not self.is_fitted:
            # Простой прогноз на основе среднего последних значений
            if self._last_values is not None and len(self._last_values) > 0:
                last_value = np.mean(self._last_values)
                return np.full(steps, last_value)
            return np.zeros(steps)

        try:
            # Получаем прогноз от MA модели
            forecast = self.model.forecast(steps).values

            # Проверка на разумность прогноза
            if any(np.isnan(forecast)):
                raise ValueError("Прогноз содержит NaN значения")

            # Ограничение экстремальных значений на основе исторического диапазона
            if self._last_values is not None:
                historical_range = np.ptp(self._last_values)
                if historical_range > 0:
                    forecast_mean = np.mean(forecast)
                    forecast = np.clip(
                        forecast,
                        forecast_mean - historical_range,
                        forecast_mean + historical_range
                    )

            return forecast

        except Exception as e:
            print(f"  - Ошибка прогнозирования MA: {str(e)}")
            # Возвращаем среднее последних значений
            if self._last_values is not None and len(self._last_values) > 0:
                return np.full(steps, np.mean(self._last_values))
            return np.zeros(steps)


class VanillaRNN(BaseModel):
    """Улучшенная RNN модель для ряда C с оптимизированной архитектурой"""

    def __init__(self, units=64, epochs=100, window_size=12, batch_size=32, name="Improved Vanilla RNN"):
        super().__init__(name)
        self.window_size = window_size
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size

        # Улучшенная архитектура с BatchNormalization и улучшенной инициализацией
        self.model = Sequential([
            SimpleRNN(units,
                      input_shape=(self.window_size, 1),
                      kernel_initializer='he_normal',
                      return_sequences=False),
            BatchNormalization(),
            Dense(1, kernel_initializer='he_normal')
        ])

        # Оптимизированный оптимизатор с настроенным learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Добавлена метрика MAE для лучшего контроля обучения
        self.model.compile(optimizer=optimizer,
                           loss='mse',
                           metrics=['mae'])

        self.last_window = None
        self.is_fitted = False
        self.scaler = StandardScaler()

    def fit(self, train_data):
        train_values = train_data.values.reshape(-1, 1)

        # Нормализация данных
        self.scaler.fit(train_values)
        scaled_values = self.scaler.transform(train_values).flatten()

        # Генерация окон с использованием генератора для экономии памяти
        X, y = [], []
        for i in range(len(scaled_values) - self.window_size):
            X.append(scaled_values[i:i + self.window_size])
            y.append(scaled_values[i + self.window_size])

        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)

        # Добавлен EarlyStopping для предотвращения переобучения
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )

        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[early_stop]
        )

        self.last_window = scaled_values[-self.window_size:]
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

        # Обратное преобразование нормализованных данных
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()

        return predictions


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
                NaiveModel()  # Базовая модель для сравнения
            ]

        elif series_name == 'B':
            return [
                ARIMAModel(order=(1, 1, 0)),
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
                SESModel(smoothing_level=0.5),
                VanillaRNN(units=64, epochs=50, window_size=18, batch_size=16),
                GRUModel(units=64, epochs=12, window_size=24, batch_size=16),
                LSTMModel(units=16, epochs=200, window_size=12, batch_size=32),
                #NaiveModel()
            ]
        else:
            raise ValueError(f"Unknown series name: {series_name}")