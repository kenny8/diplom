# evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


class Evaluator:
    def __init__(self, scaler=None):
        self.scaler = scaler

    def calculate_metrics(self, y_true, y_pred):
        """Вычисляет метрики качества прогноза"""
        # Преобразование в numpy массивы
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Проверка размеров
        if len(y_true) != len(y_pred):
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        # Обратное преобразование данных
        if self.scaler:
            y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Вычисление метрик
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

        return {
            'MAE': mae,
            'RMSE': rmse,
            'sMAPE': smape
        }

    def run_statistical_tests(self, residuals):
        """Выполняет статистические тесты на остатках"""
        residuals = np.asarray(residuals)
        if len(residuals) == 0:
            return {}

        # Тест Люнга-Бокса на автокорреляцию
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].values[0]

        # Тест Дарбина-Уотсона на автокорреляцию
        dw_stat = durbin_watson(residuals)

        return {
            'Ljung-Box_pvalue': lb_pvalue,
            'Durbin-Watson': dw_stat
        }

    def plot_results(self, y_true, y_pred, model_name, save_path=None):
        """Визуализирует фактические и прогнозные значения"""
        try:
            # Преобразование в numpy массивы
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            # Проверка данных
            if y_true.size == 0 or y_pred.size == 0:
                print(f"  - Ошибка визуализации для {model_name}: нет данных для построения графика")
                return

            # Проверка совпадения размеров
            if len(y_true) != len(y_pred):
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

            # Создание фигуры
            plt.figure(figsize=(12, 6))

            # Создание временного индекса
            x_vals = np.arange(len(y_true))

            # Построение графиков
            plt.plot(x_vals, y_true, label='Фактические значения', color='blue')
            plt.plot(x_vals, y_pred, label='Прогноз', color='red', linestyle='--')

            # Настройка оформления
            plt.title(f'Факт vs Прогноз: {model_name}')
            plt.xlabel('Период')
            plt.ylabel('Значение')
            plt.legend()
            plt.grid(True)

            # Сохранение или отображение
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"  - Ошибка при построении графика для {model_name}: {str(e)}")

    def plot_residuals(self, residuals, model_name, save_path=None):
        """Визуализирует остатки модели"""
        try:
            residuals = np.asarray(residuals)
            if residuals.size == 0:
                return

            plt.figure(figsize=(12, 6))
            plt.plot(residuals)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title(f'Остатки модели: {model_name}')
            plt.xlabel('Период')
            plt.ylabel('Ошибка')
            plt.grid(True)

            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"  - Ошибка при построении остатков для {model_name}: {str(e)}")

    # evaluation.py (обновленная функция plot_components)
    def plot_components(self, trend, seasonal, residual, model_name, save_path=None):
        """Визуализирует компоненты временного ряда (если они доступны)"""
        try:
            # Проверяем, что все компоненты существуют и не пустые
            if trend is None or seasonal is None or residual is None:
                return

            # Преобразуем в numpy arrays
            trend = np.asarray(trend)
            seasonal = np.asarray(seasonal)
            residual = np.asarray(residual)

            # Проверяем, что есть данные для построения
            if len(trend) == 0 or len(seasonal) == 0 or len(residual) == 0:
                return

            # Создаем фигуру
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))

            # Общий тренд
            ax1.plot(trend)
            ax1.set_title('Тренд')
            ax1.grid(True)

            # Сезонность
            ax2.plot(seasonal)
            ax2.set_title('Сезонность')
            ax2.grid(True)

            # Остатки
            ax3.plot(residual)
            ax3.axhline(y=0, color='r', linestyle='-')
            ax3.set_title('Остатки')
            ax3.grid(True)

            # Все компоненты вместе
            ax4.plot(trend + seasonal + residual)
            ax4.set_title('Тренд + Сезонность + Остатки')
            ax4.grid(True)

            plt.suptitle(f'Декомпозиция ряда: {model_name}')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"  - Ошибка при построении компонентов для {model_name}: {str(e)}")


