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
        # Обратное преобразование данных, если задан scaler
        if self.scaler:
            y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Вычисление метрик
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

        return {
            'MAE': mae,
            'RMSE': rmse,
            'sMAPE': smape
        }

    def run_statistical_tests(self, residuals):
        """Выполняет статистические тесты на остатках"""
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
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Фактические значения', color='blue')
        plt.plot(y_pred, label='Прогноз', color='red', linestyle='--')
        plt.title(f'Факт vs Прогноз: {model_name}')
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_residuals(self, residuals, model_name, save_path=None):
        """Визуализирует остатки модели"""
        plt.figure(figsize=(12, 6))
        plt.plot(residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(f'Остатки модели: {model_name}')
        plt.xlabel('Время')
        plt.ylabel('Ошибка')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_components(self, trend, seasonal, residual, model_name, save_path=None):
        """Визуализирует компоненты временного ряда"""
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



