"""
TransitPulse - Time Series Forecasting Models
Implements:
1. Naive Forecast (last observation)
2. Seasonal Naive Forecast (same day previous week)
3. Moving Average Forecast (rolling mean baseline)
4. ARIMA (Autoregressive Integrated Moving Average)
5. SARIMA (Seasonal ARIMA with weekly cycle s=7)

All models adhere to a unified scikit-learn style API: fit() and predict().
"""

from typing import Dict, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NaiveForecaster:
    """
    Baseline 1: Naive Forecast
    Predicts the last observed historical value for all future steps.
    y_hat_{t+h} = y_t
    """
    def __init__(self):
        self.last_value = None
        self.name = "Naive (Last Value)"

    def fit(self, y: Union[pd.Series, np.ndarray]):
        arr = np.asarray(y, dtype=float)
        if len(arr) == 0:
            raise ValueError("Training series cannot be empty.")
        self.last_value = float(arr[-1])
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self.last_value is None:
            raise RuntimeError("Model must be fitted before predict().")
        preds = np.full(steps, self.last_value, dtype=float)
        return np.clip(preds, a_min=0, a_max=None)


class SeasonalNaiveForecaster:
    """
    Baseline 2: Seasonal Naive Forecast
    Predicts the value observed in the identical seasonal cycle period (s=7 for weekly seasonality).
    y_hat_{t+h} = y_{t + h - s * ceil(h/s)}
    """
    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.last_cycle = None
        self.name = f"Seasonal Naive (s={season_length})"

    def fit(self, y: Union[pd.Series, np.ndarray]):
        arr = np.asarray(y, dtype=float)
        if len(arr) < self.season_length:
            raise ValueError(f"Series length {len(arr)} is shorter than seasonal length {self.season_length}")
        self.last_cycle = arr[-self.season_length:].copy()
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self.last_cycle is None:
            raise RuntimeError("Model must be fitted before predict().")
        # Tile the seasonal cycle for steps
        repeats = (steps // self.season_length) + 1
        preds = np.tile(self.last_cycle, repeats)[:steps]
        return np.clip(preds, a_min=0, a_max=None)


class MovingAverageForecaster:
    """
    Baseline 3: Moving Average Forecast
    Predicts the unweighted arithmetic mean of the last window observations.
    y_hat_{t+h} = mean(y_{t-window+1:t})
    """
    def __init__(self, window: int = 7):
        self.window = window
        self.ma_value = None
        self.name = f"Moving Average (k={window})"

    def fit(self, y: Union[pd.Series, np.ndarray]):
        arr = np.asarray(y, dtype=float)
        if len(arr) < self.window:
            raise ValueError(f"Series length {len(arr)} is shorter than window {self.window}")
        self.ma_value = float(np.mean(arr[-self.window:]))
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self.ma_value is None:
            raise RuntimeError("Model must be fitted before predict().")
        preds = np.full(steps, self.ma_value, dtype=float)
        return np.clip(preds, a_min=0, a_max=None)


class ARIMAForecaster:
    """
    Model 4: Autoregressive Integrated Moving Average
    Order: (p, d, q)
    Uses statsmodels ARIMA engine.
    """
    def __init__(self, order: Tuple[int, int, int] = (7, 1, 1)):
        self.order = order
        self.model_fitted = None
        self.name = f"ARIMA{order}"

    def fit(self, y: Union[pd.Series, np.ndarray]):
        arr = np.asarray(y, dtype=float)
        try:
            model = ARIMA(arr, order=self.order)
            self.model_fitted = model.fit()
        except Exception as exc:
            logger.warning(f"ARIMA{self.order} failed to converge with error: {exc}. Falling back to ARIMA(1,1,1)")
            self.order = (1, 1, 1)
            model = ARIMA(arr, order=self.order)
            self.model_fitted = model.fit()
        return self

    def predict(self, steps: int, return_conf_int: bool = False, alpha: float = 0.05) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.model_fitted is None:
            raise RuntimeError("Model must be fitted before predict().")
        forecast_res = self.model_fitted.get_forecast(steps=steps)
        preds = np.clip(forecast_res.predicted_mean, a_min=0, a_max=None)

        if return_conf_int:
            conf_int = forecast_res.conf_int(alpha=alpha)
            lower = np.clip(conf_int[:, 0], a_min=0, a_max=None)
            upper = np.clip(conf_int[:, 1], a_min=0, a_max=None)
            return preds, lower, upper
        return preds


class SARIMAForecaster:
    """
    Model 5: Seasonal ARIMA
    Order: (p, d, q) x (P, D, Q, s)
    Captures both non-seasonal autoregression and weekly seasonality (s=7).
    """
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fitted = None
        self.name = f"SARIMA{order}x{seasonal_order}"

    def fit(self, y: Union[pd.Series, np.ndarray]):
        arr = np.asarray(y, dtype=float)
        try:
            model = SARIMAX(
                arr,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            # Limit maxiter for fast, stable fitting
            self.model_fitted = model.fit(disp=False, maxiter=50)
        except Exception as exc:
            logger.warning(f"SARIMA fitting encountered {exc}. Trying fallback seasonal order (0, 1, 1, 7).")
            self.seasonal_order = (0, 1, 1, 7)
            model = SARIMAX(
                arr,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fitted = model.fit(disp=False, maxiter=50)
        return self

    def predict(self, steps: int, return_conf_int: bool = False, alpha: float = 0.05) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.model_fitted is None:
            raise RuntimeError("Model must be fitted before predict().")
        forecast_res = self.model_fitted.get_forecast(steps=steps)
        preds = np.clip(forecast_res.predicted_mean, a_min=0, a_max=None)

        if return_conf_int:
            conf_int = forecast_res.conf_int(alpha=alpha)
            lower = np.clip(conf_int[:, 0], a_min=0, a_max=None)
            upper = np.clip(conf_int[:, 1], a_min=0, a_max=None)
            return preds, lower, upper
        return preds
