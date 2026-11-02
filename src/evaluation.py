"""
TransitPulse - Evaluation Module
Provides strict chronological train-test splitting and robust time series performance metrics:
MAE, RMSE, MAPE, sMAPE, and R2.
Guarantees zero data leakage.
"""

from typing import Dict, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_test_split_time_series(
    df: pd.DataFrame,
    test_size: float = 0.20,
    date_col: str = "date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series strictly chronologically.
    Never uses random or shuffled splitting.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    split_idx = int(n * (1.0 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    # Safety check: no temporal overlap
    assert train_df[date_col].max() < test_df[date_col].min(), "Data leakage: Train date overlaps test date!"
    
    logger.info(
        f"Chronological Split: Train ({len(train_df)} rows, {train_df[date_col].min().strftime('%Y-%m-%d')} "
        f"to {train_df[date_col].max().strftime('%Y-%m-%d')}) | "
        f"Test ({len(test_df)} rows, {test_df[date_col].min().strftime('%Y-%m-%d')} "
        f"to {test_df[date_col].max().strftime('%Y-%m-%d')})"
    )
    return train_df, test_df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute time series forecasting metrics.
    Handles zero values gracefully to avoid infinite MAPE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Safe MAPE: exclude zeroes or tiny values
    mask = y_true > 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = 0.0

    # Symmetric MAPE (sMAPE)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    valid = denominator > 1e-8
    if np.any(valid):
        smape = float(np.mean(np.abs(y_pred[valid] - y_true[valid]) / denominator[valid]) * 100.0)
    else:
        smape = 0.0

    # R-squared
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE": round(float(mape), 2),
        "sMAPE": round(float(smape), 2),
        "R2": round(float(r2), 4)
    }


def compare_models(results_dict: Dict[str, Dict[str, float]], sort_by: str = "MAE") -> pd.DataFrame:
    """
    Format and rank multiple model metrics into a comparison DataFrame.
    """
    records = []
    for model_name, metrics in results_dict.items():
        rec = {"Model": model_name}
        rec.update(metrics)
        records.append(rec)
    
    df_comp = pd.DataFrame(records)
    if sort_by in df_comp.columns:
        df_comp = df_comp.sort_values(sort_by, ascending=True).reset_index(drop=True)
    df_comp["Rank"] = range(1, len(df_comp) + 1)
    
    cols = ["Rank", "Model", "MAE", "RMSE", "MAPE", "sMAPE", "R2"]
    return df_comp[[c for c in cols if c in df_comp.columns]]
