"""
TransitPulse - Model Utilities Module
Provides serialization, deserialization, metric tracking, and export utilities.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import logging
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def save_model(model: Any, filename: str, models_dir: Optional[Path] = None) -> Path:
    """
    Serialize a trained model or forecasting pipeline to disk.
    """
    target_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / filename
    joblib.dump(model, out_path)
    logger.info(f"Model successfully saved to {out_path}")
    return out_path


def load_model(filename: str, models_dir: Optional[Path] = None) -> Any:
    """
    Load a serialized model from disk.
    """
    target_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    in_path = target_dir / filename
    if not in_path.exists():
        raise FileNotFoundError(f"Model file not found: {in_path}")
    model = joblib.load(in_path)
    logger.info(f"Loaded model from {in_path}")
    return model


def save_metrics(metrics: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
    """
    Save evaluation metrics dictionary to JSON.
    """
    target_path = Path(filepath) if filepath else DEFAULT_REPORTS_DIR / "metrics.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to {target_path}")
    return target_path


def load_metrics(filepath: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load metrics dictionary from JSON.
    """
    target_path = Path(filepath) if filepath else DEFAULT_REPORTS_DIR / "metrics.json"
    if not target_path.exists():
        return {}
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_comparison_table(df_comp: pd.DataFrame, filepath: Optional[Path] = None) -> Path:
    """
    Save model performance comparison DataFrame to CSV.
    """
    target_path = Path(filepath) if filepath else DEFAULT_REPORTS_DIR / "model_comparison.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(target_path, index=False)
    logger.info(f"Saved model comparison table to {target_path}")
    return target_path


def save_forecast_results(df_forecast: pd.DataFrame, filepath: Optional[Path] = None) -> Path:
    """
    Save future forecast results DataFrame to CSV.
    """
    target_path = Path(filepath) if filepath else DEFAULT_REPORTS_DIR / "forecast_results.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df_forecast.to_csv(target_path, index=False)
    logger.info(f"Saved forecast results to {target_path}")
    return target_path
