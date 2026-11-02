"""
TransitPulse - Preprocessing Module
Extracts, cleans, transforms, and feature-engineers the daily subway ridership time series.
Enforces chronological ordering, handles missing dates, and prevents data leakage.
"""

from pathlib import Path
from typing import Optional, Tuple
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DEFAULT_PROCESSED_FILE = DEFAULT_PROCESSED_DIR / "mta_subway_ridership_clean.csv"


def extract_subway_series(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the raw long-format MTA data for subway ridership only.
    Standardizes column names and formats.
    """
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "mode" not in df.columns or "date" not in df.columns:
        raise ValueError(f"Expected 'mode' and 'date' in columns, found: {list(df.columns)}")

    # Standardize mode column
    df["mode_clean"] = df["mode"].astype(str).str.strip().str.lower()
    subway_mask = df["mode_clean"] == "subway"
    df_subway = df.loc[subway_mask].copy()

    if df_subway.empty:
        raise ValueError("No records found for mode='Subway' in raw data.")

    # Target column is 'count'
    count_col = "count" if "count" in df_subway.columns else [c for c in df_subway.columns if "ridership" in c or "count" in c][0]
    
    df_clean = pd.DataFrame({
        "date": pd.to_datetime(df_subway["date"], errors="coerce"),
        "subway_ridership": pd.to_numeric(df_subway[count_col], errors="coerce")
    })

    # Drop records where date or ridership is null
    df_clean = df_clean.dropna(subset=["date", "subway_ridership"])
    
    # Ensure non-negative ridership
    df_clean["subway_ridership"] = df_clean["subway_ridership"].clip(lower=0)
    
    # Sort chronologically
    df_clean = df_clean.sort_values("date").reset_index(drop=True)
    
    # Handle duplicate dates if any (average)
    if df_clean["date"].duplicated().any():
        logger.warning("Duplicate dates found in raw subway records. Aggregating by mean.")
        df_clean = df_clean.groupby("date", as_index=False)["subway_ridership"].mean()

    return df_clean


def regularize_daily_calendar(df_subway: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the series has a continuous daily calendar without skipped days.
    Small missing gaps are interpolated via linear time interpolation.
    """
    df = df_subway.copy().set_index("date")
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    
    missing_dates = full_idx.difference(df.index)
    if len(missing_dates) > 0:
        logger.warning(f"Detected {len(missing_dates)} missing calendar days. Interpolating missing days linearly.")
        df = df.reindex(full_idx)
        df["subway_ridership"] = df["subway_ridership"].interpolate(method="time")
    else:
        df = df.reindex(full_idx)
        
    df.index.name = "date"
    df = df.reset_index()
    return df


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar, lag, and rolling statistics for exploratory analysis.
    Note: Lags and rolling metrics are strictly computed on historical data.
    """
    df_feat = df.copy()
    
    # Calendar features
    df_feat["day_of_week"] = df_feat["date"].dt.dayofweek # 0=Monday, 6=Sunday
    df_feat["day_name"] = df_feat["date"].dt.day_name()
    df_feat["is_weekend"] = df_feat["day_of_week"].isin([5, 6]).astype(int)
    df_feat["month"] = df_feat["date"].dt.month
    df_feat["month_name"] = df_feat["date"].dt.strftime("%b")
    df_feat["year"] = df_feat["date"].dt.year
    df_feat["day_of_year"] = df_feat["date"].dt.dayofyear
    df_feat["quarter"] = df_feat["date"].dt.quarter

    # Rolling averages (7-day and 30-day for trend smoothing)
    df_feat["rolling_7"] = df_feat["subway_ridership"].rolling(window=7, min_periods=1).mean()
    df_feat["rolling_30"] = df_feat["subway_ridership"].rolling(window=30, min_periods=1).mean()
    
    # 7-day seasonal lag
    df_feat["lag_7"] = df_feat["subway_ridership"].shift(7)
    df_feat["pct_change_7"] = df_feat["subway_ridership"].pct_change(7) * 100

    return df_feat


def preprocess_data(
    df_raw: pd.DataFrame,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Filter Subway mode
    2. Regularize daily frequency
    3. Engineer temporal features
    4. Save to processed cache
    """
    logger.info("Starting subway ridership preprocessing pipeline...")
    df_subway = extract_subway_series(df_raw)
    df_daily = regularize_daily_calendar(df_subway)
    df_processed = engineer_temporal_features(df_daily)

    dest = Path(save_path) if save_path else DEFAULT_PROCESSED_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(dest, index=False)
    logger.info(f"Processed dataset saved to {dest} with {len(df_processed)} rows and {df_processed.shape[1]} columns.")
    logger.info(f"Date range: {df_processed['date'].min().strftime('%Y-%m-%d')} to {df_processed['date'].max().strftime('%Y-%m-%d')}")
    
    return df_processed


def load_processed_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load cleaned and preprocessed dataset.
    """
    path = Path(file_path) if file_path else DEFAULT_PROCESSED_FILE
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found at {path}. Run preprocessing first.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df
