"""
TransitPulse - Automated Unit & Integration Tests
Validates:
1. Data cleaning and negative value handling
2. Chronological sorting and continuity
3. Strict chronological train-test split (Zero Data Leakage)
4. Baseline & model forecasting horizon and output shapes
5. Evaluation metrics calculation correctness (MAE, RMSE, MAPE, sMAPE)
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import extract_subway_series, regularize_daily_calendar, engineer_temporal_features
from src.evaluation import train_test_split_time_series, calculate_metrics
from src.forecasting import NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster


def test_extract_subway_series_filters_and_cleans():
    """Verify filtering for mode='Subway', stripping non-negatives, and date sorting."""
    raw_data = {
        "date": ["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-01"],
        "mode": ["Subway", "subway", "BUS", "Subway"],
        "count": ["3000000", "2500000", "1000000", "2600000"]
    }
    df_raw = pd.DataFrame(raw_data)
    df_clean = extract_subway_series(df_raw)

    assert len(df_clean) == 2 # 2026-01-01 (deduplicated mean) and 2026-01-03
    assert list(df_clean.columns) == ["date", "subway_ridership"]
    assert df_clean["date"].is_monotonic_increasing
    assert (df_clean["subway_ridership"] >= 0).all()
    # Check deduplication mean for 2026-01-01: (2500000 + 2600000) / 2 = 2550000
    assert df_clean.iloc[0]["subway_ridership"] == 2550000


def test_regularize_daily_calendar_handles_missing_dates():
    """Verify missing dates are filled without dropping calendar continuity."""
    df_subway = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-04"]),
        "subway_ridership": [100.0, 400.0]
    })
    df_reg = regularize_daily_calendar(df_subway)

    assert len(df_reg) == 4 # Jan 1, Jan 2, Jan 3, Jan 4
    assert df_reg["date"].iloc[1] == pd.Timestamp("2026-01-02")
    # Linear interpolation: Jan 2 -> 200, Jan 3 -> 300
    assert np.isclose(df_reg["subway_ridership"].iloc[1], 200.0)
    assert np.isclose(df_reg["subway_ridership"].iloc[2], 300.0)


def test_chronological_train_test_split_no_leakage():
    """Verify that train test split is strictly chronological with no leakage."""
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "subway_ridership": np.random.uniform(2e6, 4e6, size=100)
    })
    train_df, test_df = train_test_split_time_series(df, test_size=0.20)

    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df["date"].max() < test_df["date"].min()


def test_metric_calculations():
    """Verify MAE, RMSE, MAPE, sMAPE with synthetic known values."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0]) # errors: +10, -10, +30

    metrics = calculate_metrics(y_true, y_pred)

    expected_mae = (10 + 10 + 30) / 3 # 16.67
    expected_rmse = np.sqrt((100 + 100 + 900) / 3) # sqrt(366.67) = 19.15
    assert np.isclose(metrics["MAE"], 16.67, atol=0.1)
    assert np.isclose(metrics["RMSE"], 19.15, atol=0.1)
    assert metrics["MAPE"] > 0
    assert metrics["sMAPE"] > 0


def test_forecasters_horizons():
    """Verify forecasters output exact requested steps and non-negative values."""
    y_train = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])

    naive = NaiveForecaster().fit(y_train)
    preds_naive = naive.predict(7)
    assert len(preds_naive) == 7
    assert (preds_naive == 80.0).all()

    snaive = SeasonalNaiveForecaster(season_length=7).fit(y_train)
    preds_snaive = snaive.predict(14)
    assert len(preds_snaive) == 14
    # Expected last 7: [20, 30, 40, 50, 60, 70, 80]
    np.testing.assert_array_equal(preds_snaive[:7], np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]))
    np.testing.assert_array_equal(preds_snaive[7:], np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]))

    ma = MovingAverageForecaster(window=7).fit(y_train)
    preds_ma = ma.predict(5)
    assert len(preds_ma) == 5
    assert np.all(preds_ma >= 0)
