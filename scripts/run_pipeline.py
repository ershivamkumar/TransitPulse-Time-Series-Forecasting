"""
TransitPulse - Master End-to-End Execution Pipeline
Runs data ingestion, preprocessing, EDA figure generation, time series model training,
chronological evaluation, model selection, residual diagnostics, and future forecasting.
"""

from pathlib import Path
import sys
import logging
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import download_mta_data, load_raw_data, validate_raw_schema
from src.preprocessing import preprocess_data
from src.eda import (
    plot_historical_trend,
    plot_rolling_average,
    plot_monthly_average,
    plot_weekly_seasonality,
    plot_monthly_seasonality,
    plot_year_over_year,
    plot_decomposition,
    plot_acf_pacf,
    plot_actual_vs_predicted,
    plot_future_forecast,
    plot_model_comparison,
    plot_residual_analysis
)
from src.evaluation import train_test_split_time_series, calculate_metrics, compare_models
from src.forecasting import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    ARIMAForecaster,
    SARIMAForecaster
)
from src.model_utils import (
    save_model,
    save_metrics,
    save_comparison_table,
    save_forecast_results
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TransitPulsePipeline")


def run_full_pipeline(test_size: float = 0.15, forecast_horizon: int = 7):
    logger.info("=" * 70)
    logger.info("TRANSITPULSE: STARTING END-TO-END PIPELINE")
    logger.info("=" * 70)

    # ----------------------------------------------------
    # STEP 1: DATA INGESTION
    # ----------------------------------------------------
    logger.info(">>> STEP 1: Data Ingestion & Schema Validation")
    raw_path = PROJECT_ROOT / "data" / "raw" / "mta_daily_ridership.csv"
    download_mta_data(dest_path=raw_path)
    df_raw = load_raw_data(raw_path)
    
    is_valid, msg = validate_raw_schema(df_raw)
    if not is_valid:
        raise ValueError(f"Schema validation failed: {msg}")
    logger.info(f"Raw data verified: {len(df_raw):,} records.")

    # ----------------------------------------------------
    # STEP 2: PREPROCESSING
    # ----------------------------------------------------
    logger.info(">>> STEP 2: Preprocessing & Temporal Feature Engineering")
    processed_path = PROJECT_ROOT / "data" / "processed" / "mta_subway_ridership_clean.csv"
    df_clean = preprocess_data(df_raw, save_path=processed_path)
    logger.info(f"Clean series spans {df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}")
    logger.info(f"Total valid daily observations: {len(df_clean):,}")

    # ----------------------------------------------------
    # STEP 3: EXPLORATORY DATA ANALYSIS FIGURES
    # ----------------------------------------------------
    logger.info(">>> STEP 3: Generating Professional EDA Figures")
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_historical_trend(df_clean, output_dir=fig_dir)
    plot_rolling_average(df_clean, output_dir=fig_dir)
    plot_monthly_average(df_clean, output_dir=fig_dir)
    plot_weekly_seasonality(df_clean, output_dir=fig_dir)
    plot_monthly_seasonality(df_clean, output_dir=fig_dir)
    plot_year_over_year(df_clean, output_dir=fig_dir)
    plot_decomposition(df_clean, output_dir=fig_dir)
    plot_acf_pacf(df_clean, output_dir=fig_dir)
    logger.info("Completed generating historical and seasonal EDA figures.")

    # ----------------------------------------------------
    # STEP 4: CHRONOLOGICAL TRAIN-TEST SPLIT
    # ----------------------------------------------------
    logger.info(">>> STEP 4: Chronological Train-Test Split (Zero Data Leakage)")
    train_df, test_df = train_test_split_time_series(df_clean, test_size=test_size)
    y_train = train_df["subway_ridership"].values
    y_test = test_df["subway_ridership"].values
    test_dates = test_df["date"]
    steps = len(y_test)
    logger.info(f"Training set: {len(y_train)} days | Holdout test set: {steps} days")

    # ----------------------------------------------------
    # STEP 5: TIME SERIES MODEL TRAINING & EVALUATION
    # ----------------------------------------------------
    logger.info(">>> STEP 5: Training & Evaluating Forecasting Models")
    
    models = {
        "Naive (Last Value)": NaiveForecaster(),
        "Seasonal Naive (s=7)": SeasonalNaiveForecaster(season_length=7),
        "Moving Average (k=7)": MovingAverageForecaster(window=7),
        "ARIMA (7,1,1)": ARIMAForecaster(order=(7, 1, 1)),
        "SARIMA (1,1,1)x(1,1,1,7)": SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    }

    test_predictions = {}
    metrics_summary = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(y_train)
        preds = model.predict(steps=steps)
        # Ensure 1D array
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds).flatten()[:steps]
        test_predictions[name] = preds

        metrics = calculate_metrics(y_test, preds)
        metrics_summary[name] = metrics
        logger.info(f"{name} -> MAE: {metrics['MAE']:,} | RMSE: {metrics['RMSE']:,} | MAPE: {metrics['MAPE']}% | sMAPE: {metrics['sMAPE']}%")

    df_comp = compare_models(metrics_summary, sort_by="MAE")
    save_comparison_table(df_comp)
    save_metrics(metrics_summary)
    logger.info("\n" + df_comp.to_string(index=False))

    best_model_name = df_comp.iloc[0]["Model"]
    logger.info(f">>> WINNING MODEL: {best_model_name} (Lowest holdout MAE: {df_comp.iloc[0]['MAE']:,})")

    # ----------------------------------------------------
    # STEP 6: EVALUATION VISUALIZATIONS
    # ----------------------------------------------------
    logger.info(">>> STEP 6: Generating Evaluation Visualizations")
    plot_actual_vs_predicted(test_dates, y_test, test_predictions, output_dir=fig_dir)
    plot_model_comparison(df_comp, output_dir=fig_dir)
    
    # Residual analysis for the best model
    best_pred = test_predictions[best_model_name]
    best_residuals = y_test - best_pred
    plot_residual_analysis(best_residuals, test_dates, model_name=best_model_name, output_dir=fig_dir)

    # ----------------------------------------------------
    # STEP 7: FUTURE FORECASTING (7 & 30 DAYS)
    # ----------------------------------------------------
    logger.info(">>> STEP 7: Generating Future Forecast on Full Historical Dataset")
    full_y = df_clean["subway_ridership"].values
    last_date = df_clean["date"].max()
    
    # Fit winning model on 100% of historical data
    refit_model = models[best_model_name]
    refit_model.fit(full_y)
    save_model(refit_model, "best_forecaster.joblib")

    # Predict future 7 and 30 days
    future_dates_30 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq="D")
    
    try:
        if hasattr(refit_model, "predict") and "return_conf_int" in refit_model.predict.__code__.co_varnames:
            preds_30, lower_30, upper_30 = refit_model.predict(30, return_conf_int=True)
        else:
            preds_30 = refit_model.predict(30)
            # Standard error heuristic based on train residuals
            train_err = np.std(full_y[7:] - full_y[:-7]) if len(full_y) > 7 else np.std(full_y)
            lower_30 = np.clip(preds_30 - 1.96 * train_err, a_min=0, a_max=None)
            upper_30 = preds_30 + 1.96 * train_err
    except Exception as exc:
        logger.warning(f"Error during future confidence interval calculation: {exc}")
        preds_30 = refit_model.predict(30)
        lower_30 = np.clip(preds_30 * 0.90, a_min=0, a_max=None)
        upper_30 = preds_30 * 1.10

    df_future = pd.DataFrame({
        "date": future_dates_30,
        "predicted_ridership": np.round(preds_30).astype(int),
        "lower_bound_95": np.round(lower_30).astype(int),
        "upper_bound_95": np.round(upper_30).astype(int),
        "day_name": future_dates_30.day_name()
    })
    save_forecast_results(df_future)
    logger.info("\nUpcoming 7-Day Subway Ridership Forecast:")
    logger.info("\n" + df_future.head(7).to_string(index=False))

    # Generate future forecast plot (7 days)
    plot_future_forecast(
        hist_dates=df_clean["date"].tail(28),
        hist_values=df_clean["subway_ridership"].tail(28).values,
        forecast_dates=future_dates_30[:7],
        forecast_values=preds_30[:7],
        lower_bound=lower_30[:7],
        upper_bound=upper_30[:7],
        model_name=best_model_name,
        output_dir=fig_dir
    )

    logger.info("=" * 70)
    logger.info("[SUCCESS] TransitPulse pipeline executed successfully!")
    logger.info(f"Artifacts saved in: {PROJECT_ROOT / 'reports'}")
    logger.info("=" * 70)
    return df_comp, df_future


if __name__ == "__main__":
    run_full_pipeline()
