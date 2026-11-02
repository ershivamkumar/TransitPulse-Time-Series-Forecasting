"""
TransitPulse - Exploratory Data Analysis & Visualization Module
Generates professional, publication-grade figures for ridership trends,
seasonality, decomposition, autocorrelation, residual diagnostics, and forecast performance.
"""

from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

# Professional typography and color palette
PALETTE = {
    "bg_dark": "#0F172A",
    "card_dark": "#1E293B",
    "primary": "#00B4D8",
    "secondary": "#0077B6",
    "accent": "#F59E0B",
    "highlight": "#10B981",
    "danger": "#EF4444",
    "text_light": "#F8FAFC",
    "text_muted": "#94A3B8",
    "grid": "#334155"
}

def apply_clean_style():
    """Apply consistent styling across all generated figures."""
    sns.set_theme(style="darkgrid", rc={
        "figure.facecolor": PALETTE["bg_dark"],
        "axes.facecolor": PALETTE["card_dark"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text_light"],
        "text.color": PALETTE["text_light"],
        "xtick.color": PALETTE["text_muted"],
        "ytick.color": PALETTE["text_muted"],
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold"
    })


def plot_historical_trend(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """1. Historical Daily Ridership Trend"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig_path = out / "historical_trend.png"

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.plot(df["date"], df["subway_ridership"] / 1e6, color=PALETTE["primary"], linewidth=1.2, alpha=0.85, label="Daily Ridership")
    
    ax.set_title("MTA New York Subway Daily Ridership (2020 – Present)", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Ridership (Millions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend(facecolor=PALETTE["card_dark"], edgecolor=PALETTE["grid"])
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_rolling_average(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """2. Seven-Day and 30-Day Rolling Averages"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "rolling_average_trend.png"

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.plot(df["date"], df["subway_ridership"] / 1e6, color=PALETTE["text_muted"], alpha=0.3, linewidth=0.8, label="Daily Observations")
    ax.plot(df["date"], df["rolling_7"] / 1e6, color=PALETTE["primary"], linewidth=1.8, label="7-Day Rolling Mean (Weekly Trend)")
    ax.plot(df["date"], df["rolling_30"] / 1e6, color=PALETTE["accent"], linewidth=2.0, linestyle="--", label="30-Day Rolling Mean (Monthly Trend)")
    
    ax.set_title("Subway Ridership Trend Smoothing (Rolling Averages)", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Ridership (Millions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend(facecolor=PALETTE["card_dark"], edgecolor=PALETTE["grid"])
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_monthly_average(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """3. Monthly Average Ridership Trend"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "monthly_average_ridership.png"

    df_monthly = df.set_index("date").resample("ME")["subway_ridership"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.bar(df_monthly["date"], df_monthly["subway_ridership"] / 1e6, width=22, color=PALETTE["secondary"], edgecolor=PALETTE["primary"], alpha=0.85)
    ax.plot(df_monthly["date"], df_monthly["subway_ridership"] / 1e6, color=PALETTE["accent"], linewidth=2.0, marker="o", markersize=4)

    ax.set_title("Monthly Mean Daily Subway Ridership", pad=15)
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Avg Daily Ridership (Millions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_weekly_seasonality(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """4. Weekly Seasonality: Weekdays vs Weekends"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "weekly_seasonality.png"

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_stats = df.groupby("day_name")["subway_ridership"].agg(["mean", "std"]).reindex(day_order).reset_index()

    colors = [PALETTE["primary"] if d not in ["Saturday", "Sunday"] else PALETTE["accent"] for d in weekly_stats["day_name"]]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.bar(weekly_stats["day_name"], weekly_stats["mean"] / 1e6, color=colors, edgecolor=PALETTE["text_light"], linewidth=0.8, alpha=0.9)
    
    # Value annotations on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}M",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=PALETTE["text_light"])

    ax.set_title("Weekly Demand Profile: Commuter Surge vs Weekend Drop", pad=15)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Ridership (Millions)")
    ax.set_ylim(0, max(weekly_stats["mean"] / 1e6) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_monthly_seasonality(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """5. Monthly Seasonality (Calendar Month Patterns)"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "monthly_seasonality.png"

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_agg = df.groupby("month_name")["subway_ridership"].mean().reindex(months).reset_index()

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.plot(monthly_agg["month_name"], monthly_agg["subway_ridership"] / 1e6, marker="s", markersize=8, color=PALETTE["highlight"], linewidth=2.2)
    ax.fill_between(range(len(months)), 0, monthly_agg["subway_ridership"] / 1e6, color=PALETTE["highlight"], alpha=0.15)

    ax.set_title("Seasonal Month-of-Year Ridership Profile", pad=15)
    ax.set_xlabel("Calendar Month")
    ax.set_ylabel("Average Ridership (Millions)")
    ax.set_ylim(0, max(monthly_agg["subway_ridership"].dropna() / 1e6) * 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_year_over_year(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """6. Year-over-Year Ridership Comparison"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "year_over_year.png"

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    years = sorted(df["year"].unique())
    year_palette = sns.color_palette("mako", n_colors=len(years))

    for idx, yr in enumerate(years):
        sub = df[df["year"] == yr]
        ax.plot(sub["day_of_year"], sub["rolling_7"] / 1e6, label=str(yr), color=year_palette[idx], linewidth=1.8)

    ax.set_title("Year-over-Year Daily Ridership Trajectory (7-Day Smoothed)", pad=15)
    ax.set_xlabel("Day of Year (1 - 365)")
    ax.set_ylabel("Ridership (Millions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend(title="Year", facecolor=PALETTE["card_dark"], edgecolor=PALETTE["grid"])
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_decomposition(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """7. Classical Time Series Decomposition (Trend, Seasonal, Residual)"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "time_series_decomposition.png"

    # Use the last 365 days for crystal clear seasonal visualization
    recent_series = df.set_index("date")["subway_ridership"].tail(365)
    decomp = seasonal_decompose(recent_series, model="additive", period=7)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, dpi=300)
    
    axes[0].plot(decomp.observed / 1e6, color=PALETTE["primary"], linewidth=1.2)
    axes[0].set_ylabel("Observed")
    axes[0].set_title("Additive Time Series Decomposition (s=7 Weekly Period)", pad=10)

    axes[1].plot(decomp.trend / 1e6, color=PALETTE["accent"], linewidth=1.8)
    axes[1].set_ylabel("Trend")

    axes[2].plot(decomp.seasonal / 1e6, color=PALETTE["highlight"], linewidth=1.0)
    axes[2].set_ylabel("Seasonal")

    axes[3].scatter(recent_series.index, decomp.resid / 1e6, color=PALETTE["text_muted"], alpha=0.6, s=12)
    axes[3].axhline(0, color=PALETTE["danger"], linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")

    for ax in axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_acf_pacf(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """8. Autocorrelation (ACF) and Partial Autocorrelation (PACF)"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "autocorrelation_acf_pacf.png"

    series = df["subway_ridership"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    plot_acf(series, lags=30, ax=axes[0], color=PALETTE["primary"], vlines_kwargs={"colors": PALETTE["primary"]})
    axes[0].set_title("Autocorrelation Function (ACF - 30 Lags)")
    axes[0].set_xlabel("Lag (Days)")
    axes[0].set_ylabel("Correlation")

    plot_pacf(series, lags=30, ax=axes[1], color=PALETTE["accent"], vlines_kwargs={"colors": PALETTE["accent"]}, method="ywm")
    axes[1].set_title("Partial Autocorrelation Function (PACF - 30 Lags)")
    axes[1].set_xlabel("Lag (Days)")
    axes[1].set_ylabel("Partial Correlation")

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_dir: Optional[Path] = None
) -> Path:
    """9. Actual vs Predicted on Holdout Test Window"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "actual_vs_predicted.png"

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    ax.plot(dates, y_true / 1e6, color=PALETTE["text_light"], linewidth=2.2, label="Actual Observations", marker="o", markersize=3)

    model_colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["highlight"], "#EC4899", "#A855F7"]
    for idx, (m_name, m_pred) in enumerate(predictions.items()):
        col = model_colors[idx % len(model_colors)]
        ax.plot(dates, m_pred / 1e6, linewidth=1.6, linestyle="--", color=col, label=f"Predicted ({m_name})")

    ax.set_title("Model Forecast vs Actual Ridership on Holdout Test Period", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Ridership (Millions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend(facecolor=PALETTE["card_dark"], edgecolor=PALETTE["grid"], loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_future_forecast(
    hist_dates: pd.Series,
    hist_values: np.ndarray,
    forecast_dates: pd.Series,
    forecast_values: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    model_name: str = "Best Model",
    output_dir: Optional[Path] = None
) -> Path:
    """10. Future Ridership Forecast with Confidence Interval"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "future_forecast.png"

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    
    # Historical context: last 30 days
    ax.plot(hist_dates, hist_values / 1e6, color=PALETTE["text_muted"], linewidth=1.8, marker="o", markersize=4, label="Historical Observations (Recent)")
    
    # Forecasted points
    ax.plot(forecast_dates, forecast_values / 1e6, color=PALETTE["primary"], linewidth=2.5, marker="s", markersize=6, label=f"Future Forecast ({model_name})")

    if lower_bound is not None and upper_bound is not None:
        ax.fill_between(
            forecast_dates,
            lower_bound / 1e6,
            upper_bound / 1e6,
            color=PALETTE["primary"],
            alpha=0.25,
            label="95% Forecast Interval"
        )

    # Connecting line between last observed and first forecast point
    h_last = hist_dates.iloc[-1] if hasattr(hist_dates, "iloc") else hist_dates[-1]
    f_first = forecast_dates.iloc[0] if hasattr(forecast_dates, "iloc") else forecast_dates[0]
    ax.plot([h_last, f_first], [hist_values[-1] / 1e6, forecast_values[0] / 1e6], color=PALETTE["primary"], linestyle=":", linewidth=1.5)

    ax.set_title(f"TransitPulse: Future Daily Subway Ridership Forecast ({len(forecast_dates)} Days)", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Ridership (Millions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend(facecolor=PALETTE["card_dark"], edgecolor=PALETTE["grid"])
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_model_comparison(df_comparison: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """11. Model Evaluation Metrics Comparison (MAE, RMSE, MAPE)"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "model_performance_comparison.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # MAE
    bars1 = axes[0].barh(df_comparison["Model"], df_comparison["MAE"] / 1e3, color=PALETTE["primary"], edgecolor=PALETTE["text_light"], linewidth=0.7)
    axes[0].set_title("Mean Absolute Error (Lower is better)")
    axes[0].set_xlabel("MAE (Thousands)")
    for b in bars1:
        w = b.get_width()
        axes[0].annotate(f"{w:.0f}K", xy=(w, b.get_y() + b.get_height() / 2), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold", color=PALETTE["text_light"])

    # RMSE
    bars2 = axes[1].barh(df_comparison["Model"], df_comparison["RMSE"] / 1e3, color=PALETTE["accent"], edgecolor=PALETTE["text_light"], linewidth=0.7)
    axes[1].set_title("Root Mean Squared Error (Lower is better)")
    axes[1].set_xlabel("RMSE (Thousands)")
    for b in bars2:
        w = b.get_width()
        axes[1].annotate(f"{w:.0f}K", xy=(w, b.get_y() + b.get_height() / 2), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold", color=PALETTE["text_light"])

    # MAPE
    bars3 = axes[2].barh(df_comparison["Model"], df_comparison["MAPE"], color=PALETTE["highlight"], edgecolor=PALETTE["text_light"], linewidth=0.7)
    axes[2].set_title("Mean Absolute Percentage Error (%)")
    axes[2].set_xlabel("MAPE (%)")
    for b in bars3:
        w = b.get_width()
        axes[2].annotate(f"{w:.1f}%", xy=(w, b.get_y() + b.get_height() / 2), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold", color=PALETTE["text_light"])

    for ax in axes:
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_residual_analysis(residuals: np.ndarray, dates: pd.Series, model_name: str, output_dir: Optional[Path] = None) -> Path:
    """12. Residual Diagnostic Plots (Error Time Series & Error Distribution)"""
    apply_clean_style()
    out = Path(output_dir) if output_dir else DEFAULT_FIGURES_DIR
    fig_path = out / "residual_analysis.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Residuals over time
    axes[0].plot(dates, residuals / 1e3, color=PALETTE["primary"], linewidth=1.2)
    axes[0].axhline(0, color=PALETTE["danger"], linestyle="--", linewidth=1.5)
    axes[0].set_title(f"Residual Errors Over Time ({model_name})")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Error (Thousands)")

    # Residual distribution
    sns.histplot(residuals / 1e3, kde=True, ax=axes[1], color=PALETTE["accent"], bins=20)
    axes[1].axvline(0, color=PALETTE["danger"], linestyle="--", linewidth=1.5)
    axes[1].set_title(f"Residual Error Distribution ({model_name})")
    axes[1].set_xlabel("Forecast Error (Thousands)")
    axes[1].set_ylabel("Density / Count")

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path
