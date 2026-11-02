"""
TransitPulse – Transit Ridership Forecasting & Trend Intelligence
Interactive Streamlit Dashboard
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="TransitPulse | Transit Ridership Forecasting",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# CUSTOM STYLING (DARK NAVY / MODERN MINIMALIST)
# ----------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Metrics / Cards */
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0B1120 !important;
        border-right: 1px solid #1E293B;
    }
    
    /* Header badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(14, 165, 233, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(14, 165, 233, 0.3);
        margin-right: 6px;
    }
    
    /* Dataframe tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# PATHS & DATA LOADING WITH CACHING
# ----------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT_DIR / "data" / "processed" / "mta_subway_ridership_clean.csv"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_FILE = REPORTS_DIR / "metrics.json"
COMP_FILE = REPORTS_DIR / "model_comparison.csv"
FORECAST_FILE = REPORTS_DIR / "forecast_results.csv"


@st.cache_data(show_spinner=False)
def load_data():
    """Load preprocessed subway ridership dataset."""
    if not DATA_PROCESSED.exists():
        return None
    df = pd.read_csv(DATA_PROCESSED)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_artifacts():
    """Load model metrics, comparison table, and future forecast results."""
    metrics = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    df_comp = pd.read_csv(COMP_FILE) if COMP_FILE.exists() else None
    df_forecast = pd.read_csv(FORECAST_FILE) if FORECAST_FILE.exists() else None
    if df_forecast is not None:
        df_forecast["date"] = pd.to_datetime(df_forecast["date"])

    return metrics, df_comp, df_forecast


df_clean = load_data()
metrics_dict, df_comp, df_forecast = load_artifacts()

# ----------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 10px 0 20px 0;">
    <h2 style="color: #38BDF8; margin: 0; font-size: 1.5rem; font-weight: 800;">TransitPulse</h2>
    <p style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">Subway Ridership Intelligence</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Historical Analysis",
        "Trend & Seasonality",
        "Forecasting",
        "Model Performance",
        "Dataset Information"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #64748B;">
    <b>Data Source:</b> NY State / MTA<br>
    <b>Cadence:</b> Daily Frequency<br>
    <b>Horizon:</b> 7 to 30 Days<br>
    <b>Model:</b> Seasonal Time Series
</div>
""", unsafe_allow_html=True)

# Safety check for pipeline execution
if df_clean is None:
    st.error("⚠️ Processed dataset not found. Please run the pipeline script first: `python scripts/run_pipeline.py`")
    st.stop()

# ----------------------------------------------------
# PAGE 1: OVERVIEW
# ----------------------------------------------------
if menu == "Overview":
    st.markdown("""
    <div>
        <span class="badge">Production ML</span>
        <span class="badge">MTA New York</span>
        <span class="badge">Time Series Intelligence</span>
    </div>
    <h1 style="color: #F8FAFC; margin-top: 8px; font-weight: 800; font-size: 2.2rem;">
        TransitPulse – Transit Ridership Forecasting
    </h1>
    <p style="color: #94A3B8; font-size: 1.05rem; margin-bottom: 24px;">
        Transforming historical public transportation demand patterns into actionable, high-precision demand forecasts.
    </p>
    """, unsafe_allow_html=True)

    # Key Performance Indicators
    latest_row = df_clean.iloc[-1]
    latest_val = latest_row["subway_ridership"]
    latest_date_str = latest_row["date"].strftime("%b %d, %Y")

    avg_val = df_clean["subway_ridership"].mean()
    max_val = df_clean["subway_ridership"].max()
    max_date_str = df_clean.loc[df_clean["subway_ridership"] == max_val, "date"].iloc[0].strftime("%b %d, %Y")
    min_val = df_clean["subway_ridership"].min()
    min_date_str = df_clean.loc[df_clean["subway_ridership"] == min_val, "date"].iloc[0].strftime("%b %d, %Y")

    # 7-day change
    if len(df_clean) > 7:
        val_7d_ago = df_clean.iloc[-8]["subway_ridership"]
        pct_7d = ((latest_val - val_7d_ago) / val_7d_ago) * 100 if val_7d_ago > 0 else 0
        delta_str = f"{pct_7d:+.1f}% vs last week"
    else:
        delta_str = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Observed Daily", f"{latest_val/1e6:.2f}M", delta=delta_str, help=f"Recorded on {latest_date_str}")
    c2.metric("Historical Mean", f"{avg_val/1e6:.2f}M", help="Mean daily ridership across entire series")
    c3.metric("Peak Ridership", f"{max_val/1e6:.2f}M", help=f"Recorded on {max_date_str}")
    c4.metric("Trough Ridership", f"{min_val/1e6:.2f}M", help=f"Recorded on {min_date_str}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Overview Chart
    st.subheader("Historical Demand Curve & 7-Day Trend")
    fig_overview = go.Figure()

    fig_overview.add_trace(go.Scatter(
        x=df_clean["date"],
        y=df_clean["subway_ridership"],
        name="Daily Observations",
        mode="lines",
        line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
        hoverinfo="x+y"
    ))

    fig_overview.add_trace(go.Scatter(
        x=df_clean["date"],
        y=df_clean["rolling_7"],
        name="7-Day Rolling Trend",
        mode="lines",
        line=dict(color="#00B4D8", width=2.5),
        hoverinfo="x+y"
    ))

    fig_overview.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        margin=dict(l=20, r=20, t=30, b=20),
        height=450,
        xaxis=dict(showgrid=True, gridcolor="#334155"),
        yaxis=dict(title="Ridership", showgrid=True, gridcolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_overview, use_container_width=True)

    # Automated Intelligence Callout
    st.info(f"""
    💡 **Trend Intelligence Summary:**
    - **Total Records:** {len(df_clean):,} daily observations ranging from **{df_clean['date'].min().strftime('%B %d, %Y')}** to **{df_clean['date'].max().strftime('%B %d, %Y')}**.
    - **Current Volume:** Latest observed subway ridership stands at **{latest_val:,.0f} passengers** ({latest_row['day_name']}).
    - **Weekly Dynamic:** Ridership exhibits severe weekly periodicity, with weekday commuter peaks approaching 4.5M+ and weekend recreational demand dropping ~40-50%.
    """)

# ----------------------------------------------------
# PAGE 2: HISTORICAL ANALYSIS
# ----------------------------------------------------
elif menu == "Historical Analysis":
    st.title("Historical Ridership Analysis")
    st.write("Explore daily transit volumes across customized date windows and temporal aggregation levels.")

    col_filter1, col_filter2 = st.columns([2, 1])
    min_date = df_clean["date"].min().date()
    max_date = df_clean["date"].max().date()

    with col_filter1:
        date_range = st.date_input(
            "Select Date Horizon",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    with col_filter2:
        agg_choice = st.selectbox("Temporal Aggregation", ["Daily", "Weekly (Mean)", "Monthly (Mean)"])

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_d, end_d = date_range
        mask = (df_clean["date"].dt.date >= start_d) & (df_clean["date"].dt.date <= end_d)
        df_filtered = df_clean.loc[mask].copy()
    else:
        df_filtered = df_clean.copy()

    # Aggregation
    if agg_choice == "Weekly (Mean)":
        df_plot = df_filtered.set_index("date").resample("W-SUN")["subway_ridership"].mean().reset_index()
    elif agg_choice == "Monthly (Mean)":
        df_plot = df_filtered.set_index("date").resample("ME")["subway_ridership"].mean().reset_index()
    else:
        df_plot = df_filtered.copy()

    fig_hist = px.line(
        df_plot,
        x="date",
        y="subway_ridership",
        labels={"subway_ridership": "Subway Ridership", "date": "Date"},
        title=f"Filtered Historical Subway Ridership ({agg_choice})"
    )
    fig_hist.update_traces(line_color="#38BDF8", line_width=2)
    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Statistical summary
    st.subheader("Descriptive Statistics")
    col_stat1, col_stat2 = st.columns([2, 1])
    with col_stat1:
        stats_df = df_filtered["subway_ridership"].describe().to_frame().T
        stats_df.columns = ["Count", "Mean", "Std Dev", "Min", "25%", "Median (50%)", "75%", "Max"]
        st.dataframe(stats_df.style.format("{:,.0f}"), use_container_width=True)

    with col_stat2:
        csv_data = df_filtered[["date", "subway_ridership"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name="transitpulse_filtered_ridership.csv",
            mime="text/csv",
            use_container_width=True
        )

# ----------------------------------------------------
# PAGE 3: TREND & SEASONALITY
# ----------------------------------------------------
elif menu == "Trend & Seasonality":
    st.title("Trend & Seasonality Intelligence")
    st.write("Deconstruct historical ridership into systematic cyclical patterns and long-term momentum.")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.subheader("Weekly Day-of-Week Profile")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_df = df_clean.groupby("day_name")["subway_ridership"].mean().reindex(day_order).reset_index()

        fig_dow = px.bar(
            dow_df,
            x="day_name",
            y="subway_ridership",
            color="subway_ridership",
            color_continuous_scale="Blues",
            labels={"day_name": "Day", "subway_ridership": "Avg Ridership"}
        )
        fig_dow.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            showlegend=False,
            height=380,
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155")
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_s2:
        st.subheader("Calendar Month Seasonality")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        moy_df = df_clean.groupby("month_name")["subway_ridership"].mean().reindex(months).reset_index()

        fig_moy = px.line(
            moy_df,
            x="month_name",
            y="subway_ridership",
            markers=True,
            labels={"month_name": "Month", "subway_ridership": "Avg Ridership"}
        )
        fig_moy.update_traces(line_color="#F59E0B", line_width=2.5, marker=dict(size=8, color="#F59E0B"))
        fig_moy.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            height=380,
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155")
        )
        st.plotly_chart(fig_moy, use_container_width=True)

    st.subheader("Time Series Classical Decomposition (Additive, Period=7)")
    decomp_img = REPORTS_DIR / "figures" / "time_series_decomposition.png"
    if decomp_img.exists():
        st.image(str(decomp_img), caption="Statistical Decomposition: Observed = Trend + Seasonal (7d) + Residual", use_column_width=True)
    else:
        st.info("Run the pipeline script to render high-resolution decomposition plots.")

# ----------------------------------------------------
# PAGE 4: FORECASTING
# ----------------------------------------------------
elif menu == "Forecasting":
    st.title("Future Subway Ridership Forecasting")
    st.write("Operational projections generated by the validated winning model fitted on all historical observations.")

    if df_forecast is None:
        st.warning("⚠️ No future forecast generated yet. Run `python scripts/run_pipeline.py`.")
    else:
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            horizon = st.radio("Forecast Horizon", ["7 Days (Recommended)", "30 Days (Extended)"])
            h_steps = 7 if "7" in horizon else 30
            df_sub_f = df_forecast.head(h_steps).copy()

            st.metric("Horizon Length", f"{h_steps} Days")
            st.metric("Projected Total Demand", f"{df_sub_f['predicted_ridership'].sum()/1e6:.2f}M")
            st.metric("Projected Daily Average", f"{df_sub_f['predicted_ridership'].mean()/1e6:.2f}M")

            f_csv = df_sub_f.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Forecast CSV",
                f_csv,
                f"transitpulse_forecast_{h_steps}d.csv",
                "text/csv",
                use_container_width=True
            )

        with col_f2:
            fig_f = go.Figure()

            # Past 21 days for context
            df_recent = df_clean.tail(21)
            fig_f.add_trace(go.Scatter(
                x=df_recent["date"],
                y=df_recent["subway_ridership"],
                name="Recent Observed",
                mode="lines+markers",
                line=dict(color="#94A3B8", width=2),
                marker=dict(size=4)
            ))

            # Forecast
            fig_f.add_trace(go.Scatter(
                x=df_sub_f["date"],
                y=df_sub_f["predicted_ridership"],
                name="Predicted Ridership",
                mode="lines+markers",
                line=dict(color="#00B4D8", width=3),
                marker=dict(size=6, symbol="diamond")
            ))

            # Confidence interval
            fig_f.add_trace(go.Scatter(
                x=pd.concat([df_sub_f["date"], df_sub_f["date"][::-1]]),
                y=pd.concat([df_sub_f["upper_bound_95"], df_sub_f["lower_bound_95"][::-1]]),
                fill="toself",
                fillcolor="rgba(0, 180, 216, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Forecast Interval"
            ))

            fig_f.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                height=450,
                xaxis=dict(title="Date", gridcolor="#334155"),
                yaxis=dict(title="Ridership", gridcolor="#334155"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_f, use_container_width=True)

        st.subheader("Tabular Forecast Matrix")
        st.dataframe(
            df_sub_f.style.format({
                "predicted_ridership": "{:,.0f}",
                "lower_bound_95": "{:,.0f}",
                "upper_bound_95": "{:,.0f}"
            }),
            use_container_width=True
        )

# ----------------------------------------------------
# PAGE 5: MODEL PERFORMANCE
# ----------------------------------------------------
elif menu == "Model Performance":
    st.title("Model Performance & Holdout Evaluation")
    st.write("Rigorous, strictly chronological holdout evaluation (Zero Data Leakage). Models are ranked by holdout MAE.")

    if df_comp is not None:
        best_name = df_comp.iloc[0]["Model"]
        st.success(f"🏆 **Top-Performing Model:** {best_name} (Lowest Holdout MAE: {df_comp.iloc[0]['MAE']:,})")
        st.dataframe(
            df_comp.style.format({
                "MAE": "{:,.2f}",
                "RMSE": "{:,.2f}",
                "MAPE": "{:.2f}%",
                "sMAPE": "{:.2f}%",
                "R2": "{:.4f}"
            }),
            use_container_width=True
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_mae = px.bar(
                df_comp,
                x="Model",
                y="MAE",
                color="MAE",
                color_continuous_scale="Viridis_r",
                title="Mean Absolute Error (Lower is Better)"
            )
            fig_mae.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                showlegend=False,
                height=350,
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155")
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        with col_m2:
            fig_mape = px.bar(
                df_comp,
                x="Model",
                y="MAPE",
                color="MAPE",
                color_continuous_scale="Plasma_r",
                title="Mean Absolute Percentage Error (%) (Lower is Better)"
            )
            fig_mape.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                showlegend=False,
                height=350,
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155")
            )
            st.plotly_chart(fig_mape, use_container_width=True)

        # Static High-Resolution Visuals
        st.subheader("Holdout Diagnostic Visualizations")
        c_diag1, c_diag2 = st.columns(2)
        act_img = REPORTS_DIR / "figures" / "actual_vs_predicted.png"
        resid_img = REPORTS_DIR / "figures" / "residual_analysis.png"

        with c_diag1:
            if act_img.exists():
                st.image(str(act_img), caption="Actual vs Model Predictions on Test Horizon", use_column_width=True)
        with c_diag2:
            if resid_img.exists():
                st.image(str(resid_img), caption="Residual Error Distribution & Series", use_column_width=True)

    else:
        st.info("Run `python scripts/run_pipeline.py` to produce model comparison metrics.")

# ----------------------------------------------------
# PAGE 6: DATASET INFORMATION
# ----------------------------------------------------
elif menu == "Dataset Information":
    st.title("Dataset Information & Governance")
    st.markdown("""
    ### Official Dataset Attribution
    - **Dataset Name:** MTA Daily Ridership and Traffic: Beginning 2020
    - **Publishing Authority:** Metropolitan Transportation Authority (MTA), State of New York
    - **Open Data Portal:** [data.ny.gov (Dataset sayj-mze2)](https://data.ny.gov/d/sayj-mze2)
    - **Socrata API Endpoint:** `https://data.ny.gov/resource/sayj-mze2.csv?$limit=50000`
    
    ### Scope & Definitions
    - **Target Series:** Daily Subway Ridership (`mode == 'Subway'`)
    - **Measurement:** Total estimated passenger throughput per calendar day.
    - **Temporal Resolution:** Daily observations with zero synthetic data invention.
    - **Exclusions:** Staten Island Railway (SIR) is tracked as a distinct operational entity and is isolated from subway metrics to prevent double counting.
    
    ### Known Real-World Limitations
    1. **Preliminary Estimates:** The most recent 2-5 days in MTA feeds are periodically revised as turnstile data reconciles.
    2. **Structural Regime Shifts:** The 2020 pandemic caused a severe structural rupture in transit ridership. Subsequent years show a steady return-to-office trajectory.
    3. **Exogenous Shocks:** Extreme weather (snowstorms, hurricane remnants) and federal holidays introduce sharp one-off variance that pure univariate time series models cannot anticipate without exogenous calendar flags.
    """)

    st.markdown("---")
    st.markdown("TransitPulse • Open-Source Machine Learning Portfolio Project • MIT License")
