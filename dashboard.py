"""
🚨 Real-Time KPI Alert System — Interactive Dashboard

This Streamlit app provides a live, interactive demo of the entire
KPI monitoring and anomaly detection pipeline. Visitors can:
  - View real-time KPI data (simulated or Yahoo Finance)
  - Watch anomaly detection run live
  - Explore alerts by severity, type, and KPI
  - Adjust detection thresholds dynamically
  - Browse raw data and database records

Deploy: https://share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import json

# ─── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="KPI Alert System — Live Demo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Add project root to path ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings, KPIThreshold
from src.database.schema import initialize_database
from src.database.operations import DatabaseOperations
from src.ingestion.simulated import SimulatedDataSource
from src.detection.anomaly_detector import AnomalyDetector


# ─── Initialize Session State ──────────────────────────────────
# WHY: Streamlit reruns the entire script on every interaction.
# Session state preserves data between reruns so we don't re-fetch
# and re-detect on every button click.

if "data" not in st.session_state:
    st.session_state.data = None
if "anomalies" not in st.session_state:
    st.session_state.anomalies = []
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
if "run_metadata" not in st.session_state:
    st.session_state.run_metadata = {}


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR — Controls & Configuration
# ═══════════════════════════════════════════════════════════════

def render_sidebar():
    """Sidebar with data source selection, threshold controls, and run button."""

    st.sidebar.markdown("#  KPI Alert System")
    st.sidebar.markdown("### Interactive Demo")
    st.sidebar.markdown("---")

    # ─── Data Source ───────────────────────────────────────
    st.sidebar.markdown("##  Data Source")

    data_source = st.sidebar.selectbox(
        "Choose Data Source",
        ["Simulated (Recommended)", "Yahoo Finance (Live)"],
        help="Simulated data includes injected anomalies for demo purposes"
    )

    if data_source == "Simulated (Recommended)":
        num_days = st.sidebar.slider("Days of History", 30, 180, 90)
        anomaly_rate = st.sidebar.slider(
            "Anomaly Injection Rate",
            0.01, 0.20, 0.05,
            help="Higher = more anomalies injected into simulated data"
        )
    else:
        num_days = 90
        anomaly_rate = 0.05
        symbols = st.sidebar.multiselect(
            "Stock Symbols",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
            default=["AAPL", "MSFT", "GOOGL"]
        )

    st.sidebar.markdown("---")

    # ─── Detection Thresholds ──────────────────────────────
    st.sidebar.markdown("## 🔧 Detection Settings")

    z_score_thresh = st.sidebar.slider(
        "Z-Score Threshold",
        1.0, 5.0, 2.5, 0.1,
        help="Flag values this many std devs from mean"
    )

    pct_change_thresh = st.sidebar.slider(
        "% Change Threshold",
        0.01, 0.50, 0.10, 0.01,
        help="Flag day-over-day changes exceeding this %"
    )

    rolling_window = st.sidebar.slider(
        "Rolling Window (days)",
        5, 50, 20,
        help="Lookback period for rolling average calculations"
    )

    st.sidebar.markdown("---")

    # ─── Run Pipeline Button ───────────────────────────────
    run_clicked = st.sidebar.button(
        " Run Pipeline",
        type="primary",
        use_container_width=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Built by [Niv Patel](https://github.com/Nivpatel23/Real_Time_Kpi_Alert_System)**\n\n"
        "[📂 View Source Code](https://github.com/Nivpatel23/Real_Time_Kpi_Alert_System)"
    )

    return {
        "data_source": data_source,
        "num_days": num_days,
        "anomaly_rate": anomaly_rate,
        "symbols": symbols if data_source != "Simulated (Recommended)" else [],
        "z_score_threshold": z_score_thresh,
        "pct_change_threshold": pct_change_thresh,
        "rolling_window": rolling_window,
        "run_clicked": run_clicked,
    }


# ═══════════════════════════════════════════════════════════════
#  DATA GENERATION & DETECTION
# ═══════════════════════════════════════════════════════════════

def run_pipeline(config: dict):
    """Execute the full pipeline and store results in session state."""

    start_time = datetime.now()

    # ─── Step 1: Generate / Fetch Data ─────────────────────
    with st.spinner("📊 Fetching KPI data..."):
        if config["data_source"] == "Simulated (Recommended)":
            source = SimulatedDataSource(
                num_days=config["num_days"],
                anomaly_rate=config["anomaly_rate"],
                seed=None  # Random each run for variety
            )
            data = source.fetch_data()
        else:
            try:
                from src.ingestion.yahoo_finance import YahooFinanceDataSource
                source = YahooFinanceDataSource(
                    symbols=config["symbols"],
                    period="3mo"
                )
                data = source.fetch_data()
            except Exception as e:
                st.error(f"Failed to fetch Yahoo Finance data: {e}")
                st.info("Falling back to simulated data...")
                source = SimulatedDataSource(
                    num_days=config["num_days"],
                    anomaly_rate=config["anomaly_rate"]
                )
                data = source.fetch_data()

    st.session_state.data = data

    # ─── Step 2: Run Anomaly Detection ─────────────────────
    with st.spinner("🔍 Running anomaly detection..."):
        detector = AnomalyDetector()

        # Override thresholds with sidebar values
        for kpi_name in data["kpi_name"].unique():
            detector.detection_config.default_thresholds[kpi_name] = KPIThreshold(
                kpi_name=kpi_name,
                z_score_threshold=config["z_score_threshold"],
                pct_change_threshold=config["pct_change_threshold"],
                rolling_window=config["rolling_window"],
            )

        anomalies = detector.analyze_batch(data)
        anomaly_dicts = [a.to_dict() for a in anomalies]

    st.session_state.anomalies = anomaly_dicts
    st.session_state.pipeline_run = True

    # ─── Step 3: Store metadata ────────────────────────────
    duration = (datetime.now() - start_time).total_seconds()
    st.session_state.run_metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": config["data_source"],
        "rows_ingested": len(data),
        "kpis_monitored": data["kpi_name"].nunique(),
        "anomalies_detected": len(anomaly_dicts),
        "duration_seconds": round(duration, 2),
    }

    # ─── Step 4: Store to database ─────────────────────────
    try:
        initialize_database()
        db_ops = DatabaseOperations()
        db_ops.insert_kpi_readings(data)
        for alert in anomaly_dicts:
            db_ops.insert_alert(alert)
    except Exception:
        pass  # DB storage is optional for the demo


# ═══════════════════════════════════════════════════════════════
#  MAIN PAGE — Header & Metrics
# ═══════════════════════════════════════════════════════════════

def render_header():
    """Top section with title and pipeline status cards."""

    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px 0;">
        <h1 style="margin-bottom:0;">🚨 Real-Time KPI Alert System</h1>
        <p style="color:gray; font-size:1.1em;">
            Multi-Strategy Anomaly Detection • SQL Storage • Automated Alerting
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Status Cards
    if st.session_state.pipeline_run:
        meta = st.session_state.run_metadata
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("📊 Data Points", f"{meta['rows_ingested']:,}")
        col2.metric("📈 KPIs Monitored", meta["kpis_monitored"])
        col3.metric(
            "🚨 Anomalies",
            meta["anomalies_detected"],
            delta=f"{meta['anomalies_detected']} detected",
            delta_color="inverse" if meta["anomalies_detected"] > 0 else "off"
        )
        col4.metric("⏱️ Duration", f"{meta['duration_seconds']}s")
        col5.metric("🕐 Last Run", meta["timestamp"].split(" ")[1])
    else:
        st.info("👈 Configure settings in the sidebar and click **Run Pipeline** to start the demo.")


# ═══════════════════════════════════════════════════════════════
#  TAB 1: KPI Time Series Charts
# ═══════════════════════════════════════════════════════════════

def render_kpi_charts():
    """Interactive time series charts with anomaly markers."""

    data = st.session_state.data
    anomalies = st.session_state.anomalies

    if data is None:
        return

    st.markdown("## 📈 KPI Time Series with Anomaly Detection")

    # KPI selector
    kpi_names = sorted(data["kpi_name"].unique())
    selected_kpi = st.selectbox("Select KPI to Visualize", kpi_names)

    # Filter data for selected KPI
    kpi_data = data[data["kpi_name"] == selected_kpi].copy()
    kpi_data["timestamp"] = pd.to_datetime(kpi_data["timestamp"])
    kpi_data = kpi_data.sort_values("timestamp")

    # Check if there are multiple symbols
    has_symbols = "symbol" in kpi_data.columns and kpi_data["symbol"].notna().any()

    if has_symbols:
        symbols = sorted(kpi_data["symbol"].dropna().unique())
        if len(symbols) > 1:
            selected_symbols = st.multiselect(
                "Filter by Symbol",
                symbols,
                default=symbols[:3]
            )
            kpi_data = kpi_data[kpi_data["symbol"].isin(selected_symbols)]

    # ─── Build Chart ───────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{selected_kpi} — Time Series",
            f"{selected_kpi} — Value Distribution"
        )
    )

    if has_symbols and kpi_data["symbol"].notna().any():
        for symbol in kpi_data["symbol"].dropna().unique():
            sym_data = kpi_data[kpi_data["symbol"] == symbol]
            fig.add_trace(
                go.Scatter(
                    x=sym_data["timestamp"],
                    y=sym_data["value"],
                    mode="lines",
                    name=symbol,
                    line=dict(width=2),
                ),
                row=1, col=1
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=kpi_data["timestamp"],
                y=kpi_data["value"],
                mode="lines",
                name=selected_kpi,
                line=dict(color="#1f77b4", width=2),
            ),
            row=1, col=1
        )

    # ─── Add Rolling Average Line ──────────────────────────
    if len(kpi_data) >= 20:
        kpi_data["rolling_mean"] = kpi_data["value"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=kpi_data["timestamp"],
                y=kpi_data["rolling_mean"],
                mode="lines",
                name="20-Day Rolling Avg",
                line=dict(color="orange", width=1, dash="dash"),
                opacity=0.7,
            ),
            row=1, col=1
        )

    # ─── Add Anomaly Markers ──────────────────────────────
    kpi_anomalies = [a for a in anomalies if a["kpi_name"] == selected_kpi]

    if kpi_anomalies:
        # Get timestamps of anomalous values
        anomaly_values = [a["kpi_value"] for a in kpi_anomalies]
        severity_colors = {
            "CRITICAL": "red",
            "HIGH": "orange",
            "MEDIUM": "yellow",
            "LOW": "lightblue",
        }

        for anomaly in kpi_anomalies:
            color = severity_colors.get(anomaly["severity"], "gray")
            # Find the closest timestamp for this anomaly value
            closest_idx = (kpi_data["value"] - anomaly["kpi_value"]).abs().idxmin()
            anomaly_ts = kpi_data.loc[closest_idx, "timestamp"]

            fig.add_trace(
                go.Scatter(
                    x=[anomaly_ts],
                    y=[anomaly["kpi_value"]],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=color,
                        symbol="x",
                        line=dict(width=2, color="black")
                    ),
                    name=f"{anomaly['severity']} — {anomaly['alert_type']}",
                    hovertemplate=(
                        f"<b>{anomaly['severity']}</b><br>"
                        f"Type: {anomaly['alert_type']}<br>"
                        f"Value: {anomaly['kpi_value']:.4f}<br>"
                        f"{anomaly['message']}<extra></extra>"
                    ),
                    showlegend=True,
                ),
                row=1, col=1
            )

    # ─── Distribution Histogram ────────────────────────────
    fig.add_trace(
        go.Histogram(
            x=kpi_data["value"],
            nbinsx=40,
            marker_color="#1f77b4",
            opacity=0.7,
            name="Distribution",
            showlegend=False,
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ─── Statistics Box ────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    values = kpi_data["value"]
    col1.metric("Mean", f"{values.mean():.4f}")
    col2.metric("Std Dev", f"{values.std():.4f}")
    col3.metric("Min", f"{values.min():.4f}")
    col4.metric("Max", f"{values.max():.4f}")


# ═══════════════════════════════════════════════════════════════
#  TAB 2: Alert Dashboard
# ═══════════════════════════════════════════════════════════════

def render_alert_dashboard():
    """Interactive alert explorer with filters and charts."""

    anomalies = st.session_state.anomalies

    if not anomalies:
        st.info("No anomalies detected in this run. Try increasing the anomaly rate or lowering thresholds.")
        return

    st.markdown("## 🚨 Alert Dashboard")

    alert_df = pd.DataFrame(anomalies)

    # ─── Summary Row ───────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    critical = len(alert_df[alert_df["severity"] == "CRITICAL"])
    high = len(alert_df[alert_df["severity"] == "HIGH"])
    medium = len(alert_df[alert_df["severity"] == "MEDIUM"])
    low = len(alert_df[alert_df["severity"] == "LOW"])

    col1.metric("🔴 Critical", critical)
    col2.metric("🟠 High", high)
    col3.metric("🟡 Medium", medium)
    col4.metric("🔵 Low", low)

    st.markdown("---")

    # ─── Charts Row ────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Alerts by Severity — Pie Chart
        severity_counts = alert_df["severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]

        color_map = {
            "CRITICAL": "#dc3545",
            "HIGH": "#fd7e14",
            "MEDIUM": "#ffc107",
            "LOW": "#17a2b8",
        }

        fig_pie = px.pie(
            severity_counts,
            values="Count",
            names="Severity",
            title="Alerts by Severity",
            color="Severity",
            color_discrete_map=color_map,
            hole=0.4,
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        # Alerts by Type — Bar Chart
        type_counts = alert_df["alert_type"].value_counts().reset_index()
        type_counts.columns = ["Detection Method", "Count"]

        fig_bar = px.bar(
            type_counts,
            x="Detection Method",
            y="Count",
            title="Alerts by Detection Method",
            color="Count",
            color_continuous_scale="Reds",
        )
        fig_bar.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ─── Alerts by KPI ─────────────────────────────────────
    kpi_severity = alert_df.groupby(["kpi_name", "severity"]).size().reset_index(name="count")

    fig_kpi = px.bar(
        kpi_severity,
        x="kpi_name",
        y="count",
        color="severity",
        title="Alert Distribution by KPI and Severity",
        color_discrete_map=color_map,
        barmode="stack",
    )
    fig_kpi.update_layout(height=400)
    st.plotly_chart(fig_kpi, use_container_width=True)

    # ─── Filters ───────────────────────────────────────────
    st.markdown("### 🔎 Filter Alerts")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        severity_filter = st.multiselect(
            "Severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        )

    with filter_col2:
        type_filter = st.multiselect(
            "Detection Type",
            alert_df["alert_type"].unique().tolist(),
            default=alert_df["alert_type"].unique().tolist()
        )

    with filter_col3:
        kpi_filter = st.multiselect(
            "KPI Name",
            alert_df["kpi_name"].unique().tolist(),
            default=alert_df["kpi_name"].unique().tolist()
        )

    # Apply filters
    filtered = alert_df[
        (alert_df["severity"].isin(severity_filter)) &
        (alert_df["alert_type"].isin(type_filter)) &
        (alert_df["kpi_name"].isin(kpi_filter))
    ]

    # ─── Alert Table ───────────────────────────────────────
    st.markdown(f"### 📋 Alert Details ({len(filtered)} alerts)")

    display_cols = ["severity", "kpi_name", "alert_type", "kpi_value", "z_score", "message"]
    available_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[available_cols].style.apply(
            lambda row: [
                f"background-color: {'#ffcccc' if row['severity'] == 'CRITICAL' else '#fff3cd' if row['severity'] in ('HIGH', 'MEDIUM') else '#cce5ff'}"
                for _ in row
            ],
            axis=1
        ),
        use_container_width=True,
        height=400,
    )


# ═══════════════════════════════════════════════════════════════
#  TAB 3: Raw Data Explorer
# ═══════════════════════════════════════════════════════════════

def render_data_explorer():
    """Browse raw KPI data with search and download."""

    data = st.session_state.data

    if data is None:
        return

    st.markdown("## 🗃️ Data Explorer")

    # ─── Data Summary ──────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Dataset Info")
        st.write(f"- **Total Rows:** {len(data):,}")
        st.write(f"- **KPIs:** {data['kpi_name'].nunique()}")
        st.write(f"- **Date Range:** {data['timestamp'].min()} → {data['timestamp'].max()}")
        st.write(f"- **Source:** {data['source'].unique().tolist()}")

    with col2:
        st.markdown("### Per-KPI Row Counts")
        kpi_counts = data["kpi_name"].value_counts()
        for kpi, count in kpi_counts.items():
            st.write(f"- **{kpi}:** {count} rows")

    with col3:
        st.markdown("### Value Statistics")
        for kpi in data["kpi_name"].unique():
            vals = data[data["kpi_name"] == kpi]["value"]
            st.write(f"**{kpi}:** μ={vals.mean():.2f}, σ={vals.std():.2f}")

    st.markdown("---")

    # ─── Correlation Matrix ────────────────────────────────
    st.markdown("### 📊 KPI Correlation Matrix")

    pivot = data.pivot_table(
        index="timestamp",
        columns="kpi_name",
        values="value",
        aggfunc="mean"
    )

    if len(pivot.columns) >= 2:
        corr = pivot.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="KPI Correlation Heatmap",
            aspect="auto",
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # ─── Raw Data Table ────────────────────────────────────
    st.markdown("### 🔍 Browse Raw Data")

    # Filter controls
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        kpi_filter = st.selectbox("Filter by KPI", ["All"] + sorted(data["kpi_name"].unique().tolist()))
    with fcol2:
        sort_order = st.selectbox("Sort by Timestamp", ["Newest First", "Oldest First"])

    display_data = data.copy()
    if kpi_filter != "All":
        display_data = display_data[display_data["kpi_name"] == kpi_filter]

    ascending = sort_order == "Oldest First"
    display_data = display_data.sort_values("timestamp", ascending=ascending)

    st.dataframe(display_data, use_container_width=True, height=400)

    # ─── Download Button ───────────────────────────────────
    csv = display_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"kpi_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════
#  TAB 4: Detection Engine Deep Dive
# ═══════════════════════════════════════════════════════════════

def render_detection_deep_dive():
    """Visual explanation of each detection strategy."""

    data = st.session_state.data

    if data is None:
        return

    st.markdown("## 🔬 Detection Engine — Deep Dive")
    st.markdown(
        "Explore how each anomaly detection strategy works on your data. "
        "Select a KPI to see all four methods in action."
    )

    # KPI selector
    kpi_names = sorted(data["kpi_name"].unique())
    selected_kpi = st.selectbox("Select KPI for Analysis", kpi_names, key="deep_dive_kpi")

    kpi_data = data[data["kpi_name"] == selected_kpi].copy()
    kpi_data["timestamp"] = pd.to_datetime(kpi_data["timestamp"])
    kpi_data = kpi_data.sort_values("timestamp").reset_index(drop=True)
    values = kpi_data["value"].values

    # ─── Strategy 1: Z-Score ──────────────────────────────
    st.markdown("### 📐 Strategy 1: Z-Score Analysis")
    st.markdown(
        "> Z-score = (value - mean) / std. Flags values that are statistically "
        "unusual compared to the entire history."
    )

    mean_val = np.mean(values)
    std_val = np.std(values)
    z_scores = (values - mean_val) / std_val if std_val > 0 else np.zeros_like(values)

    kpi_data["z_score"] = z_scores

    fig_z = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                          subplot_titles=("KPI Values with Mean ± 2.5σ Bands", "Z-Score Over Time"))

    fig_z.add_trace(go.Scatter(
        x=kpi_data["timestamp"], y=values,
        mode="lines", name="Value", line=dict(color="#1f77b4")
    ), row=1, col=1)

    fig_z.add_hline(y=mean_val, line_dash="dash", line_color="green",
                    annotation_text="Mean", row=1, col=1)
    fig_z.add_hline(y=mean_val + 2.5 * std_val, line_dash="dot",
                    line_color="red", annotation_text="+2.5σ", row=1, col=1)
    fig_z.add_hline(y=mean_val - 2.5 * std_val, line_dash="dot",
                    line_color="red", annotation_text="-2.5σ", row=1, col=1)

    # Z-Score plot
    colors = ["red" if abs(z) > 2.5 else "#1f77b4" for z in z_scores]
    fig_z.add_trace(go.Bar(
        x=kpi_data["timestamp"], y=z_scores,
        marker_color=colors, name="Z-Score", showlegend=False
    ), row=2, col=1)

    fig_z.add_hline(y=2.5, line_dash="dash", line_color="red", row=2, col=1)
    fig_z.add_hline(y=-2.5, line_dash="dash", line_color="red", row=2, col=1)

    fig_z.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig_z, use_container_width=True)

    outlier_count = np.sum(np.abs(z_scores) > 2.5)
    st.success(f"**Z-Score Results:** {outlier_count} values exceed ±2.5σ threshold")

    # ─── Strategy 2: Rolling Average ──────────────────────
    st.markdown("### 📊 Strategy 2: Rolling Average Deviation")
    st.markdown(
        "> Compares each value against the recent rolling mean. Catches local "
        "anomalies that global z-score might miss."
    )

    window = 20
    kpi_data["rolling_mean"] = kpi_data["value"].rolling(window).mean()
    kpi_data["rolling_std"] = kpi_data["value"].rolling(window).std()
    kpi_data["upper_band"] = kpi_data["rolling_mean"] + 2.5 * kpi_data["rolling_std"]
    kpi_data["lower_band"] = kpi_data["rolling_mean"] - 2.5 * kpi_data["rolling_std"]

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=kpi_data["timestamp"], y=kpi_data["value"],
        mode="lines", name="Value", line=dict(color="#1f77b4")
    ))
    fig_roll.add_trace(go.Scatter(
        x=kpi_data["timestamp"], y=kpi_data["rolling_mean"],
        mode="lines", name=f"{window}-Day Rolling Mean",
        line=dict(color="orange", dash="dash")
    ))
    fig_roll.add_trace(go.Scatter(
        x=kpi_data["timestamp"], y=kpi_data["upper_band"],
        mode="lines", name="Upper Band (+2.5σ)",
        line=dict(color="red", dash="dot"), opacity=0.5
    ))
    fig_roll.add_trace(go.Scatter(
        x=kpi_data["timestamp"], y=kpi_data["lower_band"],
        mode="lines", name="Lower Band (-2.5σ)",
        line=dict(color="red", dash="dot"), opacity=0.5,
        fill="tonexty", fillcolor="rgba(255,0,0,0.05)"
    ))

    fig_roll.update_layout(
        height=400, template="plotly_white",
        title=f"Rolling Average Analysis (window={window})"
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    # ─── Strategy 3: % Change ─────────────────────────────
    st.markdown("### 📉 Strategy 3: Percentage Change Detection")
    st.markdown(
        "> Measures day-over-day velocity of change. Catches sudden shocks "
        "regardless of absolute level."
    )

    kpi_data["pct_change"] = kpi_data["value"].pct_change() * 100
    threshold_pct = 10  # 10%

    fig_pct = go.Figure()
    pct_colors = [
        "red" if abs(p) > threshold_pct else "#1f77b4"
        for p in kpi_data["pct_change"].fillna(0)
    ]
    fig_pct.add_trace(go.Bar(
        x=kpi_data["timestamp"],
        y=kpi_data["pct_change"],
        marker_color=pct_colors,
        name="% Change"
    ))
    fig_pct.add_hline(y=threshold_pct, line_dash="dash", line_color="red",
                      annotation_text=f"+{threshold_pct}%")
    fig_pct.add_hline(y=-threshold_pct, line_dash="dash", line_color="red",
                      annotation_text=f"-{threshold_pct}%")

    fig_pct.update_layout(
        height=350, template="plotly_white",
        title="Day-over-Day % Change",
        yaxis_title="% Change"
    )
    st.plotly_chart(fig_pct, use_container_width=True)

    spikes = kpi_data["pct_change"].abs() > threshold_pct
    st.success(f"**% Change Results:** {spikes.sum()} values exceed ±{threshold_pct}% threshold")


# ═══════════════════════════════════════════════════════════════
#  TAB 5: Architecture & About
# ═══════════════════════════════════════════════════════════════

def render_architecture():
    """System architecture and project information."""

    st.markdown("## 🏗️ System Architecture")

    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────┐
    │              REAL-TIME KPI ALERT SYSTEM                   │
    ├──────────────────────────────────────────────────────────┤
    │                                                          │
    │   DATA SOURCES                                           │
    │   ├── Yahoo Finance API (AAPL, MSFT, GOOGL)             │
    │   └── Simulated Generator (with anomaly injection)       │
    │              │                                            │
    │              ▼                                            │
    │   INGESTION LAYER                                        │
    │   ├── Fetch & normalize data                             │
    │   ├── Validate (NaN, types, infinities)                  │
    │   └── Retry with exponential backoff                     │
    │              │                                            │
    │              ▼                                            │
    │   SQL DATABASE (SQLite)                                   │
    │   ├── kpi_readings (indexed by kpi_name, timestamp)      │
    │   ├── alerts (severity, type, audit trail)               │
    │   └── kpi_thresholds (configurable per KPI)              │
    │              │                                            │
    │              ▼                                            │
    │   ANOMALY DETECTION ENGINE                               │
    │   ├── Static Threshold Breach                            │
    │   ├── Z-Score Outlier Detection                          │
    │   ├── Rolling Average Deviation                          │
    │   └── % Change Spike Detection                           │
    │              │                                            │
    │              ▼                                            │
    │   ALERT MANAGER                                          │
    │   ├── Console / Log (always active)                      │
    │   ├── SQL Audit Trail (always active)                    │
    │   └── Email Digest via SMTP (configurable)               │
    │              │                                            │
    │              ▼                                            │
    │   THIS DASHBOARD (Streamlit)                             │
    └──────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("---")

    # Detection Methods Explanation
    st.markdown("## 🔍 Detection Methods Explained")

    methods = {
        "Static Threshold": {
            "icon": "🎯",
            "formula": "value < lower_bound OR value > upper_bound",
            "strength": "Simple, interpretable, good for known limits",
            "example": "Revenue below \$5,000 = something is wrong",
        },
        "Z-Score": {
            "icon": "📐",
            "formula": "|z| = |(x - μ) / σ| > threshold",
            "strength": "Adapts to data scale, statistically rigorous",
            "example": "A value 3 std devs from mean is rare (0.3% probability)",
        },
        "Rolling Average": {
            "icon": "📊",
            "formula": "|value - rolling_mean| / rolling_std > threshold",
            "strength": "Catches local anomalies, adapts to trends",
            "example": "Recent average is \$15K but today is \$8K",
        },
        "% Change": {
            "icon": "📉",
            "formula": "|Δ%| = |(current - previous) / previous| > threshold",
            "strength": "Catches sudden shocks regardless of level",
            "example": "Stock drops 8% in one day",
        },
    }

    cols = st.columns(2)
    for i, (method, info) in enumerate(methods.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="border:1px solid #ddd; border-radius:10px; 
                        padding:15px; margin:10px 0; background:#fafafa;">
                <h4>{info['icon']} {method}</h4>
                <p><code>{info['formula']}</code></p>
                <p><strong>Strength:</strong> {info['strength']}</p>
                <p><em>Example: {info['example']}</em></p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Tech Stack
    st.markdown("## 🛠️ Tech Stack")

    tech_cols = st.columns(4)
    tech_cols[0].markdown("**🐍 Python**\n- pandas\n- numpy\n- yfinance\n- smtplib")
    tech_cols[1].markdown("**🗄️ Database**\n- SQLite\n- Indexed tables\n- Audit trails")
    tech_cols[2].markdown("**📊 Visualization**\n- Streamlit\n- Plotly\n- Interactive charts")
    tech_cols[3].markdown("**⚙️ Engineering**\n- Modular design\n- Error handling\n- Unit tests")


# ═══════════════════════════════════════════════════════════════
#  MAIN APP — Compose Everything
# ═══════════════════════════════════════════════════════════════

def main():
    """Main app entry point — compose all sections."""

    # Render sidebar and get config
    config = render_sidebar()

    # Run pipeline if button clicked
    if config["run_clicked"]:
        run_pipeline(config)

    # Render header
    render_header()

    if not st.session_state.pipeline_run:
        # Show architecture while waiting for first run
        st.markdown("---")
        render_architecture()
        return

    # ─── Tab Navigation ────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 KPI Charts",
        "🚨 Alert Dashboard",
        "🔬 Detection Deep Dive",
        "🗃️ Data Explorer",
        "🏗️ Architecture",
    ])

    with tab1:
        render_kpi_charts()

    with tab2:
        render_alert_dashboard()

    with tab3:
        render_detection_deep_dive()

    with tab4:
        render_data_explorer()

    with tab5:
        render_architecture()


if __name__ == "__main__":
    main()
