"""
Streamlit dashboard for inspecting simulation state and metrics over time.
Expects logs/state_t*.json files (e.g. from bazaar_env runs). Run from project root.
"""
import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import glob

# Support both package import and running this file directly via streamlit run
try:
    from .chart_builder import AltairChartBuilder
except ImportError:
    from chart_builder import AltairChartBuilder

st.set_page_config(page_title="Agent Bazaar Dashboard", layout="wide")

st.title("🏛️ Agent Bazaar: Civilization Simulacra Dashboard")

# Discover state snapshots: state_t0.json, state_t1.json, ... (sorted by timestep)
log_dir = "logs"
state_files = sorted(
    glob.glob(os.path.join(log_dir, "state_t*.json")),
    key=lambda x: int(os.path.basename(x).replace("state_t", "").replace(".json", "")),
)

if not state_files:
    st.warning("No state files found in logs/ directory. Run a simulation first!")
else:
    # Which snapshot to show in the first three tabs (single-timestep view)
    timestep = st.select_slider(
        "Timestep",
        options=list(range(len(state_files))),
        value=len(state_files) - 1,
    )

    with open(state_files[timestep], "r") as f:
        state = json.load(f)

    st.header(f"Timestep {state['timestep']}")

    # Top-row summary metrics for the selected timestep
    col1, col2, col3, col4 = st.columns(4)
    total_cash = sum(state["ledger"]["money"].values())
    col1.metric("Total Money Supply", f"${total_cash:.2f}")

    # Gini coefficient: 0 = perfect equality, 1 = all wealth in one agent
    cash_values = sorted(state["ledger"]["money"].values())
    n = len(cash_values)
    index = np.arange(1, n + 1)
    gini = (
        (np.sum((2 * index - n - 1) * cash_values)) / (n * np.sum(cash_values))
        if total_cash > 0
        else 0
    )
    col2.metric("Gini Coefficient", f"{gini:.4f}")

    col3.metric(
        "Firms in Business", len([f for f in state["firms"] if f["in_business"]])
    )
    col4.metric(
        "Avg Consumer Utility",
        f"{pd.DataFrame(state['consumers'])['utility'].mean():.2f}",
    )

    # Tab 1–3: single-timestep view. Tab 4: time-series charts across all state files.
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Wealth Distribution", "🏢 Firms", "👥 Consumers", "📊 Charts"])

    with tab1:
        st.subheader("Cash by Agent")
        # Bar chart of ledger.money for the selected timestep
        cash_df = pd.DataFrame(
            state["ledger"]["money"].items(), columns=["Agent", "Cash"]
        )
        st.bar_chart(cash_df, x="Agent", y="Cash")

    with tab2:
        st.subheader("Firm States")
        firms_df = pd.DataFrame(state["firms"])
        # Drop columns that PyArrow can't serialize (nested lists/dicts break st.table)
        for col in ["diary", "prices", "inventory"]:
            if col in firms_df.columns:
                firms_df = firms_df.drop(columns=[col])
        st.table(firms_df)

    with tab3:
        st.subheader("Consumer States")
        cons_df = pd.DataFrame(state["consumers"])
        st.table(cons_df.drop(columns=["diary"]))

        # Diary is stored as list of [timestep, "entry text"]; show last entry as text
        selected_consumer = st.selectbox(
            "Select Consumer for Diary", [c["name"] for c in state["consumers"]]
        )
        for c in state["consumers"]:
            if c["name"] == selected_consumer:
                diary = c.get("diary") or []
                if isinstance(diary, list) and diary and isinstance(diary[-1], (list, tuple)):
                    last_entry = diary[-1][-1] if len(diary[-1]) > 1 else str(diary[-1])
                else:
                    last_entry = diary if isinstance(diary, str) else (diary[-1] if diary else "—")
                st.info(f"Last Diary Entry: {last_entry}")
                
    with tab4:
        # Build one row per timestep with aggregate metrics (used for line chart)
        def load_all_states(state_files):
            rows = []
            for path in state_files:
                with open(path, "r") as f:
                    s = json.load(f)
                t = s["timestep"]
                cash = sum(s["ledger"]["money"].values())
                n = len(s["ledger"]["money"].values())
                vals = sorted(s["ledger"]["money"].values())
                gini = (np.sum((2 * np.arange(1, n + 1) - n - 1) * vals) / (n * sum(vals))) if cash else 0
                rows.append({"timestep": t, "total_cash": cash, "gini": gini})
            return pd.DataFrame(rows)

        time_df = load_all_states(state_files)
        # Long format: one row per (timestep, metric), so Altair can color by metric
        time_long = time_df.melt(
            id_vars=["timestep"],
            value_vars=["total_cash", "gini"],
            var_name="metric",
            value_name="value",
        )
        # Display names in legend; domain for the chart is taken from data so renames don't break lines
        time_long["metric"] = time_long["metric"].replace(
            {"total_cash": "Total cash in circulation", "gini": "Gini coefficient"}
        )

        # Color scale domain = unique metric names in data order; one picker per metric
        metric_order = time_long["metric"].unique().tolist()
        default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        st.subheader("Metrics over time")
        color_cols = st.columns(min(len(metric_order), 5))
        color_by_metric = {}
        for i, name in enumerate(metric_order):
            with color_cols[i % len(color_cols)]:
                color_by_metric[name] = st.color_picker(
                    f"{name}", default_colors[i % len(default_colors)], key=f"chart_color_{i}_{name}"
                )
        colors_in_order = [color_by_metric[m] for m in metric_order]

        # Builder takes the long-format df; domain/range_ align colors with metric names
        chart = (
            AltairChartBuilder(time_long)
            .x("timestep", title="Timestep")
            .y("value", title="Value")
            .color(
                "metric",
                domain=metric_order,
                range_=colors_in_order,
                legend_title="Metric",
            )
            .mark_line(strokeWidth=2)
            .build()
        )
        st.altair_chart(chart, use_container_width=True)