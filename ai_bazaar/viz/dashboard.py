"""
Streamlit dashboard for inspecting simulation state and metrics over time.
Expects logs/state_t*.json files (e.g. from bazaar_env runs). Run from project root.
"""
import json
import os
import sys

import pandas as pd
import numpy as np
import glob
import streamlit as st

from chart_builder import AltairChartBuilder
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
from ai_bazaar.utils.dataframe_builder import DataFrameBuilder

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
        cash_df = DataFrameBuilder.value_by_agent(
            state, ledger_field="money", agent_label="Agent", value_label="Cash"
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
        df_builder = DataFrameBuilder(state_files=state_files)

        # ---- Chart 1: Total cash in circulation ----
        cash_long = df_builder.metrics_over_time_long(
            value_vars=["total_cash"],
            renames={"total_cash": "Total cash in circulation"},
        )
        st.subheader("Total cash in circulation")
        color_cash = st.color_picker("Line color", "#1f77b4", key="chart_total_cash_color")
        chart_cash_agg = (
            AltairChartBuilder(cash_long)
            .x("timestep", title="Timestep")
            .y("value", title="Total cash")
            .color(
                "metric",
                domain=["Total cash in circulation"],
                range_=[color_cash],
                legend_title="Metric",
            )
            .mark_line(strokeWidth=2)
            .build()
        )
        st.altair_chart(chart_cash_agg, use_container_width=True)

        # ---- Chart 2: Gini coefficient ----
        gini_long = df_builder.metrics_over_time_long(
            value_vars=["gini"],
            renames={"gini": "Gini coefficient"},
        )
        st.subheader("Gini coefficient")
        color_gini = st.color_picker("Line color", "#ff7f0e", key="chart_gini_color")
        chart_gini = (
            AltairChartBuilder(gini_long)
            .x("timestep", title="Timestep")
            .y("value", title="Gini coefficient")
            .color(
                "metric",
                domain=["Gini coefficient"],
                range_=[color_gini],
                legend_title="Metric",
            )
            .mark_line(strokeWidth=2)
            .build()
        )
        st.altair_chart(chart_gini, use_container_width=True)

        # ---- Chart 3: Profit per firm (all firms on same chart) ----
        st.subheader("Profit per firm")
        profit_per_firm = df_builder.profit_per_firm_over_time()
        chart_profit_firm = (
            AltairChartBuilder(profit_per_firm)
            .x("timestep", title="Timestep")
            .y("value", title="Profit")
            .color("firm", legend_title="Firm")
            .mark_line(strokeWidth=2)
            .build()
        )
        st.altair_chart(chart_profit_firm, use_container_width=True)

        # ---- Chart 4: Cash per firm (all firms on same chart) ----
        st.subheader("Cash per firm")
        cash_per_firm = df_builder.cash_per_firm_over_time()
        chart_cash_firm = (
            AltairChartBuilder(cash_per_firm)
            .x("timestep", title="Timestep")
            .y("value", title="Cash")
            .color("firm", legend_title="Firm")
            .mark_line(strokeWidth=2)
            .build()
        )
        st.altair_chart(chart_cash_firm, use_container_width=True)

        # ---- Chart 5: Cash per consumer (all consumers on same chart) ----
        st.subheader("Cash per consumer")
        cash_per_consumer = df_builder.cash_per_consumer_over_time()
        chart_cash = (
            AltairChartBuilder(cash_per_consumer)
            .x("timestep", title="Timestep")
            .y("value", title="Cash")
            .color("consumer", legend_title="Consumer")
            .mark_line(strokeWidth=2)
            .build()
        )
        st.altair_chart(chart_cash, use_container_width=True)

        # ---- Chart 6: Food inventory per consumer (all consumers on same chart) ----
        st.subheader("Food inventory per consumer")
        food_per_consumer = df_builder.food_inventory_per_consumer_over_time()
        chart_food = (
            AltairChartBuilder(food_per_consumer)
            .x("timestep", title="Timestep")
            .y("value", title="Food inventory")
            .color("consumer", legend_title="Consumer")
            .mark_line(strokeWidth=2)
            .build()
        )
        st.altair_chart(chart_food, use_container_width=True)