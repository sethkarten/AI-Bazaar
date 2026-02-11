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

# Discover runs: logs/ (legacy) and logs/<run_name>/ (run subfolders)
log_dir = "logs"
run_dirs = []
if os.path.isdir(log_dir):
    # Legacy: state files directly in logs/
    if glob.glob(os.path.join(log_dir, "state_t*.json")):
        run_dirs.append((log_dir, "logs (root)"))
    for name in sorted(os.listdir(log_dir)):
        sub = os.path.join(log_dir, name)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "state_t*.json")):
            run_dirs.append((sub, name))
run_dir = run_dirs[0][0] if run_dirs else None
run_label = run_dirs[0][1] if run_dirs else None
if len(run_dirs) > 1:
    run_label = st.selectbox("Run", options=[r[1] for r in run_dirs], index=0)
    run_dir = next(r[0] for r in run_dirs if r[1] == run_label)
state_files = sorted(
    glob.glob(os.path.join(run_dir, "state_t*.json")) if run_dir else [],
    key=lambda x: int(os.path.basename(x).replace("state_t", "").replace(".json", "")),
)
# Consumer attributes from the selected run folder (optional)
consumer_attributes_path = os.path.join(run_dir, "consumer_attributes.json") if run_dir else None
consumer_attributes_list = None
if consumer_attributes_path and os.path.isfile(consumer_attributes_path):
    with open(consumer_attributes_path, "r") as f:
        consumer_attributes_list = json.load(f)

if not state_files:
    st.warning("No state files found in logs/ (or in any logs/<run_name>/ folder). Run a simulation first!")
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

    # WEALTH DISTRIBUTION TAB: Cash by agent.
    with tab1:
        st.subheader("Cash by Agent")
        cash_df = DataFrameBuilder.value_by_agent(
            state, ledger_field="money", agent_label="Agent", value_label="Cash"
        )
        st.bar_chart(cash_df, x="Agent", y="Cash")

    # FIRM TAB: Firm states and diary entries.
    with tab2:
        st.subheader("Firm States")
        firms_df = pd.DataFrame(state["firms"])
        # Drop columns that PyArrow can't serialize (nested lists/dicts break st.table)
        for col in ["diary", "prices", "inventory"]:
            if col in firms_df.columns:
                firms_df = firms_df.drop(columns=[col])
        st.table(firms_df)

    # CONSUMER TAB: Consumer states and diary entries.
    with tab3:
        consumer_names = [c["name"] for c in state["consumers"]]
        selected_consumer = st.selectbox(
            "Select consumer",
            consumer_names,
            key="tab3_consumer_select",
        )

        st.subheader(f"Consumer: {selected_consumer}")

        # Table of consumer attributes from consumer_attributes.json (matches selected consumer)
        if consumer_attributes_list is not None:
            attrs_for_consumer = next(
                (a for a in consumer_attributes_list if a.get("name") == selected_consumer),
                None,
            )
            if attrs_for_consumer is not None:
                st.subheader("Consumer attributes (from run init)")
                # Build two-column table: Attribute | Value (skip "name" to avoid redundancy)
                attr_rows = []
                for key, val in attrs_for_consumer.items():
                    if key == "name":
                        continue
                    if isinstance(val, dict):
                        display_val = json.dumps(val) if val else ""
                    elif isinstance(val, list):
                        display_val = ", ".join(str(x) for x in val) if val else "[]"
                    else:
                        display_val = val if val is not None else "—"
                    attr_rows.append({"Attribute": key, "Value": display_val})
                if attr_rows:
                    st.table(pd.DataFrame(attr_rows))
            else:
                st.caption(f"No attributes on file for {selected_consumer}.")
        else:
            st.caption("No consumer_attributes.json found in logs/ (run a simulation to generate it).")

        st.subheader("Consumer state (this timestep)")
        cons_df = pd.DataFrame(state["consumers"])
        cons_df_filtered = cons_df[cons_df["name"] == selected_consumer]
        if "diary" in cons_df_filtered.columns:
            cons_df_filtered = cons_df_filtered.drop(columns=["diary"])
        st.table(cons_df_filtered)

        # ---- Goods, cash, labor, and total utility for selected consumer ----
        st.subheader("Goods, cash, labor, and total utility")
        df_builder_consumer = DataFrameBuilder(state_files=state_files)
        utility_components = df_builder_consumer.consumer_utility_components_over_time(
            consumer_name=selected_consumer
        )
        if not utility_components.empty:
            chart_utility_components = (
                AltairChartBuilder(utility_components)
                .x("timestep", title="Timestep")
                .y("value", title="Value")
                .color(
                    "metric",
                    domain=["Goods utility", "Cash utility", "Labor disutility", "Total utility"],
                    range_=["#2ca02c", "#1f77b4", "#d62728", "#9467bd"],
                    legend_title="Metric",
                )
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_utility_components, use_container_width=True)

        # ---- Consumer diary: all entries across all state files ----
        st.subheader("Consumer diary (all timesteps)")
        diary_entries_all = []
        for path in state_files:
            with open(path, "r") as f:
                snap = json.load(f)
            file_ts = snap.get("timestep", 0)
            for c in snap.get("consumers", []):
                if c["name"] != selected_consumer:
                    continue
                diary = c.get("diary") or []
                if isinstance(diary, list) and diary:
                    for entry in diary:
                        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            ts, text = entry[0], entry[1]
                            diary_entries_all.append({"Timestep": ts, "Entry": text})
                        else:
                            diary_entries_all.append({"Timestep": file_ts, "Entry": str(entry)})
                elif isinstance(diary, str):
                    diary_entries_all.append({"Timestep": file_ts, "Entry": diary})
                break
        diary_all_df = pd.DataFrame(diary_entries_all)
        if diary_entries_all:
            st.table(diary_all_df)
        else:
            st.info(f"No diary entries for {selected_consumer} across the loaded state files.")
            
    # CHARTS TAB: Time-series charts across all state files.            
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

        # ---- Chart 4b: Reputation per firm (all firms on same chart) ----
        st.subheader("Reputation per firm")
        reputation_per_firm = df_builder.reputation_per_firm_over_time()
        if not reputation_per_firm.empty:
            chart_reputation_firm = (
                AltairChartBuilder(reputation_per_firm)
                .x("timestep", title="Timestep")
                .y("value", title="Reputation")
                .color("firm", legend_title="Firm")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_reputation_firm, use_container_width=True)

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

        # ---- Chart 7: Goods, cash, labor, and total utility (avg across consumers) ----
        st.subheader("Goods, cash, labor, and total utility (avg)")
        utility_components = df_builder.consumer_utility_components_over_time()
        if not utility_components.empty:
            chart_utility_components = (
                AltairChartBuilder(utility_components)
                .x("timestep", title="Timestep")
                .y("value", title="Value")
                .color(
                    "metric",
                    domain=["Goods utility (avg)", "Cash utility (avg)", "Labor disutility (avg)", "Total utility (avg)"],
                    range_=["#2ca02c", "#1f77b4", "#d62728", "#9467bd"],
                    legend_title="Metric",
                )
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_utility_components, use_container_width=True)