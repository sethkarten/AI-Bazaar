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

# Experiment arguments from the selected run folder (optional)
experiment_args_path = os.path.join(run_dir, "experiment_args.json") if run_dir else None
experiment_args_dict = None
if experiment_args_path and os.path.isfile(experiment_args_path):
    with open(experiment_args_path, "r") as f:
        experiment_args_dict = json.load(f)

# Firm attributes from the selected run folder (optional)
firm_attributes_path = os.path.join(run_dir, "firm_attributes.json") if run_dir else None
firm_attributes_list = None
if firm_attributes_path and os.path.isfile(firm_attributes_path):
    with open(firm_attributes_path, "r") as f:
        firm_attributes_list = json.load(f)

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

    # Tab 0: General. Tab 1–3: single-timestep view. Tab 4: Charts. Tab 5: Lemon Market.
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 General", "💰 Wealth Distribution", "🏢 Firms", "👥 Consumers", "📊 Charts", "🍋 Lemon Market"])

    # GENERAL TAB: Experiment arguments.
    with tab0:
        st.subheader("Experiment arguments")
        if experiment_args_dict is not None:
            # Copy-paste rerun command (if present)
            rerun_cmd = experiment_args_dict.get("rerun_command")
            if rerun_cmd:
                st.subheader("Rerun command")
                st.code(rerun_cmd, language="bash")
            # Key/value table (flat; nested dicts shown as JSON string)
            rows = []
            for key, val in sorted(experiment_args_dict.items()):
                if isinstance(val, dict):
                    display_val = json.dumps(val)
                elif isinstance(val, list):
                    display_val = json.dumps(val) if val else "[]"
                else:
                    display_val = val if val is not None else "—"
                rows.append({"Argument": key, "Value": str(display_val)})
            if rows:
                st.table(pd.DataFrame(rows))
            with st.expander("Raw JSON"):
                st.code(json.dumps(experiment_args_dict, indent=2), language="json")
        else:
            st.caption("No experiment_args.json found in this run. Run a simulation with --use-env to generate it.")

    # WEALTH DISTRIBUTION TAB: Cash by agent.
    with tab1:
        st.subheader("Cash by Agent")
        cash_df = DataFrameBuilder.value_by_agent(
            state, ledger_field="money", agent_label="Agent", value_label="Cash"
        )
        st.bar_chart(cash_df, x="Agent", y="Cash")

    # FIRM TAB: Firm states, firm attributes (supply unit costs), supply-by-good chart.
    with tab2:
        st.subheader("Firm States")
        firms_df = pd.DataFrame(state["firms"])
        # Drop columns that PyArrow can't serialize (nested lists/dicts break st.table)
        for col in ["diary", "prices", "inventory"]:
            if col in firms_df.columns:
                firms_df = firms_df.drop(columns=[col])
        st.table(firms_df)

        st.subheader("Firm attributes (supply unit costs)")
        if firm_attributes_list is not None:
            # Collect all goods across firms for column headers
            all_goods = []
            for fa in firm_attributes_list:
                costs = fa.get("supply_unit_costs") or {}
                if isinstance(costs, dict):
                    for g in costs:
                        if g not in all_goods:
                            all_goods.append(g)
                for g in fa.get("goods") or []:
                    if g not in all_goods:
                        all_goods.append(g)
            all_goods = sorted(all_goods)
            # Build table: Firm | good1 | good2 | ...
            rows = []
            for fa in firm_attributes_list:
                row = {"Firm": fa.get("name", "")}
                costs = fa.get("supply_unit_costs") or {}
                if not isinstance(costs, dict):
                    costs = {}
                for good in all_goods:
                    val = costs.get(good)
                    row[good] = f"{val:.2f}" if isinstance(val, (int, float)) else "—"
                rows.append(row)
            if rows:
                st.table(pd.DataFrame(rows))
            else:
                st.caption("No firm attributes in this run.")
        else:
            st.caption("No firm_attributes.json found in this run. Run a simulation with --use-env to generate it.")

        st.subheader("Supply purchases by good (selected firm)")
        firm_names = [f["name"] for f in state["firms"]]
        selected_firm = st.selectbox(
            "Select firm",
            firm_names,
            key="tab2_firm_select",
        )
        df_builder_firm = DataFrameBuilder(state_files=state_files)
        supply_by_good_df = df_builder_firm.supply_purchases_by_good_over_time(selected_firm)
        if not supply_by_good_df.empty:
            chart_supply_by_good = (
                AltairChartBuilder(supply_by_good_df)
                .x("timestep", title="Timestep")
                .y("value", title="Total cost")
                .color("good", legend_title="Good")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_supply_by_good, use_container_width=True)
        else:
            st.caption(f"No supply_by_good data for {selected_firm} in this run.")

        st.subheader("Average profit (last 3 timesteps)")
        profit_rolling = df_builder_firm.profit_rolling_avg_per_firm_over_time(window=3)
        profit_rolling_firm = profit_rolling[profit_rolling["firm"] == selected_firm] if not profit_rolling.empty else pd.DataFrame()
        if not profit_rolling_firm.empty:
            chart_profit_rolling = (
                AltairChartBuilder(profit_rolling_firm)
                .x("timestep", title="Timestep")
                .y("value", title="Avg profit (3-step)")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_profit_rolling, use_container_width=True)
        else:
            st.caption(f"No profit data for {selected_firm} in this run.")

        st.subheader("Orders filled")
        filled_by_firm = df_builder_firm.filled_orders_count_by_firm_over_time()
        filled_selected = filled_by_firm[filled_by_firm["firm"] == selected_firm] if not filled_by_firm.empty else pd.DataFrame()
        if not filled_selected.empty:
            chart_filled = (
                AltairChartBuilder(filled_selected)
                .x("timestep", title="Timestep")
                .y("value", title="Orders filled")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_filled, use_container_width=True)
        else:
            st.caption(f"No filled-orders data for {selected_firm} (run with --use-env and state files that include filled_orders_count_by_firm).")

    # CONSUMER TAB: Consumer states and diary entries.
    with tab3:
        st.subheader("Consumer CES params")
        if consumer_attributes_list is not None:
            # Collect all goods across consumers for column headers
            all_goods_ces = []
            for ca in consumer_attributes_list:
                params = ca.get("ces_params") or {}
                if isinstance(params, dict):
                    for g in params:
                        if g not in all_goods_ces:
                            all_goods_ces.append(g)
            all_goods_ces = sorted(all_goods_ces)
            # Build table: Consumer | good1 | good2 | ...
            rows_ces = []
            for ca in consumer_attributes_list:
                row = {"Consumer": ca.get("name", "")}
                params = ca.get("ces_params") or {}
                if not isinstance(params, dict):
                    params = {}
                for good in all_goods_ces:
                    val = params.get(good)
                    row[good] = f"{val:.2f}" if isinstance(val, (int, float)) else "—"
                rows_ces.append(row)
            if rows_ces:
                st.table(pd.DataFrame(rows_ces))
            else:
                st.caption("No consumer CES params in this run.")
        else:
            st.caption("No consumer_attributes.json found in this run. Run a simulation with --use-env to generate it.")

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

        # ---- eWTP by good for selected consumer ----
        st.subheader("eWTP by good")
        ewtp_by_good = df_builder_consumer.consumer_ewtp_by_good_over_time(
            consumer_name=selected_consumer
        )
        if not ewtp_by_good.empty:
            chart_ewtp = (
                AltairChartBuilder(ewtp_by_good)
                .x("timestep", title="Timestep")
                .y("value", title="eWTP")
                .color("good", legend_title="Good")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_ewtp, use_container_width=True)
        else:
            st.caption("No eWTP data for this consumer (eWTP is tracked for CES consumers).")

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

        # ---- Chart 1b: Total filled orders per timestep ----
        st.subheader("Total filled orders per timestep")
        filled_total_df = df_builder.filled_orders_count_over_time()
        if not filled_total_df.empty:
            chart_filled_total = (
                AltairChartBuilder(filled_total_df)
                .x("timestep", title="Timestep")
                .y("value", title="Filled orders")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_filled_total, use_container_width=True)
        else:
            st.caption("No filled-orders data (run with --use-env and state files that include filled_orders_count).")

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

        # ---- Chart 3b: Average profit (last 3 timesteps) per firm ----
        st.subheader("Average profit (last 3 timesteps) per firm")
        profit_rolling_per_firm = df_builder.profit_rolling_avg_per_firm_over_time(window=3)
        if not profit_rolling_per_firm.empty:
            chart_profit_rolling = (
                AltairChartBuilder(profit_rolling_per_firm)
                .x("timestep", title="Timestep")
                .y("value", title="Avg profit (3-step)")
                .color("firm", legend_title="Firm")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_profit_rolling, use_container_width=True)
        else:
            st.caption("No profit data in this run.")

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

        # ---- Chart 4c: Sales per firm (all firms on same chart) ----
        st.subheader("Sales per firm")
        sales_per_firm = df_builder.sales_per_firm_over_time()
        chart_sales_firm = (
            AltairChartBuilder(sales_per_firm)
            .x("timestep", title="Timestep")
            .y("value", title="Sales (quantity sold)")
            .color("firm", legend_title="Firm")
            .mark_line(strokeWidth=2)
            .build()
        )
        st.altair_chart(chart_sales_firm, use_container_width=True)

        # ---- Chart 4d: Price per firm for a good (good selectable) ----
        st.subheader("Price per firm (by good)")
        good_names = df_builder._all_good_names()
        if good_names:
            selected_good = st.selectbox(
                "Good",
                good_names,
                key="chart_price_good_select",
            )
            price_per_firm = df_builder.price_per_firm_over_time(selected_good)
            chart_price_firm = (
                AltairChartBuilder(price_per_firm)
                .x("timestep", title="Timestep")
                .y("value", title=f"Price ({selected_good})")
                .color("firm", legend_title="Firm")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_price_firm, use_container_width=True)
        else:
            st.caption("No goods found in firm prices.")

        # ---- Chart 4e: Sales this step per firm (good selectable) ----
        st.subheader("Sales this step per firm (by good)")
        good_names_sales = df_builder._all_good_names()
        if good_names_sales:
            selected_good_sales = st.selectbox(
                "Good (sales this step)",
                good_names_sales,
                key="chart_sales_this_step_good_select",
            )
            sales_this_step_per_firm = df_builder.sales_this_step_per_firm_over_time(selected_good_sales)
            chart_sales_this_step = (
                AltairChartBuilder(sales_this_step_per_firm)
                .x("timestep", title="Timestep")
                .y("value", title=f"Sales this step ({selected_good_sales})")
                .color("firm", legend_title="Firm")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_sales_this_step, use_container_width=True)
        else:
            st.caption("No goods found for sales this step.")

        # ---- Chart 4f: Inventory per firm (good selectable; excludes cash) ----
        st.subheader("Inventory per firm (by good)")
        inventory_good_names = df_builder._all_inventory_good_names()
        if inventory_good_names:
            selected_good_inv = st.selectbox(
                "Good (inventory)",
                inventory_good_names,
                key="chart_inventory_firm_good_select",
            )
            inventory_per_firm = df_builder.inventory_per_firm_over_time(selected_good_inv)
            chart_inventory_firm = (
                AltairChartBuilder(inventory_per_firm)
                .x("timestep", title="Timestep")
                .y("value", title=f"Inventory ({selected_good_inv})")
                .color("firm", legend_title="Firm")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_inventory_firm, use_container_width=True)
        else:
            st.caption("No inventory goods found for firms.")

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

        # ---- Chart 5b: Consumer surplus per consumer (all consumers on same chart) ----
        st.subheader("Consumer surplus per consumer")
        surplus_per_consumer = df_builder.consumer_surplus_per_consumer_over_time()
        if not surplus_per_consumer.empty:
            chart_surplus = (
                AltairChartBuilder(surplus_per_consumer)
                .x("timestep", title="Timestep")
                .y("value", title="Consumer surplus")
                .color("consumer", legend_title="Consumer")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_surplus, use_container_width=True)

        # ---- Chart 5c: Average eWTP by good ----
        st.subheader("Average eWTP by good")
        avg_ewtp_by_good = df_builder.avg_ewtp_by_good_over_time()
        if not avg_ewtp_by_good.empty:
            chart_avg_ewtp = (
                AltairChartBuilder(avg_ewtp_by_good)
                .x("timestep", title="Timestep")
                .y("value", title="Avg eWTP")
                .color("good", legend_title="Good")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_avg_ewtp, use_container_width=True)
        else:
            st.caption("No eWTP data in this run (eWTP is tracked for CES consumers).")

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

    # LEMON MARKET TAB: All posted listings (only for runs with consumer_scenario LEMON_MARKET).
    with tab5:
        is_lemon_run = (
            experiment_args_dict is not None
            and experiment_args_dict.get("consumer_scenario") == "LEMON_MARKET"
        )
        if not is_lemon_run:
            st.info("Current selection is not a Lemon Market run.")
        else:
            all_listings = []
            for path in state_files:
                with open(path, "r") as f:
                    snap = json.load(f)
                for row in snap.get("lemon_market_new_listings", []):
                    all_listings.append(row)
            if all_listings:
                st.subheader("All posted listings")
                df_listings = pd.DataFrame(all_listings)
                # Order columns: timestep_posted first, then id, firm_id, etc.
                cols = ["timestep_posted", "id", "firm_id", "quality", "quality_value", "price", "reputation", "description"]
                cols = [c for c in cols if c in df_listings.columns]
                extra = [c for c in df_listings.columns if c not in cols]
                df_listings = df_listings[cols + extra]
                st.dataframe(df_listings, use_container_width=True)
            else:
                st.caption("No lemon market listings found in state files.")