"""
Streamlit dashboard for inspecting simulation state and metrics over time.
Expects logs/**/state_t*.json (any depth under logs/, e.g. logs/exp1_<model>/<run>/).
Run from project root.
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


def _filter_seller_prompt_records(records, seller_name: str):
    """Rows from lemon_agent_prompts.jsonl for one seller (honest name or sybil identity)."""
    out = []
    for r in records:
        if r.get("role") != "seller":
            continue
        ag = r.get("agent", "")
        sid = r.get("sybil_identity")
        if ag == seller_name:
            out.append(r)
        elif sid == seller_name:
            out.append(r)
        elif (
            seller_name.startswith("sybil_")
            and ag == "sybil_principal"
            and r.get("call") == "sybil_tier"
        ):
            out.append(r)
    out.sort(key=lambda x: (x.get("timestep", 0), str(x.get("call", ""))))
    return out


def _filter_buyer_prompt_records(records, buyer_name: str):
    out = []
    for r in records:
        if r.get("role") == "buyer" and r.get("agent") == buyer_name:
            out.append(r)
    out.sort(key=lambda x: (x.get("timestep", 0), str(x.get("call", ""))))
    return out


def _render_lemon_prompt_navigator(records: list, widget_key_prefix: str) -> None:
    """Select one logged LLM exchange and show system / user / response."""
    if not records:
        return
    n = len(records)
    labels = [
        f"t={r.get('timestep')} · {r.get('call')} · {i + 1}/{n}"
        for i, r in enumerate(records)
    ]
    sk = f"{widget_key_prefix}_nav"
    if sk not in st.session_state:
        st.session_state[sk] = 0
    st.session_state[sk] = int(np.clip(st.session_state[sk], 0, n - 1))

    nav_a, nav_b = st.columns([1, 1])
    with nav_a:
        if st.button("◀ Prev", key=f"{widget_key_prefix}_prev"):
            st.session_state[sk] = max(0, st.session_state[sk] - 1)
    with nav_b:
        if st.button("Next ▶", key=f"{widget_key_prefix}_next"):
            st.session_state[sk] = min(n - 1, st.session_state[sk] + 1)

    idx = st.selectbox(
        "Jump to prompt",
        options=list(range(n)),
        format_func=lambda i: labels[i],
        key=sk,
    )
    r = records[idx]
    extra = (
        f"**{r.get('call')}** · timestep **{r.get('timestep')}**"
        + (
            f" · retry depth {r.get('depth')}"
            if r.get("depth") not in (None, 0)
            else ""
        )
    )
    st.markdown(extra)
    if r.get("sybil_identity"):
        st.caption(f"Sybil identity: {r['sybil_identity']}")
    st.text_area(
        "System prompt",
        value=r.get("system_prompt") or "",
        height=140,
        disabled=True,
        key=f"{widget_key_prefix}_sys_{idx}",
    )
    st.text_area(
        "User prompt",
        value=r.get("user_prompt") or "",
        height=220,
        disabled=True,
        key=f"{widget_key_prefix}_usr_{idx}",
    )
    st.text_area(
        "Response",
        value=r.get("response") or "",
        height=220,
        disabled=True,
        key=f"{widget_key_prefix}_rsp_{idx}",
    )


st.set_page_config(page_title="Agent Bazaar Dashboard", layout="wide")

st.title("🏛️ Agent Bazaar: Civilization Simulacra Dashboard")


def discover_run_dirs(log_root: str) -> list[tuple[str, str]]:
    """Find every directory under log_root that contains state_t*.json.

    Returns sorted list of (absolute_path, display_label). Labels use forward slashes
    relative to log_root (e.g. exp1_model/run_name) so nested experiment layouts are clear.
    """
    runs: list[tuple[str, str]] = []
    if not os.path.isdir(log_root):
        return runs
    log_root = os.path.normpath(log_root)
    for dirpath, _dirnames, _filenames in os.walk(log_root):
        if glob.glob(os.path.join(dirpath, "state_t*.json")):
            rel = os.path.relpath(dirpath, log_root)
            if rel in (".", ""):
                label = "logs (root)"
            else:
                label = rel.replace(os.sep, "/")
            runs.append((os.path.abspath(dirpath), label))
    runs.sort(key=lambda x: x[1].lower())
    return runs


# Discover runs at any depth under logs/ (e.g. logs/exp1_<model>/<run_name>/)
log_dir = "logs"
run_dirs = discover_run_dirs(log_dir)

with st.sidebar:
    st.subheader("Run directory")
    if not run_dirs:
        st.info(f"No runs found under `{log_dir}/` (need directories containing state_t*.json).")
    run_filter = st.text_input(
        "Filter path",
        value="",
        placeholder="e.g. sonnet, dlc3, baseline",
        help="Case-insensitive substring match on the path under logs/",
        disabled=not run_dirs,
    )
    needle = (run_filter or "").strip().lower()
    filtered = (
        run_dirs
        if not needle
        else [(p, lab) for p, lab in run_dirs if needle in lab.lower()]
    )
    if needle and not filtered and run_dirs:
        st.warning("No runs match the filter; showing all runs.")
        filtered = run_dirs

    if len(filtered) == 1:
        run_dir, run_label = filtered[0]
        st.caption(f"**{run_label}**")
    elif len(filtered) > 1:
        labels = [lab for _p, lab in filtered]
        run_label = st.selectbox(
            "Run",
            options=labels,
            index=0,
            help="Path is relative to logs/. Nested layouts (e.g. exp1) appear as parent/run.",
        )
        run_dir = next(p for p, lab in filtered if lab == run_label)
    else:
        run_dir, run_label = None, None
state_files = sorted(
    glob.glob(os.path.join(run_dir, "state_t*.json")) if run_dir else [],
    key=lambda x: int(os.path.basename(x).replace("state_t", "").replace(".json", "")),
)
if run_label:
    st.caption(f"Selected run: `{log_dir}/{run_label}`" if run_label != "logs (root)" else f"Selected run: `{log_dir}/` (root)")
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

# Token usage summary for this run (optional; written by ai_bazaar.main)
token_usage_files = (
    sorted(glob.glob(os.path.join(run_dir, "*_token_usage.json")))
    if run_dir
    else []
)

# Optional LEMON_MARKET LLM prompt log (--log-buyer-prompts / --log-seller-prompts)
lemon_prompt_records: list = []
prompt_log_path = os.path.join(run_dir, "lemon_agent_prompts.jsonl") if run_dir else None
if prompt_log_path and os.path.isfile(prompt_log_path):
    with open(prompt_log_path, "r", encoding="utf-8") as _pf:
        for _line in _pf:
            _line = _line.strip()
            if not _line:
                continue
            try:
                lemon_prompt_records.append(json.loads(_line))
            except json.JSONDecodeError:
                pass

# Optional THE_CRASH firm prompt log (--log-crash-firm-prompts)
crash_prompt_records: list = []
crash_prompt_log_path = os.path.join(run_dir, "crash_agent_prompts.jsonl") if run_dir else None
if crash_prompt_log_path and os.path.isfile(crash_prompt_log_path):
    with open(crash_prompt_log_path, "r", encoding="utf-8") as _cpf:
        for _line in _cpf:
            _line = _line.strip()
            if not _line:
                continue
            try:
                crash_prompt_records.append(json.loads(_line))
            except json.JSONDecodeError:
                pass

if not state_files:
    st.warning(
        "No state files found under logs/ (searched recursively for state_t*.json). "
        "Run a simulation from the project root or adjust the logs/ layout."
    )
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

    # Tab 0: General. Tab 1–3: single-timestep view. Tab 4: Charts. Tab 5: Lemon Market. Tab 6: Discovery. Tab 7: Token usage.
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "📋 General",
            "💰 Wealth Distribution",
            "🏢 Firms",
            "👥 Consumers",
            "📊 Charts",
            "🍋 Lemon Market",
            "🔍 Discovery",
            "🧮 Token Usage",
        ]
    )

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

        st.subheader("Firms in business over time")
        df_builder_firms_count = DataFrameBuilder(state_files=state_files)
        firms_in_business_df = df_builder_firms_count.firms_in_business_over_time()
        if not firms_in_business_df.empty:
            chart_firms_in_business = (
                AltairChartBuilder(firms_in_business_df)
                .x("timestep", title="Timestep")
                .y("value", title="Firms in business")
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_firms_in_business, use_container_width=True)
        else:
            st.caption("No firm count data in this run.")

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

        firm_names = [f["name"] for f in state["firms"]]
        selected_firm = st.selectbox(
            "Select firm",
            firm_names,
            key="tab2_firm_select",
        )

        st.subheader("System prompt (selected firm)")
        if firm_attributes_list is not None:
            fa_for_firm = next((fa for fa in firm_attributes_list if fa.get("name") == selected_firm), None)
            prompt_text = fa_for_firm.get("system_prompt") if fa_for_firm else None
            if prompt_text:
                st.text_area("", value=prompt_text, height=300, key="tab2_system_prompt", disabled=True)
            else:
                st.caption("No system prompt for this firm (e.g. fixed-price firm).")
        else:
            st.caption("No firm_attributes.json; system prompts not available.")

        st.subheader("Supply purchases by good (selected firm)")
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

        # THE_CRASH firm prompt navigator (only when crash_agent_prompts.jsonl is present)
        if crash_prompt_records:
            firm_crash_recs = [r for r in crash_prompt_records if r.get("agent") == selected_firm]
            st.markdown("**LLM prompts & responses**")
            if firm_crash_recs:
                st.caption(f"{len(firm_crash_recs)} logged exchange(s) for **{selected_firm}** in this run.")
                _render_lemon_prompt_navigator(firm_crash_recs, f"crash_firm_{selected_firm}")
            else:
                st.caption(f"No prompt records for **{selected_firm}** in crash_agent_prompts.jsonl.")
        elif crash_prompt_log_path and os.path.isfile(crash_prompt_log_path):
            st.caption("crash_agent_prompts.jsonl is present but empty.")

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

        # ---- Overlay charts: combine multiple series on one chart ----
        def _long_firms_in_business(db):
            df = db.firms_in_business_over_time()
            if df.empty:
                return df
            df = df.copy()
            df["metric"] = "Firms in business"
            return df

        def _long_total_cash(db):
            return db.metrics_over_time_long(
                value_vars=["total_cash"],
                renames={"total_cash": "Total cash"},
            )

        def _long_gini(db):
            return db.metrics_over_time_long(
                value_vars=["gini"],
                renames={"gini": "Gini coefficient"},
            )

        def _long_filled_orders(db):
            df = db.filled_orders_count_over_time()
            if df.empty:
                return df
            df = df.copy()
            df["metric"] = "Filled orders"
            return df

        def _long_total_profit(db):
            return db.metrics_over_time_long(
                value_vars=["total_profit"],
                renames={"total_profit": "Total profit"},
            )

        def _long_avg_reputation(db):
            df = db.reputation_per_firm_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].mean()
            agg["metric"] = "Avg reputation (firms)"
            return agg[["timestep", "metric", "value"]]

        def _long_total_sales(db):
            df = db.sales_per_firm_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].sum()
            agg["metric"] = "Total sales (quantity)"
            return agg[["timestep", "metric", "value"]]

        def _long_avg_consumer_surplus(db):
            df = db.consumer_surplus_per_consumer_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].mean()
            agg["metric"] = "Avg consumer surplus"
            return agg[["timestep", "metric", "value"]]

        def _long_total_utility_avg(db):
            df = db.consumer_utility_components_over_time()
            if df.empty:
                return df
            part = df[df["metric"] == "Total utility (avg)"].copy()
            part["metric"] = "Total utility (avg)"
            return part[["timestep", "metric", "value"]]

        def _long_avg_food_inventory(db):
            df = db.food_inventory_per_consumer_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].mean()
            agg["metric"] = "Avg food inventory"
            return agg[["timestep", "metric", "value"]]

        def _long_avg_ewtp(db):
            df = db.avg_ewtp_by_good_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].mean()
            agg["metric"] = "Avg eWTP (across goods)"
            return agg[["timestep", "metric", "value"]]

        def _long_total_cash_firms(db):
            df = db.cash_per_firm_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].sum()
            agg["metric"] = "Total cash (firms)"
            return agg[["timestep", "metric", "value"]]

        def _long_avg_cash_firms(db):
            df = db.cash_per_firm_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].mean()
            agg["metric"] = "Avg cash (firms)"
            return agg[["timestep", "metric", "value"]]

        def _long_avg_profit_firms(db):
            df = db.profit_per_firm_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].mean()
            agg["metric"] = "Avg profit (firms)"
            return agg[["timestep", "metric", "value"]]

        def _long_total_cash_consumers(db):
            df = db.cash_per_consumer_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].sum()
            agg["metric"] = "Total cash (consumers)"
            return agg[["timestep", "metric", "value"]]

        def _long_avg_cash_consumers(db):
            df = db.cash_per_consumer_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].mean()
            agg["metric"] = "Avg cash (consumers)"
            return agg[["timestep", "metric", "value"]]

        def _long_avg_price(db):
            good_names = db._all_good_names()
            if not good_names:
                return pd.DataFrame({"timestep": [], "metric": [], "value": []})
            parts = []
            for g in good_names:
                df = db.price_per_firm_over_time(g)
                if not df.empty:
                    parts.append(df)
            if not parts:
                return pd.DataFrame({"timestep": [], "metric": [], "value": []})
            combined = pd.concat(parts, ignore_index=True)
            combined = combined[combined["value"] > 0]  # only positive prices
            if combined.empty:
                return pd.DataFrame({"timestep": [], "metric": [], "value": []})
            agg = combined.groupby("timestep", as_index=False)["value"].mean()
            agg = agg.assign(metric="Avg price (all goods)")
            return agg[["timestep", "metric", "value"]]

        def _long_utility_component(db, component_name):
            df = db.consumer_utility_components_over_time()
            if df.empty:
                return df
            part = df[df["metric"] == component_name].copy()
            part["metric"] = component_name
            return part[["timestep", "metric", "value"]]

        def _long_goods_utility_avg(db):
            return _long_utility_component(db, "Goods utility (avg)")

        def _long_cash_utility_avg(db):
            return _long_utility_component(db, "Cash utility (avg)")

        def _long_labor_disutility_avg(db):
            return _long_utility_component(db, "Labor disutility (avg)")

        def _long_avg_sales_per_firm(db):
            df = db.sales_per_firm_over_time()
            if df.empty:
                return df
            agg = df.groupby("timestep", as_index=False)["value"].mean()
            agg["metric"] = "Avg sales per firm"
            return agg[["timestep", "metric", "value"]]

        def _long_lemon_metric(db, metric_name):
            df = db.lemon_market_metrics_over_time()
            if df.empty:
                return df
            part = df[df["metric"] == metric_name].copy()
            part["metric"] = metric_name
            return part[["timestep", "metric", "value"]]

        overlay_options = [
            ("Total cash", _long_total_cash),
            ("Total cash (firms)", _long_total_cash_firms),
            ("Avg cash (firms)", _long_avg_cash_firms),
            ("Total cash (consumers)", _long_total_cash_consumers),
            ("Avg cash (consumers)", _long_avg_cash_consumers),
            ("Total profit", _long_total_profit),
            ("Avg profit (firms)", _long_avg_profit_firms),
            ("Gini coefficient", _long_gini),
            ("Firms in business", _long_firms_in_business),
            ("Filled orders", _long_filled_orders),
            ("Avg reputation (firms)", _long_avg_reputation),
            ("Total sales (quantity)", _long_total_sales),
            ("Avg sales per firm", _long_avg_sales_per_firm),
            ("Avg price (all goods)", _long_avg_price),
            ("Avg consumer surplus", _long_avg_consumer_surplus),
            ("Total utility (avg)", _long_total_utility_avg),
            ("Goods utility (avg)", _long_goods_utility_avg),
            ("Cash utility (avg)", _long_cash_utility_avg),
            ("Labor disutility (avg)", _long_labor_disutility_avg),
            ("Avg food inventory", _long_avg_food_inventory),
            ("Avg eWTP (across goods)", _long_avg_ewtp),
            ("Lemon: Listings", lambda db: _long_lemon_metric(db, "Listings")),
            ("Lemon: Bids", lambda db: _long_lemon_metric(db, "Bids")),
            ("Lemon: Passes", lambda db: _long_lemon_metric(db, "Passes")),
        ]
        overlay_labels = [o[0] for o in overlay_options]
        overlay_getters = {o[0]: o[1] for o in overlay_options}

        st.subheader("Overlay charts")
        overlay_selected = st.multiselect(
            "Select two or more series to overlay on one chart",
            options=overlay_labels,
            default=[],
            key="charts_overlay_select",
        )
        overlay_normalize = st.checkbox(
            "Normalize each series to 0–1 for comparison",
            value=False,
            key="charts_overlay_normalize",
        )
        if len(overlay_selected) >= 2:
            combined_rows = []
            for label in overlay_selected:
                getter = overlay_getters[label]
                part = getter(df_builder)
                if part.empty or "metric" not in part.columns:
                    part = part.copy()
                    part["metric"] = label
                if "value" in part.columns and "timestep" in part.columns:
                    combined_rows.append(part[["timestep", "metric", "value"]])
            if combined_rows:
                overlay_df = pd.concat(combined_rows, ignore_index=True)
                if overlay_normalize:
                    overlay_df = overlay_df.copy()
                    for m in overlay_df["metric"].unique():
                        mask = overlay_df["metric"] == m
                        v = overlay_df.loc[mask, "value"]
                        lo, hi = v.min(), v.max()
                        if hi > lo:
                            overlay_df.loc[mask, "value"] = (v - lo) / (hi - lo)
                        else:
                            overlay_df.loc[mask, "value"] = 0.0
                chart_overlay = (
                    AltairChartBuilder(overlay_df)
                    .x("timestep", title="Timestep")
                    .y("value", title="Value" + (" (normalized 0–1)" if overlay_normalize else ""))
                    .color("metric", legend_title="Series")
                    .mark_line(strokeWidth=2)
                    .build()
                )
                st.altair_chart(chart_overlay, use_container_width=True)
            else:
                st.caption("No data for selected series.")
        elif overlay_selected:
            st.caption("Select at least two series to overlay.")

        st.divider()

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

        # ---- Chart 1c: Firms in business over time ----
        st.subheader("Firms in business over time")
        firms_in_business_chart_df = df_builder.firms_in_business_over_time()
        if not firms_in_business_chart_df.empty:
            firms_in_business_chart_df = firms_in_business_chart_df.copy()
            firms_in_business_chart_df["metric"] = "Firms in business"
            color_firms = st.color_picker("Line color", "#2ca02c", key="chart_firms_in_business_color")
            chart_firms_in_business_tab4 = (
                AltairChartBuilder(firms_in_business_chart_df)
                .x("timestep", title="Timestep")
                .y("value", title="Firms in business")
                .color(
                    "metric",
                    domain=["Firms in business"],
                    range_=[color_firms],
                    legend_title="Metric",
                )
                .mark_line(strokeWidth=2)
                .build()
            )
            st.altair_chart(chart_firms_in_business_tab4, use_container_width=True)
        else:
            st.caption("No firm count data in this run.")

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

    # LEMON MARKET TAB: Listings, bids/passes, and quality/price structure (LEMON_MARKET runs only).
    with tab5:
        is_lemon_run = (
            experiment_args_dict is not None
            and experiment_args_dict.get("consumer_scenario") == "LEMON_MARKET"
        )
        if not is_lemon_run:
            st.info("Current selection is not a Lemon Market run.")
        else:
            import altair as alt

            # ---- Section 1: Listings, bids, passes over time ----
            df_builder_lemon = DataFrameBuilder(state_files=state_files)
            lemon_metrics = df_builder_lemon.lemon_market_metrics_over_time()
            st.subheader("Listings, bids, and passes over time")
            if not lemon_metrics.empty:
                chart_lemon = (
                    AltairChartBuilder(lemon_metrics)
                    .x("timestep", title="Timestep")
                    .y("value", title="Count")
                    .color(
                        "metric",
                        domain=["Listings", "Bids", "Passes"],
                        range_=["#1f77b4", "#2ca02c", "#ff7f0e"],
                        legend_title="Metric",
                    )
                    .mark_line(strokeWidth=2)
                    .build()
                )
                st.altair_chart(chart_lemon, use_container_width=True)
            else:
                st.caption("No lemon market metrics found in state files for this run.")

            # ---- Section 2: Current timestep summary + listings ----
            st.subheader("Current timestep summary")
            col_a, col_b, col_c, col_d = st.columns(4)
            listings_now = state.get("lemon_market_listings_count", 0)
            bids_now = state.get("lemon_market_bids_count", 0)
            passes_now = state.get("lemon_market_passes_count", 0)
            filled_now = state.get("filled_orders_count", 0)
            col_a.metric("Listings this step", int(listings_now) if isinstance(listings_now, (int, float)) else 0)
            col_b.metric("Bids this step", int(bids_now) if isinstance(bids_now, (int, float)) else 0)
            col_c.metric("Passes this step", int(passes_now) if isinstance(passes_now, (int, float)) else 0)
            col_d.metric("Filled orders this step", int(filled_now) if isinstance(filled_now, (int, float)) else 0)
            col_e, col_f = st.columns(2)
            sybil_rev_now = state.get("lemon_market_sybil_revenue_share")
            avg_cs_now = state.get("lemon_market_avg_consumer_surplus")
            col_e.metric(
                "Sybil revenue share",
                f"{sybil_rev_now:.1%}" if isinstance(sybil_rev_now, (int, float)) else "—",
            )
            col_f.metric(
                "Avg consumer surplus",
                f"${avg_cs_now:,.0f}" if isinstance(avg_cs_now, (int, float)) else "—",
            )

            def _fmt_vote_cell(v):
                if not isinstance(v, (int, float)):
                    return "—"
                v = float(v)
                return int(v) if abs(v - round(v)) < 1e-9 else round(v, 2)

            st.subheader("Seller rating summary (this timestep)")
            vote_rows = []
            for f in state.get("firms", []):
                name = f.get("name")
                if not name:
                    continue
                u = f.get("upvotes")
                d = f.get("downvotes")
                rep = f.get("reputation")
                is_active = f.get("in_business", True)
                vote_rows.append({
                    "Seller": name,
                    "Sybil": bool(f.get("sybil", False)),
                    "Status": "Active" if is_active else "Retired",
                    "Created (t)": f.get("timestep_created"),
                    "Retired (t)": f.get("timestep_retired"),
                    "Total Sales": f.get("sales_by_good", {}).get("car", 0),
                    "Upvotes": _fmt_vote_cell(u) if u is not None else "—",
                    "Downvotes": _fmt_vote_cell(d) if d is not None else "—",
                    "Reputation": f"{rep:.4f}" if isinstance(rep, (int, float)) else "—",
                })
            if vote_rows:
                vote_df = pd.DataFrame(vote_rows)
                # Sort: sybil first, then honest; within each group active before retired
                vote_df["_s"] = vote_df["Sybil"].map({True: 0, False: 1})
                vote_df["_r"] = vote_df["Status"].map({"Active": 0, "Retired": 1})
                vote_df = vote_df.sort_values(["_s", "_r"]).drop(columns=["_s", "_r"]).reset_index(drop=True)
                st.dataframe(vote_df, use_container_width=True)
            else:
                st.caption("No firm data for vote counts.")

            # ---- Sybil identity lifecycle section ----
            st.subheader("Sybil identity lifecycle (all identities this run)")
            registry_df = df_builder_lemon.build_sybil_identity_registry()
            if not registry_df.empty:
                st.dataframe(registry_df, use_container_width=True)
                n_retired = int((registry_df["Status"] == "Retired").sum())
                n_active = int((registry_df["Status"] == "Active").sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Active identities", n_active)
                c2.metric("Retired identities", n_retired)
                total_created = state.get("deceptive_principal", {}).get("total_identities_created", "—")
                c3.metric("Total identities created", total_created)
            else:
                st.info("No sybil identities found in this run.")

            # ---- Sybil rotation events timeline ----
            rotation_data = [
                {
                    "timestep": s.get("timestep"),
                    "Rotations": s.get("deceptive_principal", {}).get("retired_this_step", 0),
                }
                for s in df_builder_lemon.states
                if s.get("deceptive_principal") is not None
            ]
            if rotation_data:
                rot_df = pd.DataFrame(rotation_data).set_index("timestep")
                if rot_df["Rotations"].sum() > 0:
                    st.subheader("Sybil identity rotations over time")
                    st.bar_chart(rot_df)

            # ---- Seller detail panel ----
            st.subheader("Seller detail")
            # Collect all seller names and their last-seen snapshot across all states
            _all_seller_snaps: dict = {}  # name -> last snapshot
            for _s in df_builder_lemon.states:
                _t = _s.get("timestep", 0)
                for _f in _s.get("firms", []):
                    _n = _f.get("name")
                    if _n:
                        _all_seller_snaps[_n] = {**_f, "_last_t": _t}

            if _all_seller_snaps:
                # Sort: sybil first, then honest; within sybil sort by created timestep
                def _seller_sort_key(n):
                    snap = _all_seller_snaps[n]
                    is_sybil = 0 if snap.get("sybil", False) else 1
                    created = snap.get("timestep_created") or 999
                    return (is_sybil, created, n)

                _seller_names = sorted(_all_seller_snaps.keys(), key=_seller_sort_key)
                _selected = st.selectbox("Select seller", _seller_names, key="lemon_seller_detail")

                if _selected:
                    _snap = _all_seller_snaps[_selected]
                    _is_sybil = _snap.get("sybil", False)
                    _is_active = _snap.get("in_business", True)

                    # Info metrics row
                    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                    _mc1.metric("Type", "Sybil" if _is_sybil else "Honest")
                    _mc2.metric("Status", "Active" if _is_active else "Retired")
                    _rep = _snap.get("reputation")
                    _mc3.metric("Reputation", f"{_rep:.4f}" if isinstance(_rep, (int, float)) else "—")
                    _mc4.metric("Total Sales", _snap.get("sales_by_good", {}).get("car", 0))

                    _mv1, _mv2, _mv3, _mv4 = st.columns(4)
                    _up = _snap.get("upvotes")
                    _dn = _snap.get("downvotes")
                    _mv1.metric("Upvotes", f"{_up:.1f}" if isinstance(_up, (int, float)) else "—")
                    _mv2.metric("Downvotes", f"{_dn:.1f}" if isinstance(_dn, (int, float)) else "—")
                    if _is_sybil:
                        _mv3.metric("Created (t)", _snap.get("timestep_created", "—"))
                        _mv4.metric("Retired (t)", _snap.get("timestep_retired", "—") if not _is_active else "—")

                    # Reputation over time for this seller
                    _rep_rows = []
                    for _s in df_builder_lemon.states:
                        _t = _s.get("timestep", 0)
                        for _f in _s.get("firms", []):
                            if _f.get("name") == _selected:
                                _rep_rows.append({
                                    "timestep": _t,
                                    "Reputation": _f.get("reputation"),
                                    "Upvotes": _f.get("upvotes"),
                                    "Downvotes": _f.get("downvotes"),
                                })
                    if _rep_rows:
                        _rep_df = pd.DataFrame(_rep_rows).sort_values("timestep")
                        _rep_chart = (
                            alt.Chart(_rep_df)
                            .mark_line(point=True, strokeWidth=2, color="#1f77b4")
                            .encode(
                                x=alt.X("timestep:Q", title="Timestep"),
                                y=alt.Y("Reputation:Q", title="Reputation", scale=alt.Scale(domain=[0, 1])),
                                tooltip=["timestep:Q", "Reputation:Q", "Upvotes:Q", "Downvotes:Q"],
                            )
                            .properties(height=200, title=f"Reputation over time — {_selected}")
                        )
                        st.altair_chart(_rep_chart, use_container_width=True)

                    # Listing history for this seller across all states
                    _listing_rows = []
                    for _s in df_builder_lemon.states:
                        _t = _s.get("timestep", 0)
                        for _row in (_s.get("lemon_market_new_listings") or []):
                            if isinstance(_row, dict) and _row.get("firm_id") == _selected:
                                # Find seller rep at this timestep from firm snapshot
                                _seller_rep_at_t = None
                                for _f in _s.get("firms", []):
                                    if _f.get("name") == _selected:
                                        _seller_rep_at_t = _f.get("reputation")
                                        break
                                _listing_rows.append({
                                    "Timestep": _t,
                                    "Listing ID": _row.get("id"),
                                    "True Quality": _row.get("quality"),
                                    "Quality Value": _row.get("quality_value"),
                                    "Price": _row.get("price"),
                                    "Reputation at Post": _seller_rep_at_t,
                                    "Description": _row.get("description"),
                                })
                    if _listing_rows:
                        st.markdown(f"**Listing history ({len(_listing_rows)} listings)**")
                        _listing_df = pd.DataFrame(_listing_rows).sort_values("Timestep")
                        st.dataframe(_listing_df, use_container_width=True)
                    else:
                        st.caption("No listing history found for this seller in the current state files.")

                    st.markdown("**LLM prompts & responses**")
                    _seller_prompt_recs = _filter_seller_prompt_records(
                        lemon_prompt_records, _selected
                    )
                    if prompt_log_path and os.path.isfile(prompt_log_path):
                        st.caption(
                            f"{len(_seller_prompt_recs)} logged exchange(s) for **{_selected}** in this run."
                        )
                    else:
                        st.caption(
                            "No `lemon_agent_prompts.jsonl` in this run folder. "
                            "Re-run with `--log-seller-prompts` (and/or `--log-buyer-prompts`) to capture LLM I/O."
                        )
                    _render_lemon_prompt_navigator(
                        _seller_prompt_recs,
                        f"lemon_seller_prompt_{_selected}",
                    )
            else:
                st.info("No seller data available.")

            # ---- Buyer detail panel (LEMON_MARKET buyers) ----
            st.subheader("Buyer detail")
            _lemon_buyer_names = [c["name"] for c in state.get("consumers", [])]
            if _lemon_buyer_names:
                _buyer_sel = st.selectbox(
                    "Select buyer",
                    _lemon_buyer_names,
                    key="lemon_buyer_detail_select",
                )
                _buyer_snap = next(
                    (c for c in state.get("consumers", []) if c.get("name") == _buyer_sel),
                    None,
                )
                if _buyer_snap:
                    _bc1, _bc2, _bc3 = st.columns(3)
                    _seen = _buyer_snap.get("sybil_seen_total")
                    _passed = _buyer_snap.get("sybil_passed_total")
                    _bc1.metric(
                        "Sybil listings seen (total)",
                        int(_seen) if isinstance(_seen, (int, float)) else "—",
                    )
                    _bc2.metric(
                        "Sybil listings passed (total)",
                        int(_passed) if isinstance(_passed, (int, float)) else "—",
                    )
                    _util = _buyer_snap.get("utility")
                    _bc3.metric(
                        "Utility (proxy)",
                        f"{_util:.2f}" if isinstance(_util, (int, float)) else "—",
                    )
                st.markdown("**LLM prompts & responses**")
                _buyer_prompt_recs = _filter_buyer_prompt_records(
                    lemon_prompt_records, _buyer_sel
                )
                if prompt_log_path and os.path.isfile(prompt_log_path):
                    st.caption(
                        f"{len(_buyer_prompt_recs)} logged exchange(s) for **{_buyer_sel}** in this run."
                    )
                else:
                    st.caption(
                        "No `lemon_agent_prompts.jsonl` in this run folder. "
                        "Re-run with `--log-buyer-prompts` (and/or `--log-seller-prompts`) to capture LLM I/O."
                    )
                _render_lemon_prompt_navigator(
                    _buyer_prompt_recs,
                    f"lemon_buyer_prompt_{_buyer_sel}",
                )
            else:
                st.caption("No buyers (consumers) in this run.")

            st.subheader("Seller upvotes and downvotes over time")
            df_votes_long = df_builder_lemon.seller_vote_counts_long_over_time()
            df_plot = df_votes_long.dropna(subset=["value"]) if not df_votes_long.empty else pd.DataFrame()
            if not df_plot.empty:
                chart_votes_ts = (
                    alt.Chart(df_plot)
                    .mark_line(point=True, strokeWidth=2)
                    .encode(
                        x=alt.X("timestep:Q", title="Timestep"),
                        y=alt.Y("value:Q", title="Count"),
                        color=alt.Color(
                            "metric:N",
                            title="Metric",
                            scale=alt.Scale(
                                domain=["Upvotes", "Downvotes"],
                                range=["#2ca02c", "#d62728"],
                            ),
                        ),
                    )
                    .properties(width=200, height=150)
                    .facet(
                        alt.Facet("firm:N", title="Seller"),
                        columns=4,
                    )
                )
                st.altair_chart(chart_votes_ts, use_container_width=True)
            else:
                st.caption(
                    "No upvotes/downvotes in state files for this run. "
                    "Re-run the simulation after updating the env so state snapshots include vote counts."
                )

            # Firm metadata (sybil flag + current reputation) for enrichment
            firm_meta = {}
            for f in state.get("firms", []):
                name = f.get("name")
                if not name:
                    continue
                firm_meta[name] = {
                    "sybil": bool(f.get("sybil", False)),
                    "reputation": f.get("reputation"),
                }

            st.subheader("Listings in this timestep")
            new_listings = state.get("lemon_market_new_listings") or []
            unsold_pool = state.get("lemon_market_unsold_listings") or []
            rows_step = []
            if isinstance(new_listings, list):
                for row in new_listings:
                    if not isinstance(row, dict):
                        continue
                    enriched = dict(row)
                    meta = firm_meta.get(row.get("firm_id"), {})
                    enriched["sybil"] = meta.get("sybil", False)
                    enriched["firm_reputation"] = meta.get("reputation")
                    enriched["status"] = "new"
                    enriched["timestep"] = state.get("timestep", 0)
                    rows_step.append(enriched)
            if isinstance(unsold_pool, list):
                for row in unsold_pool:
                    if not isinstance(row, dict):
                        continue
                    enriched = dict(row)
                    meta = firm_meta.get(row.get("firm_id"), {})
                    enriched["sybil"] = meta.get("sybil", False)
                    enriched["firm_reputation"] = meta.get("reputation")
                    enriched["status"] = "unsold_pool"
                    enriched["timestep"] = state.get("timestep", 0)
                    rows_step.append(enriched)

            if rows_step:
                df_step = pd.DataFrame(rows_step)
                base_cols = [
                    "timestep",
                    "timestep_posted",
                    "status",
                    "id",
                    "firm_id",
                    "sybil",
                    "quality",
                    "quality_value",
                    "price",
                    "reputation",
                    "firm_reputation",
                    "description",
                ]
                cols = [c for c in base_cols if c in df_step.columns]
                extra = [c for c in df_step.columns if c not in cols]
                st.dataframe(df_step[cols + extra], use_container_width=True)
            else:
                st.caption("No lemon market listings recorded for this timestep.")

            # ---- Section 3: Listings and unsold listings across the full run ----
            all_listings = []
            all_unsold = []
            for path in state_files:
                with open(path, "r") as f:
                    snap = json.load(f)
                t = snap.get("timestep", 0)
                firm_meta_snap = {}
                for f in snap.get("firms", []):
                    name = f.get("name")
                    if not name:
                        continue
                    firm_meta_snap[name] = {
                        "sybil": bool(f.get("sybil", False)),
                        "reputation": f.get("reputation"),
                    }
                for row in snap.get("lemon_market_new_listings", []) or []:
                    if not isinstance(row, dict):
                        continue
                    enriched = dict(row)
                    meta = firm_meta_snap.get(row.get("firm_id"), {})
                    enriched["sybil"] = meta.get("sybil", False)
                    enriched["firm_reputation"] = meta.get("reputation")
                    enriched["timestep"] = t
                    all_listings.append(enriched)
                for row in snap.get("lemon_market_unsold_listings", []) or []:
                    if not isinstance(row, dict):
                        continue
                    enriched = dict(row)
                    meta = firm_meta_snap.get(row.get("firm_id"), {})
                    enriched["sybil"] = meta.get("sybil", False)
                    enriched["firm_reputation"] = meta.get("reputation")
                    enriched["timestep"] = t
                    all_unsold.append(enriched)

            if all_listings:
                st.subheader("All posted listings (entire run)")
                df_all = pd.DataFrame(all_listings)
                base_cols_all = [
                    "timestep",
                    "timestep_posted",
                    "id",
                    "firm_id",
                    "sybil",
                    "quality",
                    "quality_value",
                    "price",
                    "reputation",
                    "firm_reputation",
                    "description",
                ]
                cols_all = [c for c in base_cols_all if c in df_all.columns]
                extra_all = [c for c in df_all.columns if c not in cols_all]
                st.dataframe(df_all[cols_all + extra_all], use_container_width=True)

                # Scatter: true quality vs. price, colored by seller type, sized by reputation
                if {"quality_value", "price"}.issubset(df_all.columns):
                    df_scatter = df_all.copy()
                    df_scatter["seller_type"] = np.where(df_scatter["sybil"], "Sybil", "Honest")
                    chart_scatter = (
                        alt.Chart(df_scatter)
                        .mark_circle(opacity=0.7)
                        .encode(
                            x=alt.X("quality_value:Q", title="True quality value"),
                            y=alt.Y("price:Q", title="Listing price"),
                            color=alt.Color(
                                "seller_type:N",
                                title="Seller type",
                                scale=alt.Scale(
                                    domain=["Honest", "Sybil"],
                                    range=["#1f77b4", "#ffd92f"],
                                ),
                            ),
                            size=alt.Size("reputation:Q", title="Listing-time reputation", scale=alt.Scale(range=[20, 300])),
                            tooltip=[
                                "timestep:Q",
                                "id:N",
                                "firm_id:N",
                                "seller_type:N",
                                "quality:N",
                                "quality_value:Q",
                                "price:Q",
                                "reputation:Q",
                            ],
                        )
                    )
                    st.subheader("Price vs. true quality by seller type")
                    st.altair_chart(chart_scatter, use_container_width=True)
            else:
                st.caption("No lemon market listings found in state files for this run.")

            if all_unsold:
                st.subheader("All unsold listings (carried over between timesteps)")
                df_unsold = pd.DataFrame(all_unsold)
                base_cols_unsold = [
                    "timestep",
                    "id",
                    "firm_id",
                    "sybil",
                    "quality",
                    "quality_value",
                    "price",
                    "reputation",
                    "firm_reputation",
                    "description",
                ]
                cols_unsold = [c for c in base_cols_unsold if c in df_unsold.columns]
                extra_unsold = [c for c in df_unsold.columns if c not in cols_unsold]
                st.dataframe(df_unsold[cols_unsold + extra_unsold], use_container_width=True)

            # ---- Section 4: Lemon Market time-series metrics ----
            st.subheader("Sybil revenue share over time")
            df_sybil_rev = df_builder_lemon.lemon_sybil_revenue_share_over_time()
            if not df_sybil_rev.empty:
                chart_sybil_rev = (
                    AltairChartBuilder(df_sybil_rev)
                    .x("timestep", title="Timestep")
                    .y("value", title="Sybil revenue share")
                    .mark_line(strokeWidth=2)
                    .build()
                )
                st.altair_chart(chart_sybil_rev, use_container_width=True)
            else:
                st.caption("No sybil revenue data (run needs --sybil-cluster-size > 0).")

            st.subheader("Avg consumer surplus over time")
            df_avg_cs = df_builder_lemon.lemon_avg_consumer_surplus_over_time()
            if not df_avg_cs.empty:
                chart_avg_cs = (
                    AltairChartBuilder(df_avg_cs)
                    .x("timestep", title="Timestep")
                    .y("value", title="Avg consumer surplus ($)")
                    .mark_line(strokeWidth=2)
                    .build()
                )
                st.altair_chart(chart_avg_cs, use_container_width=True)
            else:
                st.caption("No consumer surplus data yet (no purchases recorded).")

            st.subheader("Cumulative consumer surplus per buyer")
            df_cum_cs = df_builder_lemon.lemon_consumer_surplus_cumulative_per_buyer_over_time()
            if not df_cum_cs.empty:
                chart_cum_cs = (
                    AltairChartBuilder(df_cum_cs)
                    .x("timestep", title="Timestep")
                    .y("value", title="Cumulative CS ($)")
                    .color("consumer", legend_title="Buyer")
                    .mark_line(strokeWidth=2)
                    .build()
                )
                st.altair_chart(chart_cum_cs, use_container_width=True)
            else:
                st.caption("No cumulative consumer surplus data yet.")

            st.subheader("Sybil pass rate per buyer over time")
            df_pass_rate = df_builder_lemon.lemon_sybil_pass_rate_per_buyer_over_time()
            if not df_pass_rate.empty:
                chart_pass_rate = (
                    AltairChartBuilder(df_pass_rate)
                    .x("timestep", title="Timestep")
                    .y("value", title="Sybil pass rate")
                    .color("consumer", legend_title="Buyer")
                    .mark_line(strokeWidth=2)
                    .build()
                )
                st.altair_chart(chart_pass_rate, use_container_width=True)
            else:
                st.caption("No sybil pass rate data (no sybil listings seen yet).")

            # ---- Section 5: Buyer sybil encounter summary table ----
            st.subheader("Buyer sybil encounter summary")
            buyer_summary_rows = []
            for c in state.get("consumers", []):
                name = c.get("name", "")
                seen = c.get("sybil_seen_total", 0)
                passed = c.get("sybil_passed_total", 0)
                rate = passed / seen if seen > 0 else None
                buyer_summary_rows.append({
                    "Buyer": name,
                    "Sybil listings seen (total)": seen,
                    "Sybil listings passed (total)": passed,
                    "Pass rate (run-level)": f"{rate:.1%}" if rate is not None else "—",
                })
            if buyer_summary_rows:
                st.dataframe(pd.DataFrame(buyer_summary_rows), use_container_width=True)
            else:
                st.caption("No buyer data available.")

    # DISCOVERY TAB: Consumer-firm exposure (THE_CRASH discovery-limit mechanic).
    with tab6:
        import altair as alt

        views_by_firm_state = state.get("views_by_firm", {})
        has_discovery = bool(views_by_firm_state)

        if not has_discovery:
            st.info("No discovery data at this timestep. Discovery tracking is active for THE_CRASH scenario runs.")
        else:
            # --- Section 1: Exposure summary table ---
            st.subheader("Exposure summary")
            summary_rows = []
            for f in state["firms"]:
                if not f.get("in_business", True):
                    continue
                prices = f.get("prices") or {}
                price_val = list(prices.values())[0] if prices else None
                views = f.get("views_this_step", 0)
                sales = len(f.get("sales_info") or [])
                conv = f.get("conversion_rate")
                summary_rows.append({
                    "Firm": f["name"],
                    "Price": f"${price_val:.2f}" if isinstance(price_val, (int, float)) else "—",
                    "Views": views,
                    "Orders": sales,
                    "Conversion %": f"{conv * 100:.1f}%" if conv is not None else "—",
                })
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            # --- Section 2: Bipartite discovery graph ---
            st.subheader("Bipartite discovery graph")

            # Determine good to display (support multi-good runs)
            goods_with_data = set()
            for c in state["consumers"]:
                for g in (c.get("discovery_this_step") or {}).keys():
                    goods_with_data.add(g)
            goods_list_disc = sorted(goods_with_data)

            if not goods_list_disc:
                st.caption("No per-consumer discovery data (discovery_this_step not recorded for this run).")
            else:
                if len(goods_list_disc) > 1:
                    selected_good_disc = st.selectbox("Good", options=goods_list_disc, key="disc_good_select")
                else:
                    selected_good_disc = goods_list_disc[0]

                active_firms = [f for f in state["firms"] if f.get("in_business", True)]
                all_consumers_disc = state["consumers"]
                n_firms = max(1, len(active_firms))
                n_cons = max(1, len(all_consumers_disc))

                # Stable y positions — compress into [0.2, 0.8] so nodes cluster visually
                def _spread(i, n, lo=0.2, hi=0.8):
                    if n <= 1:
                        return (lo + hi) / 2
                    return lo + (hi - lo) * i / (n - 1)

                firm_y = {f["name"]: _spread(i, n_firms) for i, f in enumerate(active_firms)}
                consumer_y = {c["name"]: _spread(i, n_cons) for i, c in enumerate(all_consumers_disc)}

                # Firm nodes
                firm_node_rows = []
                for f in active_firms:
                    prices = f.get("prices") or {}
                    price_val = list(prices.values())[0] if prices else 0.0
                    if not isinstance(price_val, (int, float)):
                        price_val = 0.0
                    views = f.get("views_this_step", 0) or 0
                    conv = f.get("conversion_rate") or 0.0
                    firm_node_rows.append({
                        "x": 1.0, "y": firm_y[f["name"]], "label": f["name"],
                        "views": float(views), "conversion": float(conv),
                        "price": float(price_val), "node_type": "firm",
                    })
                firm_nodes = pd.DataFrame(firm_node_rows)

                # Consumer nodes
                consumer_node_rows = []
                for c in all_consumers_disc:
                    disc = (c.get("discovery_this_step") or {}).get(selected_good_disc, {})
                    ordered_from = disc.get("ordered")
                    participating = bool(c.get("discovery_this_step"))
                    consumer_node_rows.append({
                        "x": 0.0, "y": consumer_y[c["name"]], "label": c["name"],
                        "ordered_from": str(ordered_from) if ordered_from else "",
                        "participating": participating,
                        "node_type": "consumer",
                    })
                consumer_nodes = pd.DataFrame(consumer_node_rows)

                # Edges
                edge_rows = []
                for c in all_consumers_disc:
                    disc = (c.get("discovery_this_step") or {}).get(selected_good_disc, {})
                    c_y = consumer_y.get(c["name"], 0.0)
                    for firm_id in disc.get("seen", []):
                        if firm_id not in firm_y:
                            continue
                        is_ordered = (firm_id == disc.get("ordered"))
                        edge_id = f"{c['name']}__{firm_id}"
                        for xval, yval in [(0.0, c_y), (1.0, firm_y[firm_id])]:
                            edge_rows.append({
                                "edge_id": edge_id, "x": xval, "y": yval,
                                "firm": firm_id, "ordered": is_ordered,
                            })
                edges_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(columns=["edge_id", "x", "y", "firm", "ordered"])

                chart_height = max(300, 40 * max(n_firms, n_cons))

                layers = []

                if not edges_df.empty:
                    seen_edges_df = edges_df[~edges_df["ordered"]].copy()
                    ordered_edges_df = edges_df[edges_df["ordered"]].copy()

                    if not seen_edges_df.empty:
                        seen_layer = alt.Chart(seen_edges_df).mark_line(
                            strokeDash=[4, 4], opacity=0.25, strokeWidth=1, color="#aaaaaa"
                        ).encode(
                            x=alt.X("x:Q", axis=None),
                            y=alt.Y("y:Q", axis=None),
                            detail="edge_id:N",
                        )
                        layers.append(seen_layer)

                    if not ordered_edges_df.empty:
                        ordered_layer = alt.Chart(ordered_edges_df).mark_line(
                            strokeWidth=2.5, opacity=0.85
                        ).encode(
                            x=alt.X("x:Q", axis=None),
                            y=alt.Y("y:Q", axis=None),
                            detail="edge_id:N",
                            color=alt.Color("firm:N", title="Firm"),
                        )
                        layers.append(ordered_layer)

                if not firm_nodes.empty:
                    firm_chart = alt.Chart(firm_nodes).mark_circle(stroke="white", strokeWidth=1.5).encode(
                        x=alt.X("x:Q", axis=None),
                        y=alt.Y("y:Q", axis=None),
                        size=alt.Size("views:Q", scale=alt.Scale(range=[200, 1200]), title="Views"),
                        color=alt.Color("conversion:Q",
                                        scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                                        title="Conversion rate"),
                        tooltip=["label:N", alt.Tooltip("price:Q", format="$.2f"),
                                 alt.Tooltip("views:Q"), alt.Tooltip("conversion:Q", format=".1%")],
                    )
                    firm_labels = alt.Chart(firm_nodes).mark_text(dx=14, align="left", fontSize=11).encode(
                        x=alt.X("x:Q", axis=None),
                        y=alt.Y("y:Q", axis=None),
                        text="label:N",
                    )
                    layers.extend([firm_chart, firm_labels])

                if not consumer_nodes.empty:
                    consumer_chart = alt.Chart(consumer_nodes).mark_circle(size=80).encode(
                        x=alt.X("x:Q", axis=None),
                        y=alt.Y("y:Q", axis=None),
                        color=alt.condition(
                            alt.datum.participating,
                            alt.Color("ordered_from:N", title="Ordered from"),
                            alt.value("#e0e0e0"),
                        ),
                        opacity=alt.condition(
                            alt.datum.participating,
                            alt.value(0.9),
                            alt.value(0.3),
                        ),
                        tooltip=["label:N", "ordered_from:N"],
                    )
                    participating_nodes = consumer_nodes[consumer_nodes["participating"]].copy()
                    if not participating_nodes.empty:
                        consumer_labels = alt.Chart(participating_nodes).mark_text(
                            dx=-14, align="right", fontSize=10
                        ).encode(
                            x=alt.X("x:Q", axis=None),
                            y=alt.Y("y:Q", axis=None),
                            text="label:N",
                        )
                        layers.extend([consumer_chart, consumer_labels])
                    else:
                        layers.append(consumer_chart)

                if layers:
                    graph = alt.layer(*layers).properties(
                        width=600,
                        height=chart_height,
                        title=f"Consumer Discovery — Timestep {state['timestep']} ({selected_good_disc})",
                    ).configure_axis(grid=False).configure_view(strokeWidth=0)
                    st.altair_chart(graph, use_container_width=True)

            # --- Section 3: Consumer-firm interaction matrix ---
            st.subheader("Consumer-firm interaction matrix")
            if goods_list_disc:
                matrix_good = goods_list_disc[0] if len(goods_list_disc) == 1 else st.selectbox(
                    "Good (matrix)", options=goods_list_disc, key="disc_matrix_good"
                )
                firm_names_active = [f["name"] for f in state["firms"] if f.get("in_business", True)]
                consumer_names_all = [c["name"] for c in state["consumers"]]
                matrix_rows = []
                for c in state["consumers"]:
                    disc = (c.get("discovery_this_step") or {}).get(matrix_good, {})
                    seen_set = set(disc.get("seen", []))
                    ordered = disc.get("ordered")
                    row = {"Consumer": c["name"]}
                    for fname in firm_names_active:
                        if fname == ordered:
                            row[fname] = "ordered"
                        elif fname in seen_set:
                            row[fname] = "seen"
                        else:
                            row[fname] = ""
                    matrix_rows.append(row)
                if matrix_rows:
                    matrix_df = pd.DataFrame(matrix_rows).set_index("Consumer")
                    st.dataframe(matrix_df, use_container_width=True)
                else:
                    st.caption("No matrix data available.")
            else:
                st.caption("No discovery data to build matrix.")

            # --- Section 4: Views over time ---
            st.subheader("Views per firm over time")
            df_builder_disc = DataFrameBuilder(state_files=state_files)
            views_df = df_builder_disc.views_per_firm_over_time()
            if not views_df.empty:
                chart_views = (
                    AltairChartBuilder(views_df)
                    .x("timestep", title="Timestep")
                    .y("value", title="Consumer views")
                    .color("firm", legend_title="Firm")
                    .mark_line(strokeWidth=2)
                    .build()
                )
                st.altair_chart(chart_views, use_container_width=True)
            else:
                st.caption("No views-over-time data (only populated for THE_CRASH scenario).")

    # TOKEN USAGE TAB: Aggregate and per-run token usage statistics.
    with tab7:
        st.subheader("Token usage summary")
        if not token_usage_files:
            st.caption(
                "No *_token_usage.json files found in this run directory. "
                "Run a simulation with LLM firms to generate token usage logs."
            )
        else:
            # If multiple usage files exist (e.g., multiple experiments in same folder),
            # allow the user to pick which one to inspect.
            usage_labels = [os.path.basename(p) for p in token_usage_files]
            selected_label = st.selectbox(
                "Token usage file",
                options=usage_labels,
                index=0,
                key="token_usage_file_select",
            )
            selected_path = next(
                p for p in token_usage_files if os.path.basename(p) == selected_label
            )

            try:
                with open(selected_path, "r") as f:
                    usage_data = json.load(f)
            except Exception as e:
                st.error(f"Failed to load token usage from {selected_label}: {e}")
                usage_data = None

            if isinstance(usage_data, dict):
                input_tokens = float(usage_data.get("input_tokens", 0) or 0)
                output_tokens = float(usage_data.get("output_tokens", 0) or 0)
                requests = int(usage_data.get("requests", 0) or 0)
                total_tokens = input_tokens + output_tokens

                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Input tokens", f"{int(input_tokens):,}")
                col_b.metric("Output tokens", f"{int(output_tokens):,}")
                col_c.metric("Total tokens", f"{int(total_tokens):,}")
                col_d.metric("LLM requests", f"{requests:,}")

                st.subheader("Raw JSON")
                st.code(json.dumps(usage_data, indent=2), language="json")
            else:
                st.caption(
                    f"{selected_label} does not contain a JSON object with token usage totals."
                )