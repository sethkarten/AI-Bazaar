"""
DataFrame builder for simulation state: builds pandas DataFrames from state JSON files
or state dicts. Caller supplies file paths or state lists; this module handles
aggregation, melting, and optional renames for viz/charting.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _gini(cash_values: List[float]) -> float:
    """Gini coefficient; 0 = perfect equality, 1 = all wealth in one agent."""
    if not cash_values or sum(cash_values) == 0:
        return 0.0
    vals = np.array(sorted(cash_values), dtype=float)
    n = len(vals)
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * vals) / (n * np.sum(vals)))


class DataFrameBuilder:
    """
    Builds DataFrames from simulation state (either state file paths or list of state dicts).
    Use for metrics-over-time (wide or long) and single-state tables like cash-by-agent.
    """

    def __init__(
        self,
        state_files: Optional[List[str]] = None,
        states: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Provide either state_files (paths to state_t*.json) or states (list of
        loaded state dicts). If state_files is given, states are loaded from disk on demand.
        """
        self._state_files = state_files or []
        self._states = states
        if self._states is None and self._state_files:
            self._states = self._load_states(self._state_files)

    @staticmethod
    def _load_states(paths: List[str]) -> List[Dict[str, Any]]:
        """Load state dicts from JSON file paths, in path order."""
        out = []
        for path in paths:
            with open(path, "r") as f:
                out.append(json.load(f))
        return out

    @property
    def states(self) -> List[Dict[str, Any]]:
        """Ordered list of state dicts (from files or as passed in)."""
        if self._states is None:
            self._states = self._load_states(self._state_files)
        return self._states

    def _all_firm_names(self) -> List[str]:
        """Union of all firm names from state["firms"] and ledger (firm_*) so charts include every firm."""
        seen = set()
        names = []
        for s in self.states:
            for f in s.get("firms", []):
                name = f.get("name")
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
            for key in s.get("ledger", {}).get("money", {}):
                if key.startswith("firm_") and key not in seen:
                    seen.add(key)
                    names.append(key)
        return sorted(names)

    def _all_consumer_names(self) -> List[str]:
        """Union of all consumer names from state["consumers"] and ledger (consumer_*) so charts include every consumer."""
        seen = set()
        names = []
        for s in self.states:
            for c in s.get("consumers", []): 
                name = c.get("name")
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
            for key in s.get("ledger", {}).get("money", {}):
                if key.startswith("consumer_") and key not in seen:
                    seen.add(key)
                    names.append(key)
        
        return sorted(names)

    def _all_good_names(self) -> List[str]:
        """Union of all good names from firm["prices"] across states (e.g. food, supply)."""
        seen = set()
        for s in self.states:
            for f in s.get("firms", []):
                for g in (f.get("prices") or {}).keys():
                    if g and g not in seen:
                        seen.add(g)
        return sorted(seen)

    def _all_inventory_good_names(self) -> List[str]:
        """Union of all good names from firm["inventory"] across states (e.g. food, supply)."""
        seen = set()
        for s in self.states:
            for f in s.get("firms", []):
                for g in (f.get("inventory") or {}).keys():
                    if g and g not in seen:
                        seen.add(g)
        return sorted(seen)

    def _all_consumer_ewtp_good_names(self) -> List[str]:
        """Union of all good names from consumer eWTP dicts across states."""
        seen = set()
        for s in self.states:
            for c in s.get("consumers", []):
                ewtp = c.get("eWTP") or {}
                for g in ewtp.keys():
                    if g and g not in seen:
                        seen.add(g)
        return sorted(seen)

    def consumer_ewtp_by_good_over_time(
        self, consumer_name: str
    ) -> pd.DataFrame:
        """
        Long format: one row per (timestep, good). Value is eWTP for the given consumer per good.
        Uses all goods that appear in any consumer eWTP across states; missing (timestep, good) get value 0.
        """
        goods = self._all_consumer_ewtp_good_names()
        if not goods:
            return pd.DataFrame(columns=["timestep", "good", "value"])
        rows = []
        for s in self.states:
            t = s["timestep"]
            consumer_by_name = {c.get("name"): c for c in s.get("consumers", []) if c.get("name")}
            c = consumer_by_name.get(consumer_name, {})
            ewtp = c.get("eWTP") or {}
            for good in goods:
                val = ewtp.get(good, 0.0)
                if not isinstance(val, (int, float)):
                    val = 0.0
                rows.append({"timestep": t, "good": good, "value": float(val)})
        return pd.DataFrame(rows)

    def avg_ewtp_by_good_over_time(self) -> pd.DataFrame:
        """
        Long format: one row per (timestep, good). Value is average eWTP across all consumers for that good.
        Uses all goods that appear in any consumer eWTP across states.
        """
        goods = self._all_consumer_ewtp_good_names()
        if not goods:
            return pd.DataFrame(columns=["timestep", "good", "value"])
        rows = []
        for s in self.states:
            t = s["timestep"]
            consumers = s.get("consumers", [])
            for good in goods:
                vals = []
                for c in consumers:
                    ewtp = c.get("eWTP") or {}
                    v = ewtp.get(good)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                avg = float(np.mean(vals)) if vals else 0.0
                rows.append({"timestep": t, "good": good, "value": avg})
        return pd.DataFrame(rows)

    def metrics_over_time(
        self,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Wide format: one row per timestep, columns timestep + metric names.
        Supported metrics: total_cash, gini, total_profit (sum of firm profits).
        """
        metrics = metrics or ["total_cash", "gini"]
        rows = []
        for s in self.states:
            t = s["timestep"]
            cash = sum(s["ledger"]["money"].values())
            vals = sorted(s["ledger"]["money"].values())
            row = {"timestep": t}
            if "total_cash" in metrics:
                row["total_cash"] = cash
            if "gini" in metrics:
                row["gini"] = _gini(vals)
            if "total_profit" in metrics:
                row["total_profit"] = sum(
                    f.get("profit", 0) for f in s.get("firms", [])
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def metrics_over_time_long(
        self,
        value_vars: Optional[List[str]] = None,
        var_name: str = "metric",
        value_name: str = "value",
        renames: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Long format for charting: one row per (timestep, metric). Melted from
        metrics_over_time(); renames applied to the categorical column (e.g. for legend labels).
        """
        value_vars = value_vars or ["total_cash", "gini"]
        wide = self.metrics_over_time(metrics=value_vars)
        long_df = wide.melt(
            id_vars=["timestep"],
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name,
        )
        if renames:
            long_df[var_name] = long_df[var_name].replace(renames)
        return long_df

    def lemon_market_metrics_over_time(self) -> pd.DataFrame:
        """
        Long format: one row per (timestep, metric). Metrics: Listings, Bids, Passes.
        Uses state lemon_market_listings_count, lemon_market_bids_count, lemon_market_passes_count.
        Returns empty DataFrame if no state has these keys (non–lemon runs).
        """
        rows = []
        for s in self.states:
            t = s["timestep"]
            listings = s.get("lemon_market_listings_count", 0)
            bids = s.get("lemon_market_bids_count", 0)
            passes = s.get("lemon_market_passes_count", 0)
            if not isinstance(listings, (int, float)):
                listings = 0
            if not isinstance(bids, (int, float)):
                bids = 0
            if not isinstance(passes, (int, float)):
                passes = 0
            rows.append({"timestep": t, "metric": "Listings", "value": int(listings)})
            rows.append({"timestep": t, "metric": "Bids", "value": int(bids)})
            rows.append({"timestep": t, "metric": "Passes", "value": int(passes)})
        return pd.DataFrame(rows)

    def firms_in_business_over_time(self) -> pd.DataFrame:
        """
        One row per timestep: timestep, value (number of firms with in_business True).
        """
        rows = []
        for s in self.states:
            t = s["timestep"]
            count = sum(1 for f in s.get("firms", []) if f.get("in_business", False))
            rows.append({"timestep": t, "value": int(count)})
        return pd.DataFrame(rows)

    def filled_orders_count_over_time(self) -> pd.DataFrame:
        """
        One row per timestep: timestep, value (total filled consumer orders that step).
        Uses state top-level filled_orders_count; 0 if missing (e.g. legacy state files).
        """
        rows = []
        for s in self.states:
            t = s["timestep"]
            count = s.get("filled_orders_count", 0)
            if not isinstance(count, (int, float)):
                count = 0
            rows.append({"timestep": t, "value": int(count)})
        return pd.DataFrame(rows)

    def filled_orders_count_by_firm_over_time(self) -> pd.DataFrame:
        """
        Long format: one row per (timestep, firm). Value is number of consumer orders
        filled by that firm that step. Uses union of all firm names; 0 if firm had no fills.
        """
        all_firms = self._all_firm_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            by_firm = s.get("filled_orders_count_by_firm") or {}
            for name in all_firms:
                count = by_firm.get(name, 0)
                if not isinstance(count, (int, float)):
                    count = 0
                rows.append({"timestep": t, "firm": name, "value": int(count)})
        return pd.DataFrame(rows)

    def profit_per_firm_over_time(self) -> pd.DataFrame:
        """
        Long format: one row per (timestep, firm). Value is firm["profit"] for each firm.
        Uses union of all firm names across states so every firm appears for every timestep.
        """
        all_firms = self._all_firm_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            firm_by_name = {f.get("name"): f for f in s.get("firms", []) if f.get("name")}
            for name in all_firms:
                f = firm_by_name.get(name)
                rows.append({"timestep": t, "firm": name, "value": f.get("profit", 0) if f else 0})
        return pd.DataFrame(rows)

    def profit_rolling_avg_per_firm_over_time(
        self, window: int = 3
    ) -> pd.DataFrame:
        """
        Long format: one row per (timestep, firm). Value is average of profit over the last
        `window` timesteps (including current). Uses min_periods=1 so early timesteps have
        partial averages.
        """
        df = self.profit_per_firm_over_time()
        if df.empty:
            return df
        result = []
        for firm_name, grp in df.groupby("firm"):
            grp = grp.sort_values("timestep")
            rolling = grp["value"].rolling(window=window, min_periods=1).mean()
            for t, v in zip(grp["timestep"], rolling):
                result.append({"timestep": t, "firm": firm_name, "value": float(v)})
        return pd.DataFrame(result)

    def cash_per_firm_over_time(self) -> pd.DataFrame:
        """
        Long format: one row per (timestep, firm). Value is cash from ledger.money.
        Uses union of all firm names across states so every firm appears for every timestep.
        """
        all_firms = self._all_firm_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            money = s["ledger"]["money"]
            for name in all_firms:
                rows.append({"timestep": t, "firm": name, "value": money.get(name, 0)})
        return pd.DataFrame(rows)

    def reputation_per_firm_over_time(self) -> pd.DataFrame:
        """
        Long format: one row per (timestep, firm). Value is firm["reputation"] for each firm.
        Uses union of all firm names across states. Default reputation 1.0 if missing from state.
        """
        all_firms = self._all_firm_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            firm_by_name = {f.get("name"): f for f in s.get("firms", []) if f.get("name")}
            for name in all_firms:
                f = firm_by_name.get(name)
                rows.append({"timestep": t, "firm": name, "value": f.get("reputation", 1.0) if f else 1.0})
        return pd.DataFrame(rows)

    def sales_per_firm_over_time(self) -> pd.DataFrame:
        """
        Long format: one row per (timestep, firm). Value is total quantity sold (sum of
        firm["sales_by_good"] across all goods). Uses union of all firm names across states.
        """
        all_firms = self._all_firm_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            firm_by_name = {f.get("name"): f for f in s.get("firms", []) if f.get("name")}
            for name in all_firms:
                f = firm_by_name.get(name)
                if f:
                    sales_by_good = f.get("sales_by_good") or {}
                    total_sales = sum(
                        v for v in sales_by_good.values()
                        if isinstance(v, (int, float))
                    )
                else:
                    total_sales = 0
                rows.append({"timestep": t, "firm": name, "value": total_sales})
        return pd.DataFrame(rows)

    def price_per_firm_over_time(self, good: str) -> pd.DataFrame:
        """
        Long format: one row per (timestep, firm). Value is firm["prices"].get(good) for that good.
        Uses union of all firm names across states. Missing price is 0.
        """
        all_firms = self._all_firm_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            firm_by_name = {f.get("name"): f for f in s.get("firms", []) if f.get("name")}
            for name in all_firms:
                f = firm_by_name.get(name)
                if f:
                    prices = f.get("prices") or {}
                    price = prices.get(good, 0)
                    if isinstance(price, (int, float)):
                        pass
                    else:
                        price = 0
                else:
                    price = 0
                rows.append({"timestep": t, "firm": name, "value": price})
        return pd.DataFrame(rows)

    def sales_this_step_per_firm_over_time(self, good: str) -> pd.DataFrame:
        """
        Long format: one row per (timestep, firm). Value is quantity sold this step for the given
        good: firm["sales_this_step"].get(good, 0). Uses union of all firm names across states.
        """
        all_firms = self._all_firm_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            firm_by_name = {f.get("name"): f for f in s.get("firms", []) if f.get("name")}
            for name in all_firms:
                f = firm_by_name.get(name)
                if f:
                    sales_this_step = f.get("sales_this_step") or {}
                    qty = sales_this_step.get(good, 0)
                    if not isinstance(qty, (int, float)):
                        qty = 0
                else:
                    qty = 0
                rows.append({"timestep": t, "firm": name, "value": qty})
        return pd.DataFrame(rows)

    def supply_purchases_by_good_over_time(self, firm_name: str) -> pd.DataFrame:
        """
        Long format: one row per (timestep, good). Value is total_cost from
        firm["expenses_info"]["supply_by_good"] for the given firm. Uses all goods that appear
        in any timestep for this firm; missing (timestep, good) get value 0.
        """
        rows = []
        good_seen = set()
        timesteps = [s["timestep"] for s in self.states]
        for s in self.states:
            t = s["timestep"]
            firm_by_name = {f.get("name"): f for f in s.get("firms", []) if f.get("name")}
            f = firm_by_name.get(firm_name)
            if not f:
                continue
            expenses_info = f.get("expenses_info") or {}
            supply_by_good = expenses_info.get("supply_by_good") or []
            if isinstance(supply_by_good, list):
                for entry in supply_by_good:
                    if not isinstance(entry, dict):
                        continue
                    good = entry.get("good")
                    if good is None:
                        continue
                    good_seen.add(good)
                    total_cost = entry.get("total_cost", 0)
                    if not isinstance(total_cost, (int, float)):
                        total_cost = 0
                    rows.append({"timestep": t, "good": good, "value": total_cost})
        if not good_seen:
            return pd.DataFrame(columns=["timestep", "good", "value"])
        goods = sorted(good_seen)
        df = pd.DataFrame(rows)
        full_rows = []
        for t in timesteps:
            sub = df[df["timestep"] == t] if not df.empty else pd.DataFrame()
            for good in goods:
                val = 0.0
                if not sub.empty:
                    row = sub[sub["good"] == good]
                    val = float(row["value"].sum()) if not row.empty else 0.0
                full_rows.append({"timestep": t, "good": good, "value": val})
        return pd.DataFrame(full_rows)

    def inventory_per_firm_over_time(self, good: str) -> pd.DataFrame:
        """
        Long format: one row per (timestep, firm). Value is firm["inventory"].get(good, 0) for
        that good (quantity in stock, excluding cash). Uses union of all firm names across states.
        """
        all_firms = self._all_firm_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            firm_by_name = {f.get("name"): f for f in s.get("firms", []) if f.get("name")}
            for name in all_firms:
                f = firm_by_name.get(name)
                if f:
                    inv = f.get("inventory") or {}
                    qty = inv.get(good, 0)
                    if not isinstance(qty, (int, float)):
                        qty = 0
                else:
                    qty = 0
                rows.append({"timestep": t, "firm": name, "value": qty})
        return pd.DataFrame(rows)

    def cash_per_consumer_over_time(self) -> pd.DataFrame:
        """
        Long format: one row per (timestep, consumer). Value is cash from ledger.money.
        Uses union of all consumer names across states so every consumer appears for every timestep.
        """
        all_consumers = self._all_consumer_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            money = s["ledger"]["money"]
            for name in all_consumers:
                rows.append({"timestep": t, "consumer": name, "value": money.get(name, 0)})
        return pd.DataFrame(rows)

    def consumer_surplus_per_consumer_over_time(self) -> pd.DataFrame:
        """
        Long format: one row per (timestep, consumer). Value is consumer_surplus from state["consumers"].
        Uses union of all consumer names across states so every consumer appears for every timestep.
        """
        all_consumers = self._all_consumer_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            consumer_by_name = {c.get("name"): c for c in s.get("consumers", []) if c.get("name")}
            for name in all_consumers:
                c = consumer_by_name.get(name)
                rows.append({"timestep": t, "consumer": name, "value": c.get("consumer_surplus", 0) if c else 0})
        return pd.DataFrame(rows)

    def food_inventory_per_consumer_over_time(
        self, good: str = "food"
    ) -> pd.DataFrame:
        """
        Long format: one row per (timestep, consumer). Value is inventory of `good`
        from ledger.inventories. Uses union of all consumer names across states so every
        consumer appears for every timestep.
        """
        all_consumers = self._all_consumer_names()
        rows = []
        for s in self.states:
            t = s["timestep"]
            inv = s["ledger"].get("inventories", {})
            for name in all_consumers:
                agent_inv = inv.get(name, {})
                rows.append({"timestep": t, "consumer": name, "value": agent_inv.get(good, 0)})
        return pd.DataFrame(rows)

    def consumer_utility_components_over_time(
        self, consumer_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Long format: one row per (timestep, metric). Value is goods_utility, cash_utility,
        labor_disutility, or total utility. Columns: timestep, metric, value.
        If consumer_name is None: value is mean across consumers; metrics include "(avg)" suffix.
        If consumer_name is set: value is that consumer's series.
        """
        rows = []
        for s in self.states:
            t = s["timestep"]
            consumers = s.get("consumers", [])
            if not consumers:
                if consumer_name is None:
                    rows.append({"timestep": t, "metric": "Goods utility (avg)", "value": 0.0})
                    rows.append({"timestep": t, "metric": "Cash utility (avg)", "value": 0.0})
                    rows.append({"timestep": t, "metric": "Labor disutility (avg)", "value": 0.0})
                    rows.append({"timestep": t, "metric": "Total utility (avg)", "value": 0.0})
                else:
                    rows.append({"timestep": t, "metric": "Goods utility", "value": 0.0})
                    rows.append({"timestep": t, "metric": "Cash utility", "value": 0.0})
                    rows.append({"timestep": t, "metric": "Labor disutility", "value": 0.0})
                    rows.append({"timestep": t, "metric": "Total utility", "value": 0.0})
                continue
            if consumer_name is None:
                mean_goods = np.mean([c.get("goods_utility", 0.0) for c in consumers])
                mean_cash = np.mean([c.get("cash_utility", 0.0) for c in consumers])
                mean_labor = np.mean([c.get("labor_disutility", 0.0) for c in consumers])
                mean_total = np.mean([c.get("utility", 0.0) for c in consumers])
                rows.append({"timestep": t, "metric": "Goods utility (avg)", "value": mean_goods})
                rows.append({"timestep": t, "metric": "Cash utility (avg)", "value": mean_cash})
                rows.append({"timestep": t, "metric": "Labor disutility (avg)", "value": mean_labor})
                rows.append({"timestep": t, "metric": "Total utility (avg)", "value": mean_total})
            else:
                c_by_name = {c.get("name"): c for c in consumers if c.get("name")}
                c = c_by_name.get(consumer_name, {})
                rows.append({"timestep": t, "metric": "Goods utility", "value": c.get("goods_utility", 0.0)})
                rows.append({"timestep": t, "metric": "Cash utility", "value": c.get("cash_utility", 0.0)})
                rows.append({"timestep": t, "metric": "Labor disutility", "value": c.get("labor_disutility", 0.0)})
                rows.append({"timestep": t, "metric": "Total utility", "value": c.get("utility", 0.0)})
        return pd.DataFrame(rows)

    def views_per_firm_over_time(self) -> pd.DataFrame:
        """
        Long format: one row per (timestep, firm). Value is number of consumer views
        for that firm that step. Uses state["views_by_firm"]; 0 if missing (e.g. non-THE_CRASH runs).
        """
        rows = []
        for state in self.states:
            t = state["timestep"]
            for firm_id, count in state.get("views_by_firm", {}).items():
                rows.append({"timestep": t, "firm": firm_id, "value": count})
        return pd.DataFrame(rows, columns=["timestep", "firm", "value"]) if rows else pd.DataFrame(columns=["timestep", "firm", "value"])

    @staticmethod
    def value_by_agent(
        state: Dict[str, Any], 
        ledger_field: str = "money", 
        agent_label: str = "Agent", 
        value_label: str = "Value"
    ) -> pd.DataFrame:
        """
        From a single state dict, build a two-column DataFrame: <agent_label>, <value_label>
        using values from state["ledger"][ledger_field].
        
        Args:
            state: Dict, a single simulation state.
            ledger_field: str, the field in state["ledger"] to extract (e.g. "money").
            agent_label: str, name for the agent identifier column.
            value_label: str, name for the value column.
        Returns:
            pd.DataFrame with columns [agent_label, value_label].
        """
        items = list(state["ledger"][ledger_field].items())
        return pd.DataFrame(items, columns=[agent_label, value_label])
