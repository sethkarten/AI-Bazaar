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
