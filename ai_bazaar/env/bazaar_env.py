import logging
import os
import json
import numpy as np
import random
from dataclasses import asdict
from typing import Dict, List, Any, Sequence
from collections import defaultdict
from ..market_core.market_core import Ledger, Market
from ..agents.firm import FirmAgent, FixedFirmAgent
from ..agents.consumer import CESConsumerAgent, FixedConsumerAgent
from ..utils.common import QUALITY_DICT, LEMON_MARKET_GOODS, FIRM_PERSONAS, firm_name_from_persona


from ..agents.planner import TaxPlanner, FixedTaxPlanner
from ..utils.heterogeneity import create_heterogeneity

DEFAULT_SUPPLY_UNIT_COSTS = {
    "food": 1.0,
    "clothing": 1.0,
    "electronics": 1.0,
    "furniture": 1.0,
    "car": 1.0,
}

DEFAULT_PREFERENCES = {
    "food": 0.6,
    "clothing": 0.2,
    "electronics": 0.1,
    "furniture": 0.1,
    "car": 1.0,
}

class BazaarWorld:
    def __init__(self, args, llm_model=None):
        self.args = args
        self.logger = logging.getLogger("main")
        self.ledger = Ledger()
        self.market = Market()
        # LEMON_MARKET: use lemon_market_goods and force num_goods = 1
        if getattr(args, "consumer_scenario", None) == "LEMON_MARKET":
            args.num_goods = 1
            self.goods_list = list(LEMON_MARKET_GOODS)
            self.goods = self.goods_list[:1]  # ["car"]
        else:
            self.goods_list = ["food", "clothing", "electronics", "furniture"]
            self.goods = self.goods_list[: args.num_goods]
        # default supply unit costs
        self.supply_unit_costs = {good: DEFAULT_SUPPLY_UNIT_COSTS.get(good) for good in self.goods}

        use_cost_pref_gen = getattr(args, "use_cost_pref_gen", False)
        if use_cost_pref_gen:
            self.supply_unit_costs_by_firm, self.consumer_preferences = create_heterogeneity(
                args, goods=self.goods
            )
        else:
            # Create default structure for costs and preferences to match create_heterogeneity outputs
            self.supply_unit_costs_by_firm = [
                {good: DEFAULT_SUPPLY_UNIT_COSTS.get(good, 1.0) for good in self.goods}
                for _ in range(args.num_firms)
            ]
            self.consumer_preferences = [
                {good: DEFAULT_PREFERENCES.get(good, 0.1) for good in self.goods}
                for _ in range(args.num_consumers)
            ]

        self.firms = []
        sybil_cluster_size = getattr(args, "sybil_cluster_size", 0)
        is_lemon = getattr(args, "consumer_scenario", None) == "LEMON_MARKET"
        for i in range(args.num_firms):
            # Assign distinct personas round-robin; disabled by --disable-firm-personas
            if getattr(args, "disable_firm_personas", False):
                firm_persona = None
            else:
                firm_persona = FIRM_PERSONAS[i % len(FIRM_PERSONAS)]
            name = firm_name_from_persona(i, firm_persona, FIRM_PERSONAS)
            is_sybil = is_lemon and sybil_cluster_size > 0 and i >= args.num_firms - sybil_cluster_size
            if args.firm_type == "LLM":
                firm_kw = {
                    "llm": args.llm,
                    "port": args.port,
                    "name": name,
                    "goods": self.goods,
                    "initial_cash": args.firm_initial_cash,
                    "ledger": self.ledger,
                    "market": self.market,
                    "prompt_algo": getattr(args, "prompt_algo", "io"),
                    "history_len": getattr(args, "history_len", 10),
                    "timeout": getattr(args, "timeout", 30),
                    "args": args,
                    "llm_instance": llm_model,
                    "sybil": is_sybil,
                    "persona": firm_persona,
                }
                firm_kw["supply_unit_costs"] = self.supply_unit_costs_by_firm[i]
                firm = FirmAgent(**firm_kw)
            else:
                firm = FixedFirmAgent(
                    name=name,
                    goods=self.goods,
                    initial_cash=args.firm_initial_cash,
                    ledger=self.ledger,
                    market=self.market,
                    unit_costs=self.supply_unit_costs_by_firm[i],
                    markup=args.firm_markup,
                )
            # Scale overhead by timestep length: daily (non-LEMON) = 1/7 of base; LEMON (weekly) = full base
            firm.overhead_scale = 1.0 / 7.0 if getattr(args, "consumer_scenario", None) != "LEMON_MARKET" else 1.0
            # First num_stabilizing_firms LLM firms are stabilizing firms
            num_stabilizing = getattr(args, "num_stabilizing_firms", 0)
            if args.firm_type == "LLM" and num_stabilizing > 0:
                firm.stabilizing_firm = i < num_stabilizing
            # LEMON_MARKET: initial reputation R_0 (paper uses 0.8)
            if is_lemon and getattr(args, "reputation_initial", None) is not None:
                firm.reputation = float(args.reputation_initial)
            self.firms.append(firm)

        self.consumers = []
        from ai_bazaar.utils import PERSONAS

        personas = [random.sample(PERSONAS, 1)[0] for _ in range(args.num_consumers)]

        for i in range(args.num_consumers):
            name = f"consumer_{i}"
            income = np.random.uniform(50, 200)
            if args.consumer_type == "CES":
                if args.use_gen_ces is False:
                    ces_params = self.consumer_preferences[i]
                    consumer = CESConsumerAgent(
                        name=name,
                        income_stream=income,
                        ledger=self.ledger,
                        market=self.market,
                        persona=personas[i],
                        goods=self.goods,
                        llm=args.llm,
                        port=args.port,
                        args=args,
                        ces_params=ces_params,
                        risk_aversion=getattr(args, "risk_aversion", None),
                        llm_instance=llm_model,
                    )
                else:
                    consumer = CESConsumerAgent(
                        name=name,
                        income_stream=income,
                        ledger=self.ledger,
                        market=self.market,
                        persona=personas[i],
                        goods=self.goods,
                        llm=args.llm,
                        port=args.port,
                        args=args,
                        ces_params=None,  # Use default necessity weights
                        risk_aversion=getattr(args, "risk_aversion", None),
                        llm_instance=llm_model,
                    )
            else:
                consumer = FixedConsumerAgent(
                    name=name,
                    income_stream=income,
                    ledger=self.ledger,
                    market=self.market,
                    goods=self.goods,
                    quantity_per_good=args.fixed_consumer_quantity_per_good,
                )
            self.consumers.append(consumer)

        # Income scale by scenario: LEMON_MARKET = 1 timestep per year (52); else 1 timestep = 1 day (1/365)
        if getattr(args, "consumer_scenario", None) == "LEMON_MARKET":
            for c in self.consumers:
                c.income_scale = 52
                if hasattr(c, "delta"):
                    c._labor_disutility_scale = 52 ** c.delta
        else:
            # Non-LEMON: scale income to 1 day (nominal income interpreted as annual)
            for c in self.consumers:
                c.income_scale = 1.0 / 365.0

        self._write_consumer_attributes()
        self._write_firm_attributes()
        self._write_experiment_args()

        # Marketplace platform fees (simulating Amazon/eBay)
        self.platform_fee_rate = 0.10  # 10% on revenue

        self.timestep = 0
        self.firm_prices_last_step = {}
        # LEMON_MARKET: list of new listings each step (filled in lemon_market_firm_phases)
        self.lemon_market_listings = []
        # LEMON_MARKET: unsold listings carried over after clear
        self.lemon_market_listings_unsold = []
    def _write_consumer_attributes(self):
        """Write unique attributes of all consumer agents to a JSON file (after full initialization)."""
        def _to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            return obj

        out = []
        for c in self.consumers:
            # skill: CES uses self.v, Fixed does not have it
            skill = getattr(c, "v", None)
            entry = {
                "name": c.name,
                "ces_params": _to_serializable(getattr(c, "ces_params", None)),
                "c": _to_serializable(getattr(c, "c", None)),
                "sigma": _to_serializable(getattr(c, "sigma", None)),
                "delta": _to_serializable(getattr(c, "delta", None)),
                "llm_model": getattr(c, "llm_model", None),
                "skill": _to_serializable(skill),
                "risk_aversion": _to_serializable(getattr(c, "risk_aversion", None)),
                "epsilon": _to_serializable(getattr(c, "epsilon", None)),
                "beta": _to_serializable(getattr(c, "beta", None)),
                "goods": getattr(c, "goods", None),
            }
            out.append(entry)

        log_dir = getattr(self.args, "log_dir", "logs")
        run_name = getattr(self.args, "name", None) or "simulation"
        run_dir = os.path.join(log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "consumer_attributes.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        self.logger.info("Wrote consumer attributes to %s", path)

    def _write_firm_attributes(self):
        """Write firm attributes (e.g. supply unit costs) to a JSON file for the viz Firms tab."""
        def _to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            return obj

        out = []
        for f in self.firms:
            entry = {
                "name": f.name,
                "goods": getattr(f, "goods", None),
                "supply_unit_costs": _to_serializable(getattr(f, "supply_unit_costs", None)),
                "persona": getattr(f, "persona", None),
            }
            out.append(entry)

        log_dir = getattr(self.args, "log_dir", "logs")
        run_name = getattr(self.args, "name", None) or "simulation"
        run_dir = os.path.join(log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "firm_attributes.json")
        with open(path, "w") as fp:
            json.dump(out, fp, indent=2)
        self.logger.info("Wrote firm attributes to %s", path)

    def _build_rerun_command(self, d: dict) -> str:
        """Build a copy-paste ready command line string to rerun the simulation."""
        STORE_TRUE_FLAGS = {
            "info_asymmetry", "use_gen_ces", "use_cost_pref_gen", "wandb",
            "log_firm_prompts", "use_parsing_agent", "no_diaries", "use_env",
            "disable_firm_personas",
        }
        parts = []
        for key, val in sorted(d.items()):
            opt = "--" + key.replace("_", "-")
            if key in STORE_TRUE_FLAGS:
                if val is True:
                    parts.append(opt)
            else:
                s = str(val)
                if " " in s or '"' in s or "\\" in s:
                    s = '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
                parts.append(f"{opt} {s}")
        return "python -m ai_bazaar.main " + " ".join(parts)

    def _write_experiment_args(self):
        """Write experiment arguments to a JSON file in the run directory for the viz General tab."""
        def _to_serializable(obj):
            if obj is None or isinstance(obj, (bool, str, int, float)):
                return obj
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

        d = _to_serializable(vars(self.args))
        d["rerun_command"] = self._build_rerun_command(d)
        log_dir = getattr(self.args, "log_dir", "logs")
        run_name = getattr(self.args, "name", None) or "simulation"
        run_dir = os.path.join(log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "experiment_args.json")
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        self.logger.info("Wrote experiment args to %s", path)

    def _consumers_participating_this_step(self):
        """Return the list of consumers who participate this timestep. If poisson_demand_lambda is set, k = min(Poisson(lam), N) consumers are chosen at random; else all participate. THE_CRASH scenario defaults lambda to 30 if not set."""
        lam = getattr(self.args, "poisson_demand_lambda", None)
        if lam is None:
            if getattr(self.args, "consumer_scenario", None) == "THE_CRASH":
                lam = 30
            else:
                return self.consumers
        k = np.random.poisson(lam)
        k = min(max(0, k), len(self.consumers))
        if k == 0:
            return []
        return random.sample(self.consumers, k)

    def step(self):
        """Execute one timestep of the bazaar with parallel agent actions"""
        import concurrent.futures

        start_ledger = self.ledger.copy()

        # Snapshot state at start of step for alignment-trace logging (state, action, outcome)
        if getattr(self.args, "log_alignment_traces", False):
            for f in self.firms:
                setattr(f, "_last_price_trace", None)
            self._step_trace_state = {
                "timestep": self.timestep,
                "prices_last": dict(self.firm_prices_last_step),
                "firms": [
                    {
                        "name": f.name,
                        "cash": float(self.ledger.agent_money.get(f.name, 0.0)),
                        "inventory": {k: float(v) for k, v in (self.ledger.agent_inventories.get(f.name, {}) or {}).items()},
                        "supply_unit_costs": {k: float(v) for k, v in (getattr(f, "supply_unit_costs", None) or {}).items()},
                        "in_business": getattr(f, "in_business", True),
                    }
                    for f in self.firms
                ],
            }

        # Reset step-level expenses and sales (accumulated from expenses_info / sales_info lists)
        expenses_info = []
        for firm in self.firms:
            if hasattr(firm, "expenses_info"):
                firm.expenses_info = {k: 0.0 for k in getattr(firm, "EXPENSE_KEYS", ("supply_cost", "overhead_costs", "taxes_paid", "platform_fees"))}
                firm.expenses_info["supply_by_good"] = []
            if hasattr(firm, "sales_info"):
                firm.sales_info = []

        # Defensive solvency guard: mark any active firm with negative cash as bankrupt
        # before any phase runs, so it never enters active_firms.
        for firm in self.firms:
            if getattr(firm, "in_business", True) and firm.cash < 0:
                firm.mark_out_of_business(
                    reason=(
                        f"{firm.name} entered timestep {self.timestep} with negative cash "
                        f"(${firm.cash:.2f}). Marking out of business."
                    )
                )

        # 0. Labor Phase: CES consumers choose labor at t=0; thereafter only if --dynamic-labor
        labor_phase_active = self.timestep == 0 or getattr(
            self.args, "dynamic_labor", False
        )
        if labor_phase_active:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.consumers)
            ) as executor:
                futures = []
                for consumer in self.consumers:
                    if hasattr(consumer, "choose_labor"):
                        futures.append(
                            executor.submit(
                                consumer.choose_labor, self.timestep, wage=10.0
                            )
                        )
                concurrent.futures.wait(futures)

        # firm phases
        #B2C
        def firm_phases(firm_prices, market_contexts):
            supply_unit_price = 1.0
            active_firms = [
                f for f in self.firms
                if getattr(f, "in_business", True)
            ]

            # Build market contexts for all active firms (Information Asymmetry + discovery_limit_firms)
            discovery_limit_firms = getattr(
                self.args, "discovery_limit_firms", 0
            )
            for firm in active_firms:
                competitor_names = [
                    f.name
                    for f in self.firms
                    if f.name != firm.name and getattr(f, "in_business", True)
                ]
                if discovery_limit_firms > 0 and len(competitor_names) > discovery_limit_firms:
                    competitor_names = random.sample(
                        competitor_names, discovery_limit_firms
                    )
                if getattr(self.args, "info_asymmetry", False):
                    # Firm only sees a noisy average of (possibly limited) competitor prices
                    noisy_context = {"competitor_summary": {}}
                    for good in self.goods:
                        comp_prices = [
                            self.firm_prices_last_step.get(n, {}).get(good, 10.0)
                            for n in competitor_names
                        ]
                        if comp_prices:
                            avg = np.mean(comp_prices)
                            # Add 10% noise
                            noisy_avg = avg * (1.0 + np.random.uniform(-0.1, 0.1))
                            noisy_context["competitor_summary"][good] = round(
                                noisy_avg, 2
                            )
                    market_contexts[firm.name] = noisy_context
                else:
                    # Full information: own last_prices + up to discovery_limit_firms competitors
                    last_prices = {
                        firm.name: self.firm_prices_last_step.get(
                            firm.name, {}
                        )
                    }
                    for n in competitor_names:
                        last_prices[n] = self.firm_prices_last_step.get(n, {})
                    market_contexts[firm.name] = {"last_prices": last_prices}

            # Single combined parallel phase for LLM firms; FixedFirmAgent runs three methods sequentially
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(active_firms))) as executor:
                future_to_firm = {}
                for firm in active_firms:
                    if self.args.firm_type == "LLM":
                        future_to_firm[
                            executor.submit(
                                firm.decide_firm_action,
                                self.timestep,
                                market_contexts.get(firm.name, {}),
                            )
                        ] = firm
                    else:
                        # FixedFirmAgent: run supply, production, and pricing sequentially (unchanged)
                        quantity = firm.cash * 0.5 / supply_unit_price
                        quantity = firm.purchase_supplies(quantity, supply_unit_price, self.timestep)
                        cost = quantity * supply_unit_price
                        expenses_info.append({
                            "firm_id": firm.name,
                            "expense_type": "supply",
                            "good": "supply",
                            "amount": cost,
                            "quantity": quantity,
                            "unit_price": supply_unit_price,
                        })
                        firm.produce_goods(self.timestep)
                        prices = firm.set_price(timestep=self.timestep)
                        firm_prices[firm.name] = prices
                        firm.post_quotes(prices)

                if self.args.firm_type == "LLM":
                    for future in concurrent.futures.as_completed(future_to_firm):
                        firm = future_to_firm[future]
                        result = future.result()
                        expenses_info.extend(result["supply_entries"])
                        firm_prices[firm.name] = result["prices"]
                        firm.post_quotes(result["prices"])
        # C2C Lemon Market: endow sequentially (RNG order), then create_listings in parallel (like Crash pricing)
        def lemon_market_firm_phases():
            active_firms = [
                f for f in self.firms
                if getattr(f, "in_business", True)
            ]
            # 1) Endow each firm with a car (sequential for reproducible RNG)
            for firm in active_firms:
                if not hasattr(firm, "listings"):
                    firm.listings = []
                quality_key = random.choice(list(QUALITY_DICT.keys()))
                quality_value = QUALITY_DICT[quality_key]
                firm.listings.append({
                    "quality": quality_key,
                    "quality_value": quality_value,
                    "posted": False,
                })

            # 2) Create listings in parallel (each firm only touches own state + LLM)
            new_listings = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(active_firms))
            ) as executor:
                future_to_firm = {
                    executor.submit(firm.create_listings, self.timestep): firm
                    for firm in active_firms
                }
                for future in concurrent.futures.as_completed(future_to_firm):
                    firm_new = future.result()
                    new_listings.extend(firm_new)

            # 3) Store the new listings in a list (on the world for downstream use)
            self.lemon_market_listings = new_listings
        
        # 3. Firm phases
        firm_prices = {}
        market_contexts = {}
        if self.args.consumer_scenario != "LEMON_MARKET":
            firm_phases(firm_prices, market_contexts)
        elif self.args.consumer_scenario == "LEMON_MARKET":
            lemon_market_firm_phases()
            # Merge unsold (from previous step) with new listings; post all
            unsold = getattr(self, "lemon_market_listings_unsold", [])
            all_listings = unsold + self.lemon_market_listings
            self.market.post_listings(all_listings)
            self.lemon_market_listings_count = len(self.market.listings)
            # Endow firms with 1 car only for NEW listings (unsold already have inventory)
            for L in self.lemon_market_listings:
                self.ledger.add_good(L["firm_id"], "car", 1.0)

        # 4. Income Phase: Receive labor income
        for consumer in self.consumers:
            consumer.receive_income(self.timestep)

        # 5. Consumption Phase (Parallel)
        # Get reputations for discovery (only firms in business, to match main.py)
        reputations = {
            f.name: f.reputation
            for f in self.firms
            if getattr(f, "in_business", True)
        }
        discovery_limit_consumers = getattr(
            self.args, "discovery_limit_consumers", 5
        )
        use_eWTP = getattr(self.args, "use_eWTP", False) or (
            getattr(self.args, "consumer_scenario", None) == "THE_CRASH"
        )
        if getattr(self.args, "consumer_scenario", None) == "LEMON_MARKET":
            self.lemon_market_bids_count = 0
            self.lemon_market_passes_count = 0

        participating = self._consumers_participating_this_step()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(participating))
        ) as executor:
            future_to_cons = {}
            for consumer in participating:
                if self.args.consumer_type == "CES":
                    future_to_cons[
                        executor.submit(
                            consumer.make_orders,
                            self.timestep,
                            self.args.consumer_scenario,
                            discovery_limit=discovery_limit_consumers,
                            firm_reputations=reputations,
                            use_eWTP=use_eWTP,
                        )
                    ] = consumer
                else:
                    orders = consumer.make_orders(
                        self.timestep,
                        discovery_limit=discovery_limit_consumers,
                    )
                    consumer.submit_orders(orders)

            for future in concurrent.futures.as_completed(future_to_cons):
                consumer = future_to_cons[future]
                orders = future.result()
                consumer.submit_orders(orders)
                if getattr(self.args, "consumer_scenario", None) == "LEMON_MARKET":
                    n_seen = getattr(consumer, "_lemon_listings_seen", 0)
                    n_bids = len(orders)
                    self.lemon_market_bids_count += n_bids
                    self.lemon_market_passes_count += max(0, n_seen - n_bids)

        pre_clearing_ledger = self.ledger.copy()

        # 6. Market Clearing
        filled_orders, sales_info = self.market.clear(self.ledger)
        self.filled_orders_count = len(filled_orders)
        filled_by_firm = defaultdict(int)
        for order in filled_orders:
            filled_by_firm[order.firm_id] += 1
        self.filled_orders_count_by_firm = dict(filled_by_firm)
        self.logger.info(f"Filled {len(filled_orders)} orders")

        # LEMON_MARKET: carry over unsold listings to next step
        if getattr(self.args, "consumer_scenario", None) == "LEMON_MARKET":
            self.lemon_market_listings_unsold = list(self.market.listings)

            # Update seller reputation: R_{t+1} = alpha * R_t + (1 - alpha) * q (q = quality of car sold)
            alpha = getattr(self.args, "reputation_alpha", 0.9)
            firms_by_name = {f.name: f for f in self.firms}
            for sale in sales_info:
                firm_id = sale.get("firm_id")
                q = sale.get("quality_value")
                if firm_id is None or q is None:
                    continue
                firm = firms_by_name.get(firm_id)
                if firm is not None and hasattr(firm, "update_reputation"):
                    firm.update_reputation(quality=float(q), alpha=alpha)

            # Sybil identity rotation: if R < rho_min, reset to R_0 (fresh identity)
            rho_min = getattr(self.args, "sybil_rho_min", 0.3)
            r0 = getattr(self.args, "reputation_initial", 0.8)
            for firm in self.firms:
                if getattr(firm, "sybil", False) and firm.reputation < rho_min:
                    firm.reputation = float(r0)
                    self.logger.info(f"Sybil identity rotation: {firm.name} R reset to {r0}")

        # update eWTP for all consumers
        # iterate through sales info and update eWTP for each consumer
        consumers_sold_to = []
        consumers_by_name = {c.name: c for c in self.consumers}
        for sale in sales_info:
            consumer_id = sale.get("consumer_id")
            if consumer_id is None:
                continue
            consumer = consumers_by_name.get(consumer_id)
            if consumer is not None and hasattr(consumer, "update_eWTP"):
                consumer.update_eWTP(sale)
                consumers_sold_to.append(consumer_id)
        # if the consumer is not in any sales info (they did not buy anything), update eWTP (r unchanged)
        for consumer in self.consumers:
            if consumer.name not in consumers_sold_to and hasattr(consumer, "update_eWTP"):
                consumer.update_eWTP()

        # CES-based consumer surplus this timestep (per consumer, for state file)
        consumers_by_name = {c.name: c for c in self.consumers}
        self.consumer_surplus_this_step = {}
        for order, sale in zip(filled_orders, sales_info):
            consumer = consumers_by_name.get(order.consumer_id)
            if consumer is not None and hasattr(consumer, "compute_willingness_to_pay"):
                wtp = consumer.compute_willingness_to_pay(self.timestep).get(order.good, 0.0)
                price = sale.get("price", 0.0)
                qty = sale.get("quantity_sold", 0.0)
                surplus = max(0.0, (wtp - price) * qty)
            else:
                surplus = 0.0
            self.consumer_surplus_this_step[order.consumer_id] = (
                self.consumer_surplus_this_step.get(order.consumer_id, 0.0) + surplus
            )

        # Reset step-level profit before accumulating (unit_cost 1.0 matches supply_unit_price)
        supply_unit_price = 1.0
        unit_cost = getattr(self.args, "unit_cost", supply_unit_price)
        for firm in self.firms:
            if hasattr(firm, "update_profit"):
                firm.profit = 0.0

        # Update sales tracking and reputations
        firm_sales_summary = defaultdict(lambda: {"sold": 0.0, "requested": 0.0})

        for sale in sales_info:
            firm_name = sale["firm_id"]
            good = sale["good"]
            quantity_sold = sale["quantity_sold"]
            requested_qty = sale.get("requested_quantity", quantity_sold)  # market_core returns requested_quantity
            # Use filled price from sale when available (from market.clear). LEMON_MARKET has no firm_prices.
            if getattr(self.args, "consumer_scenario", None) == "LEMON_MARKET":
                price = sale.get("price", 0.0)
            else:
                price = sale.get("price", firm_prices.get(firm_name, {}).get(good, 0.0))

            firm_sales_summary[firm_name]["sold"] += quantity_sold
            firm_sales_summary[firm_name]["requested"] += requested_qty

            for firm in self.firms:
                if firm.name == firm_name:
                    firm.total_quantity_sold_by_good[good] += quantity_sold
                    firm.total_quantity_sold_by_good_this_timestep[self.timestep][
                        good
                    ] += quantity_sold

                    # Per-firm sales_info (like expenses_info): one record per sale
                    if hasattr(firm, "sales_info"):
                        firm.sales_info.append({
                            "good": good,
                            "quantity_sold": quantity_sold,
                            "requested_quantity": requested_qty,
                            "price": price,
                        })

                    # Accumulate profit (margin per sale; unit_cost matches supply cost)
                    if hasattr(firm, "update_profit"):
                        firm.update_profit(
                            quantity_sold,
                            price,
                        )
                    break

        # Update reputations for all firms (even if no sales)
        # LEMON_MARKET uses R_{t+1} = alpha*R_t + (1-alpha)*q instead (done above)
        if getattr(self.args, "consumer_scenario", None) != "LEMON_MARKET":
            for firm in self.firms:
                summary = firm_sales_summary.get(firm.name, {"sold": 0.0, "requested": 0.0})
                if summary["requested"] > 0:
                    firm.update_reputation(summary["sold"], summary["requested"])

        # 7. Cleanup & Overhead
        self.market.quotes.clear()
        while self.market.orders:
            self.market.orders.popleft()

        for firm in self.firms:
            if not getattr(firm, "in_business", True):
                continue
            amount_paid = firm.pay_overhead_costs(self.timestep)
            expenses_info.append({
                "firm_id": firm.name,
                "expense_type": "overhead",
                "amount": amount_paid,
            })

        # 7b. Taxes (all firms including FixedFirmAgent)
        firm_tax_rate = getattr(self.args, "firm_tax_rate", 0.05)
        for firm in self.firms:
            if not getattr(firm, "in_business", True):
                continue
            if hasattr(firm, "pay_taxes"):
                taxes_paid = firm.pay_taxes(self.timestep, firm_tax_rate)
                expenses_info.append({
                    "firm_id": firm.name,
                    "expense_type": "taxes",
                    "amount": taxes_paid,
                })

        # 8. Platform Fees (Simulating Amazon/eBay)
        total_fees = 0.0
        for firm in self.firms:
            if not getattr(firm, "in_business", True):
                continue
            fee = firm.cash * 0.05
            self.ledger.credit(firm.name, -fee)
            total_fees += fee
            expenses_info.append({
                "firm_id": firm.name,
                "expense_type": "platform_fee",
                "amount": fee,
            })

        # Consume expenses_info (like sales_info): update each firm's step expenses and apply to profit
        for expense in expenses_info:
            firm_id = expense["firm_id"]
            amount = expense["amount"]
            for firm in self.firms:
                if firm.name == firm_id:
                    if hasattr(firm, "update_expenses"):
                        firm.update_expenses(
                            expense["expense_type"],
                            amount,
                            quantity=expense.get("quantity"),
                            unit_price=expense.get("unit_price"),
                            good=expense.get("good"),
                        )
                    if hasattr(firm, "apply_expense_to_profit"):
                        firm.apply_expense_to_profit(amount)
                    break

        # 9. Reflection
        for firm in self.firms:
            if not getattr(firm, "in_business", True):
                continue
            firm.reflect(self.timestep, start_ledger, pre_clearing_ledger, self.ledger)

        for consumer in self.consumers:
            if hasattr(consumer, "reflect"):
                consumer.reflect(self.timestep)

        # After step is complete, assign rewards to trajectories
        reward_type = getattr(self.args, "reward_type", "PROFIT")

        for firm in self.firms:
            if hasattr(firm, "trajectory"):
                for entry in firm.trajectory:
                    if entry["timestep"] == self.timestep and entry["reward"] is None:
                        if reward_type == "REVENUE":
                            entry["reward"] = firm.calculate_revenue(
                                self.timestep, pre_clearing_ledger, self.ledger
                            )
                        else:  # PROFIT
                            entry["reward"] = getattr(firm, "profit", 0.0)

        for consumer in self.consumers:
            if hasattr(consumer, "trajectory"):
                for entry in consumer.trajectory:
                    if entry["timestep"] == self.timestep and entry["reward"] is None:
                        entry["reward"] = getattr(consumer, "utility", 0.0)

        self.firm_prices_last_step = firm_prices.copy()
        stats = {
            "firms": {
                f.name: {
                    "cash": f.cash,
                    "profit": getattr(f, "profit", 0.0),
                    "reputation": f.reputation,
                    "prices": firm_prices.get(f.name, {}).copy(),
                    "inventory": dict(getattr(f, "inventory", {})),
                }
                for f in self.firms
            },
            "consumers": {
                c.name: {
                    "cash": c.cash,
                    "utility": c.utility,
                    "inventory": dict(getattr(c, "inventory", {})),
                }
                for c in self.consumers
            },
            "sales_count": len(filled_orders),
            "total_fees": total_fees,
        }

        self.save_state()

        # Log alignment trace (state, prompt, response, outcome) for SFT / Stabilization Traces
        if getattr(self.args, "log_alignment_traces", False) and hasattr(self, "_step_trace_state"):
            step_trace = {
                "state": self._step_trace_state,
                "firms": [
                    getattr(f, "_last_price_trace", None)
                    for f in self.firms
                    if hasattr(f, "_last_price_trace") and getattr(f, "_last_price_trace", None) is not None
                ],
                "outcome": {
                    "timestep": self.timestep,
                    "firms": [
                        {
                            "name": f.name,
                            "prices": self.firm_prices_last_step.get(f.name, {}),
                            "profit": float(getattr(f, "profit", 0.0)),
                            "in_business": getattr(f, "in_business", True),
                            "cash": float(self.ledger.agent_money.get(f.name, 0.0)),
                        }
                        for f in self.firms
                    ],
                },
            }
            log_dir = getattr(self.args, "log_dir", "logs")
            run_name = getattr(self.args, "name", None) or "simulation"
            run_dir = os.path.join(log_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            trace_path = os.path.join(run_dir, "alignment_traces.jsonl")
            with open(trace_path, "a", encoding="utf-8") as tf:
                tf.write(json.dumps(step_trace, default=str) + "\n")

        # Consumption phase: zero consumer goods (keep cash) after every consumption_interval
        consumption_interval = getattr(self.args, "consumption_interval", 1)
        if (self.timestep + 1) % consumption_interval == 0:
            for consumer in self.consumers:
                if hasattr(consumer, "consume_inventory"):
                    consumer.consume_inventory()

        self.timestep += 1
        return stats

    def save_state(self):
        """Serialize the entire world state to a JSON file.
        Firms and consumers lists are built from both self.firms/self.consumers and
        the ledger, so every agent with ledger state is included (fixes missing agents in charts).
        """
        money = self.ledger.agent_money
        inventories = self.ledger.agent_inventories
        state = {
            "timestep": self.timestep,
            "ledger": {
                "money": money.copy(),
                "inventories": {k: v.copy() for k, v in inventories.items()},
            },
            "firms": self._build_firms_state(money, inventories),
            "consumers": self._build_consumers_state(money, inventories),
            "total_fees": getattr(self, "total_fees", 0.0),
            "filled_orders_count": getattr(self, "filled_orders_count", 0),
            "filled_orders_count_by_firm": getattr(self, "filled_orders_count_by_firm", {}),
            "lemon_market_listings_count": getattr(self, "lemon_market_listings_count", 0),
            "lemon_market_bids_count": getattr(self, "lemon_market_bids_count", 0),
            "lemon_market_passes_count": getattr(self, "lemon_market_passes_count", 0),
        }
        if getattr(self.args, "consumer_scenario", None) == "LEMON_MARKET":
            # New listings posted this step (with timestep for dashboard "all listings" table)
            new_listings = []
            for d in getattr(self, "lemon_market_listings", []):
                row = {k: v for k, v in d.items() if k != "posted"}
                row["timestep_posted"] = self.timestep
                new_listings.append(row)
            state["lemon_market_new_listings"] = new_listings
            # Unsold listings at end of step (Listing dataclass -> dict)
            state["lemon_market_unsold_listings"] = [
                asdict(L) for L in getattr(self, "lemon_market_listings_unsold", [])
            ]

        log_dir = getattr(self.args, "log_dir", "logs")
        run_name = getattr(self.args, "name", None) or "simulation"
        run_dir = os.path.join(log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        filename = os.path.join(run_dir, f"state_t{self.timestep}.json")
        import json

        with open(filename, "w") as f:
            json.dump(state, f, indent=2)

    def _build_firms_state(self, money: Dict, inventories: Dict) -> List[Dict]:
        """Build firms list for state: one entry per firm in self.firms, plus any firm_* in ledger not in list. Sorted by name."""
        by_name = {}
        for f in self.firms:
            exp_info = dict(getattr(f, "expenses_info", {}))
            # Ensure supply_by_good is populated: use firm's value, or fallback to _timestep_stats (LLM) / supply_cost (Fixed)
            supply_by_good = exp_info.get("supply_by_good")
            if not supply_by_good or (isinstance(supply_by_good, list) and len(supply_by_good) == 0):
                supply_by_good = []
                supply_stats = getattr(f, "_timestep_stats", {}).get(self.timestep, {}).get("supply", {})
                by_good = supply_stats.get("by_good", {})
                if by_good:
                    for good, bg in by_good.items():
                        cost = bg.get("cost", 0.0)
                        if cost > 0:
                            supply_by_good.append({
                                "good": good,
                                "quantity": bg.get("quantity", 0.0),
                                "unit_cost": bg.get("unit_price", 0.0),
                                "total_cost": cost,
                            })
                else:
                    # Fixed firms: one aggregate supply entry
                    supply_cost = exp_info.get("supply_cost", 0.0)
                    if supply_cost > 0:
                        supply_by_good.append({
                            "good": "supply",
                            "quantity": 0.0,
                            "unit_cost": 0.0,
                            "total_cost": supply_cost,
                        })
                exp_info["supply_by_good"] = supply_by_good
            by_name[f.name] = {
                "name": f.name,
                "in_business": getattr(f, "in_business", True),
                "persona": getattr(f, "persona", None),
                "cash": money.get(f.name, 0.0),
                "profit": getattr(f, "profit", 0.0),
                "reputation": getattr(f, "reputation", 1.0),
                "expenses_info": exp_info,
                "sales_info": list(getattr(f, "sales_info", [])),
                "prices": self.firm_prices_last_step.get(f.name, {}).copy(),
                "inventory": dict(inventories.get(f.name, {})),
                "sales_by_good": dict(getattr(f, "total_quantity_sold_by_good", {})),
                "sales_this_step": dict(
                    getattr(f, "total_quantity_sold_by_good_this_timestep", {})
                    .get(self.timestep, {})
                ),
                "diary": getattr(f, "diary", [])[-1:] if hasattr(f, "diary") else [],
            }
        def is_firm_key(key: str) -> bool:
            if key.startswith("firm_"):
                return True
            if key in FIRM_PERSONAS:
                return True
            return any(key.startswith(p + "_") for p in FIRM_PERSONAS)

        for key in money:
            if is_firm_key(key) and key not in by_name:
                by_name[key] = {
                    "name": key,
                    "in_business": False,
                    "cash": money.get(key, 0.0),
                    "profit": 0.0,
                    "reputation": 1.0,
                    "expenses_info": {"supply_cost": 0.0, "overhead_costs": 0.0, "taxes_paid": 0.0, "platform_fees": 0.0, "supply_by_good": []},
                    "sales_info": [],
                    "prices": {},
                    "inventory": dict(inventories.get(key, {})),
                    "sales_by_good": {},
                    "sales_this_step": {},
                    "diary": [],
                }
        return [by_name[name] for name in sorted(by_name)]

    def _build_consumers_state(self, money: Dict, inventories: Dict) -> List[Dict]:
        """Build consumers list for state: one entry per consumer in self.consumers, plus any consumer_* in ledger not in list. Sorted by name."""
        by_name = {}
        for c in self.consumers:
            labor_disutility = c.compute_labor_disutility() if hasattr(c, "compute_labor_disutility") else 0.0
            goods_utility = c.compute_goods_utility() if hasattr(c, "compute_goods_utility") else 0.0
            cash_utility = c.compute_cash_utility() if hasattr(c, "compute_cash_utility") else 0.0
            wtp = c.compute_willingness_to_pay(self.timestep) if hasattr(c, "compute_willingness_to_pay") else {}
            ewtp = dict(getattr(c, "eWTP", {})) if hasattr(c, "eWTP") else {}
            by_name[c.name] = {
                "name": c.name,
                "labor": getattr(c, "l", 0),
                "income": getattr(c, "income", 0.0),
                "cash": money.get(c.name, 0.0),
                "utility": getattr(c, "utility", 0),
                "goods_utility": goods_utility,
                "cash_utility": cash_utility,
                "labor_disutility": labor_disutility,
                "inventory": dict(inventories.get(c.name, {})),
                "willingness_to_pay": dict(wtp),
                "eWTP": ewtp,
                "consumer_surplus": getattr(self, "consumer_surplus_this_step", {}).get(c.name, 0.0),
                "diary": getattr(c, "diary", [])[-1:] if hasattr(c, "diary") else [],
            }
        for key in money:
            if key.startswith("consumer_") and key not in by_name:
                by_name[key] = {
                    "name": key,
                    "labor": 0,
                    "income": 0.0,
                    "cash": money.get(key, 0.0),
                    "utility": 0,
                    "goods_utility": 0.0,
                    "cash_utility": 0.0,
                    "labor_disutility": 0.0,
                    "inventory": dict(inventories.get(key, {})),
                    "willingness_to_pay": {},
                    "eWTP": {},
                    "consumer_surplus": getattr(self, "consumer_surplus_this_step", {}).get(key, 0.0),
                    "diary": [],
                }
        return [by_name[name] for name in sorted(by_name)]

    def is_done(self):
        if self.timestep >= self.args.max_timesteps:
            return True
        if not any(getattr(firm, "in_business", True) for firm in self.firms):
            return True
        return False
