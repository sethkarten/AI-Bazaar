import logging
import os
import json
import numpy as np
import random
from typing import Dict, List, Any, Sequence
from collections import defaultdict
from ..market_core.market_core import Ledger, Market
from ..agents.firm import FirmAgent, FixedFirmAgent
from ..agents.consumer import CESConsumerAgent, FixedConsumerAgent


from ..agents.planner import TaxPlanner, FixedTaxPlanner
from ..utils.heterogeneity import create_heterogeneity

DEFAULT_SUPPLY_UNIT_COSTS = {
    "food": 1.0,
    "clothing": 1.0,
    "electronics": 1.0,
    "furniture": 1.0,
}

DEFAULT_PREFERENCES = {
    "food": 0.6,
    "clothing": 0.2,
    "electronics": 0.1,
    "furniture": 0.1,
}

class BazaarWorld:
    def __init__(self, args, llm_model=None):
        self.args = args
        self.logger = logging.getLogger("main")
        self.ledger = Ledger()
        self.market = Market()
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
        for i in range(args.num_firms):
            name = f"firm_{i}"
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
                )
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

        self._write_consumer_attributes()
        self._write_firm_attributes()
        self._write_experiment_args()

        # Marketplace platform fees (simulating Amazon/eBay)
        self.platform_fee_rate = 0.10  # 10% on revenue

        self.timestep = 0
        self.firm_prices_last_step = {}

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
        log_dir = getattr(self.args, "log_dir", "logs")
        run_name = getattr(self.args, "name", None) or "simulation"
        run_dir = os.path.join(log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "experiment_args.json")
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        self.logger.info("Wrote experiment args to %s", path)

    def step(self):
        """Execute one timestep of the bazaar with parallel agent actions"""
        import concurrent.futures

        start_ledger = self.ledger.copy()

        # Reset step-level expenses and sales (accumulated from expenses_info / sales_info lists)
        expenses_info = []
        for firm in self.firms:
            if hasattr(firm, "expenses_info"):
                firm.expenses_info = {k: 0.0 for k in getattr(firm, "EXPENSE_KEYS", ("supply_cost", "overhead_costs", "taxes_paid", "platform_fees"))}
                firm.expenses_info["supply_by_good"] = []
            if hasattr(firm, "sales_info"):
                firm.sales_info = []

        # 0. Labor Phase: Consumers (Workers) choose labor supply (Parallel)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.consumers)
        ) as executor:
            futures = []
            for consumer in self.consumers:
                if hasattr(consumer, "choose_labor"):
                    futures.append(
                        executor.submit(consumer.choose_labor, self.timestep, wage=10.0)
                    )
            concurrent.futures.wait(futures)

        # 1. Supply Phase (Sequential for now as it modifies ledger)
        supply_unit_price = 1.0
        for firm in self.firms:
            if not getattr(firm, "in_business", True):
                continue
            if self.args.firm_type == "LLM":
                firm.purchase_supplies(self.timestep)
                supply_stats = getattr(firm, "_timestep_stats", {}).get(self.timestep, {}).get("supply", {})
                by_good = supply_stats.get("by_good", {})
                for good, bg in by_good.items():
                    cost = bg.get("cost", 0.0)
                    if cost <= 0:
                        continue
                    expenses_info.append({
                        "firm_id": firm.name,
                        "expense_type": "supply",
                        "good": good,
                        "amount": cost,
                        "quantity": bg.get("quantity", 0.0),
                        "unit_price": bg.get("unit_price", 0.0),
                    })
            else:
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

        # 2. Production Phase
        for firm in self.firms:
            if not getattr(firm, "in_business", True):
                continue
            firm.produce_goods(self.timestep)

        # 3. Pricing Phase (Parallel)
        firm_prices = {}

        # Prepare market context for each firm (Information Asymmetry)
        market_contexts = {}
        for firm in self.firms:
            if not getattr(firm, "in_business", True):
                continue

            if getattr(self.args, "info_asymmetry", False):
                # Firm only sees a noisy average of competitor prices
                noisy_context = {"competitor_summary": {}}
                for good in self.goods:
                    comp_prices = [
                        self.firm_prices_last_step.get(f.name, {}).get(good, 10.0)
                        for f in self.firms
                        if f.name != firm.name and getattr(f, "in_business", True)
                    ]
                    if comp_prices:
                        avg = np.mean(comp_prices)
                        # Add 10% noise
                        noisy_avg = avg * (1.0 + np.random.uniform(-0.1, 0.1))
                        noisy_context["competitor_summary"][good] = round(noisy_avg, 2)
                market_contexts[firm.name] = noisy_context
            else:
                # Full information (as before)
                market_contexts[firm.name] = {"last_prices": self.firm_prices_last_step}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.firms)
        ) as executor:
            future_to_firm = {}
            for firm in self.firms:
                if not getattr(firm, "in_business", True):
                    continue
                if self.args.firm_type == "LLM":
                    future_to_firm[
                        executor.submit(
                            firm.set_price,
                            self.timestep,
                            market_data=market_contexts.get(firm.name, {}),
                        )
                    ] = firm
                else:
                    prices = firm.set_price(price=10.0, timestep=self.timestep)
                    firm_prices[firm.name] = prices
                    firm.post_quotes(prices)

            for future in concurrent.futures.as_completed(future_to_firm):
                firm = future_to_firm[future]
                prices = future.result()
                firm_prices[firm.name] = prices
                firm.post_quotes(prices)

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
        discovery_limit = getattr(self.args, "discovery_limit", 5)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.consumers)
        ) as executor:
            future_to_cons = {}
            for consumer in self.consumers:
                if self.args.consumer_type == "CES":
                    future_to_cons[
                        executor.submit(
                            consumer.make_orders,
                            self.timestep,
                            self.args.consumer_scenario,
                            discovery_limit=discovery_limit,
                            firm_reputations=reputations,
                        )
                    ] = consumer
                else:
                    orders = consumer.make_orders(
                        self.timestep, discovery_limit=discovery_limit
                    )
                    consumer.submit_orders(orders)

            for future in concurrent.futures.as_completed(future_to_cons):
                consumer = future_to_cons[future]
                orders = future.result()
                consumer.submit_orders(orders)

        pre_clearing_ledger = self.ledger.copy()

        # 6. Market Clearing
        filled_orders, sales_info = self.market.clear(self.ledger)
        self.logger.info(f"Filled {len(filled_orders)} orders")

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
            # Use filled price from sale when available (from market.clear), else fallback to firm_prices
            price = sale.get("price", firm_prices[firm_name][good])

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
        }

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
        for key in money:
            if key.startswith("firm_") and key not in by_name:
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
