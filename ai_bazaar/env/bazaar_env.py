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


class BazaarWorld:
    def __init__(self, args, llm_model=None):
        self.args = args
        self.logger = logging.getLogger("main")
        self.ledger = Ledger()
        self.market = Market()
        self.goods_list = ["food", "clothing", "electronics", "furniture"]
        self.goods = self.goods_list[: args.num_goods]

        # Necessity mapping
        self.necessity_weights = {
            "food": 0.6,
            "clothing": 0.2,
            "electronics": 0.1,
            "furniture": 0.1,
        }

        self.firms = []
        for i in range(args.num_firms):
            name = f"firm_{i}"
            if args.firm_type == "LLM":
                firm = FirmAgent(
                    llm=args.llm,
                    port=args.port,
                    name=name,
                    goods=self.goods,
                    initial_cash=args.firm_initial_cash,
                    ledger=self.ledger,
                    market=self.market,
                    prompt_algo=getattr(args, "prompt_algo", "io"),
                    history_len=getattr(args, "history_len", 10),
                    timeout=getattr(args, "timeout", 30),
                    args=args,
                    llm_instance=llm_model,
                )
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
                # Use necessity weights for CES params if not provided by LLM
                ces_params = {
                    good: self.necessity_weights.get(good, 0.1) for good in self.goods
                }
                # Normalize
                total_w = sum(ces_params.values())
                ces_params = {k: v / total_w for k, v in ces_params.items()}

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
                    ces_params=ces_params,  # Use default necessity weights
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

        # Marketplace platform fees (simulating Amazon/eBay)
        self.platform_fee_rate = 0.10  # 10% on revenue

        self.timestep = 0
        self.firm_prices_last_step = {}

    def step(self):
        """Execute one timestep of the bazaar with parallel agent actions"""
        import concurrent.futures

        start_ledger = self.ledger.copy()

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
        for firm in self.firms:
            if not getattr(firm, "in_business", True):
                continue
            supply_unit_price = 1.0
            if self.args.firm_type == "LLM":
                firm.purchase_supplies(supply_unit_price, self.timestep)
            else:
                quantity = firm.cash * 0.5 / supply_unit_price
                firm.purchase_supplies(quantity, supply_unit_price, self.timestep)

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

                    # Accumulate profit (margin per sale; unit_cost matches supply cost)
                    if hasattr(firm, "update_profit"):
                        firm.update_profit(
                            quantity_sold,
                            price,
                            unit_cost=unit_cost,
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
            firm.pay_overhead_costs(self.timestep)

        # 8. Platform Fees (Simulating Amazon/eBay)
        total_fees = 0.0
        for firm in self.firms:
            if not getattr(firm, "in_business", True):
                continue
            fee = firm.cash * 0.05
            self.ledger.credit(firm.name, -fee)
            total_fees += fee

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
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        filename = os.path.join(log_dir, f"state_t{self.timestep}.json")
        import json

        with open(filename, "w") as f:
            json.dump(state, f, indent=2)

    def _build_firms_state(self, money: Dict, inventories: Dict) -> List[Dict]:
        """Build firms list for state: one entry per firm in self.firms, plus any firm_* in ledger not in list. Sorted by name."""
        by_name = {}
        for f in self.firms:
            by_name[f.name] = {
                "name": f.name,
                "in_business": getattr(f, "in_business", True),
                "cash": money.get(f.name, 0.0),
                "profit": getattr(f, "profit", 0.0),
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
            by_name[c.name] = {
                "name": c.name,
                "labor": getattr(c, "l", 0),
                "cash": money.get(c.name, 0.0),
                "utility": getattr(c, "utility", 0),
                "inventory": dict(inventories.get(c.name, {})),
                "diary": getattr(c, "diary", [])[-1:] if hasattr(c, "diary") else [],
            }
        for key in money:
            if key.startswith("consumer_") and key not in by_name:
                by_name[key] = {
                    "name": key,
                    "labor": 0,
                    "cash": money.get(key, 0.0),
                    "utility": 0,
                    "inventory": dict(inventories.get(key, {})),
                    "diary": [],
                }
        return [by_name[name] for name in sorted(by_name)]

    def is_done(self):
        if self.timestep >= self.args.max_timesteps:
            return True
        if not any(getattr(firm, "in_business", True) for firm in self.firms):
            return True
        return False
