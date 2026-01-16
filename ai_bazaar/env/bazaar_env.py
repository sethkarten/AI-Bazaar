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
        market_context = {"last_prices": self.firm_prices_last_step}
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
                            firm.set_price, self.timestep, market_data=market_context
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
        # Get reputations for discovery
        reputations = {f.name: f.reputation for f in self.firms}
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

        # Update sales tracking and reputations
        firm_sales_summary = defaultdict(lambda: {"sold": 0.0, "requested": 0.0})

        for sale in sales_info:
            firm_name = sale["firm_id"]
            good = sale["good"]
            quantity_sold = sale["quantity_sold"]
            requested_qty = sale.get("requested_qty", quantity_sold)  # Fallback

            firm_sales_summary[firm_name]["sold"] += quantity_sold
            firm_sales_summary[firm_name]["requested"] += requested_qty

            price = firm_prices[firm_name][good]
            for firm in self.firms:
                if firm.name == firm_name:
                    firm.total_quantity_sold_by_good[good] += quantity_sold
                    firm.total_quantity_sold_by_good_this_timestep[self.timestep][
                        good
                    ] += quantity_sold

                    # Update profit if it's a FirmAgent
                    if hasattr(firm, "update_profit"):
                        firm.update_profit(
                            quantity_sold,
                            price,
                            unit_cost=getattr(self.args, "unit_cost", 2.0),
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
                }
                for f in self.firms
            },
            "consumers": {
                c.name: {"cash": c.cash, "utility": c.utility} for c in self.consumers
            },
            "sales_count": len(filled_orders),
            "total_fees": total_fees,
        }

        self.save_state()
        self.timestep += 1
        return stats

        self.timestep += 1
        return stats

    def save_state(self):
        """Serialize the entire world state to a JSON file."""
        state = {
            "timestep": self.timestep,
            "ledger": {
                "money": self.ledger.agent_money.copy(),
                "inventories": self.ledger.agent_inventories.copy(),
            },
            "firms": [
                {
                    "name": f.name,
                    "in_business": getattr(f, "in_business", True),
                    "profit": getattr(f, "profit", 0.0),
                    "diary": getattr(f, "diary", [])[-1:]
                    if hasattr(f, "diary")
                    else [],
                }
                for f in self.firms
            ],
            "consumers": [
                {
                    "name": c.name,
                    "labor": getattr(c, "l", 0),
                    "utility": getattr(c, "utility", 0),
                    "diary": getattr(c, "diary", [])[-1:]
                    if hasattr(c, "diary")
                    else [],
                }
                for c in self.consumers
            ],
            "total_fees": getattr(self, "total_fees", 0.0),
        }

        log_dir = getattr(self.args, "log_dir", "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        filename = os.path.join(log_dir, f"state_t{self.timestep}.json")
        import json

        with open(filename, "w") as f:
            json.dump(state, f, indent=2)

    def is_done(self):
        if self.timestep >= self.args.max_timesteps:
            return True
        if not any(getattr(firm, "in_business", True) for firm in self.firms):
            return True
        return False
