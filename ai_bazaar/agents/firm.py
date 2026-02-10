# FIRM
from ai_bazaar.market_core.market_core import Ledger, Market, Quote
from typing import List, Dict, Any, Optional
from .llm_agent import LLMAgent
import logging
import numpy as np
from ai_bazaar.utils.common import Message
from collections import defaultdict


class BaseFirmAgent:
    """Base class for all firms with shared functionality.

    Note: Decision methods (set_price, purchase_supplies, produce_goods) are not
    defined here because they have different signatures for Fixed vs LLM agents.
    Fixed agents take explicit parameters, while LLM agents decide autonomously.
    """

    EXPENSE_KEYS = ("supply_cost", "overhead_costs", "taxes_paid", "platform_fees")

    def __init__(self):
        self.in_business = True
        self.reputation = 1.0  # Start with perfect reputation
        self.fulfillment_history = []  # List of (successful_qty, requested_qty)
        self.profit = 0.0  # Step-level profit (reset each step, accumulated in update_profit)
        self.expenses_info = {k: 0.0 for k in BaseFirmAgent.EXPENSE_KEYS}
        self.sales_info = []  # Step-level list of sale records (reset each step, appended in env sales loop)


    # update repuation [0.0, 1.0] based on number of orders fully fulfilled
    def update_reputation(self, successful_qty: float, requested_qty: float):
        """Update historical fulfillment rate."""
        self.fulfillment_history.append((successful_qty, requested_qty))
        # Rolling average of last 10 transactions
        recent = self.fulfillment_history[-10:]
        if not recent:
            self.reputation = 1.0
        else:
            success = sum(r[0] for r in recent)
            total = sum(r[1] for r in recent)
            self.reputation = success / total if total > 0 else 1.0

    def mark_out_of_business(self, reason: Optional[str] = None) -> None:
        """Flag the firm as no longer operating and emit a warning."""
        if not self.in_business:
            return
        self.in_business = False
        logger = getattr(self, "logger", logging.getLogger("main"))
        firm_name = getattr(self, "name", "Firm")
        logger.warning(
            reason or f"{firm_name} is out of business due to insufficient funds."
        )

    def update_profit(
        self, quantity_sold: float, price: float
    ) -> float:
        """Accumulate sales margin for the current timestep. Caller must reset self.profit at step start.
        Expenses (supply, overhead, taxes, platform_fee) are then subtracted via apply_expense_to_profit
        for each entry in expenses_info, so final profit = margins - expenses (economic profit)."""
        margin = price * quantity_sold
        self.profit = getattr(self, "profit", 0.0) + margin
        return self.profit

    def update_expenses( 
        self,
        expense_type: str,
        amount: float,
        quantity: Optional[float] = None,
        unit_price: Optional[float] = None,
    ) -> None:
        """Accumulate one expense into step-level expenses_info (called from env loop over expenses_info list).
        expense_type: one of 'supply', 'overhead', 'taxes', 'platform_fee'."""
        key = {
            "supply": "supply_cost",
            "overhead": "overhead_costs",
            "taxes": "taxes_paid",
            "platform_fee": "platform_fees",
        }.get(expense_type)
        if key is None:
            return
        info = getattr(self, "expenses_info", None)
        if info is None:
            self.expenses_info = {k: 0.0 for k in self.EXPENSE_KEYS}
            info = self.expenses_info
        info[key] = info.get(key, 0.0) + amount

    def apply_expense_to_profit(self, amount: float) -> None:
        """Subtract an expense from step-level profit (called for each entry in expenses_info).
        After all sales margins are accumulated, each expense reduces profit so profit = margins - expenses."""
        self.profit = getattr(self, "profit", 0.0) - amount

    @property
    def cash(self) -> float:
        """Get current cash from ledger"""
        return self.ledger.agent_money[self.name]

    @property
    def supplies(self) -> float:
        """Get current supply amount from ledger inventory"""
        return self.inventory["supply"]

    def post_quotes(self, prices: Dict[str, float]) -> List[Quote]:
        """Shared implementation for posting quotes to market"""
        quotes = []
        for good, price in prices.items():
            if good in self.inventory and self.inventory[good] > 0:
                quote = Quote(
                    firm_id=self.name,
                    good=good,
                    price=price,
                    quantity_available=self.inventory[good],
                )
                quotes.append(quote)
                self.market.post_quote(quote)
        return quotes

    def pay_taxes(self, timestep: int, tax_rate: float) -> float:
        """Pay taxes to the government"""
        current_cash = self.cash
        taxes_due = max(0.0, current_cash * tax_rate)
        taxes_paid = taxes_due
        if current_cash < taxes_due:
            taxes_paid = current_cash
            shortfall = taxes_due - current_cash
            self.mark_out_of_business(
                reason=(
                    f"{getattr(self, 'name', 'Firm')} cannot pay taxes of ${taxes_due:.2f} "
                    f"with only ${current_cash:.2f} available (shortfall ${shortfall:.2f}). "
                    "Marking out of business."
                )
            )
        if taxes_paid > 0.0:
            self.ledger.credit(self.name, -taxes_paid)

        if hasattr(self, "_timestep_stats"):
            stats = self._timestep_stats[timestep]
            stats["taxes_paid"] = taxes_paid
            if hasattr(self, "_get_expense_record"):
                expenses = self._get_expense_record(timestep)
                expenses["taxes_paid"] = taxes_paid
        return taxes_paid

    def get_overhead_costs(self, timestep: int) -> float:
        """Get overhead costs"""
        #! Some arbitrary rent cost for now
        return 50.0  #! TODO: Replace with actual overhead costs

    def pay_overhead_costs(self, timestep: int) -> float:
        """Pay overhead costs"""
        overhead_costs = self.get_overhead_costs(timestep)
        current_cash = self.cash
        amount_paid = overhead_costs
        if current_cash < overhead_costs:
            amount_paid = current_cash
            shortfall = overhead_costs - current_cash
            self.mark_out_of_business(
                reason=(
                    f"{getattr(self, 'name', 'Firm')} cannot cover overhead of ${overhead_costs:.2f} "
                    f"with only ${current_cash:.2f} available (shortfall ${shortfall:.2f}). "
                    "Marking out of business."
                )
            )
        if amount_paid > 0.0:
            self.ledger.credit(self.name, -amount_paid)

        if hasattr(self, "_timestep_stats"):
            stats = self._timestep_stats[timestep]
            stats["overhead_costs"] = amount_paid
            if hasattr(self, "_get_expense_record"):
                expenses = self._get_expense_record(timestep)
                expenses["overhead_costs"] = amount_paid
        return amount_paid

    def reflect(self, *args, **kwargs) -> None:
        """Base reflection - override in child classes if needed"""
        pass


class FirmAgent(LLMAgent, BaseFirmAgent):
    def __init__(
        self,
        llm: str,
        port: int,
        name: str,
        prompt_algo: str = "io",
        history_len: int = 10,
        timeout: int = 10,
        goods: List[str] = None,
        initial_cash: float = 0.0,
        ledger: Ledger = None,
        market: Market = None,
        args=None,
        llm_instance=None,
    ) -> None:
        BaseFirmAgent.__init__(self)
        super().__init__(
            llm,
            port,
            name,
            prompt_algo,
            history_len,
            timeout,
            args=args,
            llm_instance=llm_instance,
        )
        self.logger = logging.getLogger("main")
        self.name = name
        self.goods = goods
        self.ledger = ledger
        self.market = market

        # Initialize ledger with cash
        self.ledger.credit(self.name, initial_cash)

        # Initialize inventory in ledger
        self.ledger.add_good(self.name, "supply", 0.0)
        for good in goods:
            self.ledger.add_good(self.name, good, 0.0)

        # Reference the ledger's inventory directly - no separate copy
        self.inventory = self.ledger.agent_inventories[self.name]

        # Running total of quantity sold by good (cumulative across all timesteps)
        self.total_quantity_sold_by_good: Dict[str, float] = {
            good: 0.0 for good in goods
        }
        # Nested dict: timestep -> good -> quantity
        self.total_quantity_sold_by_good_this_timestep: Dict[int, Dict[str, float]] = (
            defaultdict(lambda: {good: 0.0 for good in goods})
        )

        # Set system prompt for the firm
        self.system_prompt = self._create_system_prompt()

        self._timestep_stats: dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "prices": {},
                "production": {},
                "supply": {"quantity": 0.0, "unit_price": 0.0, "cost": 0.0},
                "sales": {
                    "sold_by_good": {},
                    "revenue_by_good": {},
                    "total_revenue": 0.0,
                },
                "expenses": {
                    "supply_cost": 0.0,
                    "overhead_costs": 0.0,
                    "taxes_paid": None,
                    "total_expenses": 0.0,
                },
                "profit": 0.0,
                "cash": {"start": 0.0, "pre": 0.0, "end": 0.0},
            }
        )

    def _create_system_prompt(self) -> str:
        """Create system prompt for the LLM firm agent"""
        goods_list = ", ".join(self.goods)
        return f"""You are a firm manager named {self.name} that produces and sells goods in a market economy.
You produce the following goods: {goods_list}.

Your goal is to maximize profit by making strategic decisions about:
1. Pricing: Set competitive prices for your goods to maximize revenue
2. Supply purchasing: Buy raw supplies to produce goods
3. Production: Convert supplies into finished goods efficiently

You must balance inventory management, cash flow, and market demand to succeed.
You make decisions based on historical data about your performance, market conditions, and available resources.

Key metrics to consider:
- Cash: Your available money to purchase supplies
- Supply: Raw materials available for production
- Inventory: Finished goods available to sell
- Prices: Current market prices affect demand
- Profit: Revenue from sales minus costs of supplies

CRITICAL: Always respond with a single, valid JSON object. Do not use markdown code blocks or include explanatory text. Output only the JSON object that can be parsed directly."""

    def set_price(
        self, timestep: int = None, market_data: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """LLM decides prices for each good"""
        self.logger.info(f"[ACTION] {self.name} performing action: set_price")
        self.add_message(timestep, Message.UPDATE_PRICE, market_data=market_data)

        # Create keys for each good's price
        price_keys = [f"price_{good}" for good in self.goods]

        # Call LLM to decide prices
        prices = self.act_llm(timestep, price_keys, self.parse_prices)

        # Convert to dictionary
        price_dict = {good: price for good, price in zip(self.goods, prices)}

        self._timestep_stats[timestep]["prices"] = price_dict.copy()

        self.add_message(timestep, Message.ACTION_PRICE, prices=price_dict)
        return price_dict

    def purchase_supplies(self, unit_price: float, timestep: int) -> float:
        """LLM decides how much supply to purchase"""
        self.logger.info(f"[ACTION] {self.name} performing action: purchase_supplies")
        self.add_message(timestep, Message.UPDATE_SUPPLY, unit_price=unit_price)

        # Call LLM to decide quantity
        quantity_to_purchase = self.act_llm(
            timestep, ["supply_quantity"], self.parse_supply_purchase
        )[0]

        # Execute the purchase (as much as cash allows)
        cost = quantity_to_purchase * unit_price
        total_cost = min(cost, self.cash)
        total_quantity = total_cost / unit_price

        # Deduct cost and add supply to ledger
        self.ledger.credit(self.name, -total_cost)
        self.ledger.add_good(self.name, "supply", total_quantity)

        self._timestep_stats[timestep]["supply"] = {
            "quantity": total_quantity,
            "unit_price": unit_price,
            "cost": total_cost,
        }

        self.add_message(
            timestep, Message.ACTION_SUPPLY, quantity=total_quantity, cost=total_cost
        )
        return total_quantity

    def produce_goods(self, timestep: int):
        """LLM decides how much to produce of each good"""
        self.logger.info(f"[ACTION] {self.name} performing action: produce_goods")
        self.add_message(timestep, Message.UPDATE_PRODUCTION)

        # Create keys for each good's production amount
        production_keys = [f"produce_{good}" for good in self.goods]

        # Call LLM to decide production amounts (as percentages of available supply)
        production_percentages = self.act_llm(
            timestep, production_keys, self.parse_production
        )

        supply_available = self.supplies
        if supply_available <= 0:
            return

        # Normalize percentages to sum to 100%
        total_pct = sum(production_percentages)
        if total_pct > 0:
            production_percentages = [p / total_pct for p in production_percentages]
        else:
            # Default to even distribution
            #!TODO: Reprompt the LLM to produce a valid distribution
            production_percentages = [1.0 / len(self.goods)] * len(self.goods)

        # Produce goods according to LLM's allocation
        production_dict = {}
        #! Zip may result in error if LLM does not produce a valid distribution or doesn't match 1:1 with goods list
        #! TODO: Could add token matching to map goods given by LLM to inventory goods if this is a substaintial issue
        #! Make parser smart enough to handle this and detect when a good is omitted (production % = 0)
        for good, pct in zip(self.goods, production_percentages):
            quantity = supply_available * pct
            self.ledger.add_good(self.name, good, quantity)
            production_dict[good] = quantity

        self._timestep_stats[timestep]["production"] = production_dict.copy()

        # Consume all supplies used in production
        self.ledger.add_good(self.name, "supply", -supply_available)

        self.add_message(
            timestep, Message.ACTION_PRODUCTION, production=production_dict
        )

    # Parse functions
    def parse_prices(self, items: List[str]) -> tuple:
        """Parse and validate price decisions"""
        output = []
        for item in items:
            if isinstance(item, str):
                item = item.replace("$", "").replace(",", "").replace("\n", "")
            price = float(item)
            output.append(price)
        return tuple(output)

    def parse_supply_purchase(self, items: List[str]) -> tuple:
        """Parse and validate supply purchase decision"""
        output = []
        for item in items:
            if isinstance(item, str):
                item = (
                    item.replace("$", "")
                    .replace(",", "")
                    .replace(" units", "")
                    .replace("\n", "")
                )
            quantity = float(item)
            # Clip to non-negative values
            quantity = max(0.0, quantity)
            output.append(quantity)
        return tuple(output)

    def parse_production(self, items: List[str]) -> tuple:
        """Parse and validate production allocation (as percentages)"""
        output = []
        for item in items:
            if isinstance(item, str):
                item = item.replace("%", "").replace(",", "").replace("\n", "")
            pct = float(item)
            # Clip to 0-100%
            pct = np.clip(pct, 0.0, 100.0)
            output.append(pct)
        return tuple(output)

    def _get_sales_record(self, timestep: int) -> Dict[str, Any]:
        stats = self._timestep_stats[timestep]
        sales = stats.get("sales")
        if sales is None:
            sales = {"sold_by_good": {}, "revenue_by_good": {}, "total_revenue": 0.0}
            stats["sales"] = sales
        return sales

    def _get_expense_record(self, timestep: int) -> Dict[str, Any]:
        stats = self._timestep_stats[timestep]
        expenses = stats.get("expenses")
        if expenses is None:
            expenses = {
                "supply_cost": 0.0,
                "overhead_costs": 0.0,
                "taxes_paid": None,
                "total_expenses": 0.0,
            }
            stats["expenses"] = expenses
        return expenses

    def calculate_revenue(
        self,
        timestep: int,
        pre_clearing_ledger: Ledger,
        current_ledger: Ledger,
    ) -> float:
        """Calculate and store revenue data for the timestep."""
        stats = self._timestep_stats[timestep]
        sales = self._get_sales_record(timestep)

        prices = stats.get("prices", {})
        pre_inventory = pre_clearing_ledger.agent_inventories.get(self.name, {})
        current_inventory = current_ledger.agent_inventories.get(self.name, {})

        sold_by_good: Dict[str, float] = {}
        revenue_by_good: Dict[str, float] = {}
        total_revenue = 0.0

        for good in self.goods:
            pre_qty = pre_inventory.get(good, 0.0)
            post_qty = current_inventory.get(good, 0.0)
            sold_qty = max(0.0, pre_qty - post_qty)
            sold_by_good[good] = sold_qty

            price = prices.get(good)
            revenue = sold_qty * price if price is not None else 0.0
            revenue_by_good[good] = revenue
            total_revenue += revenue

        sales["sold_by_good"] = sold_by_good
        sales["revenue_by_good"] = revenue_by_good
        sales["total_revenue"] = total_revenue

        return total_revenue

    def calculate_expenses(
        self,
        timestep: int,
        start_ledger: Ledger,
        pre_clearing_ledger: Ledger,
        current_ledger: Ledger,
        total_revenue: Optional[float] = None,
    ) -> float:
        """Calculate and store expense data for the timestep."""
        stats = self._timestep_stats[timestep]
        expenses = self._get_expense_record(timestep)
        supply_stats = stats.get(
            "supply", {"quantity": 0.0, "unit_price": 0.0, "cost": 0.0}
        )

        cash_stats = stats.get("cash")
        if cash_stats is None:
            cash_stats = {"start": 0.0, "pre": 0.0, "end": 0.0}
            stats["cash"] = cash_stats

        cash_start = start_ledger.agent_money.get(self.name, 0.0)
        cash_pre = pre_clearing_ledger.agent_money.get(self.name, 0.0)
        cash_end = current_ledger.agent_money.get(self.name, 0.0)

        cash_stats["start"] = cash_start
        cash_stats["pre"] = cash_pre
        cash_stats["end"] = cash_end

        supply_cost = supply_stats.get("cost", 0.0)
        if supply_cost <= 0.0:
            supply_cost = max(0.0, cash_start - cash_pre)
            supply_stats["cost"] = supply_cost
            quantity = supply_stats.get("quantity", 0.0)
            if quantity > 0.0:
                supply_stats["unit_price"] = supply_cost / quantity if quantity else 0.0
        expenses["supply_cost"] = supply_cost

        overhead_costs = expenses.get("overhead_costs", 0.0)
        if overhead_costs <= 0.0:
            overhead_costs = self._timestep_stats[timestep].get("overhead_costs", 0.0)
            expenses["overhead_costs"] = overhead_costs

        taxes_paid = expenses.get("taxes_paid")
        if taxes_paid is None:
            taxes_paid = self._timestep_stats[timestep].get("taxes_paid")
        if taxes_paid is None:
            if total_revenue is None:
                total_revenue = self.calculate_revenue(
                    timestep, pre_clearing_ledger, current_ledger
                )
            taxes_paid = max(0.0, (cash_pre + total_revenue) - cash_end)
            self._timestep_stats[timestep]["taxes_paid"] = taxes_paid
        expenses["taxes_paid"] = taxes_paid

        total_expenses = supply_cost + overhead_costs + (taxes_paid or 0.0)
        expenses["total_expenses"] = total_expenses
        return total_expenses

    def calculate_profit(
        self,
        timestep: int,
        start_ledger: Ledger,
        pre_clearing_ledger: Ledger,
        current_ledger: Ledger,
        total_revenue: Optional[float] = None,
        total_expenses: Optional[float] = None,
    ) -> float:
        """Calculate and store profit data for the timestep."""
        if total_revenue is None:
            total_revenue = self.calculate_revenue(
                timestep, pre_clearing_ledger, current_ledger
            )
        if total_expenses is None:
            total_expenses = self.calculate_expenses(
                timestep,
                start_ledger,
                pre_clearing_ledger,
                current_ledger,
                total_revenue=total_revenue,
            )
        profit = total_revenue - total_expenses
        self._timestep_stats[timestep]["profit"] = profit
        return profit

    def reflect(
        self,
        timestep: int,
        start_ledger: Ledger,
        pre_clearing_ledger: Ledger,
        current_ledger: Ledger,
    ) -> None:
        reflection_msg = self.build_reflection(
            timestep, start_ledger, pre_clearing_ledger, current_ledger
        )
        self.add_message(timestep, Message.REFLECTION, reflection_msg=reflection_msg)

        # Strategic Diary Entry
        if not getattr(self.args, "no_diaries", False):
            diary_prompt = (
                f"Based on this step's performance:\n{reflection_msg}\n"
                "Write a 1-2 sentence diary entry reflecting on your pricing strategy and production efficiency. "
                "How do you feel about your competitors and current cash reserves?"
            )
            diary_entry, _ = self.llm.send_msg(
                "You are a firm manager writing in your private diary.", diary_prompt
            )
            self.write_diary_entry(timestep, diary_entry)

    def build_reflection(
        self,
        timestep: int,
        start_ledger: Ledger,
        pre_clearing_ledger: Ledger,
        current_ledger: Ledger,
    ) -> str:
        stats = self._timestep_stats.get(timestep, {})

        total_revenue = self.calculate_revenue(
            timestep,
            pre_clearing_ledger,
            current_ledger,
        )
        total_expenses = self.calculate_expenses(
            timestep,
            start_ledger,
            pre_clearing_ledger,
            current_ledger,
            total_revenue=total_revenue,
        )
        profit = self.calculate_profit(
            timestep,
            start_ledger,
            pre_clearing_ledger,
            current_ledger,
            total_revenue=total_revenue,
            total_expenses=total_expenses,
        )

        sales = stats.get("sales", {})
        sold_by_good = sales.get("sold_by_good", {})
        revenue_by_good = sales.get("revenue_by_good", {})

        expenses = stats.get("expenses", {})
        taxes_paid = expenses.get("taxes_paid") or 0.0
        overhead_costs = expenses.get("overhead_costs", 0.0)
        supply_cost = expenses.get("supply_cost", 0.0)
        total_expenses = expenses.get("total_expenses", total_expenses)

        supply_stats = stats.get(
            "supply", {"quantity": 0.0, "unit_price": 0.0, "cost": 0.0}
        )
        supply_quantity = supply_stats.get("quantity", 0.0)
        supply_unit_price = supply_stats.get("unit_price", 0.0)
        if supply_unit_price == 0.0 and supply_quantity > 0.0:
            supply_unit_price = (
                supply_cost / supply_quantity if supply_quantity else 0.0
            )

        cash_stats = stats.get("cash", {})
        cash_start = cash_stats.get(
            "start", start_ledger.agent_money.get(self.name, 0.0)
        )
        cash_pre = cash_stats.get(
            "pre", pre_clearing_ledger.agent_money.get(self.name, 0.0)
        )
        cash_end = cash_stats.get("end", current_ledger.agent_money.get(self.name, 0.0))
        cash_before_taxes = cash_end + taxes_paid

        lines = [
            f"Timestep {timestep} reflection for {self.name}:",
            "Sales:",
        ]
        for good in self.goods:
            sold_qty = sold_by_good.get(good, 0.0)
            revenue = revenue_by_good.get(good, 0.0)
            lines.append(f"  {good}: sold {sold_qty:.2f} units for ${revenue:.2f}")
        lines.append(f"Total revenue: ${total_revenue:.2f}")

        lines.append(
            f"Supply purchases: bought {supply_quantity:.2f} units at ${supply_unit_price:.2f} each (total ${supply_cost:.2f})"
        )
        lines.append(f"Overhead costs: ${overhead_costs:.2f}")
        lines.append(f"Taxes paid: ${taxes_paid:.2f}")
        lines.append(f"Total expenses: ${total_expenses:.2f}")
        lines.append(f"Profit: ${profit:.2f}")
        lines.extend(
            [
                f"Cash at start: ${cash_start:.2f}",
                f"Cash before taxes: ${cash_before_taxes:.2f}",
                f"Cash after taxes: ${cash_end:.2f}",
            ]
        )

        return "\n".join(lines) + "\n"

    # Message handling for building prompts
    def add_message(self, timestep: int, m_type: Message, **kwargs) -> None:
        """Add messages to build prompts for the LLM"""
        self.add_message_history_timestep(timestep)
        if m_type == Message.UPDATE_PRICE:
            # Prepare pricing decision prompt
            self.message_history[timestep]["historical"] += f"Cash: ${self.cash:.2f}\n"
            self.message_history[timestep]["historical"] += (
                f"Current inventory: {dict(self.inventory)}\n"
            )

            market_data = kwargs.get("market_data")
            if market_data:
                self.message_history[timestep]["historical"] += (
                    f"Market data from previous step: {market_data}\n"
                )

            goods_list = ", ".join([f'"{good}"' for good in self.goods])

            if self.prompt_algo == "cot" or self.prompt_algo == "sc":
                price_format = (
                    "{"
                    + ", ".join(
                        [
                            f'"thought":"<thinking>", "price_{good}":"X"'
                            for good in self.goods
                        ]
                    )
                    + "}"
                )
            else:
                price_format = (
                    "{"
                    + ", ".join([f'"price_{good}":"X"' for good in self.goods])
                    + "}"
                )

            self.message_history[timestep]["user_prompt"] += (
                f"Decide the price for each good: {goods_list}. "
            )
            self.message_history[timestep]["user_prompt"] += (
                f"Exactly use the JSON format: {price_format}\n"
            )
            self.message_history[timestep]["expected_format"] = price_format

        elif m_type == Message.ACTION_PRICE:
            prices = kwargs.get("prices", {})
            price_str = ", ".join(
                [f"{good}: ${price:.2f}" for good, price in prices.items()]
            )
            self.message_history[timestep]["historical"] += f"Set prices: {price_str}\n"
            self.message_history[timestep]["action"] += f"Prices: {price_str}\n"
            self.message_history[timestep]["user_prompt"] = ""

        elif m_type == Message.UPDATE_SUPPLY:
            unit_price = kwargs.get("unit_price", 0)
            self.message_history[timestep]["historical"] += f"Cash: ${self.cash:.2f}\n"
            self.message_history[timestep]["historical"] += (
                f"Supply unit price: ${unit_price:.2f}\n"
            )

            if self.prompt_algo == "cot" or self.prompt_algo == "sc":
                supply_format = '{"thought":"<thinking>", "supply_quantity":"X"}'
                self.message_history[timestep]["user_prompt"] += (
                    "Decide how much supply to purchase. "
                )
                self.message_history[timestep]["user_prompt"] += (
                    f"Use the JSON format: {supply_format}\n"
                )
            else:
                supply_format = '{"supply_quantity":"X"}'
                self.message_history[timestep]["user_prompt"] += (
                    "Decide how much supply to purchase. "
                )
                self.message_history[timestep]["user_prompt"] += (
                    f"Use the JSON format: {supply_format}\n"
                )
            self.message_history[timestep]["expected_format"] = supply_format

        elif m_type == Message.ACTION_SUPPLY:
            quantity = kwargs.get("quantity", 0)
            cost = kwargs.get("cost", 0)
            self.message_history[timestep]["historical"] += (
                f"Purchased {quantity:.2f} units of supply for ${cost:.2f}\n"
            )
            self.message_history[timestep]["action"] += f"Supply: {quantity:.2f}\n"
            self.message_history[timestep]["user_prompt"] = ""

        elif m_type == Message.UPDATE_PRODUCTION:
            self.message_history[timestep]["historical"] += (
                f"Available supply: {self.supplies:.2f}\n"
            )

            goods_list = ", ".join([f'"{good}"' for good in self.goods])

            if self.prompt_algo == "cot" or self.prompt_algo == "sc":
                prod_format = (
                    "{"
                    + ", ".join(
                        [
                            f'"thought":"<thinking>", "produce_{good}":"X%"'
                            for good in self.goods
                        ]
                    )
                    + "}"
                )
            else:
                prod_format = (
                    "{"
                    + ", ".join([f'"produce_{good}":"X%"' for good in self.goods])
                    + "}"
                )

            self.message_history[timestep]["user_prompt"] += (
                f"Decide what percentage of supply to allocate to all of these goods: {goods_list}."
            )
            self.message_history[timestep]["user_prompt"] += (
                f'By replacing the "X%"s in this JSON string: {prod_format}\n'
            )
            self.message_history[timestep]["user_prompt"] += (
                f"Do not respond with any other text or fields."
            )
            self.message_history[timestep]["expected_format"] = prod_format

        elif m_type == Message.ACTION_PRODUCTION:
            production = kwargs.get("production", {})
            prod_str = ", ".join(
                [f"{good}: {qty:.2f}" for good, qty in production.items()]
            )
            self.message_history[timestep]["historical"] += f"Produced: {prod_str}\n"
            self.message_history[timestep]["action"] += f"Production: {prod_str}\n"
            self.message_history[timestep]["user_prompt"] = ""

        elif m_type == Message.REFLECTION:
            self.message_history[timestep]["historical"] += "Reflection:"
            self.message_history[timestep]["historical"] += kwargs.get(
                "reflection_msg", ""
            )


class FixedFirmAgent(BaseFirmAgent):
    def __init__(
        self,
        name: str,
        goods: List[str],
        initial_cash: float,
        ledger: Ledger,
        market: Market,
    ):
        BaseFirmAgent.__init__(self)
        self.name = name
        self.goods = goods  # List of goods this firm can produce

        self.ledger = ledger
        self.market = market
        self.logger = logging.getLogger("main")
        # self.policy = policy
        self.ledger.credit(self.name, initial_cash)

        # Initialize inventory in ledger
        self.ledger.add_good(self.name, "supply", 0.0)
        for good in goods:
            self.ledger.add_good(self.name, good, 0.0)

        # Reference the ledger's inventory directly - no separate copy
        self.inventory = self.ledger.agent_inventories[self.name]

        # Running total of quantity sold by good (cumulative across all timesteps)
        self.total_quantity_sold_by_good: Dict[str, float] = {
            good: 0.0 for good in goods
        }
        # Nested dict: timestep -> good -> quantity
        self.total_quantity_sold_by_good_this_timestep: Dict[int, Dict[str, float]] = (
            defaultdict(lambda: {good: 0.0 for good in goods})
        )

    def set_price(self, price: float, timestep: int = None) -> Dict[str, float]:
        """Set fixed prices for goods"""
        return {good: price for good in self.goods}

    def purchase_supplies(
        self, quantity_to_purchase: float, unit_price: float, timestep: int
    ) -> float:
        """Purchases aggregate supply"""
        cost = quantity_to_purchase * unit_price
        # Only spend what we can afford
        total_cost = min(cost, self.cash)
        total_quantity = total_cost / unit_price

        # Deduct cost and add supply to ledger
        self.ledger.credit(self.name, -total_cost)
        self.ledger.add_good(self.name, "supply", total_quantity)

        return total_quantity

    def produce_goods(self, timestep: int):
        """Produce goods evenly given available supplies"""
        production = {}
        supply_available = self.supplies

        if supply_available <= 0:
            return

        # Calculate production for each good
        production_per_good = supply_available / len(self.goods)

        for good in self.goods:
            # Add produced goods to inventory via ledger
            self.ledger.add_good(self.name, good, production_per_good)

        # Consume all supplies used in production
        self.ledger.add_good(self.name, "supply", -supply_available)
