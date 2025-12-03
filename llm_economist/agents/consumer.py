# CONSUMER
from llm_economist.market_core.market_core import Ledger, Market, Order, Quote
from ..utils.common import PERSONAS, ROLE_MESSAGES
from ..agents.llm_agent import LLMAgent
from typing import List, Dict
import logging


    
        

# class ConsumerAgent(LLMAgent):
#     def __init__(self, llm: str, port: int, name: str,
#                  income_stream: float, ces_params: Dict[str, float],
#                  **kwargs):
#         super().__init__(llm, port, name, **kwargs)
#         
#         self.income = income_stream
#         self.ces_params = ces_params  # CES utility parameters
#         self.inventory = {}  # Consumer's goods inventory
#         
#         self.system_prompt = self._create_system_prompt()
#         
#     def _create_system_prompt(self):
#         return f"""You are {self.name}, a consumer with income {self.income}.
# Your goal is to maximize utility by purchasing goods optimally.
# 
# CES utility parameters: {self.ces_params}
# Current inventory: {self.inventory}
# 
# Use JSON format for responses."""
#     
#     def submit_orders(self, market: Market, timestep: int) -> List[Order]:
#         """Submit orders based on CES utility maximization"""
#         # Get available quotes
#         quotes = self._get_available_quotes(market)
#         
#         # LLM decides on optimal consumption bundle
#         prompt = f"""Based on available goods and prices:
# {quotes}
# 
# Your income: {self.income}
# CES utility parameters: {self.ces_params}
# 
# Choose optimal consumption to maximize utility.
# Respond with: {{"orders": [{{"good": "good1", "quantity": q1, "max_price": p1}}, ...]}}"""
#         
#         response = self.act_llm(timestep, ['orders'], self._parse_orders)
#         orders_data = response[0]
#         
#         orders = []
#         for order_data in orders_data:
#             order = Order(
#                 consumer_id=self.name,
#                 firm_id="any",  # Would need to determine preferred firm
#                 good=order_data['good'],
#                 quantity=order_data['quantity'],
#                 max_price=order_data['max_price']
#             )
#             orders.append(order)
#             
#         return orders
    
class CESConsumerAgent(LLMAgent):
    def __init__(self, name: str, income_stream: float, ledger: 'Ledger', market: 'Market', 
                 sigma: float=5.0, persona: str=None, ces_params: Dict[str, float]=None, risk_aversion: float=None, 
                 goods: List[str]=None, llm: str=None, port: int=8000, prompt_algo: str='cot',
                 history_len: int=10, timeout: int=3, quantity_per_good: float=10.0, args=None):
        
        # Initialize LLMAgent for CES parameter generation
        super().__init__(llm_type=llm, port=port, name=name, prompt_algo=prompt_algo, 
                        history_len=history_len, timeout=timeout, args=args)
                
        # Consumer-specific attributes
        self.sigma = sigma
        self.income = income_stream
        self.ces_params = ces_params
        self.risk_aversion = risk_aversion
        self.previous_cost_of_living = None
        self.cost_of_living = 0.0
        self.persona = persona
        self.ledger = ledger
        self.market = market
        self.quantity_per_good = quantity_per_good
        self.goods = goods
        self.prices_dict = {}
        self.prices_prev = {}
        for good in goods:
            self.prices_dict[good] = 0.0
            self.prices_prev[good] = 0.0
        
        # Initialize cash in ledger (starting with 0, will receive income)
        self.ledger.credit(self.name, 0.0)
        
        # Initialize empty inventory in ledger - will be populated as consumer buys goods
        for good in goods:
            self.ledger.add_good(self.name, good, 0.0)
        
        # Reference the ledger's inventory directly - no separate copy
        self.inventory = self.ledger.agent_inventories[self.name]
        
        # Generate CES parameters using personas if not provided
        if self.persona is not None and self.llm is not None:
            self.ces_params = self.generate_ces_params(self.goods)
            # self.risk_aversion = self.generate_risk_aversion()
        elif self.ces_params is not None:
            if len(self.ces_params) != len(self.goods):
                raise ValueError(f"CES parameters must be provided for all goods. Got {len(self.ces_params)} parameters for {len(self.goods)} goods.")
            if any(param < 0.0 for param in self.ces_params.values()):
                raise ValueError(f"CES parameters must be positive. Got {self.ces_params}")
        elif self.risk_aversion is not None:
            if self.risk_aversion <= 0:
                raise ValueError(f"Risk aversion must be positive. Got {self.risk_aversion}")
        
        else:
            raise ValueError(f"CES parameters, risk aversion, or persona must be provided. Got {self.ces_params}, {self.risk_aversion}, and {self.persona}")
        
    
    def compute_utility(self) -> float:
        """Compute CES utility"""
        utility = 0.0
        total = 0.0
        inventory = self.inventory
        for good in self.goods:
            # get quantity of good
            quantity = inventory[good]
            # get alpha for good
            alpha = self.ces_params[good]
            # compute utility for good
            total += alpha * (quantity ** ((self.sigma - 1) / self.sigma))
        utility = total ** (self.sigma / (self.sigma - 1))
        return utility
    
    def compute_cost_of_living(self) -> float:
        """Compute cost of living & store average prices for each good"""
        # save previous values
        self.previous_cost_of_living = self.cost_of_living if self.cost_of_living > 0.0 else self.previous_cost_of_living
        for good in self.goods:
            self.prices_prev[good] = self.prices_dict[good] if self.prices_dict[good] > 0.0 else self.prices_prev[good]
        
        cost_of_living = 0.0
        total = 0.0
        quotes = self.market.quotes
        #! Debugging
        for quote in quotes:
            self.logger.info(f"Quote: {quote.good}: {quote.price}")
            
        for good in self.goods:
            # Get average price for this good from quotes
            # Use str() and strip() to handle type mismatches and whitespace
            good_str = str(good).strip()
            good_quotes = [q for q in quotes if str(q.good).strip() == good_str]
            self.logger.info(f"Good quotes: {good_quotes}")
            self.logger.info(f"# Good quotes: {len(good_quotes)}")
            if good_quotes:
                self.prices_dict[good] = sum(q.price for q in good_quotes) / len(good_quotes)
            else:
                self.prices_dict[good] = 0.0  # No quotes available for this good
                self.logger.warning(f"No quotes available for {good}")
            # compute cost of living for good
            if self.prices_dict[good] > 0.0:
                total += self.ces_params[good] * (self.prices_dict[good] ** (1 - self.sigma))
            else:
                total += self.ces_params[good] * (self.prices_prev[good] ** (1 - self.sigma))
        
        if total == 0.0:
            return 0.0
        cost_of_living = total ** (1 / (1 - self.sigma))
        
        self.cost_of_living = cost_of_living
        
        return self.cost_of_living
    
    def compute_demand(self) -> Dict[str, float]:
        """Compute demand for each good"""
        P = self.compute_cost_of_living()
        if P <= 0.0:
            P = self.previous_cost_of_living
        demand = {}
        for good in self.goods:
            # get alpha for good
            alpha = self.ces_params[good]
            # get price for good
            
            price = self.prices_dict[good] if self.prices_dict[good] > 0.0 else self.prices_prev[good]
            # compute demand for good
            if P is None or P <= 0.0:
                demand[good] = 0.0
            else:
                demand[good] = (self.income / P) * alpha * (price / P) ** (-self.sigma)
        return demand     
    
    @property
    def cash(self) -> float:
        """Get current cash from ledger"""
        return self.ledger.agent_money[self.name]
    
    @property
    def utility(self) -> float:
        """Get current utility from ledger"""
        return self.compute_utility()
    
    def receive_income(self, timestep: int = None):
        """Receive income payment"""
        self.ledger.credit(self.name, self.income)
    
    def create_order(self, firm_id: str, good: str, quantity: float, max_price: float) -> Order:
        """Create an order to purchase goods (without submitting)"""
        order = Order(
            consumer_id=self.name,
            firm_id=firm_id,
            good=good,
            quantity=quantity,
            max_price=max_price
        )
        return order
    
    def submit_orders(self, orders: List[Order]) -> None:
        """Submit a list of orders to the market"""
        for order in orders:
            self.market.submit_order(order)
    
    def make_orders(self, timestep: int, scenario: str) -> List[Order]:
        "Make fixed list of orders (returns orders without submitting)"
        demand = self.compute_demand()
        quotes = self.market.quotes
        orders = []
        if scenario == 'RACE_TO_BOTTOM':
            # willingness to pay = lowest quote 
            for good in self.goods:
                good_str = str(good).strip()
                lowest_quote = min((q for q in quotes if str(q.good).strip() == good_str), key=lambda q: q.price)
                orders.append(self.create_order(lowest_quote.firm_id, good, demand[good], lowest_quote.price))
            
        elif scenario == 'EARLY_BIRD':
            # find lowest available quote, submit and fill the order immediately
            NotImplementedError("EARLY_BIRD scenario not implemented")
        elif scenario == 'PRICE_DISCRIMINATION':
            # compute willingness to pay for each good and submit an order without firm discrimination
            # market will clear quotes and orders with the closest matching prices
            NotImplementedError("PRICE_DISCRIMINATION scenario not implemented")
        elif scenario == 'RATIONAL_BAZAAR':
            # willingness to pay is the average price among all quotes
            NotImplementedError("RATIONAL_BAZAAR scenario not implemented")
        elif scenario == 'BOUNDED_BAZAAR':
            # willingness to pay is average price among a random subset of quotes
            NotImplementedError("BOUNDED_BAZAAR scenario not implemented")
        else:
            raise ValueError(f"Invalid scenario: {scenario}")
        
                    
        return orders
    
    def pay_taxes(self, timestep: int, tax_rate: float) -> float:
        """Pay taxes to the government"""
        taxes_paid = self.cash * tax_rate
        self.ledger.credit(self.name, -taxes_paid)
        return taxes_paid
    
    
    def generate_ces_params(self, goods: List[str]) -> Dict[str, float]:
        """Generate CES parameters using personas"""
        params: Dict[str, float] = {}
        # Get persona description
        if self.persona not in ROLE_MESSAGES:
            raise ValueError(f"Unknown persona: {self.persona}")
        role_message = ROLE_MESSAGES[self.persona]
        
        # Create keys matching the format expected in the prompt
        weight_keys = [f'weight_{good}' for good in goods]
        params_format = '{' + ', '.join([f'"{key}":"X"' for key in weight_keys]) + '}'

        # Build prompt
        system_prompt = f"You are an expert in consumer economics and utility functions. Generate CES (Constant Elasticity of Substitution) utility parameters based on consumer personas."
        
        user_prompt = f"Based on this consumer persona:\n{role_message}\n"
        user_prompt += f"Generate CES utility parameters that reflect this persona's preferences for each of the following goods in JSON format: {', '.join(goods)}\n" 
        user_prompt += f"The weights should:\n1. Sum to 1.0\n2. All be positive values\n3. Be realistic and grounded.\n"
        user_prompt += f"Respond with the CES utility parameters in JSON format: {params_format}\n"
        
        # Set system_prompt for call_llm to use
        original_system_prompt = self.system_prompt
        self.system_prompt = system_prompt
        
        try:
            # Combine user_prompt and pass timestep=0 (initialization)
            combined_msg = user_prompt
            response = self.call_llm(combined_msg, timestep=0, keys=weight_keys, parse_func=self._parse_ces_params)
            
            # Convert the parsed weights to a dict with good names
            for i, good in enumerate(goods):
                params[good] = response[i]
            self.logger.info(f"CES parameters: {params}")
            return params
        except Exception as e:
            self.logger.warning(f"Error generating CES parameters with LLM: {e} Falling back to uniform params.")
            fixed_weight = 1.0 / len(goods)
            for good in goods:
                params[good] = fixed_weight
            return params
        finally:
            # Restore original system_prompt
            self.system_prompt = original_system_prompt

    def _parse_ces_params(self, items: List) -> tuple:
        """Parse and validate CES parameters"""
        output = []
        for item in items:
            if isinstance(item, str):
                item = item.replace('%','').replace(',','').replace('\n','')
            weight = float(item)
            if weight <= 0:
                raise ValueError(f"CES parameter must be positive, got {weight}")
            output.append(weight)
        
        return tuple(output)
    
    def generate_risk_aversion(self) -> float:
        """Generate risk aversion using persona"""
        risk_aversion_format = '"risk_aversion":"X"'
        
        # Get persona description
        if self.persona not in ROLE_MESSAGES:
            raise ValueError(f"Unknown persona: {self.persona}")
        role_message = ROLE_MESSAGES[self.persona]

        # Build prompt
        system_prompt = f"You are an expert in consumer economics and utility functions. Generate a risk aversion parameter."
        
        user_prompt = f"Based on this consumer persona:\n{role_message}\n"
        user_prompt += f"Generate a risk aversion parameter that reflects this persona's risk aversion." 
        user_prompt += f"The risk aversion parameter should be a positive value between 0 and 1.00.\n"
        user_prompt += f"Respond in JSON format: {{\"risk_aversion_param\":\"X\"}}\n"
        
        # Set system_prompt for call_llm to use
        original_system_prompt = self.system_prompt
        self.system_prompt = system_prompt
        
        try:
            combined_msg = user_prompt
            response = self.call_llm(combined_msg, timestep=0, keys=['risk_aversion_param'], parse_func=self._parse_risk_aversion)
            # call_llm returns list[float], and _parse_risk_aversion returns tuple with one element
            if isinstance(response, list):
                risk_aversion = response[0]
            else:
                risk_aversion = response
            self.logger.info(f"Risk aversion parameter: {risk_aversion}")
            return risk_aversion
        except Exception as e:
            self.logger.warning(f"Error generating a risk aversion parameter with LLM: {e} Falling back to default value of 1.0.")
            return 1.0
        finally:
            # Restore original system_prompt
            self.system_prompt = original_system_prompt
    
    def _parse_risk_aversion(self, items: List) -> float:
        """Parse and validate risk aversion parameter"""
        if len(items) != 1:
            raise ValueError(f"Expected 1 risk aversion parameter, got {len(items)}")
        
        item = items[0]
        
        # Convert to string first if needed, then clean and convert to float
        if not isinstance(item, (str, int, float)):
            item = str(item)
        
        if isinstance(item, str):
            # Clean the string: remove %, commas, newlines, whitespace
            item = item.replace('%','').replace(',','').replace('\n','').strip()
        
        # Convert to float
        try:
            risk_aversion = float(item)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not convert risk aversion parameter to float: {item}. Error: {e}")
        
        # Validate range: clamp to [0, 1]
        risk_aversion = max(0.0, min(1.0, risk_aversion))
        
        return risk_aversion
    
class FixedConsumerAgent():
    def __init__(self, name: str, income_stream: float, ledger: 'Ledger', market: 'Market', ces_params: Dict[str, float]=None, risk_aversion: float=1.0, goods: List[str]=None, quantity_per_good: float=10.0):
        self.name = name
        self.income = income_stream
        self.ledger = ledger
        self.market = market
        self.quantity_per_good = quantity_per_good
        # Initialize cash in ledger (starting with 0, will receive income)
        self.ledger.credit(self.name, 0.0)
        
        # Initialize empty inventory in ledger - will be populated as consumer buys goods
        for good in goods:
            self.ledger.add_good(self.name, good, 0.0)
        
        # Reference the ledger's inventory directly - no separate copy
        self.inventory = self.ledger.agent_inventories[self.name]
    
    @property
    def cash(self) -> float:
        """Get current cash from ledger"""
        return self.ledger.agent_money[self.name]
    
    def receive_income(self, timestep: int = None):
        """Receive income payment"""
        self.ledger.credit(self.name, self.income)
    
    def create_order(self, firm_id: str, good: str, quantity: float, max_price: float) -> Order:
        """Create an order to purchase goods (without submitting)"""
        order = Order(
            consumer_id=self.name,
            firm_id=firm_id,
            good=good,
            quantity=quantity,
            max_price=max_price
        )
        return order
    
    def submit_orders(self, orders: List[Order]) -> None:
        """Submit a list of orders to the market"""
        for order in orders:
            self.market.submit_order(order)
    
    def make_orders(self, timestep: int) -> List[Order]:
        "Make fixed list of orders (returns orders without submitting)"
        quotes = self.market.quotes
        orders = []
        for good in self.inventory:
            good_str = str(good).strip()
            for quote in quotes:
                if str(quote.good).strip() == good_str:
                    orders.append(self.create_order(quote.firm_id, good, self.quantity_per_good, quote.price))
                    break
        return orders
    
    def pay_taxes(self, timestep: int, tax_rate: float) -> float:
        """Pay taxes to the government"""
        taxes_paid = self.cash * tax_rate
        self.ledger.credit(self.name, -taxes_paid)
        return taxes_paid
    
    