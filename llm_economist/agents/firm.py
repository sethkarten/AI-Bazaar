# FIRM
from llm_economist.market_core.market_core import Ledger, Market, Quote
from typing import List, Dict
from .llm_agent import LLMAgent
import logging
import numpy as np
from llm_economist.utils.common import Message

class BaseFirmAgent():
    """Base class for all firms with shared functionality.
    
    Note: Decision methods (set_price, purchase_supplies, produce_goods) are not
    defined here because they have different signatures for Fixed vs LLM agents.
    Fixed agents take explicit parameters, while LLM agents decide autonomously.
    """
    
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
                    quantity_available=self.inventory[good]
                )
                quotes.append(quote)
                self.market.post_quote(quote)
        return quotes
    
class FirmAgent(LLMAgent, BaseFirmAgent):
    def __init__(self, llm: str, port: int, name: str, prompt_algo: str='io', 
                 history_len: int=10, timeout: int=10, 
                 goods: List[str]=None, initial_cash: float=0.0, ledger: Ledger=None, market: Market=None, args=None) -> None:
        super().__init__(llm, port, name, prompt_algo, history_len, timeout, args=args)
        self.logger = logging.getLogger('main')
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
        
        # Set system prompt for the firm
        self.system_prompt = self._create_system_prompt()
        
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

    def set_price(self, timestep: int = None) -> Dict[str, float]:
        """LLM decides prices for each good"""
        self.add_message(timestep, Message.UPDATE_PRICE)
        
        # Create keys for each good's price
        price_keys = [f"price_{good}" for good in self.goods]
        
        # Call LLM to decide prices
        prices = self.act_llm(timestep, price_keys, self.parse_prices)
        
        # Convert to dictionary
        price_dict = {good: price for good, price in zip(self.goods, prices)}
        
        self.add_message(timestep, Message.ACTION_PRICE, prices=price_dict)
        return price_dict
    
    def purchase_supplies(self, unit_price: float, timestep: int) -> float:
        """LLM decides how much supply to purchase"""
        self.add_message(timestep, Message.UPDATE_SUPPLY, unit_price=unit_price)
        
        # Call LLM to decide quantity
        quantity_to_purchase = self.act_llm(timestep, ['supply_quantity'], self.parse_supply_purchase)[0]
        
        # Execute the purchase (as much as cash allows)
        cost = quantity_to_purchase * unit_price
        total_cost = min(cost, self.cash)
        total_quantity = total_cost / unit_price
        
        # Deduct cost and add supply to ledger
        self.ledger.credit(self.name, -total_cost)
        self.ledger.add_good(self.name, "supply", total_quantity)
        
        self.add_message(timestep, Message.ACTION_SUPPLY, quantity=total_quantity, cost=total_cost)
        return total_quantity
    
    def produce_goods(self, timestep: int):
        """LLM decides how much to produce of each good"""
        self.add_message(timestep, Message.UPDATE_PRODUCTION)
        
        # Create keys for each good's production amount
        production_keys = [f"produce_{good}" for good in self.goods]
        
        # Call LLM to decide production amounts (as percentages of available supply)
        production_percentages = self.act_llm(timestep, production_keys, self.parse_production)
        
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
        
        # Consume all supplies used in production
        self.ledger.add_good(self.name, "supply", -supply_available)
        
        self.add_message(timestep, Message.ACTION_PRODUCTION, production=production_dict)
    
    # Parse functions
    def parse_prices(self, items: List[str]) -> tuple:
        """Parse and validate price decisions"""
        output = []
        for item in items:
            if isinstance(item, str):
                item = item.replace('$','').replace(',','').replace('\n','')
            price = float(item)
            output.append(price)
        return tuple(output)
    
    def parse_supply_purchase(self, items: List[str]) -> tuple:
        """Parse and validate supply purchase decision"""
        output = []
        for item in items:
            if isinstance(item, str):
                item = item.replace('$','').replace(',','').replace(' units', '').replace('\n','')
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
                item = item.replace('%','').replace(',','').replace('\n','')
            pct = float(item)
            # Clip to 0-100%
            pct = np.clip(pct, 0.0, 100.0)
            output.append(pct)
        return tuple(output)
    
    # Message handling for building prompts
    def add_message(self, timestep: int, m_type: Message, **kwargs) -> None:
        """Add messages to build prompts for the LLM"""
        self.add_message_history_timestep(timestep)
        if m_type == Message.UPDATE_PRICE:
            # Prepare pricing decision prompt
            self.message_history[timestep]['historical'] += f'Cash: ${self.cash:.2f}\n'
            self.message_history[timestep]['historical'] += f'Current inventory: {dict(self.inventory)}\n'
            
            goods_list = ", ".join([f'"{good}"' for good in self.goods])
            
            if self.prompt_algo == 'cot' or self.prompt_algo == 'sc':
                price_format = '{' + ', '.join([f'"thought":"<thinking>", "price_{good}":"X"' for good in self.goods]) + '}'
            else:
                price_format = '{' + ', '.join([f'"price_{good}":"X"' for good in self.goods]) + '}'
            
            self.message_history[timestep]['user_prompt'] += f'Decide the price for each good: {goods_list}. '
            self.message_history[timestep]['user_prompt'] += f'Exactly use the JSON format: {price_format}\n'
        
        elif m_type == Message.ACTION_PRICE:
            prices = kwargs.get('prices', {})
            price_str = ", ".join([f"{good}: ${price:.2f}" for good, price in prices.items()])
            self.message_history[timestep]['historical'] += f'Set prices: {price_str}\n'
            self.message_history[timestep]['action'] += f'Prices: {price_str}\n'
        
        elif m_type == Message.UPDATE_SUPPLY:
            unit_price = kwargs.get('unit_price', 0)
            self.message_history[timestep]['historical'] += f'Cash: ${self.cash:.2f}\n'
            self.message_history[timestep]['historical'] += f'Supply unit price: ${unit_price:.2f}\n'
            
            if self.prompt_algo == 'cot' or self.prompt_algo == 'sc':
                self.message_history[timestep]['user_prompt'] += 'Decide how much supply to purchase. '
                self.message_history[timestep]['user_prompt'] += 'Use the JSON format: {"thought":"<thinking>", "supply_quantity":"X"}\n'
            else:
                self.message_history[timestep]['user_prompt'] += 'Decide how much supply to purchase. '
                self.message_history[timestep]['user_prompt'] += 'Use the JSON format: {"supply_quantity":"X"}\n'
        
        elif m_type == Message.ACTION_SUPPLY:
            quantity = kwargs.get('quantity', 0)
            cost = kwargs.get('cost', 0)
            self.message_history[timestep]['historical'] += f'Purchased {quantity:.2f} units of supply for ${cost:.2f}\n'
            self.message_history[timestep]['action'] += f'Supply: {quantity:.2f}\n'
        
        elif m_type == Message.UPDATE_PRODUCTION:
            self.message_history[timestep]['historical'] += f'Available supply: {self.supplies:.2f}\n'
            
            goods_list = ", ".join([f'"{good}"' for good in self.goods])
            
            if self.prompt_algo == 'cot' or self.prompt_algo == 'sc':
                prod_format = '{' + ', '.join([f'"thought":"<thinking>", "produce_{good}":"X%"' for good in self.goods]) + '}'
            else:
                prod_format = '{' + ', '.join([f'"produce_{good}":"X%"' for good in self.goods]) + '}'
            
            self.message_history[timestep]['user_prompt'] += f'Decide what percentage of supply to allocate to all of these goods: {goods_list}.'
            self.message_history[timestep]['user_prompt'] += f'By replacing the \"X%\"s in this JSON string: {prod_format}\n'
            self.message_history[timestep]['user_prompt'] += f'Do not respond with any other text or fields.'
        
        elif m_type == Message.ACTION_PRODUCTION:
            production = kwargs.get('production', {})
            prod_str = ", ".join([f"{good}: {qty:.2f}" for good, qty in production.items()])
            self.message_history[timestep]['historical'] += f'Produced: {prod_str}\n'
            self.message_history[timestep]['action'] += f'Production: {prod_str}\n'
    

class FixedFirmAgent(BaseFirmAgent):
    def __init__(self, name: str, 
                 goods: List[str], initial_cash: float, ledger: Ledger, market: Market):
        self.name = name
        self.goods = goods  # List of goods this firm can produce
        
        self.ledger = ledger
        self.market = market
        # self.policy = policy
        self.ledger.credit(self.name, initial_cash)
        
        # Initialize inventory in ledger
        self.ledger.add_good(self.name, "supply", 0.0)
        for good in goods:
            self.ledger.add_good(self.name, good, 0.0)
            
        # Reference the ledger's inventory directly - no separate copy
        self.inventory = self.ledger.agent_inventories[self.name]

    def set_price(self, price: float, timestep: int = None) -> Dict[str, float]:
        """Set fixed prices for goods"""
        return {good: price for good in self.goods}

    def purchase_supplies(self, quantity_to_purchase: float, unit_price: float, timestep: int) -> float:
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
                    
    
