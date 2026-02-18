import logging
import numpy as np
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class Item:
    item_id: str
    seller_id: str
    true_quality: float  # 0.0 (Broken) to 1.0 (Mint)
    description: str     # Generated text
    price: float
    # Hidden attributes not visible to buyer directly
    
@dataclass
class Transaction:
    item: Item
    buyer_id: str
    price_paid: float
    timestep: int
    buyer_satisfaction: float = 0.0

class LemonMarketEnv:
    def __init__(self, num_sellers: int = 5, num_buyers: int = 10, args=None):
        self.args = args
        self.timestep = 0
        self.sellers = [f"seller_{i}" for i in range(num_sellers)]
        self.buyers = [f"buyer_{i}" for i in range(num_buyers)]
        
        # State
        self.market_listings: List[Item] = []
        self.reputation: Dict[str, float] = {s: 1.0 for s in self.sellers} # 0.0 to 1.0
        self.transaction_history: List[Transaction] = []
        self.buyer_cash: Dict[str, float] = {b: 1000.0 for b in self.buyers}
        self.seller_cash: Dict[str, float] = {s: 0.0 for s in self.sellers}
        
        # Config
        self.quality_levels = {
            "Mint": 1.0,
            "Good": 0.7,
            "Fair": 0.4,
            "Poor": 0.1
        }
        
    def reset(self):
        self.timestep = 0
        self.market_listings = []
        self.reputation = {s: 1.0 for s in self.sellers}
        self.transaction_history = []
        self.buyer_cash = {b: 1000.0 for b in self.buyers}
        self.seller_cash = {s: 0.0 for s in self.sellers}
        return self.get_state()

    def step(self, seller_actions: List[Dict], buyer_actions: List[Dict]):
        """
        seller_actions: List of {seller_id, generate_item: {quality, description, price}}
        buyer_actions: List of {buyer_id, buy_item_id: str}
        """
        self.timestep += 1
        logs = []
        
        # 1. Sellers list items
        self.market_listings = [] # Clear daily listings? Or accumulate? Let's say daily clearing for now.
        for action in seller_actions:
            sid = action['seller_id']
            if 'generate_item' in action:
                gen = action['generate_item']
                # Validate cost? (e.g. producing mint costs more)
                true_q = gen.get('quality', 0.5)
                desc = gen.get('description', "Item")
                price = gen.get('price', 10.0)
                
                # Use provided ID if available (for sync with buyers), else generate
                iid = gen.get('item_id', f"{sid}_{self.timestep}_{random.randint(0,999)}")
                
                item = Item(
                    item_id=iid,
                    seller_id=sid,
                    true_quality=true_q,
                    description=desc,
                    price=price
                )
                self.market_listings.append(item)
                
        # 2. Buyers Buy
        # Shuffle buyers to avoid deterministic priority
        random.shuffle(buyer_actions)
        
        for action in buyer_actions:
            bid = action['buyer_id']
            target_id = action.get('buy_item_id')
            
            if not target_id:
                continue
                
            # Find item
            item = next((i for i in self.market_listings if i.item_id == target_id), None)
            
            if item and self.buyer_cash[bid] >= item.price:
                # Execute Trade
                self.buyer_cash[bid] -= item.price
                self.seller_cash[item.seller_id] += item.price
                
                # Calculate Satisfaction (Reward for Buyer)
                # Value = Quality * 20 (Arbitrary utility scaling)
                value = item.true_quality * 20.0
                surplus = value - item.price
                
                # Update Reputation
                # If desc says "Mint" (implied 1.0) but quality is 0.1, big hit.
                # Simplified: Satisfaction = True Quality
                self.reputation[item.seller_id] = 0.9 * self.reputation[item.seller_id] + 0.1 * item.true_quality
                
                # Log
                t = Transaction(item, bid, item.price, self.timestep, buyer_satisfaction=surplus)
                self.transaction_history.append(t)
                logs.append(f"{bid} bought {item.description} (Q={item.true_quality:.2f}) from {item.seller_id} for ${item.price}")
                
                # Remove from market
                self.market_listings.remove(item)
            else:
                logs.append(f"{bid} failed to buy {target_id}")

        return self.get_state(), logs

    def get_state(self):
        return {
            "timestep": self.timestep,
            "listings": [
                {
                    "id": i.item_id, 
                    "desc": i.description, 
                    "price": i.price, 
                    "seller_rep": self.reputation[i.seller_id]
                } 
                for i in self.market_listings
            ],
            "buyer_cash": self.buyer_cash.copy(),
            "seller_reputation": self.reputation.copy()
        }
