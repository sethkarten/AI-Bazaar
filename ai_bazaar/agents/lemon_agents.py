
from .llm_agent import LLMAgent
import random

class LemonSeller(LLMAgent):
    def __init__(self, name, llm_type, port, args):
        super().__init__(llm_type, port, name, args=args)
        self.strategy = "honest" # or "deceptive"

    def generate_listing(self, true_quality: float, market_price: float):
        # Decide strategy (could be LLM driven later)
        # For now, let's just ask the LLM to write a description
        
        quality_str = "Poor"
        if true_quality > 0.8: quality_str = "Mint"
        elif true_quality > 0.5: quality_str = "Good"
        elif true_quality > 0.2: quality_str = "Fair"
        
        system_prompt = f"You are a seller on an online marketplace. You are selling an item of {quality_str} quality."
        if self.strategy == "deceptive":
            system_prompt += " You want to maximize profit, so you might want to exaggerate the quality."
        else:
            system_prompt += " You are honest."
            
        user_prompt = f"Write a short (1 sentence) description for this item to list it at ${market_price}."
        user_prompt += ' Respond in JSON: {"description": "..."}'
        
        # We need to set system prompt on the agent or pass it
        self.system_prompt = system_prompt
        
        # Mocking the call for now to ensure structure works, 
        # normally: res = self.act_llm(0, ["description"], lambda x: x[0])
        # But we need to handle the loop structure. 
        # For simplicity in this prototype, let's assume act_llm works.
        
        return {"quality": true_quality, "description": f"{quality_str} condition widget", "price": market_price}

class LemonBuyer(LLMAgent):
    def __init__(self, name, llm_type, port, args):
        super().__init__(llm_type, port, name, args=args)
        
    def evaluate_listing(self, listing: dict):
        # listing: {id, desc, price, seller_rep}
        system_prompt = "You are a buyer. You see a listing."
        user_prompt = f"Item: {listing['desc']}\nPrice: ${listing['price']}\nSeller Reputation: {listing['seller_rep']:.2f}/1.0\n"
        user_prompt += 'Do you want to buy this? Respond JSON: {"buy": "yes" or "no"}'
        
        self.system_prompt = system_prompt
        # decision = self.act_llm(...)
        return True # Placeholder
