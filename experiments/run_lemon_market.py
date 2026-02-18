import argparse
import logging
import os
import json
import numpy as np
import random
from ai_bazaar.env.lemon_env import LemonMarketEnv
from ai_bazaar.agents.lemon_agents import LemonSeller, LemonBuyer

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO)

def run_lemon_experiment(args):
    # Initialize Environment
    env = LemonMarketEnv(num_sellers=args.num_sellers, num_buyers=args.num_buyers, args=args)
    
    # Initialize Agents
    sellers = []
    for i in range(args.num_sellers):
        name = f"seller_{i}"
        seller = LemonSeller(name, args.llm, args.port, args)
        # Assign Strategy: 50% Honest, 50% Deceptive
        seller.strategy = "deceptive" if i % 2 == 0 else "honest"
        sellers.append(seller)
        
    buyers = []
    for i in range(args.num_buyers):
        name = f"buyer_{i}"
        buyer = LemonBuyer(name, args.llm, args.port, args)
        buyers.append(buyer)
        
    logging.info(f"Starting Lemon Market Experiment: {args.num_sellers} Sellers, {args.num_buyers} Buyers")
    
    results = []
    
    for t in range(args.max_timesteps):
        logging.info(f"--- Timestep {t} ---")
        
        # 1. Sellers Generate Listings
        seller_actions = []
        for seller in sellers:
            # Assign a random true quality for this item
            true_quality = random.choice([0.1, 0.4, 0.7, 1.0]) 
            
            # Pricing Strategy
            if seller.strategy == "deceptive":
                # Always price as if Mint ($20), regardless of quality
                market_price = 20.0
            else:
                # Honest pricing based on value
                market_price = true_quality * 20.0
            
            # Agent decides listing (description + price)
            # In a real run, this calls LLM. 
            # For speed/simplicity here, we use the helper we wrote which mocks the prompt structure
            listing_data = seller.generate_listing(true_quality, market_price)
            
            seller_actions.append({
                "seller_id": seller.name,
                "generate_item": listing_data
            })
            
        # 2. Buyers Evaluate and Buy
        # Get market state (listings)
        state = env.get_state() 
        
        # 1. Execute Listing Generation (Outside Env Step or Pre-Step)
        current_listings = []
        for action in seller_actions:
            sid = action['seller_id']
            gen = action['generate_item']
            # Create a temporary item object for buyers to view
            item = {
                "id": f"{sid}_{t}_{random.randint(0,999)}",
                "seller_id": sid,
                "desc": gen['description'],
                "price": gen['price'],
                "seller_rep": env.reputation[sid]
            }
            current_listings.append(item)
            
        # 2. Buyers Decide
        buyer_actions = []
        for buyer in buyers:
            # Simple heuristic: look at up to 3 random items
            visible_items = random.sample(current_listings, min(3, len(current_listings)))
            
            for item in visible_items:
                # LLM Decision
                should_buy = buyer.evaluate_listing(item)
                if should_buy:
                    buyer_actions.append({
                        "buyer_id": buyer.name,
                        "buy_item_id": item['id']
                    })
                    break # Buy one item max
        
        # 3. Step Environment
        # Updating seller_actions with the IDs we generated for buyers
        for i, action in enumerate(seller_actions):
            action['generate_item']['item_id'] = current_listings[i]['id']
            
        state, logs = env.step(seller_actions, buyer_actions)
        
        # Log Metrics
        recent_txs = [tx for tx in env.transaction_history if tx.timestep == env.timestep]
        avg_satisfaction = np.mean([tx.buyer_satisfaction for tx in recent_txs]) if recent_txs else 0.0
        
        results.append({
            "timestep": t,
            "volume": len(recent_txs),
            "avg_satisfaction": avg_satisfaction,
            "deceptive_seller_cash": sum([env.seller_cash[s.name] for s in sellers if s.strategy == "deceptive"]),
            "honest_seller_cash": sum([env.seller_cash[s.name] for s in sellers if s.strategy == "honest"])
        })
        
        logging.info(f"Step {t}: Volume={len(recent_txs)}, Sat={avg_satisfaction:.2f}")

    # Save Results
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sellers", type=int, default=4)
    parser.add_argument("--num_buyers", type=int, default=10)
    parser.add_argument("--max_timesteps", type=int, default=20)
    parser.add_argument("--llm", default="gpt-4o-mini")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", default="lemon_results.jsonl")
    # Dummy args for Agent init
    parser.add_argument("--prompt_algo", default="io")
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--service", default="vllm")
    parser.add_argument("--bracket-setting", default="three") # Legacy
    
    args = parser.parse_args()
    setup_logging("logs")
    
    run_lemon_experiment(args)
