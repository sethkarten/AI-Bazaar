"""
Main entry point for the LLM Economist marketplace simulation.

This simulation consists of:
- FirmAgent: Firms that produce goods and set prices
- ConsumerAgent: Consumers that purchase goods
"""

import argparse
import logging
import os
import sys
import wandb
import random
import numpy as np
import time
from .market_core.market_core import Ledger, Market
from .agents.firm import FirmAgent, FixedFirmAgent
from .agents.consumer import FixedConsumerAgent
from .agents.llm_agent import TestAgent


def setup_logging(args):
    """Setup logging configuration."""
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    log_filename = f'{args.log_dir}/marketplace_{args.name if args.name else "simulation"}.log'
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def run_marketplace_simulation(args):
    """Run the marketplace simulation with firms and consumers."""
    logger = logging.getLogger('main')
    
    # Test LLM connectivity if using LLM agents
    if args.firm_type == 'LLM':
        try:
            TestAgent(args.llm, args.port, args)
            logger.info(f"Successfully connected to LLM: {args.llm}")
        except Exception as e:
            logger.error(f"Failed to connect to LLM: {e}")
            sys.exit(1)
    
    # Initialize market infrastructure
    ledger = Ledger()
    market = Market()
    
    # Initialize firms
    firms = []
    goods = ['food', 'clothing', 'luxury']
    
    for i in range(args.num_firms):
        name = f"firm_{i}"
        initial_cash = args.firm_initial_cash
        
        if args.firm_type == 'LLM':
            firm = FirmAgent(
                llm=args.llm,
                port=args.port,
                name=name,
                goods=goods,
                initial_cash=initial_cash,
                ledger=ledger,
                market=market,
                prompt_algo=args.prompt_algo,
                history_len=args.history_len,
                timeout=args.timeout,
                args=args
            )
        else:
            firm = FixedFirmAgent(
                name=name,
                goods=goods,
                initial_cash=initial_cash,
                ledger=ledger,
                market=market
            )
        firms.append(firm)
        logger.info(f"Created {name}")
    
    # Initialize consumers
    consumers = []
    for i in range(args.num_consumers):
        name = f"consumer_{i}"
        income = np.random.uniform(50, 200)  # Random income between 50-200
        
        consumer = FixedConsumerAgent(
            name=name,
            income_stream=income,
            ledger=ledger,
            market=market,
            goods=goods
        )
        consumers.append(consumer)
        logger.info(f"Created {name} with income {income:.2f}")
    
    # Initialize wandb logging
    if args.wandb:
        wandb.init(
            project="llm-economist-marketplace",
            name=args.name if args.name else "marketplace_simulation",
            config=vars(args)
        )
    
    # Run simulation loop
    start_time = time.time()
    
    for timestep in range(args.max_timesteps):
        logger.info(f"TIMESTEP {timestep}")
        print(f"TIMESTEP {timestep}")
        
        wandb_logger = {}
        
        # Supply phase: Firms purchase supplies
        for firm in firms:
            supply_unit_price = 1.0  #! TODO: Make this dynamic
            if args.firm_type == 'LLM':
                firm.purchase_supplies(supply_unit_price, timestep)
            else:
                # Fixed behavior: purchase a fixed amount
                #! TODO: Make fixed purchase_supplies directly follow CES utility (is this possible?)
                quantity = firm.cash * 0.5 / supply_unit_price
                firm.purchase_supplies(quantity, supply_unit_price, timestep)
        
        # Production phase: Firms produce goods
        for firm in firms:
            firm.produce_goods(timestep)
        
        # Pricing phase: Firms set prices
        for firm in firms:
            if args.firm_type == 'LLM':
                prices = firm.set_price(timestep)
            else:
                prices = firm.set_price(price=10.0, timestep=timestep)
            
            # Post quotes to market
            firm.post_quotes(prices)
            logger.info(f"{firm.name} set prices: {prices}")
        
        # Income phase: Consumers receive income
        for consumer in consumers:
            consumer.receive_income(timestep)
        
        # Consumption phase: Consumers make orders
        for consumer in consumers:
            consumer.make_orders(timestep)
        
        # Market clearing: Execute trades
        filled_orders = market.clear(ledger)
        logger.info(f"Filled {len(filled_orders)} orders")
        
        # Clear market for next timestep
        #! TODO: This should probably be done in market.clear()
        market.quotes.clear()
        while market.orders:
            market.orders.popleft()
        
        # Log statistics
        for i, firm in enumerate(firms):
            wandb_logger[f"firm_{i}_cash"] = firm.cash
            for good in goods:
                wandb_logger[f"firm_{i}_{good}_inventory"] = firm.inventory.get(good, 0)
        
        #! TODO: does inventory.get(good, 0) exist?
        for i, consumer in enumerate(consumers):
            wandb_logger[f"consumer_{i}_cash"] = consumer.cash
            for good in goods:
                wandb_logger[f"consumer_{i}_{good}_inventory"] = consumer.inventory.get(good, 0)
        
        if args.wandb:
            wandb.log(wandb_logger)
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"Completed {timestep + 1}/{args.max_timesteps} timesteps in {elapsed:.2f}s")
        logger.info(f"Timestep {timestep + 1}/{args.max_timesteps} completed")
    
    if args.wandb:
        wandb.finish()
    
    logger.info("Marketplace simulation completed successfully!")
    print("Simulation completed!")


def create_argument_parser():
    """Create and return the argument parser for marketplace simulation."""
    parser = argparse.ArgumentParser(description='LLM Economist Marketplace Simulation')
    
    # Agent configuration
    parser.add_argument('--num-firms', type=int, default=3, help='Number of firms in the simulation')
    parser.add_argument('--num-consumers', type=int, default=10, help='Number of consumers in the simulation')
    parser.add_argument('--firm-type', default='FIXED', choices=['LLM', 'FIXED'], help='Type of firm agents')
    
    # Simulation parameters
    parser.add_argument('--max-timesteps', type=int, default=100, help='Maximum number of timesteps')
    parser.add_argument('--firm-initial-cash', type=float, default=1000.0, help='Initial cash for firms')
    
    # LLM configuration (for LLM firm agents)
    parser.add_argument('--llm', default='llama3:8b', type=str, help='Language model to use')
    parser.add_argument('--port', type=int, default=8009, help='Port for LLM service')
    parser.add_argument('--prompt-algo', default='io', choices=['io', 'cot'], help='Prompting algorithm')
    parser.add_argument('--history-len', type=int, default=10, help='Length of history')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout for LLM calls')
    parser.add_argument('--service', default='vllm', choices=['vllm', 'ollama'], help='LLM service backend')
    parser.add_argument('--bracket-setting', default='three', choices=['flat', 'three', 'US_FED'], help='Tax bracket setting (legacy, not used in marketplace)')
    
    # Logging and tracking
    parser.add_argument('--name', type=str, default='', help='Experiment name')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for log files')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args)
    
    logger = logging.getLogger('main')
    logger.info(f"Starting marketplace simulation: {args.name}")
    logger.info(args)
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Run the marketplace simulation
    run_marketplace_simulation(args)


if __name__ == '__main__':
    main()
