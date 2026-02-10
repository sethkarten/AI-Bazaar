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
from typing import Dict, Optional
from .market_core.market_core import Ledger, Market
from .agents.firm import FirmAgent, FixedFirmAgent
from .agents.consumer import CESConsumerAgent, FixedConsumerAgent
from .agents.llm_agent import TestAgent
from .env.bazaar_env import BazaarWorld


def setup_logging(args):
    """Setup logging configuration."""
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    log_filename = (
        f"{args.log_dir}/marketplace_{args.name if args.name else 'simulation'}.log"
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def run_marketplace_simulation(args, llm_instance=None):
    """Run the marketplace simulation with firms and consumers."""
    logger = logging.getLogger("main")

    # Test LLM connectivity if using LLM agents and no instance provided
    if args.firm_type == "LLM" and llm_instance is None:
        try:
            TestAgent(args.llm, args.port, args)
            logger.info(f"Successfully connected to LLM: {args.llm}")
        except Exception as e:
            logger.error(f"Failed to connect to LLM: {e}")
            sys.exit(1)

    # Optional: run via BazaarWorld (single source of truth; state files in log_dir)
    if getattr(args, "use_env", False):
        if args.wandb:
            wandb.init(
                project="ai-bazaar-marketplace",
                entity="princeton-ai",
                name=args.name if args.name else "marketplace_simulation",
                config=vars(args),
            )
        world = BazaarWorld(args, llm_model=llm_instance)
        start_time = time.time()
        try:
            while not world.is_done():
                logger.info(f"TIMESTEP {world.timestep}")
                print(f"TIMESTEP {world.timestep}")
                stats = world.step()
                if args.wandb:
                    wandb.log(stats, step=world.timestep - 1)
                elapsed = time.time() - start_time
                print(
                    f"Completed {world.timestep}/{args.max_timesteps} timesteps in {elapsed:.2f}s"
                )
                logger.info(
                    f"Timestep {world.timestep}/{args.max_timesteps} completed"
                )
        except Exception as e:
            logger.error(f"Error in simulation: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if args.wandb:
                wandb.finish()
        logger.info("Marketplace simulation completed successfully!")
        print("Simulation completed!")
        return

    # Initialize market infrastructure
    ledger = Ledger()
    market = Market()

    # Initialize firms
    firms = []

    goods_list = ["food", "clothing", "electronics", "furniture"]
    goods = []
    for i in range(args.num_goods):
        if i >= len(goods_list):
            break
        goods.append(goods_list[i])

    for i in range(args.num_firms):
        name = f"firm_{i}"
        initial_cash = args.firm_initial_cash

        if args.firm_type == "LLM":
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
                args=args,
                llm_instance=llm_instance,
            )
        else:
            firm = FixedFirmAgent(
                name=name,
                goods=goods,
                initial_cash=initial_cash,
                ledger=ledger,
                market=market,
            )
        firms.append(firm)
        logger.info(f"Created {name}")

    # Initialize consumers
    consumers = []

    from ai_bazaar.utils import PERSONAS

    personas = [random.sample(PERSONAS, 1)[0] for _ in range(args.num_consumers)]

    for i in range(args.num_consumers):
        name = f"consumer_{i}"
        income = np.random.uniform(50, 200)
        consumer = None
        if args.consumer_type == "CES":
            consumer = CESConsumerAgent(
                name=name,
                income_stream=income,
                ledger=ledger,
                market=market,
                persona=personas[i],
                ces_params=None,
                risk_aversion=None,
                goods=goods,
                llm=args.llm,
                port=args.port,
                args=args,
                llm_instance=llm_instance,
            )
        else:
            consumer = FixedConsumerAgent(
                name=name,
                income_stream=income,
                ledger=ledger,
                market=market,
                goods=goods,
                quantity_per_good=args.fixed_consumer_quantity_per_good,
            )
        consumers.append(consumer)
        logger.info(
            f"Created {name} with income {income:.2f} and persona {personas[i]}"
        )

    # Set tax rate
    tax_rate = 0.12  # 12% tax rate

    # Initialize wandb logging
    if args.wandb:
        wandb.init(
            project="ai-bazaar-marketplace",
            entity="princeton-ai",
            name=args.name if args.name else "marketplace_simulation",
            config=vars(args),
        )

    # Run simulation loop
    start_time = time.time()
    firm_prices_last_step = {}
    try:
        for timestep in range(args.max_timesteps):
            logger.info(f"TIMESTEP {timestep}")
            print(f"TIMESTEP {timestep}")

            wandb_logger = {}
            start_ledger = ledger.copy()

            # Supply phase: Firms purchase supplies
            for firm in firms:
                if not getattr(firm, "in_business", True):
                    continue
                supply_unit_price = 1.0
                if args.firm_type == "LLM":
                    firm.purchase_supplies(supply_unit_price, timestep)
                else:
                    quantity = firm.cash * 0.5 / supply_unit_price
                    firm.purchase_supplies(quantity, supply_unit_price, timestep)

            # Production phase: Firms produce goods
            for firm in firms:
                if not getattr(firm, "in_business", True):
                    continue
                firm.produce_goods(timestep)

            # Pricing phase: Firms set prices
            firm_prices: Dict[str, Dict[str, float]] = {}
            market_context = {"last_prices": firm_prices_last_step}

            for firm in firms:
                if not getattr(firm, "in_business", True):
                    continue
                if args.firm_type == "LLM":
                    prices = firm.set_price(timestep, market_data=market_context)
                else:
                    prices = firm.set_price(price=10.0, timestep=timestep)

                firm_prices[firm.name] = prices
                firm.post_quotes(prices)
                rep = getattr(firm, "reputation", 1.0)
                logger.info(f"{firm.name} set prices: {prices} reputation={rep:.3f}")

            firm_prices_last_step = firm_prices.copy()

            # Income phase: Consumers receive income
            for consumer in consumers:
                consumer.receive_income(timestep)

            # Consumption phase: Consumers make orders (discovery_limit caps firms polled per good)
            firm_reputations = {
                firm.name: firm.reputation
                for firm in firms
                if getattr(firm, "in_business", True)
            }
            consumer_orders = {}
            for consumer in consumers:
                if args.consumer_type == "CES":
                    orders = consumer.make_orders(
                        timestep,
                        args.consumer_scenario,
                        discovery_limit=args.discovery_limit,
                        firm_reputations=firm_reputations,
                    )
                else:
                    orders = consumer.make_orders(
                        timestep, discovery_limit=args.discovery_limit
                    )
                consumer_orders[consumer.name] = orders

            # Submit orders to market
            for consumer in consumers:
                orders = consumer_orders.get(consumer.name, [])
                if orders:
                    consumer.submit_orders(orders)

            pre_clearing_ledger = ledger.copy()

            # Market clearing: Execute trades
            filled_orders, sales_info = market.clear(ledger)
            logger.info(f"Filled {len(filled_orders)} orders")

            # Update sales tracking and firm reputations (fulfillment rate)
            for sale in sales_info:
                firm_name = sale["firm_id"]
                good = sale["good"]
                quantity_sold = sale["quantity_sold"]
                requested_quantity = sale.get("requested_quantity", quantity_sold)
                for firm in firms:
                    if firm.name == firm_name:
                        if good in firm.total_quantity_sold_by_good:
                            firm.total_quantity_sold_by_good[good] += quantity_sold
                            firm.total_quantity_sold_by_good_this_timestep[timestep][
                                good
                            ] += quantity_sold
                        firm.update_reputation(quantity_sold, requested_quantity)
                        break

            market.quotes.clear()
            while market.orders:
                market.orders.popleft()

            # Overhead phase: Firms pay overhead costs
            for firm in firms:
                if not getattr(firm, "in_business", True):
                    continue
                firm.pay_overhead_costs(timestep)

            # Platform fees (simulating Amazon/eBay)
            total_fees = 0.0
            for firm in firms:
                if not getattr(firm, "in_business", True):
                    continue
                fee = firm.cash * 0.05
                ledger.credit(firm.name, -fee)
                total_fees += fee

            # Log statistics
            for i, firm in enumerate(firms):
                wandb_logger[f"firm_{i}_cash"] = firm.cash
                wandb_logger[f"firm_{i}_reputation"] = getattr(
                    firm, "reputation", 1.0
                )
                for good in goods:
                    wandb_logger[f"firm_{i}_{good}_inventory"] = firm.inventory.get(
                        good, 0
                    )
                    if firm.name in firm_prices:
                        price = firm_prices[firm.name].get(good, 0.0)
                        wandb_logger[f"firm_{i}_{good}_price"] = price
                    wandb_logger[f"firm_{i}_{good}_total_sold"] = (
                        firm.total_quantity_sold_by_good.get(good, 0.0)
                    )

            for i, consumer in enumerate(consumers):
                wandb_logger[f"consumer_{i}_cash"] = consumer.cash
                wandb_logger[f"consumer_{i}_utility"] = consumer.utility
                for good in goods:
                    wandb_logger[f"consumer_{i}_{good}_inventory"] = (
                        consumer.inventory.get(good, 0)
                    )

            if args.wandb:
                wandb.log(wandb_logger)

            # Reflection phase
            for firm in firms:
                if not getattr(firm, "in_business", True):
                    continue
                firm.reflect(timestep, start_ledger, pre_clearing_ledger, ledger)

            for consumer in consumers:
                if hasattr(consumer, "reflect"):
                    consumer.reflect(timestep)

            if not any(getattr(firm, "in_business", True) for firm in firms):
                logger.info("All firms are out of business. Ending simulation early.")
                break

            elapsed = time.time() - start_time
            print(
                f"Completed {timestep + 1}/{args.max_timesteps} timesteps in {elapsed:.2f}s"
            )
            logger.info(f"Timestep {timestep + 1}/{args.max_timesteps} completed")
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if args.wandb:
            wandb.finish()

    logger.info("Marketplace simulation completed successfully!")
    print("Simulation completed!")


def create_argument_parser():
    """Create and return the argument parser for marketplace simulation."""
    parser = argparse.ArgumentParser(description="LLM Economist Marketplace Simulation")

    # Agent configuration
    parser.add_argument(
        "--num-firms", type=int, default=3, help="Number of firms in the simulation"
    )
    parser.add_argument(
        "--num-consumers",
        type=int,
        default=10,
        help="Number of consumers in the simulation",
    )
    parser.add_argument(
        "--firm-type",
        default="FIXED",
        choices=["LLM", "FIXED"],
        help="Type of firm agents",
    )
    parser.add_argument(
        "--unit-cost", type=float, default=2.0, help="Unit cost of production for firms"
    )
    parser.add_argument(
        "--reward-type",
        default="PROFIT",
        choices=["PROFIT", "REVENUE"],
        help="Type of reward signal for firms (PROFIT for bankruptcy avoidance)",
    )
    parser.add_argument(
        "--discovery-limit",
        type=int,
        default=5,
        help="Max firms (per good) a consumer can poll for prices before ordering (0 = no limit)",
    )
    parser.add_argument(
        "--info-asymmetry",
        action="store_true",
        help="Enable information asymmetry (firms see noisy competitor data)",
    )

    # Simulation parameters
    parser.add_argument(
        "--max-timesteps", type=int, default=100, help="Maximum number of timesteps"
    )
    parser.add_argument(
        "--firm-initial-cash", type=float, default=1000.0, help="Initial cash for firms"
    )
    parser.add_argument(
        "--consumer-type",
        default="CES",
        choices=["CES", "FIXED"],
        help="Type of consumer agents",
    )
    parser.add_argument(
        "--consumer-scenario",
        default="RACE_TO_BOTTOM",
        choices=[
            "RACE_TO_BOTTOM",
            "EARLY_BIRD",
            "PRICE_DISCRIMINATION",
            "RATIONAL_BAZAAR",
            "BOUNDED_BAZAAR",
        ],
        help="Consumer scenario",
    )

    # LLM configuration (for LLM firm agents)
    parser.add_argument(
        "--llm", default="llama3:8b", type=str, help="Language model to use"
    )
    parser.add_argument("--port", type=int, default=8009, help="Port for LLM service")
    parser.add_argument(
        "--prompt-algo", default="io", choices=["io", "cot"], help="Prompting algorithm"
    )
    parser.add_argument("--history-len", type=int, default=10, help="Length of history")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for LLM calls")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--service",
        default="vllm",
        choices=["vllm", "ollama"],
        help="LLM service backend",
    )
    parser.add_argument(
        "--bracket-setting",
        default="three",
        choices=["flat", "three", "US_FED"],
        help="Tax bracket setting (legacy, not used in marketplace)",
    )

    # Logging and tracking
    parser.add_argument("--name", type=str, default="", help="Experiment name")
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory for log files"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-firm-prompts",
        action="store_true",
        help="Log full user prompts sent to firm agents",
    )
    parser.add_argument(
        "--num-goods", type=int, default=1, help="Number of goods in the simulation"
    )
    parser.add_argument(
        "--fixed-consumer-quantity-per-good",
        type=float,
        default=10.0,
        help="Quantity of goods to purchase per good for fixed consumer agents",
    )
    parser.add_argument(
        "--use-parsing-agent",
        action="store_true",
        help="Use a parsing agent LLM to clean malformed JSON responses",
    )
    parser.add_argument(
        "--no-diaries",
        action="store_true",
        help="Disable strategic diary entries for agents",
    )
    parser.add_argument(
        "--use-env",
        action="store_true",
        help="Run simulation via BazaarWorld (env) instead of inline loop; state files written to log_dir.",
    )

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    setup_logging(args)
    logger = logging.getLogger("main")
    logger.info(f"Starting marketplace simulation: {args.name}")

    np.random.seed(args.seed)
    random.seed(args.seed)

    run_marketplace_simulation(args)


if __name__ == "__main__":
    main()
