"""
Main entry point for the LLM Economist marketplace simulation.

This simulation consists of:
- FirmAgent: Firms that produce goods and set prices
- ConsumerAgent: Consumers that purchase goods
"""

import argparse
import json
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
from .utils.common import LEMON_MARKET_GOODS


def setup_logging(args):
    """Setup logging configuration."""
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    log_filename = (
        f"{args.log_dir}/marketplace_{args.name if args.name else 'simulation'}.log"
    )
    if os.path.exists(log_filename):
        open(log_filename, "w").close()

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

    # Collect token usage from all agents
    all_agents = list(world.firms) + list(world.consumers)
    total = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
    for agent in all_agents:
        if hasattr(agent, "token_usage"):
            for k in total:
                total[k] += agent.token_usage[k]

    print(f"\n=== Token Usage ===")
    print(f"  Input tokens:  {total['input_tokens']:,}")
    print(f"  Output tokens: {total['output_tokens']:,}")
    print(f"  Requests:      {total['requests']:,}")

    run_name = getattr(args, "name", None) or "simulation"
    run_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    usage_path = os.path.join(run_dir, f"{run_name}_token_usage.json")
    with open(usage_path, "w") as f:
        json.dump(total, f, indent=2)
    print(f"Token usage saved to {usage_path}")

    logger.info("Marketplace simulation completed successfully!")
    print("Simulation completed!")
    return


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
        "--max-supply-unit-cost",
        type=float,
        default=10.0,
        help="Max supply unit cost per good (firm-specific costs are uniform in [1.0, max])",
    )
    parser.add_argument(
        "--reward-type",
        default="PROFIT",
        choices=["PROFIT", "REVENUE"],
        help="Type of reward signal for firms (PROFIT for bankruptcy avoidance)",
    )
    parser.add_argument(
        "--discovery-limit-consumers",
        type=int,
        default=5,
        help="Max firms (per good) a consumer can poll for prices before ordering (0 = no limit)",
    )
    parser.add_argument(
        "--discovery-limit-firms",
        type=int,
        default=0,
        help="Max competitor firms whose prices each firm sees when setting prices (0 = no limit). Applies with and without --info-asymmetry.",
    )
    parser.add_argument(
        "--poisson-demand-lambda",
        type=float,
        default=None,
        help="If set, each timestep the number of consumers who participate is min(Poisson(lambda), num_consumers). If None, all consumers participate (except THE_CRASH, which defaults lambda to 0.6 * num_consumers).",
    )
    parser.add_argument(
        "--info-asymmetry",
        action="store_true",
        help="Enable information asymmetry (firms see noisy competitor data)",
    )
    parser.add_argument(
        "--num-stabilizing-firms",
        type=int,
        default=0,
        help="Number of firms that use the Stabilizing Firm prompt and enforce price floor >= unit cost (B2C Crash). First N LLM firms are stabilizing.",
    )
    parser.add_argument(
        "--crash-rep-scoring",
        action="store_true",
        help="In THE_CRASH, score quotes by reputation/price (instead of 1/price) when choosing which firm to order from among the dlc visible quotes.",
    )
    parser.add_argument(
        "--log-alignment-traces",
        action="store_true",
        help="Log (state, prompt, response, outcome) per step to run_dir/alignment_traces.jsonl for SFT.",
    )

    # Simulation parameters
    parser.add_argument(
        "--max-timesteps", type=int, default=100, help="Maximum number of timesteps"
    )
    parser.add_argument(
        "--firm-initial-cash", type=float, default=500.0, help="Initial cash for firms"
    )
    
    parser.add_argument(
        "--firm-markup",
        type=float,
        default=0.50,
        help="FixedFirmAgent: markup over unit cost (price = unit_cost + markup); default 0.50",
    )
    parser.add_argument(
        "--firm-tax-rate",
        type=float,
        default=0.05,
        help="Tax rate on firm cash (all firms); default 0.05 (5%%)",
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
            "THE_CRASH",
            "LEMON_MARKET",
        ],
        help="Consumer scenario (THE_CRASH forces use of eWTP)",
    )
    parser.add_argument(
        "--reputation-alpha",
        type=float,
        default=0.9,
        help="LEMON_MARKET: reputation update smoothing; R_new = alpha*R_old + (1-alpha)*q; default 0.9",
    )
    parser.add_argument(
        "--sybil-cluster-size",
        type=int,
        default=0,
        help="LEMON_MARKET: number of firms that are Sybil identities (one cluster). Last K of num_firms. 0 = no Sybil.",
    )
    parser.add_argument(
        "--reputation-initial",
        type=float,
        default=None,
        help="LEMON_MARKET: initial reputation R_0 for all sellers and for Sybil rotation. Default 0.8 when LEMON_MARKET and sybil-cluster-size>0 else 1.0.",
    )
    parser.add_argument(
        "--sybil-rho-min",
        type=float,
        default=0.3,
        help="LEMON_MARKET: Sybil identity rotation threshold; when R < rho_min, reset to reputation-initial. Default 0.3",
    )
    parser.add_argument(
        "--consumption-interval",
        type=int,
        default=1,
        help="Run consumer inventory consumption (zero goods, keep cash) every N timesteps; 1 = every timestep (default)",
    )
    parser.add_argument(
        "--use-gen-ces",
        action="store_true",
        help="Generate CES parameters via LLM for consumers (otherwise use passed-in/default ces_params)",
    )
    parser.add_argument(
        "--use-cost-pref-gen",
        action="store_true",
        help="Use heterogeneity module to generate supply_unit_costs per firm and CES preferences per consumer; if disabled, firms get random supply costs and consumers share necessity-weights.",
    )

    # LLM configuration (for LLM firm agents)
    parser.add_argument(
        "--llm", default="llama3:8b", type=str,
        help="Language model (e.g. llama3:8b, gemini-2.5-flash, gemini-3-flash-preview, gpt-4o)",
    )
    parser.add_argument("--port", type=int, default=8009, help="Port for LLM service")
    parser.add_argument(
        "--gemini-backend", default=None, choices=["studio", "vertex"],
        dest="gemini_backend",
        help="Gemini backend: 'studio' (API key) or 'vertex' (Vertex AI). "
             "Defaults to 'studio' if GOOGLE_API_KEY/GEMINI_API_KEY is set, else 'vertex'.",
    )
    parser.add_argument(
        "--prompt-algo", default="io", choices=["io", "cot"], help="Prompting algorithm"
    )
    parser.add_argument("--history-len", type=int, default=3, help="Length of history window (timesteps) sent to each LLM firm")
    parser.add_argument("--best-n", type=int, default=3, help="Best-N slab size for stabilizing firms (0 to disable)")
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
        "--disable-firm-personas",
        action="store_true",
        help="Disable heterogeneous firm personas; all LLM firms get no behavioral archetype in their prompt.",
    )
    parser.add_argument(
        "--firm-personas",
        type=str,
        default=None,
        help="Comma-separated persona:count pairs for non-stabilizing firms (e.g. competitive:3,volume_seeker:2). If omitted, all non-stabilizing firms use competitive. Names get a numeric suffix when count>1 (e.g. competitive_1, competitive_2). Valid personas: competitive, volume_seeker, reactive, cautious.",
    )
    parser.add_argument(
        "--wtp-algo",
        type=str,
        default="none",
        choices=["none", "wtp", "ewtp"],
        help="WTP algorithm when making orders: 'none' = ignore WTP (always order if a quote is chosen); 'wtp' = use CES willingness-to-pay; 'ewtp' = use expected WTP (eWTP).",
    )
    parser.add_argument(
        "--dynamic-labor",
        action="store_true",
        help="Let CES consumers re-choose labor each timestep; if disabled, labor is chosen only at t=0 and held fixed.",
    )
    parser.add_argument(
        "--overhead-costs",
        type=float,
        default=50.0,
        help="Overhead costs per week for firms; default 50.0",
    )

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # LEMON_MARKET: force num_goods to 1 and only good is "car"
    if getattr(args, "consumer_scenario", None) == "LEMON_MARKET":
        args.num_goods = 1
        if getattr(args, "reputation_initial", None) is None:
            args.reputation_initial = 0.8  # paper R_0 = 0.8 for lemon experiments
        if getattr(args, "sybil_cluster_size", 0) > 0 and args.num_firms < args.sybil_cluster_size:
            args.num_firms = args.sybil_cluster_size

    # THE_CRASH: default Poisson demand lambda to 60% of num_consumers unless explicitly set
    if getattr(args, "consumer_scenario", None) == "THE_CRASH" and getattr(args, "poisson_demand_lambda", None) is None:
        args.poisson_demand_lambda = 0.6 * args.num_consumers

    setup_logging(args)
    logger = logging.getLogger("main")
    logger.info(f"Starting marketplace simulation: {args.name}")

    np.random.seed(args.seed)
    random.seed(args.seed)

    run_marketplace_simulation(args)


if __name__ == "__main__":
    main()
