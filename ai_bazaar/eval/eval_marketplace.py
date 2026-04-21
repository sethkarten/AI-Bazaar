import json
import os
import argparse
import numpy as np
from ai_bazaar.env.bazaar_env import BazaarWorld


def run_eval(args):
    results = {}
    scenarios = [
        "RACE_TO_BOTTOM",
        "PRICE_DISCRIMINATION",
        "RATIONAL_BAZAAR",
        "BOUNDED_BAZAAR",
    ]

    for scenario in scenarios:
        print(f"Evaluating scenario: {scenario}")
        args.consumer_scenario = scenario
        world = BazaarWorld(args)

        all_stats = []
        while not world.is_done():
            stats = world.step()
            all_stats.append(stats)

        results[scenario] = {"final_stats": all_stats[-1], "history": all_stats}

    output_file = os.path.join(
        args.log_dir,
        f"eval_results_{args.llm.replace(':', '_').replace('/', '_')}.json",
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    from ai_bazaar.main import create_argument_parser

    parser = create_argument_parser()
    args = parser.parse_args()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    run_eval(args)
