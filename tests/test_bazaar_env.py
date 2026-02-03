"""
Test script for running a short simulation using BazaarWorld (bazaar_env).

Run from project root:
  python -m pytest tests/test_bazaar_env.py -v -s
  # or run the script directly (no pytest):
  python tests/test_bazaar_env.py
"""
import os
import sys

# Ensure project root is on path when run as script
if __name__ == "__main__" and __package__ is None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

from ai_bazaar.main import create_argument_parser
from ai_bazaar.env.bazaar_env import BazaarWorld


def get_bazaar_env_test_args():
    """Build args for: 1 Gemini firm, 3 CES consumers, discovery=0, 5 steps, CoT, RACE_TO_BOTTOM, no wandb."""
    parser = create_argument_parser()
    argv = [
        "--firm-type", "LLM",
        "--num-firms", "1",
        "--num-consumers", "3",
        "--discovery-limit", "0",
        "--max-timesteps", "5",
        "--prompt-algo", "cot",
        "--consumer-scenario", "RACE_TO_BOTTOM",
        "--llm", "gemini-2.5-flash",
        # no --wandb
    ]
    return parser.parse_args(argv)


def run_bazaar_env_simulation():
    """Run a short BazaarWorld simulation with the test config."""
    args = get_bazaar_env_test_args()
    assert args.firm_type == "LLM"
    assert args.num_firms == 1
    assert args.num_consumers == 3
    assert args.discovery_limit == 0
    assert args.max_timesteps == 5
    assert args.prompt_algo == "cot"
    assert args.consumer_scenario == "RACE_TO_BOTTOM"
    assert args.llm == "gemini-2.5-flash"
    assert getattr(args, "wandb", False) is False

    os.makedirs(args.log_dir, exist_ok=True)

    print("Creating BazaarWorld: 1 Gemini firm, 3 CES consumers, discovery=0, 5 steps, CoT, RACE_TO_BOTTOM")
    world = BazaarWorld(args, llm_model=None)

    step_count = 0
    while not world.is_done():
        stats = world.step()
        step_count += 1
        print(f"  Step {step_count}: sales={stats['sales_count']}, fees={stats['total_fees']:.2f}")

    print(f"Done. Ran {step_count} steps.")
    return world, step_count


def test_bazaar_env_run():
    """Pytest entry: run the bazaar_env simulation and assert it completes."""
    world, step_count = run_bazaar_env_simulation()
    assert step_count <= 5
    assert step_count >= 1
    assert len(world.firms) == 1
    assert len(world.consumers) == 3


if __name__ == "__main__":
    run_bazaar_env_simulation()
