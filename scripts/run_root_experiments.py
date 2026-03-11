"""
Run the main CRASH and Lemon Market experiments used in the paper.

Usage (from project root):

    python scripts/run_root_experiments.py

This will run all experiments sequentially with the same arguments as
documented in documentation/RUN_COMMANDS.md.
"""

import subprocess
import sys
from typing import List


def run(cmd: List[str]) -> None:
    print("\n=== Running:", " ".join(cmd), "===\n", flush=True)
    # Propagate exit code if a command fails
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    # THE CRASH experiments
    print("\n########## THE CRASH experiments ##########\n")

    # Ollama 4-parallel (CRASH test)
    run(
        [
            sys.executable,
            "-m",
            "ai_bazaar.main",
            "--name",
            "crash_test_ollama_4parallel",
            "--use-cost-pref-gen",
            "--max-supply-unit-cost",
            "1",
            "--firm-type",
            "LLM",
            "--num-goods",
            "1",
            "--num-firms",
            "2",
            "--consumer-type",
            "CES",
            "--num-consumers",
            "10",
            "--max-timesteps",
            "10",
            "--firm-initial-cash",
            "1000",
            "--consumer-scenario",
            "THE_CRASH",
            "--llm",
            "gemma3:4b",
            "--service",
            "ollama",
            "--port",
            "11434",
            "--discovery-limit-consumers",
            "2",
            "--max-tokens",
            "2000",
            "--prompt-algo",
            "cot",
            "--no-diaries",
            "--seed",
            "8",
        ]
    )

    # Baseline (FIXED firm) – 50 timesteps
    run(
        [
            sys.executable,
            "-m",
            "ai_bazaar.main",
            "--name",
            "crash_baseline_test_1",
            "--use-cost-pref-gen",
            "--max-supply-unit-cost",
            "1",
            "--firm-type",
            "FIXED",
            "--num-firms",
            "5",
            "--consumer-type",
            "CES",
            "--num-consumers",
            "50",
            "--max-timesteps",
            "50",
            "--firm-initial-cash",
            "1000",
            "--consumer-scenario",
            "THE_CRASH",
            "--discovery-limit-consumers",
            "3",
            "--firm-markup",
            "50",
            "--llm",
            "gemma3:4b",
            "--service",
            "ollama",
            "--port",
            "11434",
            "--max-tokens",
            "2000",
            "--prompt-algo",
            "cot",
            "--no-diaries",
            "--seed",
            "8",
        ]
    )

    # THE CRASH – 365 timesteps (no stabilizing firm), three seeds
    for name, seed in [
        ("crash_365_seed8", "8"),
        ("crash_365_seed42", "9"),
        ("crash_365_seed123", "10"),
    ]:
        run(
            [
                sys.executable,
                "-m",
                "ai_bazaar.main",
                "--name",
                name,
                "--use-cost-pref-gen",
                "--max-supply-unit-cost",
                "1",
                "--firm-type",
                "LLM",
                "--num-goods",
                "1",
                "--num-firms",
                "5",
                "--consumer-type",
                "CES",
                "--num-consumers",
                "50",
                "--max-timesteps",
                "365",
                "--firm-initial-cash",
                "5000",
                "--consumer-scenario",
                "THE_CRASH",
                "--discovery-limit-consumers",
                "3",
                "--llm",
                "gemma3:4b",
                "--service",
                "ollama",
                "--port",
                "11434",
                "--max-tokens",
                "2000",
                "--prompt-algo",
                "cot",
                "--no-diaries",
                "--seed",
                seed,
            ]
        )

    # THE CRASH – 365 timesteps with stabilizing firm
    run(
        [
            sys.executable,
            "-m",
            "ai_bazaar.main",
            "--name",
            "crash_stabilizing_365",
            "--use-cost-pref-gen",
            "--max-supply-unit-cost",
            "1",
            "--firm-type",
            "LLM",
            "--num-goods",
            "1",
            "--num-firms",
            "5",
            "--consumer-type",
            "CES",
            "--num-consumers",
            "50",
            "--max-timesteps",
            "365",
            "--firm-initial-cash",
            "5000",
            "--consumer-scenario",
            "THE_CRASH",
            "--discovery-limit-consumers",
            "3",
            "--llm",
            "gemma3:4b",
            "--service",
            "ollama",
            "--port",
            "11434",
            "--max-tokens",
            "2000",
            "--prompt-algo",
            "cot",
            "--no-diaries",
            "--num-stabilizing-firms",
            "1",
            "--seed",
            "8",
        ]
    )

    # Lemon Market experiments
    print("\n########## Lemon Market experiments ##########\n")

    # Lemon market proto-run (short sanity check)
    run(
        [
            sys.executable,
            "-m",
            "ai_bazaar.main",
            "--name",
            "lemon_proto_1",
            "--firm-type",
            "LLM",
            "--num-firms",
            "1",
            "--consumer-type",
            "CES",
            "--num-consumers",
            "5",
            "--max-timesteps",
            "5",
            "--firm-initial-cash",
            "5000",
            "--consumer-scenario",
            "LEMON_MARKET",
            "--llm",
            "gemma3:4b",
            "--service",
            "ollama",
            "--port",
            "11434",
            "--max-tokens",
            "2000",
            "--prompt-algo",
            "cot",
            "--no-diaries",
            "--seed",
            "8",
        ]
    )

    # Lemon market (no Sybil) – base run + two extra seeds
    for name, seed in [
        ("lemon_test_nosybil_1", "8"),
        ("lemon_test_nosybil_1_seed9", "9"),
        ("lemon_test_nosybil_1_seed10", "10"),
    ]:
        run(
            [
                sys.executable,
                "-m",
                "ai_bazaar.main",
                "--name",
                name,
                "--use-cost-pref-gen",
                "--firm-type",
                "LLM",
                "--num-firms",
                "1",
                "--consumer-type",
                "CES",
                "--num-consumers",
                "50",
                "--max-timesteps",
                "10",
                "--firm-initial-cash",
                "5000",
                "--consumer-scenario",
                "LEMON_MARKET",
                "--llm",
                "gemma3:4b",
                "--service",
                "ollama",
                "--port",
                "11434",
                "--max-tokens",
                "2000",
                "--prompt-algo",
                "cot",
                "--no-diaries",
                "--seed",
                seed,
            ]
        )

    # Lemon market with reputation alpha 0.9
    run(
        [
            sys.executable,
            "-m",
            "ai_bazaar.main",
            "--name",
            "lemon_test_nosybil_2",
            "--firm-type",
            "LLM",
            "--num-firms",
            "3",
            "--consumer-type",
            "CES",
            "--num-consumers",
            "10",
            "--max-timesteps",
            "50",
            "--firm-initial-cash",
            "5000",
            "--consumer-scenario",
            "LEMON_MARKET",
            "--reputation-alpha",
            "0.9",
            "---llm",
            "gemma3:4b",
            "--service",
            "ollama",
            "--port",
            "11434",
            "--max-tokens",
            "2000",
            "--prompt-algo",
            "cot",
            "--no-diaries",
            "--seed",
            "8",
        ]
    )

    run(
        [
            sys.executable,
            "-m",
            "ai_bazaar.main",
            "--name",
            "lemon_test_nosybil_3",
            "--firm-type",
            "LLM",
            "--num-firms",
            "10",
            "--consumer-type",
            "CES",
            "--num-consumers",
            "10",
            "--max-timesteps",
            "50",
            "--firm-initial-cash",
            "1000",
            "--consumer-scenario",
            "LEMON_MARKET",
            "--reputation-alpha",
            "0.9",
            "--llm",
            "gemma3:4b",
            "--service",
            "ollama",
            "--port",
            "11434",
            "--max-tokens",
            "2000",
            "--prompt-algo",
            "cot",
            "--no-diaries",
            "--seed",
            "8",
        ]
    )

    # Lemon market with Sybil cluster
    run(
        [
            sys.executable,
            "-m",
            "ai_bazaar.main",
            "--name",
            "lemon_sybil_1",
            "--use-cost-pref-gen",
            "--firm-type",
            "LLM",
            "--num-firms",
            "10",
            "--consumer-type",
            "CES",
            "--num-consumers",
            "50",
            "--max-timesteps",
            "50",
            "--firm-initial-cash",
            "5000",
            "--consumer-scenario",
            "LEMON_MARKET",
            "--sybil-cluster-size",
            "3",
            "--llm",
            "gemma3:4b",
            "--service",
            "ollama",
            "--port",
            "11434",
            "--max-tokens",
            "2000",
            "--prompt-algo",
            "cot",
            "--no-diaries",
            "--seed",
            "8",
        ]
    )

    print("\nAll root experiments completed.")


if __name__ == "__main__":
    main()

