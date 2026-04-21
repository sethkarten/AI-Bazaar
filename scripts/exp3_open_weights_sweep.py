#!/usr/bin/env python3
"""
Run Experiment 3 across the open-weights model set (via OpenRouter).

By design, the lemon (exp3b) sub-experiment uses a fixed seller/sybil model to
avoid confounding buyer/test-LLM comparisons across the sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = PROJECT_ROOT / "documentation" / "open_weights_models.json"


def _load_models() -> list[tuple[str, float, str]]:
    """Load the open-weights model list from documentation/open_weights_models.json."""
    with open(MODELS_JSON, encoding="utf-8") as f:
        entries = json.load(f)
    return [(e["display_name"], e["params_b"], e["slug"]) for e in entries]


def _matches_any(needles: Iterable[str], haystack: str) -> bool:
    h = haystack.lower()
    return any(n.lower() in h for n in needles)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Experiment 3 across the hardcoded open-weights model list."
    )
    ap.add_argument(
        "--seller-llm",
        type=str,
        required=True,
        help="Fixed seller/sybil model for lemon runs (required).",
    )
    ap.add_argument(
        "--experiment",
        choices=["crash", "lemon", "both"],
        default="both",
        help="Which exp3 sub-experiment to run per model (default: both).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers passed through to scripts/exp3.py (default: 1).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pass through to scripts/exp3.py to skip completed runs.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="SUBSTR",
        help="Only run models whose display name or slug contains any of these substrings.",
    )
    ap.add_argument(
        "--prompt-algo",
        choices=["io", "cot", "sc"],
        default="io",
        help="Prompt algorithm passed through to scripts/exp3.py (default: io).",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Max tokens passed through to scripts/exp3.py (default: 1000).",
    )
    ap.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Optional override passed through to scripts/exp3.py.",
    )
    ap.add_argument(
        "--openrouter-provider",
        type=str,
        nargs="+",
        default=None,
        metavar="PROVIDER",
        help="Preferred OpenRouter provider order (passed through).",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print the per-model commands and exit without executing.",
    )
    cli = ap.parse_args()

    models = _load_models()
    if cli.models:
        models = [
            (disp, params, slug)
            for (disp, params, slug) in models
            if _matches_any(cli.models, f"{disp} {slug}")
        ]

    if not models:
        print("No models matched the provided filter.", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Exp3 open-weights sweep")
    print(f"Models: {len(models)} | experiment={cli.experiment} | seller-llm={cli.seller_llm}")

    for disp, params_b, test_llm in models:
        cmd = [
            sys.executable,
            "scripts/exp3.py",
            "--experiment",
            cli.experiment,
            "--workers",
            str(cli.workers),
            "--test-llm",
            test_llm,
            "--seller-llm",
            cli.seller_llm,
            "--max-tokens",
            str(cli.max_tokens),
            "--prompt-algo",
            cli.prompt_algo,
        ]
        if cli.skip_existing:
            cmd.append("--skip-existing")
        if cli.max_timesteps is not None:
            cmd += ["--max-timesteps", str(cli.max_timesteps)]
        if cli.openrouter_provider:
            cmd += ["--openrouter-provider", *cli.openrouter_provider]

        print(f"\n=== {disp} ({params_b}B) ===")
        print(" ".join(cmd))
        if cli.list:
            continue

        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"FAILED: model={test_llm} exited with code {rc}", file=sys.stderr)
            sys.exit(rc)


if __name__ == "__main__":
    main()

