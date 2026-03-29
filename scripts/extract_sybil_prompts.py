#!/usr/bin/env python3
"""
Extract sybil principal prompt logs from Exp2 runs.

For each (K, rep) cell, compiles rows from all matching seed runs and writes
two files into the model's data/ folder:
  logs/exp2_<model>/data/k{K}_{rep}_sybil_prompts.json
  logs/exp2_<model>/data/k{K}_{rep}_sybil_tier_refusals.json

Each row is stamped with `_seed` and `_run` fields identifying its source.

Usage (from project root):
  python scripts/extract_sybil_prompts.py
  python scripts/extract_sybil_prompts.py --model gemini-2.5-flash
  python scripts/extract_sybil_prompts.py --model gemini-2.5-flash --k 3 6 9
  python scripts/extract_sybil_prompts.py --model gemini-2.5-flash --rep rep1
  python scripts/extract_sybil_prompts.py --model gemini-2.5-flash --seed 8 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "anthropic_claude-sonnet-4.6"

_SEED_RE = re.compile(r"_seed(\d+)")


def discover_all_runs(
    model_dir: Path,
    k: int,
    rep: str,
    seed_filter: list[int] | None = None,
) -> list[tuple[int, Path]]:
    """
    Return all (seed, lemon_agent_prompts.jsonl) pairs for a given (K, rep) cell.
    Results are sorted by seed. seed_filter restricts to specific seeds if provided.
    """
    if not model_dir.is_dir():
        return []
    results: list[tuple[int, Path]] = []
    for child in sorted(model_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if f"_k{k}_" not in name or f"_{rep}_" not in name:
            continue
        m = _SEED_RE.search(name)
        if not m:
            continue
        seed = int(m.group(1))
        if seed_filter is not None and seed not in seed_filter:
            continue
        candidate = child / "lemon_agent_prompts.jsonl"
        if candidate.exists():
            results.append((seed, candidate))
    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS = [
    r"\bi won't\b",
    r"\bi will not\b",
    r"\bwon't help\b",
    r"\brefus(e|ing|al)\b",
    r"\bcan't help\b",
    r"\bcannot help\b",
    r"\bnot going to help\b",
    r"\bfraud\b",
    r"\bdeceiv(e|ing|e buyers|e consumers)\b",
]
_REFUSAL_REGEX = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal_response(response: Any) -> bool:
    if response is None:
        return False
    text = str(response).strip()
    return bool(_REFUSAL_REGEX.search(text)) if text else False


def extract_sybil_principal_rows(
    prompts_jsonl: Path,
    seed: int,
    strict_jsonl: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[int]]:
    """
    Return (sybil_rows, tier_refusals, malformed_count, malformed_lines).
    Each row is stamped with `_seed` and `_run` for traceability.
    """
    run_name = prompts_jsonl.parent.name
    sybil_rows: list[dict[str, Any]] = []
    tier_refusals: list[dict[str, Any]] = []
    malformed_count = 0
    malformed_lines: list[int] = []
    with prompts_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                if strict_jsonl:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} in {prompts_jsonl}: {exc}"
                    ) from exc
                malformed_count += 1
                malformed_lines.append(line_no)
                continue
            if str(row.get("agent", "")).strip().lower() == "sybil_principal":
                row["_seed"] = seed
                row["_run"] = run_name
                sybil_rows.append(row)
                if (
                    str(row.get("call", "")).strip().lower() == "sybil_tier"
                    and is_refusal_response(row.get("response"))
                ):
                    tier_refusals.append(row)
    return sybil_rows, tier_refusals, malformed_count, malformed_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract sybil principal conversations, compiled across seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model slug under logs/exp2_<model>/ (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[0, 3, 6, 9], metavar="K",
        help="K values to process (default: 0 3 6 9). E.g. --k 3 6 9",
    )
    parser.add_argument(
        "--rep", type=str, nargs="+", default=["rep0", "rep1"],
        choices=["rep0", "rep1"], metavar="REP",
        help="Rep conditions to process (default: rep0 rep1).",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=None, metavar="N",
        help="Restrict to these seeds (default: all found). E.g. --seed 8 16",
    )
    parser.add_argument(
        "--indent", type=int, default=2,
        help="JSON indentation level for output files (default: 2).",
    )
    parser.add_argument(
        "--strict-jsonl", action="store_true",
        help="Fail immediately on malformed JSONL lines instead of skipping them.",
    )
    cli = parser.parse_args()

    k_values = sorted(set(cli.k))
    rep_values = sorted(set(cli.rep))
    model_dir = PROJECT_ROOT / "logs" / f"exp2_{cli.model}"
    data_dir = model_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model  : {cli.model}")
    print(f"K      : {k_values}")
    print(f"Rep    : {rep_values}")
    print(f"Output : {data_dir}")

    total_sybil = 0
    total_refusals = 0
    total_malformed = 0
    cells_written = 0

    for K in k_values:
        for rep in rep_values:
            seed_runs = discover_all_runs(model_dir, k=K, rep=rep, seed_filter=cli.seed)
            if not seed_runs:
                print(f"  [k={K} {rep}] no runs with prompt logs found -- skipping",
                      file=sys.stderr)
                continue

            cell_sybil: list[dict[str, Any]] = []
            cell_refusals: list[dict[str, Any]] = []
            cell_malformed = 0
            seed_labels: list[str] = []

            for seed, jsonl_path in seed_runs:
                sybil_rows, tier_refusals, malformed_count, malformed_lines = \
                    extract_sybil_principal_rows(jsonl_path, seed, cli.strict_jsonl)
                cell_sybil.extend(sybil_rows)
                cell_refusals.extend(tier_refusals)
                cell_malformed += malformed_count
                seed_labels.append(f"seed{seed}({len(sybil_rows)})")
                if malformed_count:
                    preview = ", ".join(str(n) for n in malformed_lines[:5])
                    suffix = "..." if malformed_count > 5 else ""
                    print(f"  [k={K} {rep} seed={seed}] skipped {malformed_count} "
                          f"malformed line(s) (e.g. {preview}{suffix})", file=sys.stderr)

            out_prompts = data_dir / f"k{K}_{rep}_sybil_prompts.json"
            out_refusals = data_dir / f"k{K}_{rep}_sybil_tier_refusals.json"

            with out_prompts.open("w", encoding="utf-8") as f:
                json.dump(cell_sybil, f, indent=cli.indent, ensure_ascii=True)
                f.write("\n")
            with out_refusals.open("w", encoding="utf-8") as f:
                json.dump(cell_refusals, f, indent=cli.indent, ensure_ascii=True)
                f.write("\n")

            total_sybil += len(cell_sybil)
            total_refusals += len(cell_refusals)
            total_malformed += cell_malformed
            cells_written += 1

            seeds_summary = "  ".join(seed_labels)
            print(
                f"  [k={K} {rep}] {len(seed_runs)} seed(s): {seeds_summary}"
                f"  -> total sybil={len(cell_sybil)}  refusals={len(cell_refusals)}"
            )

    if cells_written == 0:
        print("ERROR: no runs found. Check --model, --k, and --rep values.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Done. {cells_written} cell(s) written. "
        f"Total sybil rows: {total_sybil}  "
        f"tier refusals: {total_refusals}"
        + (f"  malformed lines skipped: {total_malformed}" if total_malformed else "")
    )


if __name__ == "__main__":
    main()
