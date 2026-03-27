#!/usr/bin/env python3
"""
Extract sybil principal prompt logs from an Exp2 run.

Given either:
- an Exp2 model directory containing many run subdirectories, or
- a single run directory containing `lemon_agent_prompts.jsonl`, or
- a direct path to `lemon_agent_prompts.jsonl`,
this script filters entries where `agent == "sybil_principal"` and writes them
to `lemon_sybil_prompts.json` in each processed run directory.

It also writes `lemon_sybil_tier_refusals.json`, containing only `sybil_tier`
rows where the model response appears to refuse the task.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def resolve_single_run_paths(
    path_arg: str,
    output_name: str,
    refusals_output_name: str,
) -> tuple[Path, Path, Path]:
    """Resolve input JSONL path and output JSON paths for one run."""
    input_path = Path(path_arg).expanduser().resolve()
    if input_path.is_dir():
        run_dir = input_path
        prompts_jsonl = run_dir / "lemon_agent_prompts.jsonl"
    else:
        prompts_jsonl = input_path
        run_dir = prompts_jsonl.parent

    if not prompts_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {prompts_jsonl}")

    output_path = run_dir / output_name
    refusals_output_path = run_dir / refusals_output_name
    return prompts_jsonl, output_path, refusals_output_path


def discover_run_jsonls(path_arg: str) -> list[Path]:
    """Return one or more lemon_agent_prompts.jsonl paths based on input path."""
    input_path = Path(path_arg).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    # Direct JSONL input
    if input_path.is_file():
        if input_path.name != "lemon_agent_prompts.jsonl":
            raise ValueError(
                "Expected file named lemon_agent_prompts.jsonl when passing a file path."
            )
        return [input_path]

    # Directory input: if this directory itself is a run dir, process it.
    direct_jsonl = input_path / "lemon_agent_prompts.jsonl"
    if direct_jsonl.exists():
        return [direct_jsonl]

    # Otherwise treat as an Exp2 parent directory and process immediate run subdirs.
    run_jsonls: list[Path] = []
    for child in sorted(input_path.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / "lemon_agent_prompts.jsonl"
        if candidate.exists():
            run_jsonls.append(candidate)

    if not run_jsonls:
        raise FileNotFoundError(
            f"No run directories with lemon_agent_prompts.jsonl found in: {input_path}"
        )

    return run_jsonls


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
    """Heuristic detector for refusal-style responses."""
    if response is None:
        return False
    text = str(response).strip()
    if not text:
        return False
    return bool(_REFUSAL_REGEX.search(text))


def extract_sybil_principal_rows(
    prompts_jsonl: Path,
    strict_jsonl: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[int]]:
    """Return sybil rows, refusal subset, and malformed-line diagnostics."""
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
                sybil_rows.append(row)
                if str(row.get("call", "")).strip().lower() == "sybil_tier" and is_refusal_response(row.get("response")):
                    tier_refusals.append(row)

    return sybil_rows, tier_refusals, malformed_count, malformed_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract sybil principal conversations from lemon_agent_prompts.jsonl "
            "and write lemon_sybil_prompts.json in the run directory."
        )
    )
    parser.add_argument(
        "path",
        help=(
            "Path to an Exp2 run directory (containing lemon_agent_prompts.jsonl) "
            "or a direct path to lemon_agent_prompts.jsonl."
        ),
    )
    parser.add_argument(
        "--output",
        default="lemon_sybil_prompts.json",
        help=(
            "Output filename written inside the run directory "
            "(default: lemon_sybil_prompts.json)."
        ),
    )
    parser.add_argument(
        "--refusals-output",
        default="lemon_sybil_tier_refusals.json",
        help=(
            "Refusal subset filename written inside the run directory "
            "(default: lemon_sybil_tier_refusals.json)."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for output file (default: 2).",
    )
    parser.add_argument(
        "--strict-jsonl",
        action="store_true",
        help="Fail immediately on malformed JSONL lines instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts_jsonls = discover_run_jsonls(args.path)
    print(f"Discovered {len(prompts_jsonls)} run(s) to process.")

    total_sybil_rows = 0
    total_tier_refusals = 0
    total_malformed = 0
    for prompts_jsonl in prompts_jsonls:
        _, output_path, refusals_output_path = resolve_single_run_paths(
            str(prompts_jsonl), args.output, args.refusals_output
        )
        sybil_rows, tier_refusals, malformed_count, malformed_lines = extract_sybil_principal_rows(
            prompts_jsonl,
            strict_jsonl=args.strict_jsonl,
        )

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(sybil_rows, f, indent=args.indent, ensure_ascii=True)
            f.write("\n")

        with refusals_output_path.open("w", encoding="utf-8") as f:
            json.dump(tier_refusals, f, indent=args.indent, ensure_ascii=True)
            f.write("\n")

        total_sybil_rows += len(sybil_rows)
        total_tier_refusals += len(tier_refusals)
        total_malformed += malformed_count
        print(
            f"[{prompts_jsonl.parent.name}] sybil={len(sybil_rows)} "
            f"tier_refusals={len(tier_refusals)}"
        )
        if malformed_count:
            preview = ", ".join(str(n) for n in malformed_lines[:5])
            suffix = "..." if malformed_count > 5 else ""
            print(
                f"  warning: skipped {malformed_count} malformed JSONL line(s)"
                f" (e.g., line(s) {preview}{suffix})"
            )
        print(f"  output: {output_path.name}")
        print(f"  refusals output: {refusals_output_path.name}")

    print(
        f"Done. Total sybil prompts: {total_sybil_rows}; "
        f"total sybil_tier refusals: {total_tier_refusals}; "
        f"total malformed lines skipped: {total_malformed}."
    )


if __name__ == "__main__":
    main()
