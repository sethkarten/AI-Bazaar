#!/usr/bin/env python3
"""
Compile a listing corpus from completed Gemini exp2 lemon market runs.

Walks logs/exp2_gemini-3-flash-preview/ and logs/(OLD) exp2_gemini-3-flash-preview/,
extracts every lemon_market_new_listings entry from states.json, strips run-specific
fields (id, firm_id, reputation, timestep_posted), and saves a flat JSON list to
data/listing_corpus.json indexed by (is_sybil, quality).

Usage (from project root):
    python scripts/compile_listing_corpus.py
    python scripts/compile_listing_corpus.py --out data/listing_corpus.json
    python scripts/compile_listing_corpus.py --log-dirs logs/exp2_gemini-3-flash-preview
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (PROJECT_ROOT / "ai_bazaar").exists():
    PROJECT_ROOT = Path.cwd()

DEFAULT_LOG_DIRS = [
    "logs/exp2_gemini-3-flash-preview",
    "logs/(OLD) exp2_gemini-3-flash-preview",
]
DEFAULT_OUT = "data/listing_corpus.json"


def collect_listings(log_dirs: list[Path]) -> list[dict]:
    """Extract all listings from states.json files across the given log directories."""
    listings: list[dict] = []
    runs_scanned = 0
    runs_skipped = 0

    for log_dir in log_dirs:
        if not log_dir.exists():
            print(f"  [skip] {log_dir} — not found")
            continue
        run_dirs = [d for d in log_dir.iterdir() if d.is_dir()]
        for run_dir in sorted(run_dirs):
            states_file = run_dir / "states.json"
            if not states_file.exists():
                runs_skipped += 1
                continue
            try:
                with open(states_file, encoding="utf-8") as f:
                    states = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [warn] Could not read {states_file}: {e}")
                runs_skipped += 1
                continue

            run_count = 0
            for timestep_state in states:
                for entry in timestep_state.get("lemon_market_new_listings", []):
                    firm_id = entry.get("firm_id", "")
                    is_sybil = "sybil" in firm_id.lower()
                    listing = {
                        "is_sybil": is_sybil,
                        "quality": entry["quality"],
                        "quality_value": entry["quality_value"],
                        "description": entry["description"],
                        "price": entry["price"],
                    }
                    listings.append(listing)
                    run_count += 1

            runs_scanned += 1

    print(f"\nScanned {runs_scanned} run(s), skipped {runs_skipped}.")
    return listings


def print_summary(listings: list[dict]) -> None:
    counts: dict[tuple, int] = defaultdict(int)
    for entry in listings:
        key = (entry["is_sybil"], entry["quality"])
        counts[key] += 1

    print(f"\nTotal listings: {len(listings)}")
    print(f"{'Type':<10} {'Quality':<8} {'Count':>7}")
    print("-" * 28)
    for (is_sybil, quality), count in sorted(counts.items()):
        label = "sybil" if is_sybil else "honest"
        print(f"{label:<10} {quality:<8} {count:>7}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile listing corpus from Gemini exp2 lemon market log files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-dirs", nargs="+", default=None,
        help="Log directories to scan (default: exp2_gemini-3-flash-preview and (OLD) variant).",
    )
    parser.add_argument(
        "--out", type=str, default=DEFAULT_OUT,
        help=f"Output JSON file (default: {DEFAULT_OUT}).",
    )
    args = parser.parse_args()

    log_dirs_raw = args.log_dirs or DEFAULT_LOG_DIRS
    log_dirs = [PROJECT_ROOT / d for d in log_dirs_raw]

    print("Scanning log directories:")
    for d in log_dirs:
        print(f"  {d}")

    listings = collect_listings(log_dirs)

    if not listings:
        print("\nNo listings found. Check that states.json files exist in the log directories.")
        return

    print_summary(listings)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(listings, f, indent=2)
    print(f"\nCorpus written to: {out_path}  ({len(listings)} entries)")


if __name__ == "__main__":
    main()
