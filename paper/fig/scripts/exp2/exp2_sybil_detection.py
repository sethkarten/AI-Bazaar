"""
Fig Exp2-A: Sybil Detection Premium Over Time

Detection premium = honest_hit_rate − sybil_hit_rate, where:
  honest_hit_rate = honest_steps_purchased_total / honest_steps_encountered_total
  sybil_hit_rate  = sybil_steps_purchased_total  / sybil_steps_encountered_total
(per buyer with non-zero denominator; averaged across buyers per timestep)

A positive premium means buyers preferentially purchase from honest sellers.
Near zero = no discrimination; negative = buyers prefer sybil (adversarial).

Two panels side-by-side:
  Left:  reputation visible to buyers (rep1 condition)
  Right: reputation hidden from buyers (rep0 condition)
Lines coloured by K (sybil cluster size); shaded bands = ±1σ across seeds.

Directory naming (from exp2.py):
  logs/{name_prefix}/{name_prefix}_k{k}_rep1_seed{seed}   (rep visible)
  logs/{name_prefix}/{name_prefix}_k{k}_rep0_seed{seed}   (rep hidden)
where name_prefix = exp2_{model_slug}  (e.g. exp2_gemini-2.5-flash).

Usage:
    python paper/fig/scripts/exp2/exp2_sybil_detection.py
    python paper/fig/scripts/exp2/exp2_sybil_detection.py \\
        --logs-dir logs/ --name-prefix exp2_gemini-2.5-flash \\
        --output paper/fig/exp2/exp2_sybil_detection.pdf
    python paper/fig/scripts/exp2/exp2_sybil_detection.py --force
"""

import argparse
import concurrent.futures
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repo root: 5 levels up from paper/fig/scripts/exp2/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
from exp2_cache import get_data_dir, get_cache_path, is_cache_fresh, save_cache, load_cache_data, infer_name_prefix
from exp2_common import (
    SEEDS, K_VALUES, COLORS_K, LS_REP,
    resolve_run_dir, collect_all_run_dirs,
    load_state_files, build_aggregate, serialize_agg, deserialize_agg, plot_band,
)
K_ALL = [0] + K_VALUES

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.5,
    "lines.markersize":   5,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "grid.color":         "gray",
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.01,
    "text.usetex":        False,
    "pdf.fonttype":       42,
})


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def get_detection_premium_series(run_dir: str):
    """Return (ts_array, premium_array) or None.

    Detection premium = mean_over_buyers(honest_purchase_rate - sybil_purchase_rate),
    where purchase_rate = (seen_total - passed_total) / seen_total.
    Only buyers with non-zero seen_total for each type contribute to that term.
    """
    files = load_state_files(run_dir)
    if not files:
        return None

    pts = []
    for p in files:
        with open(p) as f:
            s = json.load(f)
        t = s.get("timestep")
        if t is None:
            continue

        consumers = s.get("consumers", [])
        if not consumers:
            continue

        honest_hits = []
        sybil_hits  = []
        for cdata in consumers:
            if not isinstance(cdata, dict):
                continue
            h_seen   = cdata.get("honest_seen_total",   0) or 0
            h_passed = cdata.get("honest_passed_total", 0) or 0
            s_seen   = cdata.get("sybil_seen_total",    0) or 0
            s_passed = cdata.get("sybil_passed_total",  0) or 0

            if h_seen > 0:
                honest_hits.append((h_seen - h_passed) / h_seen)
            if s_seen > 0:
                sybil_hits.append((s_seen - s_passed) / s_seen)

        if honest_hits and sybil_hits:
            premium = np.mean(honest_hits) - np.mean(sybil_hits)
            pts.append((t, premium))

    if not pts:
        return None
    pts.sort()
    return np.array([x[0] for x in pts]), np.array([x[1] for x in pts])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp2 Fig A: Sybil Detection Premium")
    parser.add_argument("--logs-dir", default="logs/")
    parser.add_argument(
        "--name-prefix", default=None,
        help="Experiment name prefix; auto-detected if omitted.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "exp2", "exp2_sybil_detection.pdf"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Ignore cache and rebuild.")
    parser.add_argument("--good", default="car")  # compatibility with exp2_run_all.py
    cli = parser.parse_args()

    name_prefix = cli.name_prefix or infer_name_prefix(cli.logs_dir)
    print(f"Auto-detected name_prefix: {name_prefix}", flush=True)

    run_dirs   = collect_all_run_dirs(cli.logs_dir, name_prefix, include_baseline=True)
    data_dir   = get_data_dir(cli.output)
    cache_path = get_cache_path(data_dir, "exp2_sybil_detection", cli.good)

    if not cli.force and is_cache_fresh(cache_path, run_dirs, cli.logs_dir, cli.good):
        print(f"Using cached data: {cache_path}", flush=True)
        agg = deserialize_agg(load_cache_data(cache_path)["agg"])
    else:
        jobs = []
        for k in K_ALL:
            for rv in [True, False]:
                for seed in SEEDS:
                    d = resolve_run_dir(cli.logs_dir, name_prefix, k, rv, seed)
                    if d:
                        jobs.append((k, rv, seed, d))
                    else:
                        print(f"  Missing: K={k} rep={int(rv)} seed={seed}", flush=True)

        print(f"Loading {len(jobs)} runs ...", flush=True)
        results: dict = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=cli.workers) as ex:
            future_map = {
                ex.submit(get_detection_premium_series, d): (k, rv, seed)
                for k, rv, seed, d in jobs
            }
            done, total = 0, len(jobs)
            for future in concurrent.futures.as_completed(future_map):
                k, rv, seed = future_map[future]
                done += 1
                data = future.result()
                results[(k, rv, seed)] = data
                print(
                    f"  [{done}/{total}] K={k} rep={int(rv)} seed={seed}"
                    f" — {'ok' if data is not None else 'empty'}",
                    flush=True,
                )

        agg = build_aggregate(results)
        cache_data = {"agg": serialize_agg(agg)}
        save_cache(cache_path, cache_data, cli.logs_dir, cli.good)
        print(f"Cached: {cache_path}", flush=True)
        agg = deserialize_agg(cache_data["agg"])

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), constrained_layout=True, sharey=True)
    fig.suptitle(
        "Do Buyers Discriminate Against Sybil Sellers?",
        fontsize=10, fontweight="bold",
    )

    for rep_visible, ax, title in [
        (True,  axes[0], "(A) Reputation visible"),
        (False, axes[1], "(B) Reputation hidden"),
    ]:
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Timestep")

        # Shaded regions with plain-language meaning
        ax.axhspan(0, 1,  alpha=0.04, color="#009E73", zorder=0)  # green = buyers prefer honest
        ax.axhspan(-1, 0, alpha=0.04, color="#D55E00", zorder=0)  # red   = buyers prefer sybil

        ax.axhline(0.0, color="#555555", lw=1.2, ls="--", alpha=0.8, zorder=2,
                   label="No discrimination ($=0$)")

        # Region labels at left margin
        ax.text(0.01, 0.97, "buyers prefer honest sellers",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5, color="#009E73", style="italic", zorder=5)
        ax.text(0.01, 0.03, "buyers prefer sybil sellers",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.5, color="#D55E00", style="italic", zorder=5)

        for k in K_ALL:
            entry = agg.get((k, rep_visible))
            if entry is None:
                continue
            sat = k / 12
            lbl = f"K={k} ({sat:.0%} sybil)"
            plot_band(ax, entry, COLORS_K[k], lbl)

        ax.legend(loc="center right", fontsize=7.5)

    axes[0].set_ylabel("Honest $-$ sybil purchase rate")
    os.makedirs(os.path.dirname(os.path.abspath(cli.output)), exist_ok=True)
    fig.savefig(cli.output)
    print(f"Saved: {cli.output}")


if __name__ == "__main__":
    main()
