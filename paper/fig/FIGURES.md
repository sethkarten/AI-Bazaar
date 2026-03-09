# Figure Generation Scripts

All scripts live in `paper/fig/scripts/` and write PDFs to `paper/fig/` by default.
Run everything from the **repo root** so that the `ai_bazaar` package is importable without any `PYTHONPATH` changes.

---

## Prerequisites

**Python packages** (same environment used to run simulations):

```
matplotlib
numpy
pandas
```

The scripts import `ai_bazaar.utils.dataframe_builder`, so the repo root must be on
the Python path. Running from the repo root with `python paper/fig/scripts/<script>.py`
handles this automatically (each script inserts the repo root into `sys.path`).

**Data** — each script reads `state_t*.json` files from one or more run directories.
These are the per-timestep snapshots written by the simulator into `logs/<run_name>/`.
The lemon-market scripts additionally read `firm_attributes.json` from each run
directory to distinguish honest from sybil firms.

---

## Script Reference

| Script | Figure | Required args | Optional args |
|--------|--------|---------------|---------------|
| `crash_price_cascade.py` | **C1** — Per-firm price trajectories + active-firm count (two panels). Bankruptcy moments marked with dotted verticals. | `--run-dirs DIR [DIR ...]` | `--good NAME` (auto-detected), `--output PATH` |
| `crash_survival_curve.py` | **C2** — Kaplan-Meier-style fraction of firms surviving over time, with shaded ±std band. | at least one of `--baseline-dirs` or `--stabilizing-dirs` | `--output PATH` |
| `crash_intervention.py` | **C3** — Three-panel comparison (avg price / active firms / mean consumer surplus) for baseline vs stabilizing-firm condition. | at least one of `--baseline-dirs` or `--stabilizing-dirs` | `--good NAME`, `--output PATH` |
| `crash_welfare.py` | **C4** — Two panels: mean consumer surplus and firm-cash Gini coefficient over time. | `--run-dirs DIR [DIR ...]` | `--output PATH` |
| `lemon_market_freeze.py` | **L1** — Listings / Bids / Passes counts over time; annotates the Passes > Bids crossover. | `--run-dirs DIR [DIR ...]` | `--output PATH` |
| `lemon_reputation_quality.py` | **L2** — Reputation trajectories (top) and avg price (bottom) split by honest vs sybil firm type. Reads `firm_attributes.json`. | `--run-dirs DIR [DIR ...]` | `--output PATH` |
| `lemon_guardian_effect.py` | **L3** — Three-panel comparison (market trust ratio / active honest firms / consumer surplus) for baseline vs guardian condition. | at least one of `--baseline-dirs` or `--guardian-dirs` | `--output PATH` |
| `lemon_consumer_welfare.py` | **L4** — Stacked area of consumer utility components; sybil-sale timesteps marked as axis ticks. Reads `firm_attributes.json`. | `--run-dirs DIR [DIR ...]` | `--output PATH` |
| `welfare_summary.py` | **S1** — Grouped bar chart of final-timestep mean consumer surplus across all four conditions. | at least one condition flag (see below) | `--output PATH` |

---

## Usage by Scenario

All examples below assume the repo root as the working directory.

### The Crash (C1–C4)

The crash logs are in `logs/crash_365_seed42`, `logs/crash_365_seed123`, and
`logs/crash_365_seed8`. Use these for real multi-seed runs.

**C1 — Price cascade, single run:**

```bash
python paper/fig/scripts/crash_price_cascade.py \
    --run-dirs logs/crash_365_seed42
```

**C1 — Price cascade, multi-seed (trajectories overlaid):**

```bash
python paper/fig/scripts/crash_price_cascade.py \
    --run-dirs logs/crash_365_seed42 logs/crash_365_seed123 logs/crash_365_seed8
```

**C2 — Survival curve, baseline vs stabilizing-firm condition:**

```bash
python paper/fig/scripts/crash_survival_curve.py \
    --baseline-dirs     logs/crash_365_seed42 logs/crash_365_seed123 logs/crash_365_seed8 \
    --stabilizing-dirs  logs/crash_stab_seed42 logs/crash_stab_seed123
```

Pass only `--baseline-dirs` or only `--stabilizing-dirs` to plot a single condition.

**C3 — Intervention comparison (3-panel):**

```bash
python paper/fig/scripts/crash_intervention.py \
    --baseline-dirs     logs/crash_365_seed42 logs/crash_365_seed123 logs/crash_365_seed8 \
    --stabilizing-dirs  logs/crash_stab_seed42 logs/crash_stab_seed123 \
    --good food
```

`--good` is auto-detected from the first valid run directory if omitted.

**C4 — Welfare cost (consumer surplus + Gini), multi-seed:**

```bash
python paper/fig/scripts/crash_welfare.py \
    --run-dirs logs/crash_365_seed42 logs/crash_365_seed123 logs/crash_365_seed8
```

---

### The Lemon Market (L1–L4)

The lemon logs are in `logs/lemon_50_flash_nosybil_1` (no-sybil baseline) and
`logs/lemon_50_flash_sybil_1` (with sybil firms). Earlier prototypes include
`logs/lemon_proto_1` and `logs/lemon_test_nosybil_1`.

**L1 — Market freeze (listings/bids/passes), single run:**

```bash
python paper/fig/scripts/lemon_market_freeze.py \
    --run-dirs logs/lemon_50_flash_sybil_1
```

**L1 — Multi-seed:**

```bash
python paper/fig/scripts/lemon_market_freeze.py \
    --run-dirs logs/lemon_50_flash_sybil_1 logs/lemon_seed123 logs/lemon_seed8
```

**L2 — Reputation by firm type (requires `firm_attributes.json`):**

```bash
python paper/fig/scripts/lemon_reputation_quality.py \
    --run-dirs logs/lemon_50_flash_sybil_1
```

If `firm_attributes.json` is absent, all firms are treated as honest.

**L3 — Guardian intervention effect (3-panel):**

```bash
python paper/fig/scripts/lemon_guardian_effect.py \
    --baseline-dirs  logs/lemon_50_flash_sybil_1 \
    --guardian-dirs  logs/lemon_guardian_seed42 logs/lemon_guardian_seed123
```

**L4 — Consumer welfare harm (stacked area + sybil-sale markers):**

```bash
python paper/fig/scripts/lemon_consumer_welfare.py \
    --run-dirs logs/lemon_50_flash_sybil_1
```

---

### Welfare Summary (S1)

Aggregates the final-timestep consumer surplus from all four conditions into a
single grouped bar chart. Supply as many seeds as available for each condition.

```bash
python paper/fig/scripts/welfare_summary.py \
    --crash-baseline-dirs    logs/crash_365_seed42 logs/crash_365_seed123 logs/crash_365_seed8 \
    --crash-stabilizing-dirs logs/crash_stab_seed42 logs/crash_stab_seed123 \
    --lemon-baseline-dirs    logs/lemon_50_flash_nosybil_1 \
    --lemon-guardian-dirs    logs/lemon_guardian_seed42 logs/lemon_guardian_seed123
```

Any condition flag can be omitted; its bar will show `n=0` with height 0.

---

## Output

By default each script saves its PDF to `paper/fig/<figure_name>.pdf` (one level
above `scripts/`). Override with `--output`:

```bash
python paper/fig/scripts/crash_price_cascade.py \
    --run-dirs logs/crash_365_seed42 \
    --output /tmp/fig_c1_draft.pdf
```

The output directory is created automatically if it does not exist.

---

## Data Requirements

| File | Required by |
|------|-------------|
| `logs/<run>/state_t*.json` | All scripts — the per-timestep simulation snapshots |
| `logs/<run>/firm_attributes.json` | `lemon_reputation_quality.py`, `lemon_guardian_effect.py`, `lemon_consumer_welfare.py` — used to label firms as honest or sybil; scripts fall back gracefully if absent |

A run directory must contain at least one `state_t*.json` file to be processed.
Scripts print a warning to stderr and skip any directory that yields no files.
When multiple run directories are passed, series are aligned to the shortest run
before averaging (mean ± std across seeds).
