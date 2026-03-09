"""
Identify runs where firms were marked out of business then later appeared in business again
("re-entry"). Reads state_t*.json from logs/<run_name>/ and reports specific instances.

Usage (from project root):
  python scripts/analyze_firm_reentry.py [--log-dir logs] [--pattern "*crash*"]
  python scripts/analyze_firm_reentry.py --log-dir "C:/path/to/logs" --pattern "*"
"""

import argparse
import glob
import json
import os


def load_run_states(run_dir: str):
    """Load state dicts from run_dir/state_t*.json, sorted by timestep."""
    pattern = os.path.join(run_dir, "state_t*.json")
    paths = glob.glob(pattern)
    if not paths:
        return []
    def timestep_from_path(p):
        base = os.path.basename(p)
        # state_t42.json -> 42
        s = base.replace("state_t", "").replace(".json", "")
        try:
            return int(s)
        except ValueError:
            return -1
    paths.sort(key=timestep_from_path)
    states = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            states.append(json.load(f))
    return states


def firm_in_business_series(states):
    """
    For each firm, return list of (timestep, in_business) in order.
    Firms are the union of all firm names appearing in any state.
    """
    firm_names = set()
    for s in states:
        for f in s.get("firms", []):
            name = f.get("name")
            if name:
                firm_names.add(name)
    series = {name: [] for name in sorted(firm_names)}
    for s in states:
        t = s.get("timestep", -1)
        by_name = {f["name"]: f.get("in_business", False) for f in s.get("firms", []) if f.get("name")}
        for name in series:
            series[name].append((t, by_name.get(name, False)))
    return series


def find_reentries(series):
    """
    Find re-entries: firm was out (False) at some timestep then in (True) at a later timestep.
    Returns list of dicts: firm_name, timestep_went_out, timestep_came_back, timestep_went_out_again (optional).
    """
    results = []
    for firm_name, points in series.items():
        points = sorted(points, key=lambda x: x[0])
        i = 0
        while i < len(points):
            t, in_biz = points[i]
            if not in_biz:
                # Find next timestep where in_business is True (re-entry)
                j = i + 1
                while j < len(points) and not points[j][1]:
                    j += 1
                if j < len(points):
                    t_back = points[j][0]
                    # Optionally find when they went out again
                    k = j + 1
                    while k < len(points) and points[k][1]:
                        k += 1
                    t_out_again = points[k][0] if k < len(points) else None
                    results.append({
                        "firm_name": firm_name,
                        "timestep_went_out": t,
                        "timestep_came_back": t_back,
                        "timestep_went_out_again": t_out_again,
                    })
                    i = j
                else:
                    i += 1
            else:
                i += 1
    return results


def main():
    parser = argparse.ArgumentParser(description="Find CRASH (or other) runs where firms re-entered after going out of business.")
    parser.add_argument("--log-dir", default="logs", help="Log directory containing run subdirs (default: logs)")
    parser.add_argument("--pattern", default="*crash*", help="Glob pattern for run dir names (default: *crash*)")
    parser.add_argument("--run", default=None, help="Single run name (subdir of log-dir); overrides --pattern")
    args = parser.parse_args()

    if args.run:
        run_dirs = [os.path.join(args.log_dir, args.run)]
        if not os.path.isdir(run_dirs[0]):
            print(f"Run dir not found: {run_dirs[0]}")
            return
    else:
        pattern = os.path.join(args.log_dir, args.pattern)
        run_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
        run_dirs.sort()

    if not run_dirs:
        print(f"No run dirs found under {args.log_dir} matching {args.pattern}")
        return

    total_reentries = 0
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir.rstrip(os.sep))
        states = load_run_states(run_dir)
        if not states:
            continue
        series = firm_in_business_series(states)
        reentries = find_reentries(series)
        if reentries:
            total_reentries += len(reentries)
            print(f"\nRun: {run_name}")
            print(f"  Dir: {run_dir}")
            for r in reentries:
                extra = f" (out again at t={r['timestep_went_out_again']})" if r.get("timestep_went_out_again") is not None else ""
                print(f"  Firm {r['firm_name']}: out at t={r['timestep_went_out']}, back at t={r['timestep_came_back']}{extra}")

    if total_reentries == 0:
        print("No re-entries found (no firm had in_business False then True again in any run).")
    else:
        print(f"\nTotal re-entry instances: {total_reentries}")


if __name__ == "__main__":
    main()
