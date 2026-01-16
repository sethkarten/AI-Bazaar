import json
import glob
import os
import numpy as np


def analyze_states(log_dir="logs"):
    state_files = sorted(
        glob.glob(os.path.join(log_dir, "state_t*.json")),
        key=lambda x: int(x.split("_t")[-1].split(".")[0]),
    )
    if not state_files:
        print("No state files found.")
        return

    results = []
    for f in state_files:
        with open(f, "r") as f_in:
            state = json.load(f_in)
            t = state["timestep"]
            avg_utility = np.mean([c["utility"] for c in state["consumers"]])
            avg_profit = np.mean([f["profit"] for f in state["firms"] if "profit" in f])
            total_cash = sum(state["ledger"]["money"].values())
            results.append((t, avg_utility, avg_profit, total_cash))

    print("Timestep | Avg Utility | Avg Profit | Total Cash")
    print("-" * 50)
    for r in results:
        print(f"{r[0]:8} | {r[1]:11.2f} | {r[2]:10.2f} | {r[3]:10.2f}")


if __name__ == "__main__":
    analyze_states()
