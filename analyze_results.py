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

    print(
        f"Timestep | Active Firms | Avg Utility | Avg Profit | Avg Reputation | Total Cash | Gini"
    )
    print("-" * 90)

    results = []
    for f in state_files:
        with open(f, "r") as f_in:
            state = json.load(f_in)
            t = state["timestep"]

            # Basic metrics
            active_firms = len([fi for fi in state["firms"] if fi["in_business"]])
            avg_utility = np.mean([c["utility"] for c in state["consumers"]])

            # Profit logic
            profits = [
                fi.get("profit", 0.0) for fi in state["firms"] if fi["in_business"]
            ]
            avg_profit = np.mean(profits) if profits else 0.0

            # Reputation logic
            reputations = [
                fi.get("reputation", 1.0) for fi in state["firms"] if fi["in_business"]
            ]
            avg_rep = np.mean(reputations) if reputations else 1.0

            total_cash = sum(state["ledger"]["money"].values())

            # Gini Coefficient Calculation
            cash_values = sorted(state["ledger"]["money"].values())
            n = len(cash_values)
            index = np.arange(1, n + 1)
            gini = (
                (np.sum((2 * index - n - 1) * cash_values)) / (n * np.sum(cash_values))
                if total_cash > 0
                else 0
            )

            print(
                f"{t:8} | {active_firms:12} | {avg_utility:11.2f} | {avg_profit:10.2f} | {avg_rep:14.2f} | {total_cash:10.2f} | {gini:.4f}"
            )
            results.append(state)


if __name__ == "__main__":
    analyze_states()
