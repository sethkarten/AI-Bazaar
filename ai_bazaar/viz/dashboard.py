import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import glob

st.set_page_config(page_title="Agent Bazaar Dashboard", layout="wide")

st.title("🏛️ Agent Bazaar: Civilization Simulacra Dashboard")

# Load states
log_dir = "logs"
state_files = sorted(
    glob.glob(os.path.join(log_dir, "state_t*.json")),
    key=lambda x: int(os.path.basename(x).replace("state_t", "").replace(".json", "")),
)

if not state_files:
    st.warning("No state files found in logs/ directory. Run a simulation first!")
else:
    timestep = st.slider("Timestep", 0, len(state_files) - 1, len(state_files) - 1)

    with open(state_files[timestep], "r") as f:
        state = json.load(f)

    st.header(f"Timestep {state['timestep']}")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    total_cash = sum(state["ledger"]["money"].values())
    col1.metric("Total Money Supply", f"${total_cash:.2f}")

    # Calculate Gini
    cash_values = sorted(state["ledger"]["money"].values())
    n = len(cash_values)
    index = np.arange(1, n + 1)
    gini = (
        (np.sum((2 * index - n - 1) * cash_values)) / (n * np.sum(cash_values))
        if total_cash > 0
        else 0
    )
    col2.metric("Gini Coefficient", f"{gini:.4f}")

    col3.metric(
        "Firms in Business", len([f for f in state["firms"] if f["in_business"]])
    )
    col4.metric(
        "Avg Consumer Utility",
        f"{pd.DataFrame(state['consumers'])['utility'].mean():.2f}",
    )

    # Tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["💰 Wealth Distribution", "🏢 Firms", "👥 Consumers"])

    with tab1:
        st.subheader("Cash by Agent")
        cash_df = pd.DataFrame(
            state["ledger"]["money"].items(), columns=["Agent", "Cash"]
        )
        st.bar_chart(cash_df, x="Agent", y="Cash")

    with tab2:
        st.subheader("Firm States")
        st.table(pd.DataFrame(state["firms"]))

    with tab3:
        st.subheader("Consumer States")
        cons_df = pd.DataFrame(state["consumers"])
        st.table(cons_df.drop(columns=["diary"]))

        selected_consumer = st.selectbox(
            "Select Consumer for Diary", [c["name"] for c in state["consumers"]]
        )
        for c in state["consumers"]:
            if c["name"] == selected_consumer:
                st.info(f"Last Diary Entry: {c['diary']}")
