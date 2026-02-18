# The Agent Bazaar
## Benchmarking Economic Alignment in High-Frequency Multi-Agent Ecosystems

**Agent Bazaar** is a research framework for evaluating the safety and alignment of autonomous agents in decentralized marketplaces. Unlike prior benchmarks that focus on single-agent business capability (e.g., VendingBench), Agent Bazaar simulates **multi-agent dynamics** where high-frequency interactions lead to emergent systemic failures.

This repository contains the code for the COLM 2026 submission: *"The Agent Bazaar: Benchmarking Economic Alignment in High-Frequency Multi-Agent Ecosystems"*.

### 🚨 Systemic Failure Modes
We model two distinct market structures to test agent alignment:

1.  **The Crash (Amazon/B2C):** A high-frequency pricing game where myopic optimization by agents leads to volatility spirals and market collapse.
2.  **The Lemon Market (eBay/C2C):** A reputation-based market where information asymmetry allows deceptive agents to flood the ecosystem with "Lemons" (low-quality goods), destroying trust.

### 🛡️ Economic Alignment
We propose a **Dual-Alignment Strategy**:
*   **Stable Market Makers:** Firms finetuned to prioritize liquidity and stability over predatory undercutting.
*   **Skeptical Guardians:** Buyer agents finetuned to detect deceptive patterns (Sybil attacks, text-reputation mismatches) in seller networks.

### 📂 Repository Structure
*   `ai_bazaar/`: Core simulation logic.
    *   `env/`: `LemonMarketEnv` (eBay) and `CrashMarketEnv` (Amazon).
    *   `agents/`: `FirmAgent`, `ConsumerAgent`, `LemonSeller`, `LemonBuyer`.
*   `experiments/`: Scripts to run the benchmark scenarios.
    *   `run_lemon_market.py`: Runs the Sybil/Fraud experiment.
*   `paper/`: LaTeX source for the COLM submission.
*   `models/`: Local model weights (Gemma 3, Qwen 3).

### 🚀 Quick Start

1.  **Install Dependencies:**
    ```bash
    uv sync  # Or pip install -r requirements.txt
    ```

2.  **Run the Lemon Market Experiment:**
    ```bash
    PYTHONPATH=. uv run experiments/run_lemon_market.py --num_sellers 4 --num_buyers 10 --max_timesteps 20
    ```

3.  **Build the Paper:**
    ```bash
    cd paper && make
    ```

### CITATION
If you use this codebase, please cite:
```bibtex
@inproceedings{karten2026agentbazaar,
  title={The Agent Bazaar: Benchmarking Economic Alignment in High-Frequency Multi-Agent Ecosystems},
  author={Karten, Seth and Crow, Cameron and Jin, Chi},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```
