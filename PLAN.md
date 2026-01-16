# Agent Bazaar: Agentic Marketplace Dynamics (COLM 2026)

This document outlines the roadmap for the Agent Bazaar framework, focusing on firm and consumer behavior in digital marketplaces like Amazon and eBay.

## Phase 1: Marketplace Integration (Refocused)
- [x] **Store Operations**: 
    - Firms manage inventory, pricing, and cash flow to avoid bankruptcy.
    - Implement platform fees (Amazon/eBay style) as a cost of business.
- [x] **Realistic Demand**:
    - Consumers choose labor hours to generate income and optimize utility over multiple goods.
    - Necessities (Food) prioritized over luxury items.
- [ ] **Phase 1.5: Advanced Platform Features**:
    - **Discovery Friction**: Implement "Discovery" where consumers only see a subset (e.g., top 5) of firms.
    - **Recommendation Algorithm**: Lightweight ranking based on price and reputation.
    - **Reputation System**: Track firm fulfillment rates and historical "quality" ratings.
    - **Information Asymmetry**: Firms don't know the exact demand curves or competitor costs.

## Phase 2: Agentic Social Layer (Interviews + Reflections)
- [x] **Agent Interview API**:
    - Create a wrapper to query agent internal state/reasoning in character.
- [x] **Strategic Reflection**:
    - "Diary Entries" for agents to store reasoning and retrospective analysis.
    - Reflections used as in-context memory.
- [x] **State Serialization**:
    - Full state snapshotting verified.

## Phase 3: Model Training & Evaluation
- [x] **Baseline Benchmarking**:
    - Evaluate Gemma 3-4B and Llama 3.1 8B (In-Context) on local 2x5090.
- [ ] **Phase 3.5: Comparative Ablations**:
    - **Reward Signal**: Compare "Revenue Maximization" vs. "Bankruptcy Avoidance" (Profit).
    - **Reasoning Impact**: Ablation test of performance with and without "Diary Entries."
    - **Model Comparison**: Benchmark Gemma 3 4B, Qwen 3 (7B), Ministral 3 (8B), and OLMo 3 (7B).
- [x] **REINFORCE++ Finetuning**:
    - Integrated `unsloth` for LoRA-based training.
    - Implemented `UnslothModel` for fast in-process trajectory generation.
    - Resolved Gemma 3 processor subscriptable errors.
    - Optimized speed via **Parallel Agent Actions** and **vLLM Batching**.
    - Launched finetuning on `della-ailab` cluster (Job ID: 3887442).
    - **Current Progress**: Iteration 34/100 (Jan 16).
- [x] **Reward Modeling**:
    - Linked Individual Utility and Social Welfare to the RL reward signal.
- [x] **Environment Robustness**:
    - Configured `uv` environment with CUDA 12.8 support on cluster.
    - Implemented heartbeat monitoring and ETA timers.
    - **Monitoring Timer Hooks**: Set up internal `_monitor_loop` thread and external `training_watcher.py` to detect job hangs and provide progress updates.

## Phase 4: Visualization & UX
- [x] **Interactive Dashboard**:
    - Build Streamlit UI for real-time visualization of Gini, wealth flow, and utility.
- [x] **State Serialization**:
    - Automatic state snapshotting at each timestep verified.
- [x] **Scenario Testing**:
    - Preliminary results for "Race to Bottom" and "Price Discrimination" integrated into paper.
    - Full stress-testing in progress via active training job (Job ID: 3887442).

## Phase 5: Paper Iteration (AGENT_LOOP)
- [x] **Drafting**: Completed Methodology and Experiments sections in `paper/root.tex`. Added Notation Table and Equilibrium Analysis.
- [x] **Agentic Review**: Iteration 1 complete (Prof_MAS, Dr_Clarity). Meta-review priorities addressed.
- [ ] **Refinement Loop**: Automatically apply improvements via `AGENT_IMPROVER.md` once final results are ready.

---
*Status: Initialized Plan (Jan 15, 2026)*
