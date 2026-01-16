# Agent Bazaar: COLM 2026 Development Plan

This document outlines the roadmap for improving the Agent Bazaar framework and preparing the research paper for COLM 2026.

## Phase 1: Core Marketplace Integration (Labor + Governance)
- [x] **Labor Market Integration**: 
    - Link `CESConsumerAgent` to labor supply logic from `Worker` class.
    - Implement wage payments from `Firms` to `Workers`.
- [x] **Governance & Redistribution**:
    - Integrate `TaxPlanner` into `BazaarWorld`.
    - Implement tax collection and "Social Safety Net" redistribution.
- [x] **Granular Necessity Metrics**:
    - Map goods to Food, Housing, and Utilities.
    - Update utility functions with necessity weights and penalties.

## Phase 2: Agentic Social Layer (Interviews + Reflections)
- [x] **Agent Interview API**:
    - Create a wrapper to trigger conversations with agents during simulation pauses.
- [x] **Strategic Reflection**:
    - Implement "Diary Entries" for agents to store reasoning and retrospective analysis.
    - Use reflections as in-context memory for future timesteps.
- [x] **State Serialization**:
    - Implement full state snapshotting (inventory + mindset) at each timestep.

## Phase 3: Model Training & Evaluation
- [x] **Baseline Benchmarking**:
    - Evaluate Gemma 3-4B and Llama 3.1 8B (In-Context) on local 2x5090.
- [x] **REINFORCE++ Finetuning**:
    - Integrated `unsloth` for LoRA-based training.
    - Implemented `UnslothModel` for fast in-process trajectory generation.
    - Resolved Gemma 3 processor subscriptable errors.
    - Optimized speed via **Parallel Agent Actions** and **vLLM Batching**.
    - Launched finetuning on `della-ailab` cluster (Job ID: 3884330).
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
