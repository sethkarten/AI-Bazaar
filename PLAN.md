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
    - Launched finetuning on `della-ailab` cluster (Job ID: 3879110).
- [x] **Reward Modeling**:
    - Linked Individual Utility and Social Welfare to the RL reward signal.
- [x] **Environment Robustness**:
    - Configured `uv` environment with CUDA 12.8 support on cluster.
    - Implemented heartbeat monitoring and ETA timers.

## Phase 4: Visualization & UX
- [x] **Interactive Dashboard**:
    - Build Streamlit UI for real-time visualization of Gini, wealth flow, and utility.
- [ ] **Scenario Testing**:
    - Stress-test scenarios (Race to Bottom, etc.) using finetuned models.

## Phase 5: Paper Iteration (AGENT_LOOP)
- [ ] **Drafting**: Complete Methodology and Experiments sections in `paper/root.tex`.
- [ ] **Agentic Review**: Run the `agents/` pool (Prof_MAS, Dr_Clarity, etc.) on the draft.
- [ ] **Refinement Loop**: Automatically apply improvements via `AGENT_IMPROVER.md`.

---
*Status: Initialized Plan (Jan 15, 2026)*
