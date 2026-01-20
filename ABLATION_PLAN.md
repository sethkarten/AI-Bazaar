# Ablation Plan: Agentic Marketplace Dynamics (COLM 2026)

This document details the systematic ablation study for the Agent Bazaar framework, evaluating the impact of marketplace mechanics and agentic features on market stability, firm survival, and social welfare.

## 1. Primary Research Questions
1.  **Reward Alignment**: Does optimizing for **PROFIT** (long-term survival) vs. **REVENUE** (short-term greed) lead to more stable markets?
2.  **Agentic Reasoning**: Do **Strategic Diaries** (retrospective self-reflection) help agents avoid catastrophic price wars?
3.  **Discovery Friction**: How does limited visibility (**Discovery Limit**) affect market consolidation and the Gini coefficient?
4.  **Information Asymmetry**: Can agents remain competitive when provided with **Noisy Market Reports** instead of perfect information?
5.  **Model Scaling**: How do different **Version 3** open-source models (Gemma 3, Qwen 3, OLMo 3, Ministral 3) compare in economic rationality?

---

## 2. Experimental Matrix

All variants use **REINFORCE++** policy gradient finetuning on 4-bit quantized models unless specified otherwise.

| Variant Name | Base Model | Reward Signal | Diaries? | Discovery Limit | Info Asymmetry? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline` | Gemma 3 4B | PROFIT | Yes | 5 | No |
| `revenue` | Gemma 3 4B | REVENUE | Yes | 5 | No |
| `nodiaries` | Gemma 3 4B | PROFIT | **No** | 5 | No |
| `nofriction` | Gemma 3 4B | PROFIT | Yes | **0 (Full)** | No |
| `asymmetry` | Gemma 3 4B | PROFIT | Yes | 5 | **Yes (Noisy)** |
| `final-gemma`| Gemma 3 4B | PROFIT | Yes | 5 | Yes |
| `qwen3` | Qwen 3 8B | PROFIT | Yes | 5 | Yes |
| `olmo3` | OLMo 3 7B | PROFIT | Yes | 5 | Yes |
| `ministral3` | Ministral 3 8B | PROFIT | Yes | 5 | Yes |

---

## 3. Key Metrics
For each ablation run, we track:
- **Survival Rate**: Percentage of firms that remain in business until Timestep 100.
- **Gini Coefficient**: Measure of wealth inequality within the marketplace.
- **Price Stability**: Variance in pricing across timesteps.
- **Social Welfare**: Cumulative utility of all consumers.
- **Parsing Robustness**: Percentage of LLM calls requiring "Salvage Parsing."

---

## 4. Execution Roadmap
1.  **Local Debugging (RTX 5090)**:
    - Validate `baseline` throughput with 10 parallel episodes.
    - Confirm **Format Reward** effectively trains agents to follow JSON protocols.
    - Verify no CUDA OOMs during the transition to the backward pass.
2.  **Production Matrix (Pikachu Cluster)**:
    - Submit all 9 variants via `submit_matrix.py --production`.
    - 120s staggered submission delay per GPU.
    - 48-hour time limit per job for convergence.
3.  **Analysis**:
    - Use `analyze_results.py` to compare Pareto frontiers of model capability vs. economic utility.
