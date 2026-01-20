# Agent Bazaar Progress Report (COLM 2026)

## Project Overview
Evaluating **Version 3 open-source models** (Gemma 3, Qwen 3, OLMo 3) in a decentralized marketplace using **REINFORCE++ Policy Gradient** optimization. The framework simulates autonomous firms competing for survival against platform fees, fixed overhead, and information asymmetry.

---

## 1. The Blockade: Local RTX 5090 Verification
**CRITICAL: The 9-variant ablation matrix on the Pikachu Cluster is PAUSED.** 
We will not proceed to the cluster until the local high-throughput engine and REINFORCE++ reward model are validated on 1x RTX 5090.

### Current Debugging Focus: The "Stall" Issue
Local tests (Gemma 3 4B) were encountering a "Stall" where timesteps failed to clear.
- **Diagnosis**: Instruction-following failure. Gemma 3 was providing long-form reasoning instead of raw JSON. This triggered sequential retries in the environment phases (Labor -> Pricing), which effectively neutralized the benefits of parallel episode collection.
- **Impact**: GPU utilization dropped to <15% because it was waiting on sequential CPU-bound retry loops.

---

## 2. Technical Solution: High-Throughput Engine

### A. Dynamic Batching with Thinking Budget
- **Implementation**: `UnslothModel` now uses a background thread that pulls from a 65-agent queue (5 episodes * 13 agents).
- **Thinking Budget**: We now explicitly allow `max_new_tokens=512`. This gives Gemma 3 the "scratchpad" it needs to reason before outputting the final action JSON.
- **Optimization**: `batch_timeout_ms` reduced to **20ms** to ensure the RTX 5090 is fed as soon as the queue has critical mass.

### B. "No-Retry" Robust Parsing
To eliminate the stall-inducing retry loops:
1. **Aggressive Salvage**: The parser now searches for the last `{...}` block in the response, bypassing reasoning text.
2. **Regex Extraction**: If JSON parsing still fails, a new "Ultra-Aggressive" regex layer extracts numbers directly from phrases like `"labor is 40 hours"` or `"I will produce 10 units"`.
3. **Format Reward**: Instead of retrying on the model's time, we provide feedback via the **RL Loop**:
   - **Valid JSON**: $+5.0$ Bonus Reward.
   - **Malformed JSON**: $-5.0$ Penalty.
   - This "Reward Shaping" encourages the model to internalize the format constraints during the training step.

---

## 3. REINFORCE++ Reward Model Validation

We are currently validating the composite reward function:
$$Total\_Reward = Environmental\_Utility + Format\_Bonus$$

- **Goal**: Stabilize the policy so agents learn to optimize Profit while *simultaneously* maintaining a valid communication protocol (JSON).
- **Local Test Case**: 5 parallel episodes, 1 iteration.
- **Success Criteria**:
  - [ ] GPU power draw > 300W (Indicates saturation).
  - [ ] Completion of Iteration 0 in < 45 minutes.
  - [ ] "Salvage Count" decreases as training progresses.

---

## 4. Roadmap to Cluster Launch

1. **[CURRENT]** Debug 1 Local Job: Finalize the regex extractors and verify the `format_reward` gradient doesn't dominate the environmental signal.
2. **Performance Audit**: Confirm `state_ep*_t*.json` production rates.
3. **Cluster Sync**: Update Pikachu `uv` environments with the hardened `unsloth` import fixes.
4. **Approval**: Move to `submit_matrix.py --production`.

---

## 5. Current Steps (Build Mode)
- **Active**: Patching `ai_bazaar/agents/llm_agent.py` with the "Fuzzy Logic" extractor.
- **Active**: Monitoring `logs/local_uv_debug/main.log` for real-time batch sizes.
- **Pending**: Calculating `tokens_per_second` on the RTX 5090.
