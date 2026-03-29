# EAS vs. Model Size: Open-Source Model Reference

This document catalogs open-source (open-weight) models for the EAS-vs-model-size plot, including parameter counts and OpenRouter pricing as of March 2026.

---

## All Models — Sorted by Active Parameters

For MoE models, active parameters reflect per-forward-pass compute. Dense models have active = total.


| Model                         | Developer       | Total Params | Active Params | MoE?           | OR Input ($/M) | OR Output ($/M) | Notes                                  | Include/Exclude |
| ----------------------------- | --------------- | ------------ | ------------- | -------------- | -------------- | --------------- | -------------------------------------- | --------------- |
| Llama 3.2 3B Instruct         | Meta            | 3B           | 3B            | —              | $0.00 (free)   | $0.00 (free)    | Edge/mobile; free on OpenRouter        |                 |
| Nemotron 3 Nano 30B           | NVIDIA          | 30B          | ~3B           | MoE            | $0.05          | $0.20           | 262K context; free tier available      |                 |
| Gemma 3 4B                    | Google DeepMind | 4B           | 4B            | —              | $0.04          | $0.08           | Free tier also available               |                 |
| GPT-OSS-120B                  | OpenAI          | 117B         | ~5B           | MoE            | $0.039         | $0.190          | OpenAI's first open-weight; Apache 2.0 |                 |
| Mistral Small 4               | Mistral AI      | 119B         | ~6B           | MoE (128E)     | $0.15          | $0.60           | Reasoning + vision + coding; 262K ctx  |                 |
| Llama 3.1 8B Instruct         | Meta            | 8B           | 8B            | —              | $0.02          | $0.05           | 16K context                            |                 |
| Qwen3 8B                      | Alibaba         | 8.2B         | 8.2B          | —              | $0.05          | $0.40           | Hybrid thinking/non-thinking modes     |                 |
| MiniMax M2.7                  | MiniMax         | ~230B        | ~10B          | MoE            | $0.30          | $1.20           | 204K context; "self-evolving" arch     |                 |
| Gemma 3 12B                   | Google DeepMind | 12B          | 12B           | —              | $0.04          | $0.13           | Free tier also available               |                 |
| Nemotron 3 Super 120B         | NVIDIA          | 120B         | ~12B          | MoE            | $0.10          | $0.50           | Hybrid Mamba-Transformer; 1M ctx       |                 |
| Trinity Large Preview         | Arcee AI        | ~400B        | ~13B          | MoE (4-of-256) | $0.00 (free)   | $0.00 (free)    | 512K context; fully free               |                 |
| Mixtral 8x7B Instruct         | Mistral AI      | ~56B         | ~14B          | MoE            | $0.54          | $0.54           | Classic MoE; 32K context               |                 |
| Phi-4                         | Microsoft       | 14B          | 14B           | —              | $0.065         | $0.14           | Strong reasoning for size              |                 |
| DeepSeek R1 Distill Qwen 14B  | DeepSeek        | 14B          | 14B           | —              | $0.875         | $0.875          | Reasoning distillation                 |                 |
| Llama 4 Scout                 | Meta            | 109B         | ~17B          | MoE (16E)      | $0.08          | $0.30           | 327K context                           |                 |
| Llama 4 Maverick              | Meta            | ~400B        | ~17B          | MoE (128E)     | $0.15          | $0.60           | 1M context                             |                 |
| Qwen3.5 397B A17B             | Alibaba         | 397B         | ~17B          | MoE            | $0.39          | $2.34           | Multimodal; 262K context               |                 |
| Mistral 7B Instruct           | Mistral AI      | 7.3B         | 7.3B          | —              | $0.14          | $0.20           | 32K context                            |                 |
| Mistral Small 3.1 24B         | Mistral AI      | 24B          | 24B           | —              | $0.03          | $0.11           | 131K context; free tier available      |                 |
| Qwen3 235B A22B               | Alibaba         | 235B         | ~22B          | MoE            | $0.455         | $1.82           | Hybrid thinking; 131K context          |                 |
| Gemma 3 27B                   | Google DeepMind | 27B          | 27B           | —              | $0.08          | $0.16           | Vision-language capable                |                 |
| DeepSeek R1 Distill Qwen 32B  | DeepSeek        | 32B          | 32B           | —              | $0.29          | $0.29           | Reasoning distillation                 |                 |
| Kimi K2.5                     | Moonshot AI     | ~1T          | ~32B          | MoE            | $0.45          | $2.20           | 262K context                           |                 |
| DeepSeek V3 0324              | DeepSeek        | 685B         | ~37B          | MoE            | $0.20          | $0.77           | 163K context                           |                 |
| DeepSeek R1                   | DeepSeek        | 671B         | ~37B          | MoE            | $0.70          | $2.50           | Full reasoning model; MIT license      |                 |
| Mixtral 8x22B Instruct        | Mistral AI      | ~141B        | ~39B          | MoE            | $2.00          | $6.00           | 65K context                            |                 |
| Mistral Large 3 2512          | Mistral AI      | 675B         | ~41B          | MoE            | $0.50          | $1.50           | 262K context                           |                 |
| GLM-5                         | Z.ai (Zhipu AI) | 745B         | ~44B          | MoE            | $0.72          | $2.30           | Released Feb 2026                      |                 |
| Llama 3.1 Nemotron Ultra 253B | NVIDIA          | 253B         | ~53B          | MoE            | $0.60          | $1.80           | Reasoning, RAG, tool-calling           |                 |
| Llama 3.3 70B Instruct        | Meta            | 70B          | 70B           | —              | $0.10          | $0.32           | 131K context; free tier available      |                 |
| Llama 3.1 70B Instruct        | Meta            | 70B          | 70B           | —              | $0.40          | $0.40           | 131K context                           |                 |
| DeepSeek R1 Distill Llama 70B | DeepSeek        | 70B          | 70B           | —              | $0.70          | $0.80           | Reasoning distillation                 |                 |
| Llama 3.1 Nemotron 70B        | NVIDIA          | 70B          | 70B           | —              | $1.20          | $1.20           | RLHF fine-tune of Llama 3.1 70B        |                 |
| Qwen2.5 72B Instruct          | Alibaba         | 72B          | 72B           | —              | $0.12          | $0.39           | Strong general-purpose                 |                 |
| Command R+                    | Cohere          | 104B         | 104B          | —              | $2.50          | $10.00          | 128K context; strong RAG/tool use      |                 |
| Llama 3.1 405B Instruct       | Meta            | 405B         | 405B          | —              | $4.00          | $4.00           | Largest dense open Llama; 131K ctx     |                 |
| Hermes 3 405B Instruct        | NousResearch    | 405B         | 405B          | —              | $1.00          | $1.00           | Fine-tune of Llama 3.1 405B            |                 |
| Hermes 4 405B                 | NousResearch    | 405B         | 405B          | —              | $1.00          | $3.00           | Hybrid thinking/direct modes           |                 |


---

## Dense Models Only — Sorted by Parameters

Excludes all MoE architectures. Useful for clean scaling-law comparisons.


| Model                         | Developer       | Parameters | OR Model ID                                      | OR Input ($/M) | OR Output ($/M) | Notes                                 | Include/Exclude |
| ----------------------------- | --------------- | ---------- | ------------------------------------------------ | -------------- | --------------- | ------------------------------------- | --------------- |
| Llama 3.2 3B Instruct         | Meta            | 3B         | `meta-llama/llama-3.2-3b-instruct`               | $0.00 (free)   | $0.00 (free)    | Edge/mobile; free on OpenRouter       | 1               |
| Gemma 3 4B                    | Google DeepMind | 4B         | `google/gemma-3-4b-it`                           | $0.04          | $0.08           | Free tier also available              | 1               |
| Mistral 7B Instruct           | Mistral AI      | 7.3B       | `mistralai/mistral-7b-instruct-v0.1`             | $0.14          | $0.20           | 32K context                           | 1               |
| Llama 3.1 8B Instruct         | Meta            | 8B         | `meta-llama/llama-3.1-8b-instruct`               | $0.02          | $0.05           | 16K context                           | 1               |
| Qwen3 8B                      | Alibaba         | 8.2B       | `qwen/qwen3-8b`                                  | $0.05          | $0.40           | Hybrid thinking/non-thinking modes    | 1               |
| Gemma 3 12B                   | Google DeepMind | 12B        | `google/gemma-3-12b-it`                          | $0.04          | $0.13           | Free tier also available              | 1               |
| Phi-4                         | Microsoft       | 14B        | `microsoft/phi-4`                                | $0.065         | $0.14           | Strong reasoning for size             | 1               |
| DeepSeek R1 Distill Qwen 14B  | DeepSeek        | 14B        | ~~`deepseek/deepseek-r1-distill-qwen-14b`~~ — removed from OpenRouter | — | — | Reasoning distillation | 0 |
| Mistral Small 3.1 24B         | Mistral AI      | 24B        | `mistralai/mistral-small-3.1-24b-instruct`       | $0.03          | $0.11           | 131K context; free tier available     | 1               |
| Gemma 3 27B                   | Google DeepMind | 27B        | `google/gemma-3-27b-it`                          | $0.08          | $0.16           | Vision-language capable               | 1               |
| DeepSeek R1 Distill Qwen 32B  | DeepSeek        | 32B        | `deepseek/deepseek-r1-distill-qwen-32b`          | $0.29          | $0.29           | Reasoning distillation                | 1               |
| Llama 3.3 70B Instruct        | Meta            | 70B        | `meta-llama/llama-3.3-70b-instruct`              | $0.10          | $0.32           | 131K context; free tier available     | 1               |
| Llama 3.1 70B Instruct        | Meta            | 70B        | `meta-llama/llama-3.1-70b-instruct`              | $0.40          | $0.40           | 131K context                          | 1               |
| DeepSeek R1 Distill Llama 70B | DeepSeek        | 70B        | `deepseek/deepseek-r1-distill-llama-70b`         | $0.70          | $0.80           | Reasoning distillation                | 1               |
| Llama 3.1 Nemotron 70B        | NVIDIA          | 70B        | `nvidia/llama-3.1-nemotron-70b-instruct`         | $1.20          | $1.20           | RLHF fine-tune of Llama 3.1 70B       | 1               |
| Qwen2.5 72B Instruct          | Alibaba         | 72B        | `qwen/qwen-2.5-72b-instruct`                     | $0.12          | $0.39           | Strong general-purpose                | 1               |
| Command R+                    | Cohere          | 104B       | `cohere/command-r-plus`                          | $2.50          | $10.00          | 128K context; strong RAG/tool use     | 0               |
| Llama 3.1 405B Instruct       | Meta            | 405B       | ~~`meta-llama/llama-3.1-405b-instruct`~~ — removed from OpenRouter | — | — | Largest dense open Llama; 131K ctx | 0 |
| Hermes 3 405B Instruct        | NousResearch    | 405B       | `nousresearch/hermes-3-llama-3.1-405b`           | $1.00          | $1.00           | Fine-tune of Llama 3.1 405B           | 1               |
| Hermes 4 405B                 | NousResearch    | 405B       | `nousresearch/hermes-4-405b`                     | $1.00          | $3.00           | Hybrid thinking/direct modes          | 1               |


### Clarifications on Requested Models

- **"Qwen 3.5 9B"** — This model does not exist. Sizes in the Qwen3 family are 1.7B, 4B, 8B, 14B, 30B, 32B, 235B. Closest match is **Qwen3 8B** (used above).
- **"Nemotron"** — NVIDIA has two distinct Nemotron lines: standalone Nemotron 3 (Nano 30B, Super 120B) and fine-tunes on Llama 3.1 (70B, Ultra 253B). Both are included.
- **"MiniMax 2.5"** — The current listing on OpenRouter is **MiniMax M2.7** (March 2026 update); its predecessor M2.5 has been superseded.
- **"Kimi"** — Refers to **Kimi K2.5** by Moonshot AI, a 1T/32B MoE model.
- **"Trinity"** — **Trinity Large Preview** by Arcee AI, a free 400B/13B MoE model.
- **"GLM 5"** — **GLM-5** from Z.ai (formerly Zhipu AI / ChatGLM), a 745B/44B MoE model.
- **"Hermes"** — NousResearch's Hermes line (Hermes 3 and Hermes 4) are fine-tunes of Llama 3.1 405B.

---

## Added Models for Size Coverage

The following were added to improve spread across the parameter range:


| Size Range | Models Added                                             | Rationale                                |
| ---------- | -------------------------------------------------------- | ---------------------------------------- |
| ~3B        | Llama 3.2 3B                                             | Anchors the small end; free              |
| ~4B        | Gemma 3 4B                                               | Dense small from Google                  |
| ~14B       | Phi-4, DeepSeek R1 Distill 14B                           | Mid-small dense; strong reasoning        |
| ~24B       | Mistral Small 3.1 24B                                    | Fills gap between 14B and 27B            |
| ~100–120B  | Command R+, Llama 4 Scout, Mistral Small 4               | Mid-large range; distinct architectures  |
| ~235–400B  | Qwen3 235B, Qwen3.5 397B, MiniMax M2.7, Llama 4 Maverick | Fills MoE mid-range before ~670B cluster |


---

## Parameters vs. Pricing as the Size Axis

### The Problem

We want to plot EAS on the y-axis against a measure of "model size" on the x-axis, and include both open-source and frontier closed models (e.g., GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro). This creates a tension: the two most natural axes — **parameter count** and **API pricing** — each have serious drawbacks.

### Option 1: Number of Parameters

**Pros:**

- The most direct, physically meaningful measure of model capacity.
- Allows apples-to-apples comparison of architectural scale.
- Well-established in the scaling laws literature (Kaplan et al., Chinchilla).
- Unaffected by market dynamics, provider margins, or commercial strategy.

**Cons:**

- **Frontier model parameters are not disclosed.** GPT-4o, Claude 3.5/3.7 Sonnet, Gemini 1.5/2.0 Pro, and o-series models all have unknown (or at best estimated) parameter counts. Leaked estimates exist but are unreliable. This makes parameter count unusable as a universal x-axis when frontier models are included.
- MoE models complicate the story: a 671B-total DeepSeek V3 has only ~37B active parameters per forward pass, making raw total count misleading as a compute-efficiency or intelligence proxy.
- Active parameters (per-forward-pass compute) and total parameters (memory/capacity) measure different things.

**Verdict:** Ideal for a pure open-source comparison, but breaks down when frontier models are included.

### Option 2: API Pricing (Input $/M tokens)

**Pros:**

- Universal — every model, open or closed, has a price.
- Implicitly encodes compute cost, and providers generally price proportionally to inference cost.
- Practical for the audience (researchers, builders) who care about cost-effectiveness.
- Has precedent in industry benchmarking (e.g., cost-vs-quality scatter plots).

**Cons:**

- Pricing reflects business strategy, not just model size. A frontier model may be cheaper than an open-source model due to scale economies, loss-leader pricing, or subsidized compute.
- Prices change. A snapshot from March 2026 will be stale within months.
- Free/open-weight models self-hosted have near-zero cost but are not "small" — e.g., DeepSeek R1 (671B) is self-hostable.
- Pricing collapses MoE and dense models: a 1T MoE model priced for its active compute looks similar to a 30B dense model.

**Verdict:** The only option that includes all frontier models, but it measures cost-efficiency, not raw scale.

### Recommendation

**Use a dual-axis or two-panel approach:**

1. **Primary plot: Pricing as x-axis.** This is the only axis that spans both open-source and frontier models. Use **input price ($/M tokens)** as the x-axis for the main EAS-vs-cost figure. This directly answers "how much alignment do you get per dollar?" — a practically relevant research question.
2. **Secondary plot (open-source only): Active parameters as x-axis.** For the subset of open-source models where parameter counts are known, produce a companion figure with active parameter count on the x-axis (log scale). This answers the scaling-law question: does economic alignment improve predictably with model scale?
3. **Label MoE models distinctly** (e.g., hollow vs. filled markers) on both plots, since their total vs. active parameter gap is large enough to mislead visual intuition.

The pricing axis has one key advantage beyond universality: it reframes the research question from "do smarter models align better?" to "do more expensive models align better?" — which is arguably more useful for practitioners deciding what model to deploy in an AI-Bazaar-style system.