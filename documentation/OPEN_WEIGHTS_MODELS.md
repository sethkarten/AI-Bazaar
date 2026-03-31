# Open-Weights Models — Exp1 EAS Sweep

15 dense open-weight models run via OpenRouter (`scripts/exp1_eas_sweep.py`).

| Display Name     | Params (B) | OpenRouter Slug                                  |
|------------------|------------|--------------------------------------------------|
| Llama 3.2 3B     | 3.0        | `meta-llama/llama-3.2-3b-instruct`               |
| Gemma 3 4B       | 4.0        | `google/gemma-3-4b-it`                           |
| Mistral 7B       | 7.3        | `mistralai/mistral-7b-instruct-v0.1`             |
| Llama 3.1 8B     | 8.0        | `meta-llama/llama-3.1-8b-instruct`               |
| Qwen3 8B         | 8.2        | `qwen/qwen3-8b`                                  |
| Gemma 3 12B      | 12.0       | `google/gemma-3-12b-it`                          |
| Phi-4            | 14.0       | `microsoft/phi-4`                                |
| Mistral Small 24B| 24.0       | `mistralai/mistral-small-3.1-24b-instruct`       |
| Gemma 3 27B      | 27.0       | `google/gemma-3-27b-it`                          |
| DS-R1-D 32B      | 32.0       | `deepseek/deepseek-r1-distill-qwen-32b`          |
| Llama 3.3 70B    | 70.0       | `meta-llama/llama-3.3-70b-instruct`              |
| Llama 3.1 70B    | 70.0       | `meta-llama/llama-3.1-70b-instruct`              |
| DS-R1-D 70B      | 70.0       | `deepseek/deepseek-r1-distill-llama-70b`         |
| Nemotron 70B     | 70.0       | `nvidia/llama-3.1-nemotron-70b-instruct`         |
| Qwen2.5 72B      | 72.0       | `qwen/qwen-2.5-72b-instruct`                     |
| Hermes 3 405B    | 405.0      | `nousresearch/hermes-3-llama-3.1-405b`           |
| Hermes 4 405B    | 405.0      | `nousresearch/hermes-4-405b`                     |

## Removed / Unavailable

| Display Name   | Params (B) | OpenRouter Slug                               | Reason                    |
|----------------|------------|-----------------------------------------------|---------------------------|
| DS-R1-D 14B    | 14.0       | `deepseek/deepseek-r1-distill-qwen-14b`       | Removed from OpenRouter   |
| Llama 3.1 405B | 405.0      | `meta-llama/llama-3.1-405b-instruct`          | Removed from OpenRouter   |
