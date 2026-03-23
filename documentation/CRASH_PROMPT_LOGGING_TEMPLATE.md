# Lemon Prompt Logging + Dashboard Navigator (Template for THE_CRASH)

This document explains the current prompt/response logging and dashboard navigation implementation that is intentionally scoped to `LEMON_MARKET` and to buyer/seller agents.

Use this as a template to implement the same capability for `THE_CRASH`.

## Scope verification (current behavior)

The implementation is Lemon-only and does not run for THE_CRASH:

- Logging helper hard-gates on `consumer_scenario == "LEMON_MARKET"`.
- Logging helper also gates by role-specific flags (`--log-buyer-prompts`, `--log-seller-prompts`).
- Dashboard prompt navigation UI appears only inside the Lemon Market tab block (`is_lemon_run`).
- Logged roles are explicitly `buyer`/`seller`; no other agent roles are written to `lemon_agent_prompts.jsonl`.

## Files and responsibilities

### 1) `ai_bazaar/utils/agent_prompt_log.py`

- `append_lemon_agent_prompt(...)`
  - Appends one JSON record per line to:
    - `logs/<run_name>/lemon_agent_prompts.jsonl`
- `maybe_append_lemon_agent_prompt(...)`
  - Central gatekeeper for:
    - scenario check (`LEMON_MARKET`)
    - role check (`buyer` or `seller`)
    - role flag check (`log_buyer_prompts` / `log_seller_prompts`)

This file is the single best place to preserve scenario isolation.

### 2) `ai_bazaar/main.py`

CLI flags:

- `--log-buyer-prompts`
- `--log-seller-prompts`

These are optional and default OFF. They control whether JSONL logging is written.

### 3) `ai_bazaar/agents/llm_agent.py`

Shared LLM path instrumentation:

- `_maybe_log_lemon_prompt(...)`
- `_maybe_log_lemon_prompt_from_call_llm(...)`
- `call_llm(...)` invokes logging immediately after each `send_msg(...)` call.

This captures buyer bidding calls and sybil-principal tier calls that pass through `LLMAgent`.

### 4) `ai_bazaar/agents/buyer.py`

Buyer-specific direct calls instrumented:

- `self.lemon_agent_role = "buyer"`
- post-transaction review logging (`call="review"`)
- diary logging (`call="diary"`)

### 5) `ai_bazaar/agents/seller.py`

LLM seller listing generation instrumented:

- direct `send_msg(...)` in `_llm_description(...)` logs with `role="seller"`, `call="listing"`.

### 6) `ai_bazaar/agents/sybil.py`

Sybil principal logging:

- `self.lemon_agent_role = "seller"` for principal-level `call_llm` paths.
- direct listing description generation logs with `call="sybil_listing"` and `sybil_identity` metadata.

### 7) `ai_bazaar/viz/dashboard.py`

- Loads `lemon_agent_prompts.jsonl` from selected run folder.
- Seller detail section now includes a prompt navigator.
- Buyer detail section is added and includes the same navigator.
- Navigation supports prev/next and jump-to-record selection.

## JSONL record schema

Each line is one JSON object:

- `role`: `"buyer" | "seller"`
- `agent`: string
- `call`: string (examples: `bid`, `review`, `listing`, `sybil_tier`, `sybil_listing`, `diary`)
- `timestep`: int
- `depth`: int (retry/recursion depth where relevant)
- `system_prompt`: string
- `user_prompt`: string
- `response`: string
- optional metadata fields (for example `sybil_identity`)

## Template for THE_CRASH implementation

The cleanest approach is to duplicate this pattern with scenario-specific names to avoid accidental cross-scenario coupling.

### Step A: Add scenario-specific logging helper

Create a new helper (for example `crash_agent_prompts.jsonl`) with:

- function analogous to `maybe_append_lemon_agent_prompt(...)`
- strict gate: `consumer_scenario == "THE_CRASH"`
- role set appropriate to THE_CRASH (likely `consumer`, `firm`; optionally `planner` if desired)
- role-specific flags (for example `--log-crash-consumer-prompts`, `--log-crash-firm-prompts`)

### Step B: Add CLI flags in `main.py`

- add THE_CRASH-specific prompt logging flags
- keep defaults OFF
- include in any rerun-command serialization sets (where boolean flags are enumerated)

### Step C: Instrument agent call sites

Mirror Lemon strategy:

1. shared `LLMAgent.call_llm(...)` path (for agents that use `act_llm`)
2. direct `send_msg(...)` call sites in crash-specific logic

Do not rely only on shared path; direct calls must be explicitly logged.

### Step D: Dashboard integration

- add prompt file load for crash (separate file recommended)
- in THE_CRASH tab/sections, add:
  - `Firm detail` prompt navigator
  - `Consumer detail` prompt navigator
- keep filtering role-specific and scenario-specific

### Step E: Keep scenario isolation explicit

- avoid a single mixed JSONL unless you include explicit `scenario` field and strict filtering
- separate files are simpler:
  - `lemon_agent_prompts.jsonl`
  - `crash_agent_prompts.jsonl`

## Minimal acceptance checklist for THE_CRASH port

- Running LEMON_MARKET does not create crash prompt logs.
- Running THE_CRASH with crash flags ON creates crash prompt logs in `logs/<run_name>/`.
- Running THE_CRASH with flags OFF writes no crash prompt logs.
- Dashboard can navigate prompts for selected crash firm/consumer.
- Prompt navigator handles missing/empty files gracefully.

## Suggested command pattern

Keep the same ergonomics as Lemon:

- `python -m ai_bazaar.main --consumer-scenario THE_CRASH ... --log-crash-firm-prompts --log-crash-consumer-prompts`

(Flag names above are examples; choose final names consistently.)
