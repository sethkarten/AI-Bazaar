"""
compile_firm_responses.py
─────────────────────────
Compile LLM firm action responses across specified runs into a structured JSON.

Output schema
─────────────
{
  "<run_name>": {
    "metadata": {
      "experiment_args": {...},        # from experiment_args.json
      "firm_attributes": [...]         # from firm_attributes.json
    },
    "timesteps": {
      "<t>": {
        "<firm_name>": {
          "in_business": bool,
          "prices":       {good: price},
          "profit":       float,
          "cash":         float,
          "sales_count":  int,
          "sales_info":   [...],
          # ── present when alignment_traces.jsonl exists ─────────────────
          "system_prompt": str,
          "user_prompt":   str,
          "llm_response":  str,        # raw LLM text
          "thought":       str | null, # extracted from response JSON if present
          "action":        {...},      # parsed action dict from trace
          # ── present when log parsing succeeds (best-effort) ────────────
          "log_response":  str,        # raw matched LLM OUTPUT text
          "log_thought":   str | null
        },
        ...
      },
      ...
    }
  },
  ...
}

Attribution strategy
────────────────────
1. Alignment traces (alignment_traces.jsonl):
   Firm order matches state["state"]["firms"] order; firm name also embedded
   in system_prompt ("You are a firm manager named <name>"). Clean & reliable.

2. Log file (marketplace_<run>.log or logs/<run>/<run>_*.log):
   LLM OUTPUT lines are interleaved from parallel threads, so direct
   positional attribution is unreliable. Instead, for each timestep block,
   price-setting responses are matched to firms by comparing the parsed
   price values against the prices recorded in the corresponding state file.
   Ties (two firms set identical prices) are flagged with a note.
   Supply/produce responses cannot be attributed per-firm and are stored
   in a top-level "unattributed_outputs" list for the timestep.

3. State files (state_t*.json):
   Always used as the authoritative backbone for outcomes.

Usage
─────
Edit RUNS below, then run from the project root:

    python scripts/compile_firm_responses.py

Output is written to logs/<run_name>/firm_responses.json for each run,
and a combined file to logs/firm_responses_all.json.
"""

import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ── Configuration ─────────────────────────────────────────────────────────────

LOG_DIR = "logs"

# List the run names you want to process (subdirectory names under LOG_DIR).
# Set to None to process every subfolder that contains state_t*.json files.
RUNS: Optional[List[str]] = [
    "crash_100_flash_1",
    "crash_365_seed42",
    "crash_365_seed123",
    "crash_365_seed8",
    # "crash_stabilizing_proto_test_1",   # uncomment to include
]

OUTPUT_COMBINED = os.path.join(LOG_DIR, "firm_responses_all.json")

# ── Helpers ────────────────────────────────────────────────────────────────────

_TS_COMPLETE_RE = re.compile(r"Timestep (\d+)/\d+ completed")
_ACTION_RE      = re.compile(r"\[ACTION\] ([\w.]+) performing action: (\w+)")
_OUTPUT_RE      = re.compile(r"LLM OUTPUT RECURSE \d+\t(.*)")
_FIRM_NAME_RE   = re.compile(r"You are a firm manager named ([\w.]+)")


def _load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _extract_thought(response_text: str) -> Optional[str]:
    """Try to extract 'thought' field from a JSON LLM response string."""
    if not response_text:
        return None
    try:
        obj = json.loads(response_text)
        return obj.get("thought") or obj.get("reasoning") or obj.get("rationale")
    except (json.JSONDecodeError, AttributeError):
        # Try regex fallback for malformed JSON
        m = re.search(r'"thought"\s*:\s*"([^"]*)"', response_text)
        return m.group(1) if m else None


def _extract_prices_from_response(response_text: str) -> Dict[str, float]:
    """Extract price_<good> entries from a JSON LLM response string."""
    out = {}
    try:
        obj = json.loads(response_text)
        for k, v in obj.items():
            if k.startswith("price_"):
                good = k[len("price_"):]
                try:
                    out[good] = float(v)
                except (ValueError, TypeError):
                    pass
    except (json.JSONDecodeError, AttributeError):
        # Regex fallback
        for m in re.finditer(r'"price_(\w+)"\s*:\s*([\d.]+)', response_text):
            out[m.group(1)] = float(m.group(2))
    return out


def _prices_match(extracted: Dict[str, float], state_prices: Dict[str, float],
                  tol: float = 1e-4) -> bool:
    if not extracted or not state_prices:
        return False
    for good, price in extracted.items():
        sp = state_prices.get(good)
        if sp is None or abs(price - sp) > tol:
            return False
    return True


# ── Log file parsing ───────────────────────────────────────────────────────────

def _collate_multiline_outputs(lines: List[str]) -> List[str]:
    """
    Collate LLM OUTPUT lines that span multiple log lines (e.g. pretty-printed JSON).
    Returns a list of complete output strings (one per LLM call).
    """
    results = []
    current: Optional[List[str]] = None
    brace_depth = 0

    for line in lines:
        m = _OUTPUT_RE.search(line)
        if m:
            fragment = m.group(1).strip()
            # Start a new output
            if current is not None:
                results.append("".join(current))
            current = [fragment]
            brace_depth = fragment.count("{") - fragment.count("}")
            if brace_depth <= 0:
                results.append("".join(current))
                current = None
                brace_depth = 0
        elif current is not None:
            # Continuation of a multiline JSON (no timestamp prefix)
            stripped = line.rstrip("\n")
            current.append(stripped)
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0:
                results.append("".join(current))
                current = None
                brace_depth = 0
        # else: unrelated log line

    if current:
        results.append("".join(current))

    return results


def _split_log_by_timestep(log_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Parse the log file into per-timestep blocks.
    Returns dict keyed by timestep (int).
    Each value: {"action_lines": [...], "output_lines": [...], "raw_lines": [...]}
    Timestep 0 = everything before "Timestep 1/N completed".
    """
    if not os.path.exists(log_path):
        return {}

    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    blocks: Dict[int, Dict] = {}
    current_ts = 0
    block_lines: List[str] = []

    def _flush(ts, blines):
        action_lines = [l for l in blines if _ACTION_RE.search(l)]
        output_lines = [l for l in blines if _OUTPUT_RE.search(l)]
        blocks[ts] = {
            "action_lines": action_lines,
            "output_lines": _collate_multiline_outputs(blines),
            "raw_lines": blines,
        }

    for line in lines:
        m = _TS_COMPLETE_RE.search(line)
        if m:
            ts_done = int(m.group(1))
            _flush(current_ts, block_lines)
            current_ts = ts_done
            block_lines = []
        else:
            block_lines.append(line)

    if block_lines:
        _flush(current_ts, block_lines)

    return blocks


def _attribute_price_responses(
    outputs: List[str],
    state_firms: List[Dict],
) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    For each set_price LLM output that contains price_* fields, try to match
    it to a firm by comparing extracted prices against state prices.

    Returns: {firm_name: (raw_response, thought)} for successfully matched firms.
    Unmatched responses are stored under the key "__unmatched__" as a list.
    """
    attributed: Dict[str, Tuple[str, Optional[str]]] = {}
    unmatched = []

    # Index state prices: {firm_name: {good: price}}
    firm_prices = {f["name"]: f.get("prices") or {} for f in state_firms}

    price_outputs = [o for o in outputs if "price_" in o]

    for raw in price_outputs:
        extracted = _extract_prices_from_response(raw)
        if not extracted:
            unmatched.append(raw)
            continue

        matched_firms = [
            name for name, sp in firm_prices.items()
            if _prices_match(extracted, sp)
        ]

        if len(matched_firms) == 1:
            name = matched_firms[0]
            if name not in attributed:
                attributed[name] = (raw, _extract_thought(raw))
            else:
                # Already matched — shouldn't happen unless same price twice
                unmatched.append(raw)
        else:
            unmatched.append(raw)

    if unmatched:
        attributed["__unmatched__"] = unmatched  # type: ignore[assignment]

    return attributed


# ── Alignment trace parsing ────────────────────────────────────────────────────

def _parse_alignment_traces(trace_path: str) -> Dict[int, List[Dict]]:
    """
    Parse alignment_traces.jsonl.
    Returns: {timestep: [firm_trace_dict, ...]} ordered by firm index.
    Each firm_trace_dict has keys from _last_price_trace + parsed firm name.
    """
    result: Dict[int, List[Dict]] = {}
    if not os.path.exists(trace_path):
        return result

    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = entry.get("outcome", {}).get("timestep")
            if ts is None:
                ts = entry.get("state", {}).get("timestep")
            if ts is None:
                continue

            state_firms = entry.get("state", {}).get("firms", [])
            firm_traces = entry.get("firms", [])

            enriched = []
            for i, trace in enumerate(firm_traces):
                # Firm name: from system_prompt or state_firms order
                sp = trace.get("system_prompt", "")
                nm = _FIRM_NAME_RE.search(sp)
                name = nm.group(1) if nm else (
                    state_firms[i]["name"] if i < len(state_firms) else f"firm_{i}"
                )
                raw_resp = trace.get("response", "")
                enriched.append({
                    "firm_name":     name,
                    "system_prompt": sp,
                    "user_prompt":   trace.get("user_prompt", ""),
                    "llm_response":  raw_resp,
                    "thought":       _extract_thought(raw_resp),
                    "action":        trace.get("action"),
                    "unit_costs":    trace.get("unit_costs"),
                })
            result[int(ts)] = enriched

    return result


# ── State file loading ─────────────────────────────────────────────────────────

def _load_state_files(run_dir: str) -> Dict[int, Dict]:
    """Load all state_t*.json files. Returns {timestep: state_dict}."""
    pattern = os.path.join(run_dir, "state_t*.json")
    states: Dict[int, Dict] = {}
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        m = re.match(r"state_t(\d+)\.json", basename)
        if not m:
            continue
        t = int(m.group(1))
        data = _load_json(path)
        if data:
            states[t] = data
    return states


# ── Per-run compilation ────────────────────────────────────────────────────────

def _find_log_file(run_name: str, run_dir: str) -> Optional[str]:
    """Try common log file locations for a run."""
    candidates = [
        os.path.join(LOG_DIR, f"marketplace_{run_name}.log"),
        os.path.join(run_dir, f"{run_name}.log"),
    ]
    # Also search for any *.log inside the run dir
    candidates += glob.glob(os.path.join(run_dir, "*.log"))
    # And logs/<run_name>_*.log patterns from batch scripts
    candidates += glob.glob(os.path.join(LOG_DIR, f"{run_name}_*.log"))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def compile_run(run_name: str) -> Dict[str, Any]:
    run_dir = os.path.join(LOG_DIR, run_name)
    if not os.path.isdir(run_dir):
        print(f"  [SKIP] {run_name}: directory not found")
        return {}

    # ── Metadata ──────────────────────────────────────────────────────────────
    experiment_args = _load_json(os.path.join(run_dir, "experiment_args.json")) or {}
    firm_attrs      = _load_json(os.path.join(run_dir, "firm_attributes.json")) or []

    # ── State files ───────────────────────────────────────────────────────────
    states = _load_state_files(run_dir)
    if not states:
        print(f"  [SKIP] {run_name}: no state files found")
        return {}
    print(f"  Loaded {len(states)} state files for {run_name}")

    # ── Alignment traces ──────────────────────────────────────────────────────
    trace_path = os.path.join(run_dir, "alignment_traces.jsonl")
    traces_by_ts = _parse_alignment_traces(trace_path)
    if traces_by_ts:
        print(f"  Found alignment traces: {len(traces_by_ts)} timesteps")

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = _find_log_file(run_name, run_dir)
    if log_path:
        log_blocks = _split_log_by_timestep(log_path)
        print(f"  Found log: {os.path.basename(log_path)} ({len(log_blocks)} timestep blocks)")
    else:
        log_blocks = {}
        print(f"  No log file found for {run_name}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    timesteps_out: Dict[str, Dict[str, Any]] = {}

    for t in sorted(states.keys()):
        state = states[t]
        firms_state = state.get("firms", [])

        ts_key = str(t)
        timesteps_out[ts_key] = {}

        # Base firm records from state file
        firm_records: Dict[str, Dict] = {}
        for f in firms_state:
            name = f["name"]
            firm_records[name] = {
                "in_business": f.get("in_business", True),
                "prices":      f.get("prices") or {},
                "profit":      f.get("profit", 0.0),
                "cash":        f.get("cash", 0.0),
                "sales_count": len(f.get("sales_info") or []),
                "sales_info":  f.get("sales_info") or [],
            }

        # Enrich from alignment traces (most reliable)
        if t in traces_by_ts:
            for trace in traces_by_ts[t]:
                name = trace["firm_name"]
                if name in firm_records:
                    firm_records[name].update({
                        "system_prompt": trace["system_prompt"],
                        "user_prompt":   trace["user_prompt"],
                        "llm_response":  trace["llm_response"],
                        "thought":       trace["thought"],
                        "action":        trace["action"],
                    })

        # Enrich from log file (price-matched attribution)
        if t in log_blocks:
            block = log_blocks[t]
            attributed = _attribute_price_responses(
                block["output_lines"], firms_state
            )
            for name, val in attributed.items():
                if name == "__unmatched__":
                    # Store unattributed price outputs at timestep level
                    timesteps_out[ts_key]["__unattributed_price_outputs__"] = val
                elif name in firm_records and "llm_response" not in firm_records[name]:
                    raw, thought = val
                    firm_records[name]["log_response"] = raw
                    firm_records[name]["log_thought"]  = thought

            # Non-price outputs (supply/produce) stored unattributed
            non_price = [
                o for o in block["output_lines"]
                if "price_" not in o
                and any(k in o for k in ("supply_quantity", "produce_", "LABOR"))
            ]
            if non_price:
                timesteps_out[ts_key]["__unattributed_other_outputs__"] = non_price

        timesteps_out[ts_key].update(firm_records)

    # ── Summary stats ─────────────────────────────────────────────────────────
    total = attr_trace = attr_log = bankrupt_no_resp = active_no_resp = 0
    for t_data in timesteps_out.values():
        for k, v in t_data.items():
            if k.startswith("__") or not isinstance(v, dict):
                continue
            total += 1
            if "llm_response" in v:
                attr_trace += 1
            elif "log_response" in v:
                attr_log += 1
            elif not v.get("in_business", True):
                bankrupt_no_resp += 1
            else:
                active_no_resp += 1

    active_total = total - bankrupt_no_resp
    attr_total   = attr_trace + attr_log
    pct = round(100 * attr_total / active_total, 1) if active_total else 0
    summary = {
        "firm_timestep_records":        total,
        "active_firm_steps":            active_total,
        "attributed_via_trace":         attr_trace,
        "attributed_via_log":           attr_log,
        "attributed_total":             attr_total,
        "attribution_rate_active_pct":  pct,
        "bankrupt_no_response":         bankrupt_no_resp,
        "active_unattributed":          active_no_resp,
        "note": (
            "Active-firm attribution is limited by price collisions: when multiple "
            "firms set the same price in a timestep, log-based matching cannot "
            "distinguish which thought text belongs to which firm. Unattributed "
            "responses are stored in __unattributed_price_outputs__ per timestep."
        ),
    }
    print(f"  Summary: {total} records | {attr_total}/{active_total} active attributed ({pct}%) | {bankrupt_no_resp} bankrupt skipped")

    return {
        "metadata": {
            "experiment_args": experiment_args,
            "firm_attributes": firm_attrs,
        },
        "_summary": summary,
        "timesteps": timesteps_out,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if RUNS is None:
        # Auto-discover all runs with state files
        run_names = sorted(
            name for name in os.listdir(LOG_DIR)
            if os.path.isdir(os.path.join(LOG_DIR, name))
            and glob.glob(os.path.join(LOG_DIR, name, "state_t*.json"))
        )
    else:
        run_names = RUNS

    print(f"Processing {len(run_names)} run(s): {run_names}")
    combined: Dict[str, Any] = {}

    for run_name in run_names:
        print(f"\n[{run_name}]")
        result = compile_run(run_name)
        if not result:
            continue
        combined[run_name] = result

        # Per-run output
        out_path = os.path.join(LOG_DIR, run_name, "firm_responses.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({run_name: result}, f, indent=2, default=str)
        print(f"  Written: {out_path}")

    # Combined output
    with open(OUTPUT_COMBINED, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nCombined output: {OUTPUT_COMBINED}")
    print(f"Runs compiled: {list(combined.keys())}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
