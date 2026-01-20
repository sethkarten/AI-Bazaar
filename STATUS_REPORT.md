# Experiment Status Report
**Time**: 2026-01-18 19:50 EST

## 🚨 CRITICAL ISSUE IDENTIFIED

### Root Cause: VLLM Memory Exhaustion
**Problem**: Jobs are configured with `--port 0` (in-process Unsloth) but VLLM engine is still being initialized, causing memory conflicts.

**Error Message**:
```
ValueError: No available memory for the cache blocks
Available KV cache memory: -20.31 GiB (NEGATIVE!)
```

**Impact**: 2/9 jobs have FAILED, 7/9 still running but at risk

---

## Failed Jobs (2/9)

### 1. nodiaries ❌
- **Status**: Completed early (crashed)
- **Runtime**: 2h 35m (16:27 → 19:02)
- **Output**: Only 100 state files (1 episode instead of 50 iterations)
- **Error**: VLLM OOM at 00:58:43
- **Root Cause**: `-20.31 GiB` available KV cache memory

### 2. ministral3 ❌
- **Status**: Completed early (crashed)
- **Runtime**: ~13 minutes (19:33 → 19:46)
- **Output**: 0 state files (crashed immediately)
- **Likely Cause**: Same VLLM OOM issue (8B model needs more memory)

---

## Running Jobs (7/9) ⚠️

| Variant | GPU | Status | Latest Progress | At Risk? |
|---------|-----|--------|----------------|----------|
| nofriction | 1 | Running | t21/100 | ⚠️ Yes |
| asymmetry | 2 | Running | t19/100 | ⚠️ Yes |
| qwen3 | 3 | Running | t54/100 | ⚠️ Yes (8B model) |
| revenue | 4 | Running | t16/100 | ⚠️ Yes |
| final-gemma | 5 | Running | t16/100 | ⚠️ Yes |
| olmo3 | 6 | Running | t27/100 | ⚠️ Yes (7B model) |
| baseline | 7 | Running | t10/100 | ⚠️ Yes |

**All running jobs have VLLM log files**, indicating they also started VLLM engines despite `--port 0`. They may crash later.

---

## Monitoring System ✅

**Status**: Active (PID 3727512)
**Check Interval**: 5 minutes
**Log File**: `/tmp/monitor_jobs.log`

**Alerts**:
- Job failures
- Progress stalls (no new timesteps)
- VLLM errors in logs
- Early completions (<1000 timesteps)

**View live**: `tail -f /tmp/monitor_jobs.log`

---

## Technical Analysis

### Why VLLM is Starting
Despite `--port 0` flag (which should use in-process Unsloth), all jobs have `vllm_*.log` files in their result directories. This suggests:

1. Code may not properly handle `--port 0` condition
2. VLLM initialization may happen before port check
3. Possible race condition in model loading

### Memory Calculation
- **GPU Memory**: 48 GB per A6000
- **Model Size**: ~13.7 GB (4B models), ~27 GB (8B models), ~34 GB (7B models)
- **VLLM KV Cache**: Tries to allocate remaining memory
- **Conflict**: Unsloth + VLLM both trying to use same memory → OOM

---

## Recommendations

### Immediate Actions
1. ✅ **Monitor running jobs** - Watch for crashes (every 5 min)
2. ⚠️ **Expect more failures** - All jobs have VLLM logs
3. 🔍 **Check code** - Investigate why `--port 0` doesn't prevent VLLM startup

### Code Fix Needed
**File**: `ai_bazaar/train/train_reinforce.py` or model initialization
**Issue**: VLLM being initialized even with `--port 0`
**Fix**: Ensure port check happens BEFORE any model/server initialization

### Resubmission Strategy
**Option 1**: Fix code and resubmit all 9 jobs
**Option 2**: Use different port (e.g., `--port 8000`) to explicitly use VLLM
**Option 3**: Remove batching/VLLM entirely for now

---

## Next Steps

1. **Immediate**: Continue monitoring (system active)
2. **Within 1 hour**: Investigate code to find VLLM initialization
3. **If more jobs crash**: Kill all jobs and fix before resubmitting
4. **If jobs survive**: They may complete but uncertain
