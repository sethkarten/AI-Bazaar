# AI-Bazaar Experiment Progress Report
**Updated**: 2026-01-19 23:35 EST (Current time: ~26.5 hours since start)

## Summary: 6/9 Jobs Running (2 Completed Early with Errors, 1 Blocked)

---

## Currently Running (6 jobs - original batch from Jan 18)

These are the ORIGINAL jobs from before the fixes. They've been running for ~26 hours.

| Variant | GPU | Model | Runtime | Latest State | Estimated Progress |
|---------|-----|-------|---------|--------------|-------------------|
| nofriction | 1 | gemma-3-4b | 1556 min (~26h) | t55 | Still in iteration 0 |
| asymmetry | 2 | gemma-3-4b | 1555 min (~26h) | t42 | Still in iteration 0 |
| revenue | 4 | gemma-3-4b | 1551 min (~26h) | t34 | Still in iteration 0 |
| final-gemma | 5 | gemma-3-4b | 1550 min (~26h) | t44 | Still in iteration 0 |
| olmo3 | 6 | olmo3-7b | 1546 min (~26h) | t65 | Still in iteration 0 |
| baseline | 7 | gemma-3-4b | 1540 min (~26h) | t44 | Still in iteration 0 |

**Status**: All 6 jobs have 100 state files (1 full episode) but appear to still be in iteration 0.

**Issue**: These jobs are running the OLD code without the NoneType fixes, so they may be experiencing batch failures during training.

---

## Completed Early (2 jobs - with errors)

### 1. nodiaries ❌
- **Status**: Completed at 23:48 (Jan 18)
- **Runtime**: 2h 37min
- **Output**: 100 state files (only 1 iteration completed)
- **Issue**: Massive batch failures ("Batch failed: 'NoneType' object is not subscriptable")
- **Reason**: Ran OLD code before the fix was applied

### 2. qwen3 (8B model) ❌
- **Status**: Completed at 21:53 (Jan 18)
- **Runtime**: ~2h 20min
- **Output**: 100 state files (only 1 iteration completed)
- **Likely Issue**: Same batching errors as nodiaries

---

## Blocked

### ministral3 ❌
- **Issue**: Model type not supported in transformers 4.57.3
- **Status**: Not submitted

---

## Problem Analysis

### The 6 Running Jobs Are Slow
After ~26 hours, they're still in iteration 0. This is **extremely slow**. Expected performance:
- **Expected**: ~25 min/iteration → should have ~60 iterations done by now
- **Actual**: Still on iteration 0 after 26 hours

**Likely causes:**
1. **Old code without fixes**: These jobs don't have the None filtering fix (commit `3915fc0`)
2. **Training failures**: Probably experiencing same batch errors as nodiaries
3. **Stuck in training loop**: May be failing all batches but not crashing

### What Happened to nodiaries and qwen3
Both completed after only 1 iteration:
- Collected 50 episodes (100 state files per iteration)
- Tried to train on the data
- Hit massive batch failures
- Completed the iteration loop (50 iterations)
- But only generated output for iteration 0

---

## What Should Have Happened

With fixes applied (commits `3915fc0`, `e63cf93`, `435d340`):
1. ✅ None values filtered before tokenization
2. ✅ Unique logs per job
3. ✅ Proper error handling
4. ✅ ~25 min/iteration throughput
5. ✅ 50 iterations in ~20-24 hours

---

## Recommendations

### Option 1: Let Current Jobs Complete (Conservative)
- **Pro**: May finish eventually, don't lose 26 hours of compute
- **Con**: Could take another 50+ hours at current rate
- **Risk**: May fail like nodiaries/qwen3 did

### Option 2: Restart All Jobs with Fixes (Aggressive)
- **Pro**: Will complete in ~20 hours with proper fixes
- **Con**: Lose 26 hours of compute
- **Action**: Kill all 6 running jobs, resubmit with fixed code

### Option 3: Hybrid Approach
- Let 6 jobs continue
- Submit NEW jobs with fixes to free GPUs (0, 3)
- Compare results when done
- Keep whichever completes properly

---

## Current GPU Status (from nvidia-smi)

```
GPU 0:  0% util,    18 MB (IDLE - qwen3 completed)
GPU 1: 52% util, 13849 MB (nofriction - running)
GPU 2: 56% util, 13849 MB (asymmetry - running)
GPU 3:  0% util,    18 MB (IDLE - qwen3 completed)
GPU 4: 56% util, 13851 MB (revenue - running)
GPU 5: 58% util, 13849 MB (final-gemma - running)
GPU 6: 57% util, 33935 MB (olmo3 - running)
GPU 7:  7% util, 13851 MB (baseline - running)
```

**Available GPUs**: 0, 3 (can run 2 new jobs with fixes)

---

## Next Steps (Awaiting User Decision)

**Question for user**: Should we:
1. Let the 6 slow jobs continue and see if they complete?
2. Kill all 6 jobs and restart with fixes?
3. Start 2 new fixed jobs on GPUs 0 & 3 while others continue?

**If restarting**, priority variants:
1. **nodiaries** (needs rerun with fix)
2. **qwen3** (needs rerun with fix, 8B model)
3. Pick 4 from: nofriction, asymmetry, baseline, revenue, final-gemma, olmo3

**Current code status**: All fixes committed and pushed to Market-v0 branch.
