# AI-Bazaar Ablation Matrix Status
**Updated**: 2026-01-19 02:11 EST

## ✅ STATUS: 8/9 EXPERIMENTS RUNNING

All issues have been fixed and experiments are running successfully on Pikachu.

---

## Running Experiments (8/9)

| Variant | GPU | Model | Progress | Status |
|---------|-----|-------|----------|--------|
| **nodiaries** (FIXED) | 0 | gemma-3-4b | Starting | ✅ Running with all fixes |
| nofriction | 1 | gemma-3-4b | ~t75+ | ✅ Running |
| asymmetry | 2 | gemma-3-4b | ~t73+ | ✅ Running |
| qwen3 | 3 | qwen3-8b | ~t81+ | ✅ Running |
| revenue | 4 | gemma-3-4b | ~t69+ | ✅ Running |
| final-gemma | 5 | gemma-3-4b | ~t70+ | ✅ Running |
| olmo3 | 6 | olmo3-7b | ~t88+ | ✅ Running |
| baseline | 7 | gemma-3-4b | ~t65+ | ✅ Running |

**Blocked**:
- ❌ **ministral3**: Requires transformers>=4.58 (model type not supported)

---

## Fixes Applied

### 1. NoneType in Training Batches ✅
**Commits**:
- `e63cf93`: Added defensive checks for None in tokenizer output
- `3915fc0`: Filter None values from full_texts before tokenization

**Root Cause**: Tokenizer received None values in input list, causing crash
**Fix**: Filter None from full_texts/prompts/batch_total_rewards before tokenization

**Verification**: nodiaries now running without "Batch failed" errors

### 2. Per-Job Logging ✅
**Commit**: `435d340`

**Fix**: Each job now writes to unique stdout.log and stderr.log in its log directory
**Benefit**: Can debug failed jobs without logs being overwritten

---

## Technical Details

### nodiaries Fix Iterations
1. **First attempt** (`e63cf93`): Added None checks after tokenization → Still failed
2. **Root cause found**: Tokenizer itself crashes on None in input list (gemma3/processing_gemma3.py:99)
3. **Final fix** (`3915fc0`): Filter None before tokenizer call → ✅ Working

### Error Trace (Before Fix)
```
File "train_reinforce.py", line 221
    enc = self.tokenizer(full_texts, ...)
File "gemma3/processing_gemma3.py", line 99
    elif not isinstance(text, list) and not isinstance(text[0], str):
                                                       ~~~~^^^
TypeError: 'NoneType' object is not subscriptable
```

### ministral3 Limitation
- Model type: `"ministral3"` (too new)
- transformers 4.57.3 doesn't have CONFIG_MAPPING entry for ministral3
- Error: `KeyError: 'ministral3'` when loading config
- **Workaround**: Skip for this run, add in future with newer transformers

---

## Monitoring

**Active Monitor**: PID 3727512 (checking every 5 minutes)
- Status: Running
- Log: `/tmp/monitor_jobs.log`
- Alerts: Job failures, stalls, early completions

**Check progress**:
```bash
tail -f /tmp/monitor_jobs.log
ssh Pikachu "ls -lth /data3/milkkarten/AI-Bazaar/logs/results_*/state_t*.json | head -20"
```

---

## Expected Completion

- **Gemma 3 4B variants** (6 jobs): ~12-16 hours
- **Larger models** (2 jobs): ~18-24 hours
- **Total wall time**: ~24 hours for full ablation matrix

**Deliverables**:
- 8/9 variants with 50 iterations each
- 4000 episodes total (~400,000 timesteps)
- Full WandB logging for all runs
- State files for post-analysis

---

## Git Log
```
3915fc0 fix: filter None values from full_texts before tokenization
e63cf93 fix: add defensive checks for NoneType in training batches
435d340 fix: redirect stdout/stderr to unique per-job log files
```

**Branch**: Market-v0 (all changes pushed)
