# Comprehensive Logging Setup for AI-Bazaar Experiments

**Updated**: 2026-01-19 23:40 EST
**Commit**: `524d63b`

All experiments now have comprehensive logging with unique job IDs, separate log files, and extensive WandB metrics tracking.

---

## Per-Job Logging Structure

Each job creates a unique log directory with the following files:

```
logs/results_{variant}/
├── job_info.json          # Job metadata (NEW)
├── stdout.log             # Standard output
├── stderr.log             # Standard error
└── state_t*.json          # Episode state files
```

### job_info.json Contents

```json
{
  "job_id": "d8681843-4aab-49e0-8020-a8c3f003b709",
  "run_name": "v3-ablation-nodiaries",
  "log_dir": "logs/results_nodiaries",
  "wandb_mode": "online",
  "started_at": "2026-01-19T02:11:12-05:00",
  "hostname": "Pikachu",
  "pwd": "/data3/milkkarten/AI-Bazaar",
  "git_commit": "524d63b1a2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7",
  "git_branch": "Market-v0"
}
```

**Purpose**: Uniquely identify each job run and link it to git state, timestamps, and WandB logs.

---

## WandB Logging

### Environment Metrics
- `env/avg_utility` - Average consumer utility across episodes
- `env/avg_profit` - Average firm profit across episodes
- `env/total_sales` - Total sales count across episodes

### Trajectory Metrics
- `trajectories/count` - Total trajectories collected
- `trajectories/format_success_rate` - % of valid JSON responses
- `trajectories/format_failures` - Count of JSON parsing failures

### Reward Statistics
- `rewards/env_avg` - Mean environmental reward
- `rewards/env_std` - Standard deviation of rewards
- `rewards/env_min` - Minimum reward observed
- `rewards/env_max` - Maximum reward observed

### Training Metrics
- `train/loss` - Average training loss per batch
- `train/total_loss` - Total loss across all batches
- `train/successful_batches` - Number of successful gradient updates
- `train/failed_batches` - Number of failed batches (NoneType, OOM, etc.)
- `train/skipped_samples` - Samples skipped due to missing data
- `train/batch_success_rate` - % of batches that completed successfully
- `train/time_s` - Training duration in seconds
- `train/samples_per_s` - Training throughput

### GPU Metrics
- `gpu/memory_allocated_gb` - CUDA memory allocated
- `gpu/memory_reserved_gb` - CUDA memory reserved

### Performance Metrics
- `perf/collection_time_s` - Episode collection duration
- `perf/episodes_per_s` - Episode collection throughput
- `iteration_time_s` - Total iteration time
- `iteration_time_min` - Total iteration time in minutes

### Iteration Counter
- `iteration` - Current iteration number (used for x-axis in WandB)

---

## Stdout/Stderr Logging

### Stdout Format

```
================================================================================
ITERATION 1/50
================================================================================

  Starting Episode 1/50
  Starting Episode 2/50
  ...
  Episode 1/50 completed in 120.5s (100 steps, 0.20s/step, 5.0 steps/s)
  Episode 2/50 completed in 118.2s (100 steps, 0.19s/step, 5.2 steps/s)
  ...

Collected 50 episodes in 180.5s (3.61s/episode, 0.28 episodes/s)
Trajectory stats: 1250 total, 94.2% format valid, avg env reward: 3.45

Starting training step: Iteration 0, 1250 samples
Training completed: 156/157 batches successful, 1 failed, 8 samples skipped
Training time: 245.3s, GPU memory: 13.8GB allocated, 14.2GB reserved

Iteration 1 completed in 425.8s (7.10 min)
```

### Stderr Format

All warnings, errors, and tracebacks go to stderr:

```
[main|WARNING]JSON parsing failed (attempt 0): Unterminated string
[main|WARNING]LLM output was: '{"'
[main|WARNING]firm_2 cannot cover overhead of $50.00
Traceback (most recent call last):
  File "train_reinforce.py", line 221
    ...
TypeError: 'NoneType' object is not subscriptable
```

---

## Job ID Propagation

**Environment Variable**: `GPU_MANAGER_JOB_ID`

Set by GPU Manager when launching jobs. If not set, defaults to "unknown".

**Usage in Scripts**:
```bash
JOB_ID="${GPU_MANAGER_JOB_ID:-unknown}"
echo "Job ID: $JOB_ID" >> logs/job.log
```

**Finding Job IDs**:
```bash
# From GPU Manager API
curl -s http://localhost:8080/api/jobs | jq '.[] | {id, run_name, status}'

# From job_info.json files
find logs -name job_info.json -exec jq -r '"\(.run_name): \(.job_id)"' {} \;
```

---

## WandB Configuration

**Mode**: `online` (default, can override with `--wandb_mode`)
**Project**: `ai-bazaar`
**Entity**: `princeton-ai` (if logged in)

**Run URL Pattern**:
```
https://wandb.ai/princeton-ai/ai-bazaar/runs/{run_id}
```

**Printed at job start**:
```
================================================================================
Starting training: 50 iterations × 50 episodes
Run name: v3-ablation-baseline
WandB mode: online
WandB run: https://wandb.ai/princeton-ai/ai-bazaar/runs/abc123xyz
================================================================================
```

---

## Monitoring Live Experiments

### Check Job Status
```bash
# List all running jobs
curl -s http://localhost:8080/api/jobs | jq '.[] | select(.status=="running") | {run_name: .args[13], gpu: .gpu_ids}'

# Check specific job logs
tail -f logs/results_baseline/stdout.log
tail -f logs/results_baseline/stderr.log
```

### Check WandB Progress
```bash
# Get WandB run URL from job_info.json
jq -r '.run_name' logs/results_baseline/job_info.json

# Or from GPU Manager
curl -s http://localhost:8080/api/jobs/{job_id} | jq -r '.args | .[13]'
```

### Check State Files
```bash
# Count state files per variant
for variant in nofriction asymmetry baseline; do
  count=$(ls logs/results_$variant/state_t*.json 2>/dev/null | wc -l)
  latest=$(ls -t logs/results_$variant/state_t*.json 2>/dev/null | head -1 | sed 's/.*state_t//' | sed 's/.json//')
  echo "$variant: $count files (latest: t$latest)"
done
```

---

## Debugging Failed Jobs

### 1. Check job_info.json
```bash
cat logs/results_failed_job/job_info.json
```

**Gives you**: Job ID, git commit, start time, wandb mode

### 2. Check stdout.log
```bash
tail -100 logs/results_failed_job/stdout.log
```

**Look for**:
- Last completed iteration
- Batch failure counts
- "Training completed" message (if it finished)

### 3. Check stderr.log
```bash
grep -E 'ERROR|Exception|Traceback' logs/results_failed_job/stderr.log
```

**Look for**:
- Python exceptions
- CUDA OOM errors
- JSON parsing failures

### 4. Check WandB logs
Go to WandB run URL (from job_info.json or stdout) and check:
- Last logged iteration
- Training metrics trends
- GPU memory usage over time

### 5. Check GPU Manager
```bash
curl -s http://localhost:8080/api/jobs/{job_id} | jq '{status, error, exit_code}'
```

---

## Post-Experiment Analysis

### Collect All Job Metadata
```bash
find logs -name job_info.json -exec jq -s '.' {} + > all_experiments.json
```

### Link WandB Runs to Jobs
```bash
for info in logs/*/job_info.json; do
  job_id=$(jq -r '.job_id' $info)
  run_name=$(jq -r '.run_name' $info)
  echo "$run_name ($job_id): https://wandb.ai/princeton-ai/ai-bazaar/runs/{wandb_run_id}"
done
```

### Compare Metrics Across Runs
Use WandB workspace comparison:
```
https://wandb.ai/princeton-ai/ai-bazaar/workspace?workspace=user-milkkarten
```

Or download data:
```python
import wandb
api = wandb.Api()
runs = api.runs("princeton-ai/ai-bazaar")
for run in runs:
    print(f"{run.name}: iteration {run.summary.get('iteration', 'N/A')}")
```

---

## Best Practices

1. **Always use `--wandb_mode online`** for production runs
2. **Check job_info.json immediately** after job starts to confirm logging
3. **Monitor first 2 iterations** to catch early failures
4. **Save job IDs** when submitting experiments
5. **Tag WandB runs** with experiment group names
6. **Export WandB data** after completion for offline analysis

---

## Files Modified

- `cluster_launcher.sh` - Job metadata logging
- `ai_bazaar/train/train_reinforce.py` - Comprehensive metric logging

**Commit**: `524d63b`
**Branch**: `Market-v0`
