import sys
import os

# Add project root to PYTHONPATH at the very beginning
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import signal
import threading
import wandb
import subprocess
import atexit
import requests
import concurrent.futures
import numpy as np
import torch
import json
from typing import List, Dict, Any, Optional
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from ai_bazaar.models.unsloth_model import UnslothModel
from ai_bazaar.env.bazaar_env import BazaarWorld
from ai_bazaar.main import create_argument_parser


class REINFORCETrainer:
    def __init__(self, model_name: str, args):
        self.args = args
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vllm_process = None
        self.checkpoint_dir = f"checkpoints/{args.run_name or 'default'}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)

        print(f"Loading model {model_name} with Unsloth for training...", flush=True)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        # Fix for Gemma3Processor: use underlying tokenizer for encoding
        # The processor's __call__ has a bug with list inputs
        if hasattr(self.tokenizer, 'tokenizer'):
            print("Using underlying tokenizer for Gemma3Processor", flush=True)
            self.encoding_tokenizer = self.tokenizer.tokenizer
        else:
            self.encoding_tokenizer = self.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.heartbeat_file = "train_heartbeat.txt"
        self.last_activity_time = time.time()
        self.start_time = time.time()

        self.inference_model = UnslothModel(
            self.model, self.tokenizer, heartbeat_func=self.heartbeat
        )

        # Preallocate GPU memory with warmup pass to prevent reallocation during rollouts
        print("Preallocating GPU memory with warmup batch...", flush=True)
        self._warmup_gpu_memory()

        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        atexit.register(self.cleanup)

    def _warmup_gpu_memory(self):
        """Preallocate GPU memory by running a warmup batch through the model.
        This prevents memory reallocation during training and claims GPU for GPU Manager."""
        max_batch_size = 128  # Match UnslothModel max_batch_size
        warmup_text = "Test prompt for memory allocation."

        # Create dummy batch
        dummy_prompts = [warmup_text] * max_batch_size

        # Tokenize
        if hasattr(self.tokenizer, "tokenizer"):
            inputs = self.tokenizer.tokenizer(
                dummy_prompts, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")
        else:
            inputs = self.tokenizer(
                dummy_prompts, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")

        # Run forward pass to allocate memory
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.7,
                use_cache=True,
                do_sample=True,
            )

        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"GPU memory preallocated: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved", flush=True)

        # Clear intermediate tensors but keep memory reserved
        del inputs
        torch.cuda.empty_cache()

    def heartbeat(self):
        self.last_activity_time = time.time()
        with open(self.heartbeat_file, "w") as f:
            f.write(str(self.last_activity_time))

    def cleanup(self):
        self.stop_monitoring = True
        if self.vllm_process:
            self.vllm_process.terminate()

    def _monitor_loop(self):
        while not self.stop_monitoring:
            time.sleep(600)
            idle = time.time() - self.last_activity_time
            if idle > 1800:
                print(f"[MONITOR] WARNING: Idle for {idle / 60:.1f}m", flush=True)

    def collect_trajectories(self, num_episodes: int, iteration: int):
        collect_start = time.time()
        all_trajectories = []
        iter_stats = []

        # SYNCHRONIZED COLLECTION: All episodes step together for maximum batching
        print(f"  Initializing {num_episodes} episodes...", flush=True)
        worlds = [
            BazaarWorld(self.args, llm_model=self.inference_model)
            for _ in range(num_episodes)
        ]
        print(f"  All {num_episodes} episodes initialized!", flush=True)

        ep_utilities = [[] for _ in range(num_episodes)]
        ep_profits = [[] for _ in range(num_episodes)]
        ep_sales = [0 for _ in range(num_episodes)]

        step_count = 0
        step_times = []

        # Step all episodes together - this creates large batches naturally
        while any(not world.is_done() for world in worlds):
            step_start = time.time()

            # All active episodes step simultaneously
            # This sends all requests to the batcher at once → large batches!
            for ep_idx, world in enumerate(worlds):
                if not world.is_done():
                    stats = world.step()
                    ep_utilities[ep_idx].append(
                        np.mean([c["utility"] for c in stats["consumers"].values()])
                    )
                    ep_profits[ep_idx].append(
                        np.mean([f["profit"] for f in stats["firms"].values()])
                    )
                    ep_sales[ep_idx] += stats["sales_count"]

            step_times.append(time.time() - step_start)
            step_count += 1
            self.heartbeat()

            if step_count % 10 == 0:
                print(f"  Step {step_count} completed", flush=True)

        # Collect trajectories from all episodes
        for ep_idx, world in enumerate(worlds):
            ep_trajs = []
            for agent in world.firms + world.consumers:
                if hasattr(agent, "trajectory"):
                    ep_trajs.extend(agent.trajectory)
                    agent.trajectory = []

            all_trajectories.extend(ep_trajs)
            iter_stats.append({
                "avg_utility": np.mean(ep_utilities[ep_idx]) if ep_utilities[ep_idx] else 0,
                "avg_profit": np.mean(ep_profits[ep_idx]) if ep_profits[ep_idx] else 0,
                "total_sales": ep_sales[ep_idx],
            })

        avg_step_time = np.mean(step_times) if step_times else 0
        print(f"  All {num_episodes} episodes completed in {time.time() - collect_start:.2f}s ({step_count} steps, {avg_step_time:.2f}s/step)", flush=True)
        collect_time = time.time() - collect_start
        print(f"\nCollected {num_episodes} episodes in {collect_time:.2f}s ({collect_time/num_episodes:.2f}s/episode, {num_episodes/collect_time:.2f} episodes/s)", flush=True)

        # Calculate format success rate
        format_valid = sum(1 for t in all_trajectories if t.get("is_format_valid", True))
        format_success_rate = format_valid / len(all_trajectories) if all_trajectories else 0

        # Calculate reward statistics
        env_rewards = [t["reward"] for t in all_trajectories if t.get("reward") is not None]
        avg_env_reward = np.mean(env_rewards) if env_rewards else 0

        print(f"Trajectory stats: {len(all_trajectories)} total, {format_success_rate:.1%} format valid, avg env reward: {avg_env_reward:.2f}", flush=True)

        if wandb.run:
            wandb.log(
                {
                    "env/avg_utility": np.mean([s["avg_utility"] for s in iter_stats]),
                    "env/avg_profit": np.mean([s["avg_profit"] for s in iter_stats]),
                    "env/total_sales": np.sum([s["total_sales"] for s in iter_stats]),
                    "trajectories/count": len(all_trajectories),
                    "trajectories/format_success_rate": format_success_rate,
                    "trajectories/format_failures": len(all_trajectories) - format_valid,
                    "rewards/env_avg": avg_env_reward,
                    "rewards/env_std": np.std(env_rewards) if env_rewards else 0,
                    "rewards/env_min": np.min(env_rewards) if env_rewards else 0,
                    "rewards/env_max": np.max(env_rewards) if env_rewards else 0,
                    "iteration": iteration,
                    "perf/collection_time_s": collect_time,
                    "perf/episodes_per_s": num_episodes/collect_time if collect_time > 0 else 0,
                }
            )
        return all_trajectories

    def train_step(self, trajectories: List[Dict[str, Any]], iteration: int):
        train_start = time.time()
        print(
            f"Starting training step: Iteration {iteration}, {len(trajectories)} samples",
            flush=True,
        )
        self.model.train()
        total_loss = 0
        successful_batches = 0
        failed_batches = 0
        skipped_samples = 0

        # Calculate baseline from environmental rewards only
        env_rewards = [t["reward"] for t in trajectories if t.get("reward") is not None]
        baseline = np.mean(env_rewards) if env_rewards else 0

        format_weight = getattr(
            self.args, "format_reward_weight", 5.0
        )  # Scaled to be significant

        batch_size = self.args.train_batch_size
        for i in range(0, len(trajectories), batch_size):
            self.heartbeat()
            batch = trajectories[i : i + batch_size]

            full_texts, prompts, batch_total_rewards = [], [], []
            for traj in batch:
                s, u, res, rew = (
                    traj.get("system_prompt", ""),
                    traj.get("user_prompt", ""),
                    traj.get("response", ""),
                    traj.get("reward"),
                )
                is_valid = traj.get("is_format_valid", True)

                if rew is None or not res:
                    skipped_samples += 1
                    continue

                # Composite reward: Environmental success + Format compliance
                format_bonus = format_weight if is_valid else -format_weight
                total_reward = rew + format_bonus

                msg = [{"role": "system", "content": s}, {"role": "user", "content": u}]
                p = self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                full_texts.append(p + res + self.tokenizer.eos_token)
                prompts.append(p)
                batch_total_rewards.append(total_reward)

            if not full_texts:
                skipped_samples += len(batch)
                continue

            # Filter out None values that can cause tokenizer to fail
            # Zip together to maintain alignment between full_texts, prompts, and rewards
            valid_data = [
                (ft, p, r)
                for ft, p, r in zip(full_texts, prompts, batch_total_rewards)
                if ft is not None and p is not None
            ]

            if not valid_data:
                print("    Batch skipped: all texts were None")
                skipped_samples += len(full_texts)
                continue

            full_texts, prompts, batch_total_rewards = zip(*valid_data)
            full_texts = list(full_texts)
            prompts = list(prompts)
            batch_total_rewards = list(batch_total_rewards)

            try:
                # Use encoding_tokenizer (underlying tokenizer for Gemma3, regular tokenizer otherwise)
                enc = self.encoding_tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(self.device)
                p_enc = self.encoding_tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(self.device)

                # Defensive checks for None values
                if enc.input_ids is None or p_enc.input_ids is None:
                    print("    Batch skipped: tokenizer returned None for input_ids")
                    continue

                outputs = self.model(**enc)
                logits = outputs.logits

                # Check if logits is None
                if logits is None:
                    print("    Batch skipped: model returned None for logits")
                    continue

                batch_loss = 0
                for j in range(len(full_texts)):
                    # Defensive check for pad_token_id
                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is None:
                        # Use eos_token_id as fallback
                        pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else 0

                    p_len = (
                        (p_enc.input_ids[j] != pad_token_id).sum().item()
                    )
                    shift_logits = logits[j, p_len - 1 : -1, :].contiguous()
                    shift_labels = enc.input_ids[j, p_len:].contiguous()

                    if shift_labels.size(0) == 0:
                        continue

                    log_probs = torch.log_softmax(shift_logits, dim=-1)
                    selected_log_probs = torch.gather(
                        log_probs, -1, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)

                    # Policy Gradient: -log_prob * (Advantage)
                    # Advantage is (Total Reward - Environmental Baseline)
                    batch_loss += -selected_log_probs.mean() * (
                        batch_total_rewards[j] - baseline
                    )

                if isinstance(batch_loss, torch.Tensor):
                    final_loss = batch_loss / len(full_texts)
                    self.optimizer.zero_grad()
                    final_loss.backward()
                    self.optimizer.step()
                    total_loss += final_loss.item()
                    successful_batches += 1

                del outputs, logits, enc, p_enc
                if i % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"    Batch failed: {e}")
                print(f"    Exception type: {type(e).__name__}")
                traceback.print_exc()
                failed_batches += 1
                continue

        train_time = time.time() - train_start
        num_batches = (len(trajectories) + batch_size - 1) // batch_size

        # Get GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        else:
            gpu_memory_allocated = 0
            gpu_memory_reserved = 0

        print(f"Training completed: {successful_batches}/{num_batches} batches successful, {failed_batches} failed, {skipped_samples} samples skipped", flush=True)
        print(f"Training time: {train_time:.2f}s, GPU memory: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_reserved:.2f}GB reserved", flush=True)

        self.model.save_pretrained(os.path.join(self.checkpoint_dir, "latest"))

        avg_loss = total_loss / successful_batches if successful_batches > 0 else 0

        if wandb.run:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/total_loss": total_loss,
                    "train/successful_batches": successful_batches,
                    "train/failed_batches": failed_batches,
                    "train/skipped_samples": skipped_samples,
                    "train/batch_success_rate": successful_batches / num_batches if num_batches > 0 else 0,
                    "train/time_s": train_time,
                    "train/samples_per_s": len(trajectories) / train_time if train_time > 0 else 0,
                    "gpu/memory_allocated_gb": gpu_memory_allocated,
                    "gpu/memory_reserved_gb": gpu_memory_reserved,
                    "iteration": iteration,
                }
            )
        return total_loss


def main():
    parser = create_argument_parser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_episodes", type=int, default=50)  # Increased from 20 to maximize GPU utilization
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--format_reward_weight", type=float, default=1.0)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        choices=["online", "offline", "disabled"],
    )
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(
        project="ai-bazaar", name=args.run_name, config=vars(args), mode=args.wandb_mode
    )

    trainer = REINFORCETrainer(args.llm, args)
    print(f"\n{'='*80}")
    print(f"Starting training: {args.num_iterations} iterations × {args.num_episodes} episodes")
    print(f"Run name: {args.run_name}")
    print(f"WandB mode: {args.wandb_mode}")
    if wandb.run:
        print(f"WandB run: {wandb.run.url}")
    print(f"{'='*80}\n")

    for i in range(args.num_iterations):
        iter_start = time.time()
        print(f"\n{'='*80}")
        print(f"ITERATION {i+1}/{args.num_iterations}")
        print(f"{'='*80}")

        trajs = trainer.collect_trajectories(args.num_episodes, i)
        if trajs:
            trainer.train_step(trajs, i)

        iter_time = time.time() - iter_start
        print(f"\nIteration {i+1} completed in {iter_time:.2f}s ({iter_time/60:.2f} min)")

        if wandb.run:
            wandb.log({
                "iteration_time_s": iter_time,
                "iteration_time_min": iter_time / 60,
                "iteration": i,
            })

    print(f"\n{'='*80}")
    print(f"Training completed! {args.num_iterations} iterations finished")
    print(f"{'='*80}\n")

    trainer.cleanup()

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
