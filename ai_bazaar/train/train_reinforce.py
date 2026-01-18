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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.heartbeat_file = "train_heartbeat.txt"
        self.last_activity_time = time.time()
        self.start_time = time.time()

        self.inference_model = UnslothModel(
            self.model, self.tokenizer, heartbeat_func=self.heartbeat
        )

        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        atexit.register(self.cleanup)

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

        def run_episode(ep_idx):
            ep_start = time.time()
            print(f"  Starting Episode {ep_idx + 1}/{num_episodes}", flush=True)
            world = BazaarWorld(
                self.args, llm_model=self.inference_model
            )
            ep_utility, ep_profit, ep_sales = [], [], 0
            step_count = 0
            step_times = []

            while not world.is_done():
                step_start = time.time()
                stats = world.step()
                step_times.append(time.time() - step_start)
                step_count += 1
                ep_utility.append(
                    np.mean([c["utility"] for c in stats["consumers"].values()])
                )
                ep_profit.append(
                    np.mean([f["profit"] for f in stats["firms"].values()])
                )
                ep_sales += stats["sales_count"]
                self.heartbeat()

            ep_trajs = []
            for agent in world.firms + world.consumers:
                if hasattr(agent, "trajectory"):
                    ep_trajs.extend(agent.trajectory)
                    agent.trajectory = []

            ep_time = time.time() - ep_start
            avg_step_time = np.mean(step_times) if step_times else 0
            print(f"  Episode {ep_idx + 1}/{num_episodes} completed in {ep_time:.2f}s ({step_count} steps, {avg_step_time:.2f}s/step, {1/avg_step_time if avg_step_time > 0 else 0:.2f} steps/s)", flush=True)
            return ep_trajs, {
                "avg_utility": np.mean(ep_utility) if ep_utility else 0,
                "avg_profit": np.mean(ep_profit) if ep_profit else 0,
                "total_sales": ep_sales,
            }

        if num_episodes > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_episodes
            ) as executor:
                results = list(executor.map(run_episode, range(num_episodes)))
            for trajs, stats in results:
                all_trajectories.extend(trajs)
                iter_stats.append(stats)
        else:
            trajs, stats = run_episode(0)
            all_trajectories.extend(trajs)
            iter_stats.append(stats)

        collect_time = time.time() - collect_start
        print(f"\nCollected {num_episodes} episodes in {collect_time:.2f}s ({collect_time/num_episodes:.2f}s/episode, {num_episodes/collect_time:.2f} episodes/s)", flush=True)

        if wandb.run:
            wandb.log(
                {
                    "env/avg_utility": np.mean([s["avg_utility"] for s in iter_stats]),
                    "env/total_sales": np.sum([s["total_sales"] for s in iter_stats]),
                    "iteration": iteration,
                    "perf/collection_time_s": collect_time,
                    "perf/episodes_per_s": num_episodes/collect_time if collect_time > 0 else 0,
                }
            )
        return all_trajectories

    def train_step(self, trajectories: List[Dict[str, Any]], iteration: int):
        print(
            f"Starting training step: Iteration {iteration}, {len(trajectories)} samples",
            flush=True,
        )
        self.model.train()
        total_loss = 0

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
                continue

            try:
                enc = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(self.device)
                p_enc = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(self.device)

                outputs = self.model(**enc)
                logits = outputs.logits

                batch_loss = 0
                for j in range(len(full_texts)):
                    p_len = (
                        (p_enc.input_ids[j] != self.tokenizer.pad_token_id).sum().item()
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

                del outputs, logits, enc, p_enc
                if i % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"    Batch failed: {e}")
                continue

        self.model.save_pretrained(os.path.join(self.checkpoint_dir, "latest"))
        if wandb.run:
            wandb.log(
                {
                    "train/loss": total_loss / (len(trajectories) / batch_size + 1),
                    "iteration": iteration,
                }
            )
        return total_loss


def main():
    parser = create_argument_parser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_episodes", type=int, default=20)
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
    for i in range(args.num_iterations):
        trajs = trainer.collect_trajectories(args.num_episodes, i)
        if trajs:
            trainer.train_step(trajs, i)
    trainer.cleanup()


if __name__ == "__main__":
    main()
