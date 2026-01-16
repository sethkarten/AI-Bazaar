import os
import time
import sys
import signal
import threading
import wandb
import subprocess
import atexit
import requests

# Unsloth must be imported before transformers
from unsloth import FastLanguageModel
import torch
import numpy as np
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer
from ai_bazaar.models.unsloth_model import UnslothModel
from ai_bazaar.env.bazaar_env import BazaarWorld
from ai_bazaar.main import create_argument_parser

# Force offline mode for HuggingFace and WandB
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"


class REINFORCETrainer:
    def __init__(self, model_name: str, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vllm_process = None

        # Start vLLM server if port is provided and we aren't using in-process unsloth
        if args.port and args.port > 0:
            self._start_vllm_server(model_name, args.port)

        # Load model with Unsloth for training
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
        self.start_time = time.time()
        self.last_activity_time = time.time()
        self.heartbeat_file = "train_heartbeat.txt"

        # Wrapped model for in-process inference (backup if vLLM fails or not used)
        self.inference_model = UnslothModel(
            self.model, self.tokenizer, heartbeat_func=self.heartbeat
        )

        # Start monitoring thread
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        atexit.register(self.cleanup)

    def _start_vllm_server(self, model_name: str, port: int):
        """Launch a background vLLM server for high-throughput inference."""
        print(f"Starting vLLM server on port {port}...", flush=True)
        vllm_cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            "0.4",
            "--disable-log-requests",
        ]

        # In a real environment, we'd need to make sure we're in the right venv
        # But here we assume the current process is already in it.
        self.vllm_process = subprocess.Popen(
            vllm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )

        # Wait for vLLM to be ready
        max_retries = 60
        url = f"http://localhost:{port}/v1/models"
        print(f"Waiting for vLLM to be ready at {url}...", flush=True)
        for i in range(max_retries):
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    print("vLLM server is ready!", flush=True)
                    return
            except:
                pass

            if self.vllm_process.poll() is not None:
                print("vLLM server failed to start!", flush=True)
                sys.exit(1)

            time.sleep(10)
            if i % 6 == 0:
                print(f"  Still waiting... ({i * 10}s)", flush=True)

        print("vLLM server timed out!", flush=True)
        self.cleanup()
        sys.exit(1)

    def cleanup(self):
        """Ensure background processes are killed."""
        if self.vllm_process:
            print("Cleaning up vLLM server...", flush=True)
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=5)
            except:
                self.vllm_process.kill()
        self.stop_monitoring = True

    def heartbeat(self):
        self.last_activity_time = time.time()
        with open(self.heartbeat_file, "w") as f:
            f.write(str(self.last_activity_time))

    def _monitor_loop(self):
        """Background thread to log progress updates and detect hangs."""
        while not self.stop_monitoring:
            time.sleep(600)  # Every 10 minutes
            elapsed = time.time() - self.start_time
            idle_time = time.time() - self.last_activity_time

            print(
                f"\n[MONITOR] Total Elapsed: {elapsed / 3600:.2f}h | Idle: {idle_time / 60:.2f}m",
                flush=True,
            )

            if idle_time > 1800:  # 30 minutes of no activity
                print(
                    f"[MONITOR] WARNING: No activity detected for {idle_time / 60:.2f}m. Potential hang!",
                    flush=True,
                )

    def collect_trajectories(self, num_episodes: int, iteration: int):
        all_trajectories = []
        iter_stats = []
        for ep in range(num_episodes):
            ep_start = time.time()

            # If a port is provided, we assume a vLLM server is running
            # Otherwise we use the in-process UnslothModel
            llm_model = None
            if not self.args.port or self.args.port == 0:
                llm_model = self.inference_model

            world = BazaarWorld(self.args, llm_model=llm_model)
            ep_utility = []
            ep_profit = []
            ep_sales = 0

            while not world.is_done():
                stats = world.step()
                ep_utility.append(
                    np.mean([c["utility"] for c in stats["consumers"].values()])
                )
                ep_profit.append(
                    np.mean([f["profit"] for f in stats["firms"].values()])
                )
                ep_sales += stats["sales_count"]
                self.heartbeat()

            # Collect from all agents
            for agent in world.firms + world.consumers:
                if hasattr(agent, "trajectory"):
                    all_trajectories.extend(agent.trajectory)
                    agent.trajectory = []  # Clear for next episode

            ep_duration = time.time() - ep_start
            print(
                f"  Episode {ep + 1}/{num_episodes} collected in {ep_duration:.2f}s",
                flush=True,
            )

            iter_stats.append(
                {
                    "avg_utility": np.mean(ep_utility),
                    "avg_profit": np.mean(ep_profit),
                    "total_sales": ep_sales,
                    "duration": ep_duration,
                }
            )

        # Log aggregates to wandb
        if wandb.run:
            wandb.log(
                {
                    "env/avg_utility": np.mean([s["avg_utility"] for s in iter_stats]),
                    "env/avg_profit": np.mean([s["avg_profit"] for s in iter_stats]),
                    "env/total_sales": np.mean([s["total_sales"] for s in iter_stats]),
                    "env/collection_duration": np.mean(
                        [s["duration"] for s in iter_stats]
                    ),
                    "iteration": iteration,
                }
            )

        return all_trajectories

    def train_step(self, trajectories: List[Dict[str, Any]], iteration: int):
        # Set to train mode and reset inference flag for UnslothModel
        self.model.train()
        self.model._is_inference = False
        total_loss = 0

        # Simple baseline (running average of rewards)
        rewards = [t["reward"] for t in trajectories if t["reward"] is not None]
        baseline = np.mean(rewards) if rewards else 0

        step_start = time.time()
        for i, traj in enumerate(trajectories):
            self.heartbeat()

            s_prompt = traj.get("system_prompt") or ""
            u_prompt = traj.get("user_prompt") or ""
            response = traj.get("response") or ""
            reward = traj.get("reward")

            if reward is None or not response:
                continue

            prompt = s_prompt + "\n" + u_prompt

            # Tokenize with workaround for Gemma 3 processor
            full_text = prompt + response
            try:
                if hasattr(self.tokenizer, "tokenizer"):
                    encodings = self.tokenizer.tokenizer(
                        full_text, return_tensors="pt"
                    ).to(self.device)
                    prompt_encodings = self.tokenizer.tokenizer(
                        prompt, return_tensors="pt"
                    ).to(self.device)
                else:
                    encodings = self.tokenizer(full_text, return_tensors="pt").to(
                        self.device
                    )
                    prompt_encodings = self.tokenizer(prompt, return_tensors="pt").to(
                        self.device
                    )
            except Exception as e:
                print(f"    Tokenizer failed in train_step: {e}")
                continue

            prompt_len = prompt_encodings.input_ids.shape[1]

            # Forward pass
            outputs = self.model(**encodings)
            logits = outputs.logits

            # Get log-probs of the response tokens
            shift_logits = logits[..., prompt_len - 1 : -1, :].contiguous()
            shift_labels = encodings.input_ids[..., prompt_len:].contiguous()

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            selected_log_probs = torch.gather(
                log_probs, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            mean_log_prob = selected_log_probs.mean()

            # Loss: -log_prob * Advantage
            advantage = reward - baseline
            loss = -mean_log_prob * advantage

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(
                    f"    Sample {i + 1}/{len(trajectories)} processed. Avg loss: {total_loss / (i + 1):.4f}"
                )

        train_duration = time.time() - step_start
        avg_loss = total_loss / len(trajectories) if trajectories else 0
        print(f"  Train step completed in {train_duration:.2f}s")

        if wandb.run:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/duration": train_duration,
                    "train/baseline": baseline,
                    "iteration": iteration,
                }
            )

        return avg_loss


def main():
    parser = create_argument_parser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    # Initialize WandB
    run_name = args.run_name or f"bazaar-{args.reward_type}-{args.llm.split('/')[-1]}"
    wandb.init(project="ai-bazaar", name=run_name, config=vars(args), mode="offline")

    # Initialize trainer
    model_name = args.llm if args.llm != "None" else "unsloth/gemma-3-4b-it-bnb-4bit"
    print(f"Starting training on {model_name}...", flush=True)
    trainer = REINFORCETrainer(model_name, args)

    def signal_handler(sig, frame):
        print("Interrupted! Saving model...", flush=True)
        trainer.model.save_pretrained("checkpoints/interrupted")
        if wandb.run:
            wandb.finish()
        trainer.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for i in range(args.num_iterations):
        iter_start = time.time()
        trainer.heartbeat()
        print(f"Iteration {i}: Collecting trajectories...", flush=True)
        trajs = trainer.collect_trajectories(args.num_episodes, i)

        if not trajs:
            print(f"Iteration {i}: No trajectories collected. Skipping.", flush=True)
            continue

        print(f"Iteration {i}: Training on {len(trajs)} samples...", flush=True)
        loss = trainer.train_step(trajs, i)

        duration = time.time() - iter_start
        total_elapsed = time.time() - trainer.start_time
        avg_iter_time = total_elapsed / (i + 1)
        remaining_iters = args.num_iterations - (i + 1)
        eta = avg_iter_time * remaining_iters

        print(
            f"Iteration {i}: Loss = {loss:.4f} | Duration = {duration:.2f}s | Total Elapsed = {total_elapsed:.2f}s | ETA = {eta / 60:.2f}m",
            flush=True,
        )

        # Save checkpoint periodically
        if (i + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/gemma3_bazaar_iter{i}"
            trainer.model.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}", flush=True)

    if wandb.run:
        wandb.finish()
    trainer.cleanup()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
