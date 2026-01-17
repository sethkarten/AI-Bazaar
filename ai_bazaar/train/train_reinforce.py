import sys
import sys
import os
import time

# Add project root to PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

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
from typing import List, Dict, Any, Optional
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
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vllm_process = None
        self.checkpoint_dir = f"checkpoints/{args.run_name or 'default'}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Ensure log directory exists
        if args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)

        # Load model with Unsloth for training (always stays in memory)
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

    def heartbeat(self):
        self.last_activity_time = time.time()
        with open(self.heartbeat_file, "w") as f:
            f.write(str(self.last_activity_time))

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

    def _start_vllm_server(self, port: int, lora_path: Optional[str] = None):
        """Launch a background vLLM server with the latest LoRA adapter."""
        if self.vllm_process:
            self.vllm_process.terminate()
            self.vllm_process.wait()

        print(f"Starting vLLM server on port {port}...", flush=True)

        # Use a different GPU for vLLM if multi-GPU is available
        env = os.environ.copy()
        if torch.cuda.device_count() > 1:
            env["CUDA_VISIBLE_DEVICES"] = "1"
            print("  Using GPU 1 for vLLM server", flush=True)
        else:
            env["CUDA_VISIBLE_DEVICES"] = "0"
            print("  Using GPU 0 for vLLM server (sharing with training)", flush=True)

        vllm_cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            "0.8" if torch.cuda.device_count() > 1 else "0.4",
            "--disable-log-requests",
            "--trust-remote-code",
        ]

        vllm_log = open(os.path.join(self.args.log_dir, f"vllm_{port}.log"), "w")
        self.vllm_process = subprocess.Popen(
            vllm_cmd, stdout=vllm_log, stderr=subprocess.STDOUT, env=env
        )

        # Wait for vLLM to be ready
        max_retries = 60
        url = f"http://localhost:{port}/v1/models"
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
        print("vLLM server timed out!", flush=True)
        sys.exit(1)

    def collect_trajectories(self, num_episodes: int, iteration: int):
        # Start/Restart vLLM for this iteration's collection if port is provided
        llm_model = None
        if self.args.port and self.args.port > 0:
            lora_path = os.path.join(self.checkpoint_dir, "latest")
            self._start_vllm_server(
                self.args.port, lora_path if iteration > 0 else None
            )
        else:
            llm_model = self.inference_model

        all_trajectories = []
        iter_stats = []
        for ep in range(num_episodes):
            ep_start = time.time()
            world = BazaarWorld(self.args, llm_model=llm_model)
            ep_utility, ep_profit, ep_sales = [], [], 0

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

            for agent in world.firms + world.consumers:
                if hasattr(agent, "trajectory"):
                    all_trajectories.extend(agent.trajectory)
                    agent.trajectory = []

            iter_stats.append(
                {
                    "avg_utility": np.mean(ep_utility),
                    "avg_profit": np.mean(ep_profit),
                    "total_sales": ep_sales,
                }
            )
            print(f"  Episode {ep + 1}/{num_episodes} collected", flush=True)

        # Shutdown vLLM if it was used
        if self.vllm_process:
            self.vllm_process.terminate()
            self.vllm_process.wait()
            self.vllm_process = None

        if wandb.run:
            wandb.log(
                {
                    "env/avg_utility": np.mean([s["avg_utility"] for s in iter_stats]),
                    "iteration": iteration,
                }
            )
        return all_trajectories

    def train_step(self, trajectories: List[Dict[str, Any]], iteration: int):
        print(
            f"Starting training step for iteration {iteration} with {len(trajectories)} samples...",
            flush=True,
        )
        self.model.train()
        total_loss = 0

        rewards = [t["reward"] for t in trajectories if t["reward"] is not None]
        baseline = np.mean(rewards) if rewards else 0
        print(f"  Baseline reward: {baseline:.4f}", flush=True)

        step_start = time.time()
        for i, traj in enumerate(trajectories):
            self.heartbeat()
            # ...
            s_prompt = traj.get("system_prompt") or ""
            u_prompt = traj.get("user_prompt") or ""
            response = traj.get("response") or ""
            reward = traj.get("reward")
            if reward is None or not response:
                continue

            prompt = s_prompt + "\n" + u_prompt
            full_text = prompt + response
            try:
                if hasattr(self.tokenizer, "tokenizer"):
                    enc = self.tokenizer.tokenizer(full_text, return_tensors="pt").to(
                        self.device
                    )
                    p_enc = self.tokenizer.tokenizer(prompt, return_tensors="pt").to(
                        self.device
                    )
                else:
                    enc = self.tokenizer(full_text, return_tensors="pt").to(self.device)
                    p_enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            except:
                continue

            prompt_len = p_enc.input_ids.shape[1]
            outputs = self.model(**enc)
            logits = outputs.logits
            shift_logits = logits[..., prompt_len - 1 : -1, :].contiguous()
            shift_labels = enc.input_ids[..., prompt_len:].contiguous()
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            selected_log_probs = torch.gather(
                log_probs, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            loss = -selected_log_probs.mean() * (reward - baseline)
            print(
                f"    Sample {i + 1}/{len(trajectories)}: Reward={reward:.2f}, Baseline={baseline:.2f}, Loss={loss.item():.4f}",
                flush=True,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        # Save LoRA for next collection
        self.model.save_pretrained(os.path.join(self.checkpoint_dir, "latest"))

        avg_loss = total_loss / len(trajectories) if trajectories else 0
        if wandb.run:
            wandb.log({"train/loss": avg_loss, "iteration": iteration})
        return avg_loss


def main():
    parser = create_argument_parser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    wandb.init(
        project="ai-bazaar", name=args.run_name, config=vars(args), mode="offline"
    )
    trainer = REINFORCETrainer(args.llm, args)

    def signal_handler(sig, frame):
        trainer.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for i in range(args.num_iterations):
        trajs = trainer.collect_trajectories(args.num_episodes, i)
        if trajs:
            trainer.train_step(trajs, i)

    trainer.cleanup()


if __name__ == "__main__":
    main()
