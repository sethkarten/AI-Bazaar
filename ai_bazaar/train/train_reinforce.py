import os
import time
import sys
import signal

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

        # Load model with Unsloth
        print(f"Loading model {model_name} with Unsloth...")
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
        self.heartbeat_file = "train_heartbeat.txt"

        # Wrapped model for inference during collection
        self.inference_model = UnslothModel(
            self.model, self.tokenizer, heartbeat_func=self.heartbeat
        )

    def heartbeat(self):
        with open(self.heartbeat_file, "w") as f:
            f.write(str(time.time()))

    def collect_trajectories(self, num_episodes: int):
        all_trajectories = []
        for ep in range(num_episodes):
            ep_start = time.time()
            # Pass our active policy to the world
            world = BazaarWorld(self.args, llm_model=self.inference_model)
            while not world.is_done():
                world.step()
                self.heartbeat()

            # Collect from all agents
            for agent in world.firms + world.consumers:
                if hasattr(agent, "trajectory"):
                    all_trajectories.extend(agent.trajectory)
                    agent.trajectory = []  # Clear for next episode

            print(
                f"  Episode {ep + 1}/{num_episodes} collected in {time.time() - ep_start:.2f}s"
            )
        return all_trajectories

    def train_step(self, trajectories: List[Dict[str, Any]]):
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
            prompt = traj["system_prompt"] + "\n" + traj["user_prompt"]
            response = traj["response"]
            reward = traj["reward"]

            if reward is None:
                continue

            # Tokenize
            full_text = prompt + response
            encodings = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            prompt_encodings = self.tokenizer(prompt, return_tensors="pt").to(
                self.device
            )

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

        print(f"  Train step completed in {time.time() - step_start:.2f}s")
        return total_loss / len(trajectories) if trajectories else 0


def main():
    parser = create_argument_parser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--num_iterations", type=int, default=50)
    args = parser.parse_args()

    # Initialize trainer
    model_name = args.llm if args.llm != "None" else "unsloth/gemma-3-4b-it-bnb-4bit"
    print(f"Starting training on {model_name}...", flush=True)
    trainer = REINFORCETrainer(model_name, args)

    def signal_handler(sig, frame):
        print("Interrupted! Saving model...", flush=True)
        trainer.model.save_pretrained("checkpoints/interrupted")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for i in range(args.num_iterations):
        iter_start = time.time()
        trainer.heartbeat()
        print(f"Iteration {i}: Collecting trajectories...", flush=True)
        trajs = trainer.collect_trajectories(args.num_episodes)

        if not trajs:
            print(f"Iteration {i}: No trajectories collected. Skipping.", flush=True)
            continue

        print(f"Iteration {i}: Training on {len(trajs)} samples...", flush=True)
        loss = trainer.train_step(trajs)

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


if __name__ == "__main__":
    main()
