import os
import torch
import numpy as np
import json
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from ai_bazaar.env.bazaar_env import BazaarWorld
from ai_bazaar.main import create_argument_parser


class REINFORCETrainer:
    def __init__(self, model_name: str, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

    def collect_trajectories(self, num_episodes: int):
        all_trajectories = []
        for _ in range(num_episodes):
            world = BazaarWorld(self.args)
            while not world.is_done():
                world.step()

            # Collect from all agents
            for agent in world.firms + world.consumers:
                if hasattr(agent, "trajectory"):
                    all_trajectories.extend(agent.trajectory)
                    agent.trajectory = []  # Clear for next episode
        return all_trajectories

    def train_step(self, trajectories: List[Dict[str, Any]]):
        self.model.train()
        total_loss = 0

        # Simple baseline (running average of rewards)
        rewards = [t["reward"] for t in trajectories if t["reward"] is not None]
        baseline = np.mean(rewards) if rewards else 0

        for traj in trajectories:
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
            # Shift logits and labels
            # logits: [batch, seq_len, vocab_size]
            # labels: [batch, seq_len]
            shift_logits = logits[..., prompt_len - 1 : -1, :].contiguous()
            shift_labels = encodings.input_ids[..., prompt_len:].contiguous()

            # Calculate log_softmax
            log_probs = torch.log_softmax(shift_logits, dim=-1)

            # Gather the log-probs of the actual tokens
            # shift_labels needs to be [batch, seq_len, 1] for gather
            selected_log_probs = torch.gather(
                log_probs, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Mean log-prob for the whole response (REINFORCE)
            # REINFORCE++ would typically include a kl-penalty or more advanced scaling
            mean_log_prob = selected_log_probs.mean()

            # Loss: -log_prob * Advantage
            advantage = reward - baseline
            loss = -mean_log_prob * advantage

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(trajectories) if trajectories else 0


def main():
    parser = create_argument_parser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_episodes", type=int, default=5)
    args = parser.parse_args()

    # Initialize trainer with Gemma 3-4B (Placeholder name)
    model_name = args.llm if args.llm != "None" else "google/gemma-3-4b-it"
    trainer = REINFORCETrainer(model_name, args)

    for i in range(50):  # Training iterations
        print(f"Iteration {i}: Collecting trajectories...")
        trajs = trainer.collect_trajectories(args.num_episodes)
        print(f"Iteration {i}: Training on {len(trajs)} samples...")
        loss = trainer.train_step(trajs)
        print(f"Iteration {i}: Loss = {loss:.4f}")

        # Save checkpoint periodically
        if i % 10 == 0:
            trainer.model.save_pretrained(f"checkpoints/gemma3_bazaar_iter{i}")


if __name__ == "__main__":
    main()
