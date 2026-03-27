"""REINFORCE++ trainer for stabilizing firm in AI-Bazaar marketplace.

Trains a single stabilizing firm (LoRA on Qwen3.5-9B bf16) using policy
gradient with composite reward (profit + market survival + price floor).
Other firms use the base model with their assigned personas.

Usage:
    python -m ai_bazaar.train.train_reinforce \
        --llm unsloth/Qwen3.5-9B \
        --num-stabilizing-firms 1 \
        --num-firms 4 --firm-personas "competitive:2,reactive:1" \
        --num_episodes 2 --num_iterations 25 \
        --sft_warmup 200 --sft_epochs 3
"""
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import math
import time
import threading
import wandb
import random
import atexit
import concurrent.futures
import numpy as np
import torch
import json
import traceback
from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer

from ai_bazaar.models.unsloth_model import UnslothModel
from ai_bazaar.env.bazaar_env import BazaarWorld
from ai_bazaar.agents.firm import FirmAgent
from ai_bazaar.main import create_argument_parser


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class WelfordRunningStats:
    """Welford's online algorithm for running mean/variance."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def variance(self):
        return self.M2 / self.n if self.n > 1 else 1.0

    def normalize(self, x: float) -> float:
        return (x - self.mean) / (math.sqrt(self.variance) + 1e-8)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class REINFORCETrainer:
    def __init__(self, model_name: str, args):
        self.args = args
        self.model_name = model_name
        self.checkpoint_dir = f"checkpoints/{args.run_name or 'default'}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)

        num_gpus = torch.cuda.device_count()
        self.device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
        self.device_base = torch.device("cuda:1") if num_gpus >= 2 else self.device
        print(f"Training GPU: {self.device} | Inference GPU: {self.device_base} ({num_gpus} GPU(s))", flush=True)

        # ── GPU 0: 4-bit QLoRA model (stabilizing firm — trained) ────
        # Use AutoModelForCausalLM to load as text-only (Unsloth loads as
        # multimodal ConditionalGeneration which breaks in training forward)
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig
        print(f"Loading {model_name} as 4-bit CausalLM + LoRA on {self.device} …", flush=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map=str(self.device),
        )
        lora_cfg = LoraConfig(
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16, lora_dropout=0, bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.gradient_checkpointing_enable()
        self.model.print_trainable_parameters()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoding_tokenizer = (
            self.tokenizer.tokenizer if hasattr(self.tokenizer, "tokenizer") else self.tokenizer
        )

        # ── GPU 1: Frozen 4-bit base model (non-stabilizing firms — inference only)
        inference_bs = getattr(args, "inference_batch_size", 32)
        max_gen_tokens = getattr(args, "max_tokens", 256)

        if self.device_base != self.device:
            print(f"Loading frozen 4-bit base model on {self.device_base} …", flush=True)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map=str(self.device_base),
            )
            self.base_model.eval()
            for p in self.base_model.parameters():
                p.requires_grad = False
            self.inference_model_base = UnslothModel(
                self.base_model, self.tokenizer,
                heartbeat_func=self.heartbeat,
                encoding_tokenizer=self.encoding_tokenizer,
                device=self.device_base,
                max_batch_size=inference_bs,
                max_tokens=max_gen_tokens,
            )
        else:
            self.base_model = None
            self.inference_model_base = None

        # ── Stabilizing firm inference wrapper (GPU 0) ───────────────
        self.inference_model = UnslothModel(
            self.model, self.tokenizer,
            heartbeat_func=self.heartbeat,
            encoding_tokenizer=self.encoding_tokenizer,
            device=self.device,
            max_batch_size=inference_bs,
            max_tokens=max_gen_tokens,
        )

        # ── Optimizer (only LoRA params) ─────────────────────────────
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=args.lr,
        )

        # ── REINFORCE++ hyper-parameters ─────────────────────────────
        self.reward_stats = WelfordRunningStats()
        self.advantage_clip = getattr(args, "advantage_clip", 5.0)
        self.grad_clip_norm = getattr(args, "grad_clip_norm", 1.0)
        rw = getattr(args, "reward_weights", "0.3,0.3,0.3")
        self.reward_weights = [float(w) for w in rw.split(",")]
        self.survival_bonus = getattr(args, "survival_bonus", 5.0)

        # ── Resume from checkpoint if available ─────────────────────
        self.start_iteration = 0
        resume_path = os.path.join(self.checkpoint_dir, "train_state.pt")
        lora_path = os.path.join(self.checkpoint_dir, "latest")
        if getattr(args, "resume", False) and os.path.exists(resume_path):
            print(f"Resuming from {resume_path} …", flush=True)
            state = torch.load(resume_path, map_location="cpu")
            self.optimizer.load_state_dict(state["optimizer"])
            self.start_iteration = state["iteration"] + 1
            rs = state.get("reward_stats", {})
            self.reward_stats.n = rs.get("n", 0)
            self.reward_stats.mean = rs.get("mean", 0.0)
            self.reward_stats.M2 = rs.get("M2", 0.0)
            if os.path.exists(lora_path):
                from peft import PeftModel
                self.model.load_adapter(lora_path, adapter_name="default")
                print(f"Loaded LoRA from {lora_path}", flush=True)
            print(f"Resuming from iteration {self.start_iteration}", flush=True)

        mem0 = torch.cuda.memory_allocated(0) / 1024**3
        mem1 = torch.cuda.memory_allocated(1) / 1024**3 if num_gpus >= 2 else 0
        print(f"GPU memory: {mem0:.1f} GB (train) + {mem1:.1f} GB (inference)", flush=True)

        # ── Monitoring ───────────────────────────────────────────────────
        self.heartbeat_file = "train_heartbeat.txt"
        self.last_activity_time = time.time()
        self.start_time = time.time()
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        atexit.register(self.cleanup)

    # ── helpers ──────────────────────────────────────────────────────────

    def heartbeat(self):
        self.last_activity_time = time.time()
        with open(self.heartbeat_file, "w") as f:
            f.write(str(self.last_activity_time))

    def cleanup(self):
        self.stop_monitoring = True

    def _monitor_loop(self):
        while not self.stop_monitoring:
            time.sleep(600)
            idle = time.time() - self.last_activity_time
            if idle > 1800:
                print(f"[MONITOR] WARNING: Idle for {idle/60:.1f}m", flush=True)

    # ── SFT warmup ──────────────────────────────────────────────────────

    def sft_warmup(self, num_examples: int = 200, num_epochs: int = 3):
        """Supervised fine-tuning on synthetic (prompt, JSON) pairs.

        Uses the **exact same system prompt** as a stabilizing FirmAgent and
        the same user-prompt templates from firm.py:add_message().
        """
        import random as rng
        print(f"\n{'='*60}\nSFT WARMUP: {num_examples} examples × {num_epochs} epochs\n{'='*60}", flush=True)

        goods_list = ["food", "clothing", "electronics", "furniture"][:getattr(self.args, "num_goods", 1)]

        # ── Build system prompt identical to FirmAgent._create_system_prompt(stabilizing=True) ──
        from ai_bazaar.utils.common import FIRM_PERSONA_DESCRIPTIONS
        goods_str = ", ".join(goods_list)
        n_consumers = getattr(self.args, "num_consumers", 10)
        system_prompt = f"""You are a firm manager named stabilizing_firm that produces and sells goods in a market economy.
You produce the following goods: {goods_str}.
Each timestep represents one day. Consumer income and demand are on a daily scale.
This market has {n_consumers} consumers in total.

Your goal is to sustain your business by making strategic decisions about:
1. Pricing: Set prices for your goods to earn revenue and sustain your business
2. Supply purchasing: Buy raw supplies to produce goods
3. Production: Convert supplies into finished goods efficiently

You must balance inventory management, cash flow, and market demand to succeed.
You make decisions based on historical data about your performance, market conditions, and available resources.
Track your cash, inventory, prices, and competitor behavior each step.

## Your Business Strategy
You are a **stabilizing firm** — a price anchor in this market. Your strategy is to hold a stable price at a target markup above your unit cost, regardless of what competitors do.

Rules:
1. Your minimum price is always your unit cost factoring in overhead costs. Never sell below it — you will go bankrupt.
2. When competitors drop prices below your target: DO NOT FOLLOW THEM DOWN. Hold your price.
3. Buy supply conservatively — purchase only what you expect to sell at your target price.
4. Stable prices are your brand and your competitive advantage.

CRITICAL: Always respond with a single, valid JSON object. Do not use markdown code blocks or include explanatory text. Output only the JSON object that can be parsed directly."""

        # ── Generate examples matching firm.py add_message() format ──
        examples = []
        for _ in range(num_examples):
            cash = rng.uniform(50, 2000)
            inv = {g: round(rng.uniform(0, 200), 1) for g in goods_list}
            costs = {g: round(rng.uniform(0.5, 5.0), 2) for g in goods_list}
            supply_avail = round(rng.uniform(0, 500), 2)
            decision = rng.choice(["price", "supply", "produce"])

            if decision == "price":
                price_fmt = "{" + ", ".join(f'"price_{g}": 5.00' for g in goods_list) + "}"
                user = f"Cash: ${cash:.2f}\nCurrent inventory: {dict(inv)}\n"
                user += f"Set a price (positive number) for each good. "
                user += f"Respond with ONLY this JSON (replace numbers with your choices): {price_fmt}\n"
                resp = "{" + ", ".join(f'"price_{g}": {rng.uniform(costs[g]*1.2, costs[g]*2.5):.2f}' for g in goods_list) + "}"

            elif decision == "supply":
                supply_fmt = "{" + ", ".join(f'"supply_quantity_{g}": 100' for g in goods_list) + "}"
                cost_str = ", ".join(f"{g}: ${c:.2f}" for g, c in costs.items())
                user = f"Cash: ${cash:.2f}\nSupply unit costs per good: {cost_str}\n"
                user += f"Decide how many units of supply to purchase per good (positive numbers). "
                user += f"Respond with ONLY this JSON (replace numbers with your choices): {supply_fmt}\n"
                max_q = cash / (max(costs.values()) * len(goods_list))
                resp = "{" + ", ".join(f'"supply_quantity_{g}": {rng.uniform(5, min(max_q, 300)):.0f}' for g in goods_list) + "}"

            else:
                even = 100 // len(goods_list)
                prod_fmt = "{" + ", ".join(f'"produce_{g}": {even}' for g in goods_list) + "}"
                user = f"Available supply: {supply_avail:.2f}\n"
                user += f"Decide what percentage of supply to allocate to each good (positive numbers, should sum to ~100). "
                user += f"Respond with ONLY this JSON (replace numbers with your choices): {prod_fmt}\n"
                pcts = [rng.uniform(10, 50) for _ in goods_list]
                s = sum(pcts)
                resp = "{" + ", ".join(f'"produce_{g}": {p/s*100:.0f}' for g, p in zip(goods_list, pcts)) + "}"

            examples.append((system_prompt, user, resp))

        # ── Train ──
        self.model.train()
        sft_opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=2e-5)
        batch_size = getattr(self.args, "micro_batch_size", 4)  # SFT uses micro_batch_size (not train_batch_size)
        total_loss, total_batches = 0.0, 0

        for epoch in range(num_epochs):
            rng.shuffle(examples)
            ep_loss, ep_batches = 0.0, 0
            for b in range(0, len(examples), batch_size):
                batch = examples[b:b+batch_size]
                full_texts, prompt_lens = [], []
                for sp, up, rp in batch:
                    msg = [{"role": "system", "content": sp}, {"role": "user", "content": up}]
                    p = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    full_texts.append(p + rp + self.tokenizer.eos_token)
                    prompt_lens.append(len(self.encoding_tokenizer(p, truncation=True, max_length=4096).input_ids))

                try:
                    enc = self.encoding_tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(self.device)
                    logits = self.model(**enc).logits.float()  # bf16→f32 for stable loss

                    loss, cnt = torch.tensor(0.0, device=self.device), 0
                    pad_id = self.tokenizer.pad_token_id or 0
                    for j in range(len(full_texts)):
                        pl = prompt_lens[j]
                        sl = logits[j, pl-1:-1, :].contiguous()
                        lb = enc.input_ids[j, pl:].contiguous()
                        if lb.size(0) == 0:
                            continue
                        loss = loss + torch.nn.functional.cross_entropy(sl, lb, ignore_index=pad_id)
                        cnt += 1
                    if cnt > 0:
                        loss = loss / cnt
                        sft_opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
                        sft_opt.step()
                        ep_loss += loss.item(); ep_batches += 1
                    del logits, enc; torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  SFT batch failed: {e}", flush=True)
                    traceback.print_exc()

            avg = ep_loss / ep_batches if ep_batches else 0
            print(f"  SFT Epoch {epoch+1}/{num_epochs}: loss={avg:.4f} ({ep_batches} batches)", flush=True)
            total_loss += ep_loss; total_batches += ep_batches
            if wandb.run:
                wandb.log({"sft/epoch": epoch+1, "sft/loss": avg, "sft/batches": ep_batches})

        self.model.eval()
        self.model.save_pretrained(os.path.join(self.checkpoint_dir, "sft_warmup"))
        print(f"SFT warmup done: avg loss={total_loss/total_batches if total_batches else 0:.4f}\n{'='*60}\n", flush=True)

    # ── Curriculum ─────────────────────────────────────────────────────

    def get_num_stabilizing(self, iteration, ep_idx):
        """Curriculum: 5/5 → 4/5 → 3/5, then sample {1..5}."""
        if not getattr(self.args, "curriculum", True):
            return getattr(self.args, "num_stabilizing_firms", 1)
        if iteration < 20:
            return 5
        elif iteration < 40:
            return 4
        elif iteration < 60:
            return 3
        else:
            return random.Random(self.args.seed + iteration * 1000 + ep_idx).randint(1, 5)

    # ── Episode collection ──────────────────────────────────────────────

    def collect_trajectories(self, num_episodes: int, iteration: int):
        t0 = time.time()
        all_trajs, all_survived, stats_list = [], [], []
        w_profit, w_survival, w_floor = self.reward_weights[:3]
        market_seed = self.args.seed + iteration * 1000

        def run_episode(ep_idx):
            t_ep = time.time()

            # Curriculum: vary num_stabilizing_firms
            n_stab = self.get_num_stabilizing(iteration, ep_idx)
            # Thread-safe: copy args to avoid race conditions
            import copy
            ep_args = copy.copy(self.args)
            ep_args.num_stabilizing_firms = n_stab

            # Fixed market structure: same costs/preferences across episodes
            np.random.seed(market_seed)
            random.seed(market_seed)
            world = BazaarWorld(
                ep_args,
                llm_model=self.inference_model,
                llm_model_base=self.inference_model_base,
            )
            # Re-seed for episode-specific dynamics (LLM sampling, Poisson demand)
            np.random.seed(market_seed + ep_idx + 1)
            random.seed(market_seed + ep_idx + 1)

            ep_sales, steps = 0, 0
            total_stab_profit = 0.0
            steps_above_cost = 0
            max_ts = ep_args.max_timesteps

            while not world.is_done():
                try:
                    st = world.step()
                except (ValueError, RecursionError) as e:
                    print(f"  Ep {ep_idx+1}: step {steps} crashed ({type(e).__name__})", flush=True)
                    break
                steps += 1
                ep_sales += st["sales_count"]

                # Track stabilizing firm metrics
                for fn, fd in st["firms"].items():
                    for f in world.firms:
                        if f.name == fn and getattr(f, "stabilizing_firm", False):
                            total_stab_profit += fd.get("profit", 0.0)
                            prices = fd.get("prices", {})
                            costs = getattr(f, "supply_unit_costs", {})
                            if all(prices.get(g, 0) >= costs.get(g, 1.0) for g in costs):
                                steps_above_cost += 1
                self.heartbeat()

            # Collect stabilizing firm trajectories
            ep_trajs = []
            for agent in world.firms:
                if getattr(agent, "stabilizing_firm", False) and hasattr(agent, "trajectory"):
                    ep_trajs.extend(agent.trajectory)
                    agent.trajectory = []

            # ── Economic metrics for paper ──
            stab_survived = any(
                getattr(f, "in_business", True) for f in world.firms if getattr(f, "stabilizing_firm", False)
            )
            final_alive = sum(1 for f in world.firms if getattr(f, "in_business", True))
            total_firms = len(world.firms)

            # Per-firm-type survival and profit
            stab_alive = sum(1 for f in world.firms if getattr(f, "stabilizing_firm", False) and getattr(f, "in_business", True))
            stab_total = sum(1 for f in world.firms if getattr(f, "stabilizing_firm", False))
            nonstab_alive = sum(1 for f in world.firms if not getattr(f, "stabilizing_firm", False) and getattr(f, "in_business", True))
            nonstab_total = sum(1 for f in world.firms if not getattr(f, "stabilizing_firm", False))

            # Collect final cash and prices for all firms
            stab_cash = [f.cash for f in world.firms if getattr(f, "stabilizing_firm", False) and getattr(f, "in_business", True)]
            nonstab_cash = [f.cash for f in world.firms if not getattr(f, "stabilizing_firm", False) and getattr(f, "in_business", True)]

            # ── Episode return: assign SAME reward to ALL trajectories ──
            if steps > 0:
                episode_return = (
                    w_profit * (total_stab_profit / max(steps, 1))
                    + w_survival * (final_alive / total_firms)
                    + w_floor * (steps_above_cost / steps)
                    + self.survival_bonus * (1.0 if stab_survived else 0.0)
                )
            else:
                episode_return = 0.0

            for traj in ep_trajs:
                traj["reward"] = episode_return

            dt = time.time() - t_ep
            if ep_idx < 3 or ep_idx == num_episodes - 1:
                print(f"  Ep {ep_idx+1}/{num_episodes}: {steps} steps, alive={final_alive}/{total_firms}, "
                      f"stab={'alive' if stab_survived else 'DEAD'}, return={episode_return:.2f}, n_stab={n_stab}, {dt:.0f}s", flush=True)
            return ep_trajs, stab_survived, {
                "avg_profit": total_stab_profit / max(steps, 1),
                "total_sales": ep_sales, "steps": steps,
                "stab_survived": stab_survived, "episode_return": episode_return,
                "n_stab": n_stab,
                # Economic alignment metrics for paper
                "market_survival_rate": final_alive / total_firms,
                "stab_survival_rate": stab_alive / max(stab_total, 1),
                "nonstab_survival_rate": nonstab_alive / max(nonstab_total, 1),
                "stab_avg_cash": np.mean(stab_cash) if stab_cash else 0,
                "nonstab_avg_cash": np.mean(nonstab_cash) if nonstab_cash else 0,
                "price_floor_compliance": steps_above_cost / max(steps, 1),
                "bankruptcy_rate": 1.0 - final_alive / total_firms,
            }

        # Run episodes in parallel
        if num_episodes > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_episodes) as ex:
                results = list(ex.map(run_episode, range(num_episodes)))
            for tr, sv, st in results:
                all_trajs.extend(tr); all_survived.append(sv); stats_list.append(st)
        else:
            tr, sv, st = run_episode(0)
            all_trajs.extend(tr); all_survived.append(sv); stats_list.append(st)

        dt = time.time() - t0
        valid = [t for t in all_trajs if t.get("reward") is not None and t.get("response")]
        surv_rate = np.mean(all_survived) if all_survived else 0
        returns = [s["episode_return"] for s in stats_list]
        avg_steps = np.mean([s["steps"] for s in stats_list])
        mkt_surv = np.mean([s["market_survival_rate"] for s in stats_list])
        stab_surv = np.mean([s["stab_survival_rate"] for s in stats_list])
        nonstab_surv = np.mean([s["nonstab_survival_rate"] for s in stats_list])
        bankruptcy = np.mean([s["bankruptcy_rate"] for s in stats_list])
        floor_comp = np.mean([s["price_floor_compliance"] for s in stats_list])
        print(f"\nCollected {len(all_trajs)} trajs ({len(valid)} valid) in {dt:.1f}s | "
              f"stab_surv={stab_surv:.0%} nonstab_surv={nonstab_surv:.0%} mkt={mkt_surv:.0%} | "
              f"return={np.mean(returns):.2f} | steps={avg_steps:.0f} | floor={floor_comp:.0%}", flush=True)

        if wandb.run:
            wandb.log({
                "env/avg_profit": np.mean([s["avg_profit"] for s in stats_list]),
                "env/total_sales": sum(s["total_sales"] for s in stats_list),
                "env/survived_rate": surv_rate,
                # Economic alignment metrics
                "econ/market_survival": mkt_surv,
                "econ/stab_survival": stab_surv,
                "econ/nonstab_survival": nonstab_surv,
                "econ/bankruptcy_rate": bankruptcy,
                "econ/price_floor_compliance": floor_comp,
                "econ/stab_avg_cash": np.mean([s["stab_avg_cash"] for s in stats_list]),
                "econ/nonstab_avg_cash": np.mean([s["nonstab_avg_cash"] for s in stats_list]),
                "env/steps": avg_steps,
                "env/episode_return_avg": np.mean(returns),
                "env/episode_return_std": np.std(returns),
                "trajectories/count": len(all_trajs),
                "trajectories/valid": len(valid),
                "curriculum/n_stab_avg": np.mean([s["n_stab"] for s in stats_list]),
                "iteration": iteration,
                "perf/collect_time_s": dt,
            })
        return valid

    # ── REINFORCE++ training step ───────────────────────────────────────

    def train_step(self, trajectories: List[Dict[str, Any]], iteration: int):
        """REINFORCE++ with group-based advantage normalization and token-level KL penalty.

        Key differences from vanilla REINFORCE:
        1. Group trajectories by timestep — normalize advantages within each group
           (64 episodes × same timestep = 64 responses to the same prompt)
        2. Token-level KL penalty against reference model (frozen base on GPU 1)
        3. Per-token advantage weighting with clipping
        """
        t0 = time.time()
        print(f"Training: iter {iteration}, {len(trajectories)} samples", flush=True)
        torch.cuda.empty_cache()

        self.model.train()
        self.inference_model._inference_ready = False
        total_loss, total_kl_val = 0.0, 0.0
        ok_batches, fail_batches, skipped = 0, 0, 0
        grad_norms = []
        micro_bs = getattr(self.args, "micro_batch_size", 4)
        effective_bs = self.args.train_batch_size
        accum_steps = max(1, effective_bs // micro_bs)
        fmt_w = getattr(self.args, "format_reward_weight", 2.0)
        kl_coeff = getattr(self.args, "kl_coeff", 0.05)

        # ── Step 1: Prepare samples with group-based advantage normalization ──
        # Group by timestep for REINFORCE++ normalization
        from collections import defaultdict
        groups = defaultdict(list)  # timestep → list of (text, prompt, reward)
        for t in trajectories:
            s, u, r, rw = t.get("system_prompt",""), t.get("user_prompt",""), t.get("response",""), t.get("reward")
            valid = t.get("is_format_valid", True)
            if rw is None or not r:
                skipped += 1; continue
            bonus = fmt_w if valid else -fmt_w
            ts = t.get("timestep", 0)
            msg = [{"role":"system","content":s},{"role":"user","content":u}]
            p = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            full = p + r + self.tokenizer.eos_token
            # Pre-filter: skip if prompt is so long that response would be truncated
            p_len = len(self.encoding_tokenizer(p, truncation=False).input_ids)
            f_len = len(self.encoding_tokenizer(full, truncation=False).input_ids)
            if f_len - p_len < 5:
                skipped += 1; continue  # Response too short after potential truncation
            groups[ts].append({
                "text": full,
                "prompt": p,
                "reward": rw + bonus,
            })

        # Normalize advantages within each timestep group (REINFORCE++ key feature)
        all_texts, all_prompts, all_advantages = [], [], []
        for ts in sorted(groups.keys()):
            group = groups[ts]
            rewards = [g["reward"] for g in group]
            mu = np.mean(rewards)
            std = np.std(rewards) + 1e-8
            for g in group:
                adv = (g["reward"] - mu) / std
                adv = max(-self.advantage_clip, min(self.advantage_clip, adv))
                all_texts.append(g["text"])
                all_prompts.append(g["prompt"])
                all_advantages.append(adv)

        if not all_texts:
            self.model.eval()
            return 0.0

        # Sort by sequence length for efficient batching — short sequences together
        # allows larger micro-batches, better GPU utilization
        seq_lens = [len(self.encoding_tokenizer(t, truncation=False).input_ids) for t in all_texts]
        sorted_indices = sorted(range(len(all_texts)), key=lambda i: seq_lens[i])
        all_texts = [all_texts[i] for i in sorted_indices]
        all_prompts = [all_prompts[i] for i in sorted_indices]
        all_advantages = [all_advantages[i] for i in sorted_indices]
        seq_lens = [seq_lens[i] for i in sorted_indices]

        n_groups = len(groups)
        print(f"  REINFORCE++: {len(all_texts)} samples, {n_groups} groups, "
              f"seq_lens=[{min(seq_lens)}-{max(seq_lens)}], kl={kl_coeff}", flush=True)

        # ── Step 2: Gradient accumulation with KL penalty ──
        # Dynamic micro-batching: 2× micro_bs for short seqs (<2048), 1× for long
        self.optimizer.zero_grad()
        accum_loss = 0.0
        accum_kl = 0.0
        accum_count = 0

        i = 0
        while i < len(all_texts):
            self.heartbeat()
            # Dynamic batch size based on max sequence length in this chunk
            chunk_max_len = seq_lens[min(i + micro_bs * 2 - 1, len(seq_lens) - 1)]
            if chunk_max_len <= 2048:
                cur_bs = micro_bs * 2  # Double batch for short sequences
            else:
                cur_bs = micro_bs
            texts = all_texts[i:i+cur_bs]
            prompts = all_prompts[i:i+cur_bs]
            advantages = all_advantages[i:i+cur_bs]
            i += cur_bs

            try:
                enc = self.encoding_tokenizer(texts, return_tensors="pt", padding=True,
                                              truncation=True, max_length=4096).to(self.device)
                prompt_lens = [len(self.encoding_tokenizer(p, truncation=True, max_length=4096).input_ids)
                               for p in prompts]

                # Policy forward pass
                logits = self.model(**enc).logits.float()

                # Reference model forward pass for KL penalty (computed on GPU 1, results moved to GPU 0)
                ref_log_probs_all = None
                if self.base_model is not None and kl_coeff > 0:
                    with torch.no_grad():
                        ref_enc = self.encoding_tokenizer(texts, return_tensors="pt", padding=True,
                                                          truncation=True, max_length=4096).to(self.device_base)
                        ref_out = self.base_model(**ref_enc).logits.float()
                        # Compute log_softmax on GPU 1, only move selected log-probs to GPU 0
                        ref_log_probs_all = torch.log_softmax(ref_out, dim=-1)
                        del ref_out, ref_enc

                micro_loss = torch.tensor(0.0, device=self.device)
                micro_kl = 0.0
                n_valid = 0

                for j in range(len(texts)):
                    pl = prompt_lens[j]
                    sl = logits[j, pl-1:-1, :].contiguous()
                    lb = enc.input_ids[j, pl:].contiguous()
                    if lb.size(0) == 0:
                        continue

                    # Policy log-probs
                    policy_lp = torch.log_softmax(sl, dim=-1)
                    policy_selected = torch.gather(policy_lp, -1, lb.unsqueeze(-1)).squeeze(-1)
                    policy_selected = torch.clamp(policy_selected, min=-100.0)
                    if torch.isnan(policy_selected).any():
                        continue

                    adv = advantages[j]

                    # REINFORCE++ loss: -log_prob * advantage (per-token, then mean)
                    pg_loss = -(policy_selected * adv).mean()

                    # Token-level KL penalty against reference model
                    kl_loss = torch.tensor(0.0, device=self.device)
                    if ref_log_probs_all is not None:
                        ref_lp_j = ref_log_probs_all[j, pl-1:-1, :].contiguous()
                        lb_base = enc.input_ids[j, pl:].to(self.device_base)
                        ref_selected = torch.gather(ref_lp_j, -1, lb_base.unsqueeze(-1)).squeeze(-1)
                        ref_selected = torch.clamp(ref_selected, min=-100.0).to(self.device)
                        # KL(policy || ref) — ref is detached (no grad), policy gets gradient
                        kl_loss = (policy_selected - ref_selected.detach()).mean()
                        micro_kl += kl_loss.item()

                    sample_loss = pg_loss + kl_coeff * kl_loss
                    if torch.isnan(sample_loss):
                        continue
                    micro_loss = micro_loss + sample_loss
                    n_valid += 1

                if n_valid > 0:
                    loss = micro_loss / (n_valid * accum_steps)
                    if torch.isnan(loss) or torch.isinf(loss):
                        fail_batches += 1
                    else:
                        loss.backward()
                        accum_loss += loss.item() * accum_steps
                        accum_kl += micro_kl / n_valid
                        accum_count += 1
                        ok_batches += 1

                del logits, enc
                if ref_log_probs_all is not None:
                    del ref_log_probs_all
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Batch failed: {e}", flush=True)
                traceback.print_exc()
                fail_batches += 1
                continue

            # Optimizer step after accumulation
            if accum_count > 0 and accum_count % accum_steps == 0:
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        torch.nn.utils.clip_grad_norm_([p], self.grad_clip_norm)
                gn = sum(p.grad.norm().item()**2 for p in self.model.parameters()
                         if p.requires_grad and p.grad is not None)**0.5
                grad_norms.append(gn)
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += accum_loss
                total_kl_val += accum_kl
                accum_loss = 0.0
                accum_kl = 0.0

        # Final optimizer step for remaining accumulated gradients
        if accum_count % accum_steps != 0 and accum_count > 0:
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    torch.nn.utils.clip_grad_norm_([p], self.grad_clip_norm)
            gn = sum(p.grad.norm().item()**2 for p in self.model.parameters()
                     if p.requires_grad and p.grad is not None)**0.5
            grad_norms.append(gn)
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += accum_loss
            total_kl_val += accum_kl

        # Switch back to eval for inference
        self.model.eval()
        self.model.save_pretrained(os.path.join(self.checkpoint_dir, "latest"))

        # Save full training state for resumption
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "iteration": iteration,
            "reward_stats": {"n": self.reward_stats.n, "mean": self.reward_stats.mean, "M2": self.reward_stats.M2},
        }, os.path.join(self.checkpoint_dir, "train_state.pt"))

        dt = time.time() - t0
        nb = max(1, (len(trajectories) + micro_bs - 1) // micro_bs)
        avg_loss = total_loss / max(ok_batches, 1)
        avg_kl = total_kl_val / max(ok_batches, 1)
        n_updates = len(grad_norms)
        mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"Training done: {ok_batches}/{nb} ok, {fail_batches} fail, {skipped} skip | "
              f"loss={avg_loss:.4f} kl={avg_kl:.4f} | {n_updates} updates | {dt:.1f}s | {mem:.1f}GB", flush=True)

        if wandb.run:
            wandb.log({
                "train/loss": avg_loss,
                "train/kl": avg_kl,
                "train/n_updates": n_updates,
                "train/n_groups": n_groups,
                "train/ok_batches": ok_batches,
                "train/fail_batches": fail_batches,
                "train/skip": skipped,
                "train/grad_norm_avg": np.mean(grad_norms) if grad_norms else 0,
                "train/grad_norm_max": max(grad_norms) if grad_norms else 0,
                "train/reward_mean": self.reward_stats.mean,
                "train/reward_var": self.reward_stats.variance,
                "gpu/mem_gb": mem,
                "iteration": iteration,
            })
        return total_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = create_argument_parser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=64, help="Effective batch size (via gradient accumulation)")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro-batch size per forward pass (must fit in GPU memory)")
    parser.add_argument("--format_reward_weight", type=float, default=2.0)
    parser.add_argument("--inference_batch_size", type=int, default=128)
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online","offline","disabled"])
    # REINFORCE++
    parser.add_argument("--advantage_clip", type=float, default=3.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.5)
    parser.add_argument("--kl_coeff", type=float, default=0.05, help="Token-level KL penalty coefficient against reference model")
    parser.add_argument("--reward_weights", type=str, default="0.4,0.3,0.3")
    parser.add_argument("--survival_bonus", type=float, default=5.0)
    parser.add_argument("--curriculum", action="store_true", default=True, help="Enable curriculum: 5/5→4/5→3/5 then sample")
    # SFT warmup
    parser.add_argument("--sft_warmup", type=int, default=0, help="SFT examples (0=skip)")
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Training defaults: disable diaries (confuse base model), use io prompts
    args.no_diaries = True
    if not hasattr(args, 'prompt_algo') or args.prompt_algo == "cot":
        args.prompt_algo = "io"

    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(project="ai-bazaar", name=args.run_name, config=vars(args), mode=args.wandb_mode)

    trainer = REINFORCETrainer(args.llm, args)

    print(f"\n{'='*60}")
    print(f"REINFORCE++ | {args.num_iterations} iters × {args.num_episodes} eps")
    print(f"Stabilizing firms: {getattr(args, 'num_stabilizing_firms', 0)} (curriculum={'ON' if args.curriculum else 'OFF'})")
    print(f"Reward: episode return (weights={args.reward_weights}, survival_bonus={args.survival_bonus})")
    print(f"Batch: {args.train_batch_size} effective, {getattr(args, 'micro_batch_size', 4)} micro, {args.num_episodes} episodes")
    print(f"{'='*60}\n")

    # Optional SFT warmup (skip on resume — already done)
    if getattr(args, "sft_warmup", 0) > 0 and trainer.start_iteration == 0:
        trainer.sft_warmup(args.sft_warmup, getattr(args, "sft_epochs", 3))

    for i in range(trainer.start_iteration, args.num_iterations):
        t_iter = time.time()
        print(f"\n{'='*60}\nITERATION {i+1}/{args.num_iterations}\n{'='*60}")

        trajs = trainer.collect_trajectories(args.num_episodes, i)
        if len(trajs) >= 2:
            trainer.train_step(trajs, i)
        else:
            print(f"  Skip training: {len(trajs)} valid trajs (need ≥2)", flush=True)

        dt = time.time() - t_iter
        print(f"\nIteration {i+1} done in {dt:.1f}s ({dt/60:.1f} min)")
        if wandb.run:
            wandb.log({"iter_time_s": dt, "iteration": i})

    print(f"\n{'='*60}\nTraining complete! {args.num_iterations} iterations\n{'='*60}\n")
    trainer.cleanup()
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
