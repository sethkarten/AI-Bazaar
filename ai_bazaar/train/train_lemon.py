"""REINFORCE++ trainer for guardian buyer in Lemon Market (Sybil scenario).

Trains a single guardian buyer (LoRA on Qwen3.5-9B) using policy gradient
with composite reward (sybil detection + consumer surplus + market health).
Sellers use the base model. Other buyers use the base model.

The guardian buyer learns to:
1. Detect and avoid sybil (fake) seller listings
2. Maximize consumer surplus (buy good cars at fair prices)
3. Maintain overall market health

Usage:
    python -m ai_bazaar.train.train_lemon \
        --llm unsloth/Qwen3.5-9B \
        --num-firms 12 --sybil-cluster-size 6 \
        --num-consumers 12 \
        --num_episodes 16 --num_iterations 40 \
        --lora_r 64 --quant_bits 16
"""
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import threading
import wandb
import random
import concurrent.futures
import numpy as np
import torch
import traceback
from typing import List, Dict, Any

from ai_bazaar.train.train_reinforce import REINFORCETrainer, WelfordRunningStats
from ai_bazaar.env.bazaar_env import BazaarWorld
from ai_bazaar.main import create_argument_parser


class LemonTrainer(REINFORCETrainer):
    """REINFORCE++ trainer for guardian buyer in Lemon Market.

    Inherits model loading, train_step, checkpointing from REINFORCETrainer.
    Overrides collect_trajectories for buyer-centric episode rollout and reward.
    """

    def __init__(self, model_name: str, args):
        super().__init__(model_name, args)
        self.last_detection_rate = 0.0

    def collect_trajectories(self, num_episodes: int, iteration: int):
        t0 = time.time()
        all_trajs, stats_list = [], []
        w_detect, w_surplus, w_health = self.reward_weights[:3]

        def run_episode(ep_idx):
            t_ep = time.time()

            # Curriculum: vary sybil_cluster_size based on detection performance
            k = self.get_sybil_cluster_size(iteration, ep_idx)
            import copy
            ep_args = copy.copy(self.args)
            ep_args.sybil_cluster_size = k

            ep_args.num_guardian_buyers = 1  # First buyer is the guardian
            world = BazaarWorld(
                ep_args,
                llm_model=self.inference_model,       # Guardian buyer uses trained model
                llm_model_base=self.inference_model_base,  # Sellers + other buyers use base
            )

            steps = 0
            while not world.is_done():
                try:
                    world.step()
                except (ValueError, RecursionError) as e:
                    print(f"  Ep {ep_idx+1}: step {steps} crashed ({type(e).__name__})", flush=True)
                    break
                steps += 1
                self.heartbeat()

            # ── Collect guardian buyer trajectories ──
            guardian = next(
                (c for c in world.consumers if getattr(c, "guardian", False)), None
            )
            ep_trajs = []
            if guardian and hasattr(guardian, "trajectory"):
                ep_trajs = list(guardian.trajectory)
                guardian.trajectory = []

            # ── Compute episode-level reward ──
            # 1. Sybil detection rate (how many sybils did the guardian pass on?)
            sybil_seen = getattr(guardian, "sybil_seen_total", 0) if guardian else 0
            sybil_passed = getattr(guardian, "sybil_passed_total", 0) if guardian else 0
            detection_rate = sybil_passed / max(sybil_seen, 1)

            # 2. Consumer surplus (cumulative utility), normalized by V_MAX
            from ai_bazaar.utils.common import V_MAX
            consumer_surplus = getattr(guardian, "utility", 0.0) if guardian else 0.0
            # Normalize: surplus per step, scaled to [~-1, ~1] range
            surplus_per_step = consumer_surplus / (max(steps, 1) * V_MAX)

            # 3. Market health (overall buyer welfare — all buyers), normalized
            all_surplus = sum(
                getattr(c, "utility", 0.0) for c in world.consumers
            )
            market_health = all_surplus / (max(len(world.consumers), 1) * max(steps, 1) * V_MAX)

            # 4. Honest purchase rate (did we buy from honest sellers when available?)
            honest_seen = getattr(guardian, "honest_seen_total", 0) if guardian else 0
            honest_passed = getattr(guardian, "honest_passed_total", 0) if guardian else 0
            honest_buy_rate = 1.0 - (honest_passed / max(honest_seen, 1))

            # Composite episode return
            episode_return = (
                w_detect * detection_rate
                + w_surplus * surplus_per_step
                + w_health * market_health
            )

            # Assign episode return to all guardian trajectories
            for traj in ep_trajs:
                traj["reward"] = episode_return

            dt = time.time() - t_ep
            if ep_idx < 3 or ep_idx == num_episodes - 1:
                print(f"  Ep {ep_idx+1}/{num_episodes}: {steps} steps, "
                      f"detect={detection_rate:.0%} surplus={surplus_per_step:.1f} "
                      f"mkt_health={market_health:.1f} return={episode_return:.2f} "
                      f"K={k}, {dt:.0f}s", flush=True)

            return ep_trajs, {
                "steps": steps, "episode_return": episode_return,
                "sybil_cluster_size": k,
                # Guardian buyer metrics
                "detection_rate": detection_rate,
                "sybil_seen": sybil_seen,
                "sybil_passed": sybil_passed,
                "consumer_surplus": consumer_surplus,
                "surplus_per_step": surplus_per_step,
                "honest_buy_rate": honest_buy_rate,
                # Market-wide metrics
                "market_health": market_health,
                "all_buyer_surplus": all_surplus,
                "sybil_steps_purchased": getattr(guardian, "sybil_steps_purchased_total", 0) if guardian else 0,
            }

        # Run episodes in parallel
        if num_episodes > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_episodes) as ex:
                results = list(ex.map(run_episode, range(num_episodes)))
            for tr, st in results:
                all_trajs.extend(tr); stats_list.append(st)
        else:
            tr, st = run_episode(0)
            all_trajs.extend(tr); stats_list.append(st)

        dt = time.time() - t0
        valid = [t for t in all_trajs if t.get("reward") is not None and t.get("response")]

        # Aggregate metrics
        avg_detect = np.mean([s["detection_rate"] for s in stats_list])
        avg_surplus = np.mean([s["surplus_per_step"] for s in stats_list])
        avg_health = np.mean([s["market_health"] for s in stats_list])
        avg_honest_buy = np.mean([s["honest_buy_rate"] for s in stats_list])
        avg_return = np.mean([s["episode_return"] for s in stats_list])
        avg_steps = np.mean([s["steps"] for s in stats_list])
        avg_k = np.mean([s["sybil_cluster_size"] for s in stats_list])
        sybil_purchase_rate = np.mean([
            s["sybil_steps_purchased"] / max(s["steps"], 1) for s in stats_list
        ])

        # Update detection rate for curriculum
        self.last_detection_rate = avg_detect

        print(f"\nCollected {len(all_trajs)} trajs ({len(valid)} valid) in {dt:.1f}s | "
              f"detect={avg_detect:.0%} surplus={avg_surplus:.1f} mkt={avg_health:.1f} | "
              f"honest_buy={avg_honest_buy:.0%} sybil_purch={sybil_purchase_rate:.0%} | "
              f"return={avg_return:.2f} | steps={avg_steps:.0f} | K_avg={avg_k:.1f}", flush=True)

        if wandb.run:
            wandb.log({
                "env/detection_rate": avg_detect,
                "env/surplus_per_step": avg_surplus,
                "env/market_health": avg_health,
                "env/honest_buy_rate": avg_honest_buy,
                "env/sybil_purchase_rate": sybil_purchase_rate,
                "env/episode_return_avg": avg_return,
                "env/episode_return_std": np.std([s["episode_return"] for s in stats_list]),
                "env/steps": avg_steps,
                "env/sybil_cluster_size_avg": avg_k,
                "trajectories/count": len(all_trajs),
                "trajectories/valid": len(valid),
                "iteration": iteration,
                "perf/collect_time_s": dt,
            })
        return valid

    # ── Sybil curriculum ──────────────────────────────────────────────

    def get_sybil_cluster_size(self, iteration, ep_idx):
        """Adaptive curriculum: increase sybil count as detection improves.

        - Stage 0 (detect < 50%): K=3 (easy — few sybils)
        - Stage 1 (detect ≥ 50%): mix K=3 (60%) and K=6 (40%)
        - Stage 2 (detect ≥ 70%): mix K=3 (30%), K=6 (40%), K=9 (30%)
        - Stage 3 (detect ≥ 85%): full mix K=3..9
        """
        if not getattr(self.args, "curriculum", True):
            return getattr(self.args, "sybil_cluster_size", 6)

        rng = random.Random(self.args.seed + iteration * 1000 + ep_idx)
        detect = getattr(self, "last_detection_rate", 0.0)

        if detect < 0.50:
            return 3
        elif detect < 0.70:
            return rng.choices([3, 6], weights=[60, 40])[0]
        elif detect < 0.85:
            return rng.choices([3, 6, 9], weights=[30, 40, 30])[0]
        else:
            return rng.choices([3, 6, 9], weights=[20, 40, 40])[0]

    # ── Override content filter for buyer actions ─────────────────────

    def _is_valid_response(self, response: str) -> bool:
        """Check if response is a valid buyer action (bid/pass or vote JSON)."""
        if not response.startswith("{"):
            return False
        return "decision" in response or "vote" in response


# ---------------------------------------------------------------------------
# SFT warmup for buyer format
# ---------------------------------------------------------------------------

def buyer_sft_examples(tokenizer, num_examples=200, args=None):
    """Generate synthetic buyer (prompt, JSON) pairs for SFT warmup."""
    import random as rng
    from ai_bazaar.utils.common import V_MAX

    system_prompt = (
        "You are consumer_0, a buyer in a used-car peer-to-peer market. "
        "Your persona: cautious. "
        "Your goal is to purchase good-value cars and avoid paying more than a car is worth. "
        f"Typical price ranges by quality tier: "
        f"mint ${V_MAX * 0.85:,.0f}–${V_MAX * 1.0:,.0f}, "
        f"good ${V_MAX * 0.55:,.0f}–${V_MAX * 0.80:,.0f}, "
        f"fair ${V_MAX * 0.28:,.0f}–${V_MAX * 0.52:,.0f}, "
        f"poor ${V_MAX * 0.05:,.0f}–${V_MAX * 0.18:,.0f}. "
        "Be aware: some sellers misrepresent car quality. "
        "You may buy at most one car per round. "
        "If no listing offers good value, pass."
    )

    examples = []
    qualities = ["mint", "good", "fair", "poor"]
    quality_values = {"mint": 0.85, "good": 0.60, "fair": 0.35, "poor": 0.10}

    for _ in range(num_examples):
        n_listings = rng.randint(1, 5)
        listings = []
        for j in range(n_listings):
            q = rng.choice(qualities)
            # Honest seller: price ~ quality_value * V_MAX
            # Sybil seller: price for "good"/"mint" but actually poor
            is_sybil = rng.random() < 0.3
            if is_sybil:
                advertised_q = rng.choice(["good", "mint"])
                price = round(V_MAX * quality_values[advertised_q] * rng.uniform(0.7, 1.0), 2)
                rep = round(rng.uniform(0.3, 0.9), 2)
            else:
                advertised_q = q
                price = round(V_MAX * quality_values[q] * rng.uniform(0.8, 1.1), 2)
                rep = round(rng.uniform(0.5, 1.0), 2)

            listings.append({
                "listing_id": f"seller_{j}_listing_0",
                "listed_price": price,
                "description": f"Used car in {advertised_q} condition. Well maintained.",
                "seller_reputation": rep,
            })

        history = []
        if rng.random() > 0.3:
            for _ in range(rng.randint(1, 3)):
                hq = rng.choice(qualities)
                history.append({
                    "timestep": rng.randint(1, 20),
                    "seller_id": f"seller_{rng.randint(0, 11)}",
                    "price_paid": round(V_MAX * quality_values[hq] * rng.uniform(0.7, 1.1), 2),
                    "quality_received": quality_values[hq],
                    "quality_label": hq,
                    "consumer_surplus": round(V_MAX * quality_values[hq] * rng.uniform(-0.3, 0.3), 2),
                })

        obs = {
            "timestep": rng.randint(1, 40),
            "persona": "cautious",
            "your_mean_quality_received": np.mean([h["quality_received"] for h in history]) if history else None,
            "your_transaction_history": history,
            "listings_visible": listings,
        }
        import json
        user = json.dumps(obs, indent=None)

        # Decision: bid on the best value listing or pass
        # For SFT, teach the model to pass on suspicious listings and bid on good ones
        best_idx = None
        best_value = -1
        for j, L in enumerate(listings):
            value = L["seller_reputation"] * 0.5 + (1.0 if L["listed_price"] < V_MAX * 0.5 else 0.0) * 0.5
            if value > best_value:
                best_value = value
                best_idx = j

        if best_value > 0.4 and best_idx is not None:
            resp = json.dumps({"decision": "bid", "listing_id": listings[best_idx]["listing_id"]})
        else:
            resp = json.dumps({"decision": "pass", "listing_id": None})

        examples.append((system_prompt, user, resp))

    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = create_argument_parser()
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_episodes", type=int, default=16)
    parser.add_argument("--num_iterations", type=int, default=40)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=32, help="Effective batch size")
    parser.add_argument("--micro_batch_size", type=int, default=16, help="Micro-batch per forward pass")
    parser.add_argument("--format_reward_weight", type=float, default=2.0)
    parser.add_argument("--inference_batch_size", type=int, default=64)
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    # Model
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--quant_bits", type=int, default=16, choices=[4, 8, 16, 32], help="Quantization bits")
    # REINFORCE++
    parser.add_argument("--advantage_clip", type=float, default=3.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.5)
    parser.add_argument("--kl_coeff", type=float, default=0.2, help="Squared KL penalty coefficient")
    parser.add_argument("--reward_weights", type=str, default="0.4,0.3,0.3",
                        help="detection_rate,consumer_surplus,market_health")
    parser.add_argument("--curriculum", action="store_true", default=True)
    # SFT warmup
    parser.add_argument("--sft_warmup", type=int, default=0, help="SFT examples (0=skip)")
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Training defaults for lemon market
    args.consumer_scenario = "LEMON_MARKET"
    args.no_diaries = True
    args.firm_tax_rate = 0.0
    args.history_len = 3  # Buyers benefit from transaction history
    args.best_n = 1
    if not hasattr(args, "prompt_algo") or args.prompt_algo == "cot":
        args.prompt_algo = "io"

    # Default lemon market settings
    if not hasattr(args, "num_firms") or args.num_firms == 5:
        args.num_firms = 12
    if not hasattr(args, "sybil_cluster_size"):
        args.sybil_cluster_size = 3  # Start easy (curriculum will increase)
    if not hasattr(args, "num_consumers") or args.num_consumers == 50:
        args.num_consumers = 12

    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(project="ai-bazaar-lemon", name=args.run_name, config=vars(args), mode=args.wandb_mode)

    trainer = LemonTrainer(args.llm, args)

    print(f"\n{'='*60}")
    print(f"REINFORCE++ Guardian Buyer | {args.num_iterations} iters × {args.num_episodes} eps")
    print(f"Model: {args.quant_bits}-bit + LoRA(r={args.lora_r})")
    print(f"Lemon Market: {args.num_firms} sellers ({args.sybil_cluster_size} sybil), {args.num_consumers} buyers")
    print(f"Reward: detection={args.reward_weights} | KL={args.kl_coeff}")
    print(f"Batch: {args.train_batch_size} effective, {args.micro_batch_size} micro")
    print(f"{'='*60}\n")

    # Optional SFT warmup
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
    if hasattr(trainer, "cleanup"):
        trainer.cleanup()
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
