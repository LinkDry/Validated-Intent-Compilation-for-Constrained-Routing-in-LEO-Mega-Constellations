"""Training loop for GNN routing policy."""

import json
import os
import time
import numpy as np
import torch
from typing import Dict, Optional

from env.routing_env import LEORoutingEnv
from policy.routing_policy import GNNRoutingPolicy
from policy.ppo import PPO
from evaluation.metrics import compute_episode_metrics


class Trainer:
    """PPO training loop with periodic evaluation and checkpointing."""

    def __init__(
        self,
        env: LEORoutingEnv,
        ppo: PPO,
        output_dir: str = "output",
        eval_interval: int = 10,
        eval_episodes: int = 5,
        rollout_steps: int = 192,
        log_interval: int = 1,
    ):
        self.env = env
        self.ppo = ppo
        self.output_dir = output_dir
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.rollout_steps = rollout_steps
        self.log_interval = log_interval

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

        self.metrics_log = []
        self.best_pdr = -1.0

    def train(self, total_episodes: int):
        """Main training loop."""
        obs, info = self.env.reset()
        episode = 0
        global_step = 0
        ep_rewards = []
        ep_infos = []
        t_start = time.time()

        while episode < total_episodes:
            for _ in range(self.rollout_steps):
                slot_table, active_od, active_actions, log_prob, value = (
                    self.ppo.select_action(obs)
                )
                obs_next, reward, terminated, truncated, info = self.env.step(slot_table)
                self.ppo.store_transition(
                    obs, slot_table, active_od, active_actions,
                    log_prob, reward, value, terminated,
                )
                ep_rewards.append(reward)
                ep_infos.append(info)
                obs = obs_next
                global_step += 1

                if terminated or truncated:
                    ep_metrics = compute_episode_metrics(ep_infos)
                    ep_metrics["episode"] = episode
                    ep_metrics["ep_reward"] = float(np.sum(ep_rewards))
                    ep_metrics["global_step"] = global_step

                    if episode % self.log_interval == 0:
                        elapsed = time.time() - t_start
                        print(
                            f"Ep {episode:4d} | "
                            f"R={ep_metrics['ep_reward']:7.2f} | "
                            f"PDR={ep_metrics['mean_pdr']:.3f} | "
                            f"delay={ep_metrics['mean_delay_ms']:.1f}ms | "
                            f"steps={global_step} | "
                            f"{elapsed:.0f}s"
                        )

                    self.metrics_log.append(ep_metrics)
                    ep_rewards = []
                    ep_infos = []
                    episode += 1
                    obs, info = self.env.reset()

                    if episode >= total_episodes:
                        break

            update_stats = self.ppo.update()
            if update_stats and episode % self.log_interval == 0:
                print(f"  PPO: loss={update_stats['loss']:.4f} pg={update_stats['pg_loss']:.4f} "
                      f"v={update_stats['value_loss']:.4f} ent={update_stats['entropy']:.4f}")

            if episode > 0 and episode % self.eval_interval == 0:
                self._evaluate_and_checkpoint(episode)

        self._evaluate_and_checkpoint(episode, final=True)
        self._save_metrics()
        print(f"\nTraining complete. Best PDR: {self.best_pdr:.3f}")

    def _evaluate_and_checkpoint(self, episode: int, final: bool = False):
        """Run eval episodes and save checkpoint if improved."""
        eval_infos_all = []

        for ep in range(self.eval_episodes):
            obs, info = self.env.reset(seed=5000 + ep)
            ep_infos = []
            for _ in range(300):
                slot_table, _, _, _, _ = self.ppo.select_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(slot_table)
                ep_infos.append(info)
                if terminated or truncated:
                    break
            eval_infos_all.append(compute_episode_metrics(ep_infos))

        mean_pdr = float(np.mean([m["mean_pdr"] for m in eval_infos_all]))
        mean_delay = float(np.mean([m["mean_delay_ms"] for m in eval_infos_all]))
        mean_drop = float(np.mean([m["overall_drop_rate"] for m in eval_infos_all]))

        tag = "FINAL" if final else "EVAL"
        print(f"  [{tag}] ep={episode} PDR={mean_pdr:.3f} delay={mean_delay:.1f}ms drop={mean_drop:.3f}")

        path = os.path.join(self.output_dir, "checkpoints", f"ep{episode}.pt")
        self.ppo.save(path)

        if mean_pdr > self.best_pdr:
            self.best_pdr = mean_pdr
            best_path = os.path.join(self.output_dir, "checkpoints", "best.pt")
            self.ppo.save(best_path)
            print(f"  New best model saved (PDR={mean_pdr:.3f})")

    def _save_metrics(self):
        """Save training metrics to JSON."""
        path = os.path.join(self.output_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
        print(f"Metrics saved to {path}")
