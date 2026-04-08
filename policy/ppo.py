"""PPO with per-pair surrogate for destination-conditioned routing."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from .routing_policy import GNNRoutingPolicy


@dataclass
class RolloutStep:
    """Single step in a rollout."""
    obs: Dict[str, np.ndarray]
    nexthop_table: np.ndarray      # [N, N] slot indices
    active_od: np.ndarray           # [P, 2]
    active_actions: np.ndarray      # [P]
    log_probs: np.ndarray           # [P] per-pair log-probs
    reward: float
    value: float
    done: bool


class RolloutBuffer:
    """Stores rollout data for PPO updates. All data on CPU."""

    def __init__(self):
        self.steps: List[RolloutStep] = []

    def add(self, step: RolloutStep):
        self.steps.append(step)

    def clear(self):
        self.steps = []

    def __len__(self):
        return len(self.steps)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95,
    ):
        """Compute GAE advantages and discounted returns."""
        n = len(self.steps)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = False
            else:
                next_value = self.steps[t + 1].value
                next_done = self.steps[t + 1].done

            mask = 1.0 - float(self.steps[t].done)
            delta = (
                self.steps[t].reward
                + gamma * next_value * mask
                - self.steps[t].value
            )
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.steps[t].value


class PPO:
    """Proximal Policy Optimization with per-pair surrogate."""

    def __init__(
        self,
        policy: GNNRoutingPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 8,
        device: torch.device = None,
    ):
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        self.policy.to(self.device)

    @torch.no_grad()
    def select_action(self, obs: Dict[str, np.ndarray]):
        """Select action from current policy.

        Returns:
            slot_table: [N, N] np.ndarray
            active_od: [P, 2] np.ndarray
            active_actions: [P] np.ndarray
            log_probs: [P] np.ndarray (per-pair)
            value: float
        """
        self.policy.eval()
        data, mask, active_od, neighbor_table = GNNRoutingPolicy.obs_to_pyg(obs, self.device)

        slot_table, actions, log_probs, entropy, value = (
            self.policy.get_action_and_value(data, mask, active_od, neighbor_table)
        )

        return (
            slot_table.cpu().numpy(),
            active_od.cpu().numpy(),
            actions.cpu().numpy() if actions is not None else np.array([], dtype=np.int64),
            log_probs.cpu().numpy(),  # [P] per-pair
            value.item(),
        )

    @torch.no_grad()
    def get_value(self, obs: Dict[str, np.ndarray]) -> float:
        self.policy.eval()
        data, mask, _, _ = GNNRoutingPolicy.obs_to_pyg(obs, self.device)
        _, value = self.policy(data, mask)
        return value.item()

    def store_transition(
        self, obs, nexthop_table, active_od, active_actions, log_probs, reward, value, done,
    ):
        self.buffer.add(RolloutStep(
            obs, nexthop_table, active_od, active_actions, log_probs, reward, value, done,
        ))

    def update(self) -> Dict[str, float]:
        """Run PPO update with per-pair surrogate."""
        if len(self.buffer) == 0:
            return {}

        last_step = self.buffer.steps[-1]
        if last_step.done:
            last_value = 0.0
        else:
            last_value = self.get_value(last_step.obs)

        self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda,
        )

        adv = self.buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.buffer.advantages = adv

        self.policy.train()
        total_loss = 0.0
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        n = len(self.buffer)
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]

                batch_pg_loss = torch.tensor(0.0, device=self.device)
                batch_v_loss = torch.tensor(0.0, device=self.device)
                batch_entropy = torch.tensor(0.0, device=self.device)
                batch_count = 0

                for i in batch_idx:
                    step = self.buffer.steps[i]
                    data, mask, active_od, neighbor_table = (
                        GNNRoutingPolicy.obs_to_pyg(step.obs, self.device)
                    )

                    if active_od.shape[0] == 0:
                        continue

                    old_actions = torch.tensor(
                        step.active_actions, dtype=torch.long, device=self.device,
                    )
                    old_log_probs = torch.tensor(
                        step.log_probs, dtype=torch.float32, device=self.device,
                    )  # [P]

                    _, _, new_log_probs, entropy_per_pair, value = (
                        self.policy.get_action_and_value(
                            data, mask, active_od, neighbor_table, old_actions,
                        )
                    )
                    # new_log_probs: [P], entropy_per_pair: [P]

                    ret = torch.tensor(
                        self.buffer.returns[i], dtype=torch.float32, device=self.device,
                    )
                    advantage = torch.tensor(
                        self.buffer.advantages[i], dtype=torch.float32, device=self.device,
                    )

                    # Per-pair PPO surrogate, shared advantage within step
                    ratio = torch.exp(new_log_probs - old_log_probs)  # [P]
                    pg_loss1 = -advantage * ratio
                    pg_loss2 = -advantage * torch.clamp(
                        ratio, 1 - self.clip_eps, 1 + self.clip_eps,
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # mean over pairs

                    v_loss = 0.5 * (value.squeeze() - ret) ** 2
                    ent_loss = -entropy_per_pair.mean()

                    batch_pg_loss = batch_pg_loss + pg_loss
                    batch_v_loss = batch_v_loss + v_loss
                    batch_entropy = batch_entropy + ent_loss
                    batch_count += 1

                if batch_count == 0:
                    continue

                batch_pg_loss = batch_pg_loss / batch_count
                batch_v_loss = batch_v_loss / batch_count
                batch_entropy = batch_entropy / batch_count

                loss = (
                    batch_pg_loss
                    + self.value_coef * batch_v_loss
                    + self.entropy_coef * batch_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm,
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_pg_loss += batch_pg_loss.item()
                total_v_loss += batch_v_loss.item()
                total_entropy += -batch_entropy.item()
                n_updates += 1

        self.buffer.clear()

        if n_updates == 0:
            return {}

        return {
            "loss": total_loss / n_updates,
            "pg_loss": total_pg_loss / n_updates,
            "value_loss": total_v_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
