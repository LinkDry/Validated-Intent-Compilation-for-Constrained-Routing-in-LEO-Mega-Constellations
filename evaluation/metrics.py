"""Evaluation metrics for LEO routing."""

import numpy as np
from typing import Dict, List


def compute_episode_metrics(step_infos: List[Dict]) -> Dict[str, float]:
    """Aggregate per-step info dicts into episode-level metrics."""
    if not step_infos:
        return {}

    pdrs = [s["pdr"] for s in step_infos]
    delays = [s["mean_delay_ms"] for s in step_infos]
    delivered = sum(s["total_delivered"] for s in step_infos)
    injected = sum(s["total_injected"] for s in step_infos)
    dropped = sum(s["total_dropped"] for s in step_infos)

    return {
        "mean_pdr": float(np.mean(pdrs)),
        "std_pdr": float(np.std(pdrs)),
        "min_pdr": float(np.min(pdrs)),
        "mean_delay_ms": float(np.mean(delays)),
        "p95_delay_ms": float(np.percentile(delays, 95)) if delays else 0.0,
        "total_delivered": delivered,
        "total_injected": injected,
        "total_dropped": dropped,
        "overall_pdr": delivered / (injected + 1e-8),
        "overall_drop_rate": dropped / (injected + 1e-8),
        "n_steps": len(step_infos),
    }


def evaluate_policy(env, policy_fn, n_episodes: int = 5, max_steps: int = 200) -> Dict[str, float]:
    """Run evaluation episodes and return aggregated metrics.

    Args:
        env: LEORoutingEnv instance
        policy_fn: callable(obs) -> action (np.ndarray)
        n_episodes: number of evaluation episodes
        max_steps: max steps per episode
    """
    all_metrics = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        step_infos = []

        for _ in range(max_steps):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step_infos.append(info)
            if terminated or truncated:
                break

        all_metrics.append(compute_episode_metrics(step_infos))

    # Average across episodes
    keys = all_metrics[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals))

    return result
