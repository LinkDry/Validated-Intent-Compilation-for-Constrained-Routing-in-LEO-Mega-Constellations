"""Evaluate trained cost-to-go policy and compare with baselines."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import yaml
import json
import torch
import numpy as np

from env.routing_env import LEORoutingEnv
from policy.routing_policy import GNNRoutingPolicy
from baselines.shortest_path import DijkstraRouter, RandomRouter


def eval_gnn(env, policy, device, seeds, n_steps=192):
    """Evaluate GNN cost-to-go policy."""
    all_results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        pdrs, delays, drops = [], [], []
        for _ in range(n_steps):
            data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, device)
            with torch.no_grad():
                slot_table, cost = policy.get_routing_table(data, nt, nd, mask)
            obs, _, terminated, _, info = env.step(slot_table.cpu().numpy())
            pdrs.append(info["pdr"])
            delays.append(info["mean_delay_ms"])
            drops.append(info["drop_rate"])
            if terminated:
                break
        all_results.append({
            "pdr": float(np.mean(pdrs)),
            "delay": float(np.mean(delays)),
            "drop": float(np.mean(drops)),
        })
    return all_results


def eval_dijkstra(env, seeds, n_steps=192):
    """Evaluate Dijkstra baseline."""
    router = DijkstraRouter()
    all_results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        pdrs, delays, drops = [], [], []
        for _ in range(n_steps):
            node_table = router.build_nexthop_table(obs)
            slot_table = _node_table_to_slots(node_table, obs["neighbor_table"])
            obs, _, terminated, _, info = env.step(slot_table)
            pdrs.append(info["pdr"])
            delays.append(info["mean_delay_ms"])
            drops.append(info["drop_rate"])
            if terminated:
                break
        all_results.append({
            "pdr": float(np.mean(pdrs)),
            "delay": float(np.mean(delays)),
            "drop": float(np.mean(drops)),
        })
    return all_results


def eval_random(env, seeds, n_steps=192):
    """Evaluate random baseline."""
    all_results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        rng = np.random.RandomState(seed)
        pdrs, delays, drops = [], [], []
        for _ in range(n_steps):
            mask = obs["action_mask"]
            N, K = mask.shape
            slot_table = np.zeros((N, N), dtype=np.int64)
            for node in range(N):
                valid = np.where(mask[node] > 0)[0]
                if len(valid) > 0:
                    slot_table[node, :] = rng.choice(valid, size=N)
            obs, _, terminated, _, info = env.step(slot_table)
            pdrs.append(info["pdr"])
            delays.append(info["mean_delay_ms"])
            drops.append(info["drop_rate"])
            if terminated:
                break
        all_results.append({
            "pdr": float(np.mean(pdrs)),
            "delay": float(np.mean(delays)),
            "drop": float(np.mean(drops)),
        })
    return all_results


def _node_table_to_slots(node_table, neighbor_table):
    """Convert [N,N] node-ID table to [N,N] slot-index table."""
    N, K = neighbor_table.shape
    slot_table = np.zeros((N, N), dtype=np.int64)
    for node in range(N):
        for k in range(K):
            nid = neighbor_table[node, k]
            if nid >= 0:
                mask = node_table[node] == nid
                slot_table[node, mask] = k
    return slot_table


def summarize(name, results):
    pdrs = [r["pdr"] for r in results]
    delays = [r["delay"] for r in results]
    drops = [r["drop"] for r in results]
    print(f"  {name:12s}: PDR={np.mean(pdrs):.4f}+/-{np.std(pdrs):.4f}  "
          f"delay={np.mean(delays):.1f}+/-{np.std(delays):.1f}ms  "
          f"drop={np.mean(drops):.4f}")
    return {
        "name": name,
        "pdr_mean": float(np.mean(pdrs)), "pdr_std": float(np.std(pdrs)),
        "delay_mean": float(np.mean(delays)), "delay_std": float(np.std(delays)),
        "drop_mean": float(np.mean(drops)), "drop_std": float(np.std(drops)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="output/pretrain_ctg3/best.pt")
    parser.add_argument("--scenarios", nargs="+",
                        default=["uniform", "hotspot", "polar_stress", "flash"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 777, 789, 999, 1234, 2024])
    parser.add_argument("--steps", type=int, default=192)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.model, map_location=device)
    hidden_dim = ckpt.get("hidden_dim", cfg["policy"]["hidden_dim"])
    rank = ckpt.get("rank", 64)

    results = {}

    for scenario in args.scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}")

        env = LEORoutingEnv(
            num_planes=cfg["constellation"]["num_planes"],
            sats_per_plane=cfg["constellation"]["sats_per_plane"],
            altitude_km=cfg["constellation"]["altitude_km"],
            inclination_deg=cfg["constellation"]["inclination_deg"],
            max_isl_range_km=cfg["isl"]["max_range_km"],
            polar_lat_threshold_deg=cfg["isl"]["polar_lat_threshold_deg"],
            scenario=scenario,
        )

        policy = GNNRoutingPolicy(
            node_feat_dim=env.node_feat_dim,
            edge_feat_dim=env.edge_feat_dim,
            hidden_dim=hidden_dim,
            rank=rank,
        ).to(device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()

        gnn_res = eval_gnn(env, policy, device, args.seeds, args.steps)
        dij_res = eval_dijkstra(env, args.seeds, args.steps)
        rnd_res = eval_random(env, args.seeds, args.steps)

        results[scenario] = [
            summarize("GNN-CtG", gnn_res),
            summarize("Dijkstra", dij_res),
            summarize("Random", rnd_res),
        ]

    out_path = "output/eval_results.json"
    os.makedirs("output", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
