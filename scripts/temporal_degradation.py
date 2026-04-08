"""Temporal degradation: how fast does a stale routing table degrade?"""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="output/pretrain_ctg3/best.pt")
    parser.add_argument("--max-stale-steps", type=int, default=50)
    parser.add_argument("--recompute-interval", type=int, default=1,
                        help="How often GNN recomputes (1=every step, 5=every 5 steps)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = LEORoutingEnv(
        num_planes=cfg["constellation"]["num_planes"],
        sats_per_plane=cfg["constellation"]["sats_per_plane"],
        altitude_km=cfg["constellation"]["altitude_km"],
        inclination_deg=cfg["constellation"]["inclination_deg"],
        max_isl_range_km=cfg["isl"]["max_range_km"],
        polar_lat_threshold_deg=cfg["isl"]["polar_lat_threshold_deg"],
        scenario="uniform", seed=args.seed,
    )

    ckpt = torch.load(args.model, map_location=device)
    policy = GNNRoutingPolicy(
        node_feat_dim=env.node_feat_dim, edge_feat_dim=env.edge_feat_dim,
        hidden_dim=ckpt.get("hidden_dim", 128), rank=ckpt.get("rank", 64),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # Test different recompute intervals
    intervals = [1, 2, 5, 10, 20, 50]
    results = {}

    for interval in intervals:
        obs, _ = env.reset(seed=args.seed)
        pdrs, delays = [], []
        cached_slots = None

        for step in range(args.max_stale_steps):
            if step % interval == 0:
                # Recompute routing table
                data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, device)
                with torch.no_grad():
                    cached_slots, _ = policy.get_routing_table(data, nt, nd, mask)
                cached_slots_np = cached_slots.cpu().numpy()

            obs, _, terminated, _, info = env.step(cached_slots_np)
            pdrs.append(info["pdr"])
            delays.append(info["mean_delay_ms"])
            if terminated:
                break

        mean_pdr = float(np.mean(pdrs))
        mean_delay = float(np.mean(delays))
        results[interval] = {
            "pdr_mean": mean_pdr,
            "pdr_std": float(np.std(pdrs)),
            "delay_mean": mean_delay,
            "delay_std": float(np.std(delays)),
            "steps": len(pdrs),
        }
        epoch_dur = env.constellation.epoch_duration_s
        print(f"Recompute every {interval:2d} steps ({interval*epoch_dur:.0f}s): "
              f"PDR={mean_pdr:.4f} delay={mean_delay:.1f}ms")

    # Save
    out_path = "output/temporal_degradation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
