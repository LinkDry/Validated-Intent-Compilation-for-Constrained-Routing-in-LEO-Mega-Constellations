"""Ablation study: test contribution of each architectural choice."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from env.routing_env import LEORoutingEnv
from policy.routing_policy import GNNRoutingPolicy
from policy.gat_encoder import GATEncoder
from scripts.pretrain import build_distance_labels, collect_snapshots, evaluate_policy


class BilinearCostHead(nn.Module):
    """Old bilinear head for ablation comparison."""
    def __init__(self, embed_dim=128, rank=32):
        super().__init__()
        self.src_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, rank))
        self.dst_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, rank))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, node_emb):
        src = self.src_proj(node_emb)
        dst = self.dst_proj(node_emb)
        return torch.mm(src, dst.t()) + self.bias

    def derive_nexthop(self, cost, neighbor_table, edge_delays, neighbor_mask):
        nbr_idx = neighbor_table.clamp(min=0)
        nbr_cost = cost[nbr_idx]
        candidate = edge_delays.unsqueeze(-1) + nbr_cost
        inv_mask = (neighbor_mask == 0).unsqueeze(-1)
        candidate = candidate.masked_fill(inv_mask, float('inf'))
        return candidate.argmin(dim=1)


def train_ablation(env, cfg, device, snapshots, name, n_epochs=50,
                   use_bilinear=False, no_phase=False, no_bias_init=False):
    """Train one ablation variant and return final metrics."""
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"{'='*60}")

    hidden_dim = cfg["policy"].get("hidden_dim", 128)
    rank = 64

    policy = GNNRoutingPolicy(
        node_feat_dim=env.node_feat_dim, edge_feat_dim=env.edge_feat_dim,
        hidden_dim=hidden_dim, rank=rank,
    ).to(device)

    # Ablation: swap cost head to bilinear
    if use_bilinear:
        policy.cost_head = BilinearCostHead(embed_dim=hidden_dim, rank=rank).to(device)

    # Ablation: no bias init (re-init final layer)
    if no_bias_init and not use_bilinear:
        with torch.no_grad():
            policy.cost_head.scorer[-1].bias.zero_()
            policy.cost_head.scorer[-1].weight.normal_(0, 0.02)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    best_pdr = 0.0
    history = []

    for epoch in range(n_epochs):
        policy.train()
        epoch_acc = 0.0
        epoch_cost = 0.0
        epoch_count = 0

        np.random.shuffle(snapshots)

        for obs, dist_matrix, nexthop_slots in snapshots:
            data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, device)

            if use_bilinear:
                node_emb = policy.encoder(data)
                cost_pred = policy.cost_head(node_emb)
            else:
                cost_pred = policy(data)

            target = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
            diag_mask = torch.eye(target.shape[0], dtype=torch.bool, device=device)
            valid_mask = (target < 1e6) & ~diag_mask & (target > 0)
            if valid_mask.sum() == 0:
                continue

            cost_loss = F.mse_loss(cost_pred[valid_mask], target[valid_mask])

            nbr_idx = nt.clamp(min=0)
            nbr_cost = cost_pred[nbr_idx]
            candidate = nd.unsqueeze(-1) + nbr_cost
            inv_mask = (mask == 0).unsqueeze(-1)
            candidate = candidate.masked_fill(inv_mask, float('inf'))

            teacher = torch.tensor(nexthop_slots, dtype=torch.long, device=device)
            has_label = teacher >= 0

            rank_loss = torch.tensor(0.0, device=device)
            acc = 0.0
            if has_label.sum() > 0:
                pred_slots = candidate.argmin(dim=1)
                acc = (pred_slots[has_label] == teacher[has_label]).float().mean().item()
                epoch_acc += acc

                src_idx, dst_idx = torch.where(has_label)
                pair_candidates = candidate[src_idx, :, dst_idx]
                pair_teacher = teacher[src_idx, dst_idx]
                rank_loss = F.cross_entropy(-pair_candidates, pair_teacher)

            # Phased or not
            if no_phase:
                loss = cost_loss + 2.0 * rank_loss
            else:
                progress = epoch / max(1, n_epochs)
                if progress < 0.2:
                    rank_w = 0.0
                elif progress < 0.4:
                    rank_w = (progress - 0.2) / 0.2
                else:
                    rank_w = 1.0
                loss = cost_loss + rank_w * 2.0 * rank_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_cost += cost_loss.item()
            epoch_count += 1

        scheduler.step()
        avg_acc = epoch_acc / max(epoch_count, 1)
        avg_cost = epoch_cost / max(epoch_count, 1)

        if avg_acc > best_acc:
            best_acc = avg_acc
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            policy.eval()
            metrics = evaluate_policy(env, policy, device, n_steps=20)
            pdr = metrics["pdr"]
            if pdr > best_pdr:
                best_pdr = pdr
            print(f"  Epoch {epoch:3d} | cost={avg_cost:.1f} acc={avg_acc:.3f} | PDR={pdr:.3f}")
            history.append({"epoch": epoch, "acc": avg_acc, "pdr": pdr, "cost": avg_cost})

    return {
        "name": name,
        "best_acc": best_acc,
        "best_pdr": best_pdr,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--snapshots", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = LEORoutingEnv(
        num_planes=cfg["constellation"]["num_planes"],
        sats_per_plane=cfg["constellation"]["sats_per_plane"],
        altitude_km=cfg["constellation"]["altitude_km"],
        inclination_deg=cfg["constellation"]["inclination_deg"],
        max_isl_range_km=cfg["isl"]["max_range_km"],
        polar_lat_threshold_deg=cfg["isl"]["polar_lat_threshold_deg"],
        scenario="uniform", seed=args.seed,
    )

    print(f"Collecting {args.snapshots} snapshots...")
    snapshots = collect_snapshots(env, args.snapshots, args.seed)

    results = []

    # Full model (baseline)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    r = train_ablation(env, cfg, device, snapshots, "full_model",
                       n_epochs=args.epochs)
    results.append(r)

    # Ablation 1: bilinear head
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    r = train_ablation(env, cfg, device, snapshots, "bilinear_head",
                       n_epochs=args.epochs, use_bilinear=True)
    results.append(r)

    # Ablation 2: no phased loss
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    r = train_ablation(env, cfg, device, snapshots, "no_phased_loss",
                       n_epochs=args.epochs, no_phase=True)
    results.append(r)

    # Ablation 3: no bias init
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    r = train_ablation(env, cfg, device, snapshots, "no_bias_init",
                       n_epochs=args.epochs, no_bias_init=True)
    results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']:20s}: best_acc={r['best_acc']:.3f}  best_pdr={r['best_pdr']:.3f}")

    out_path = "output/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
