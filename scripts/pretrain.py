"""Supervised pretraining: distill Dijkstra cost-to-go distance field."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import networkx as nx

from env.routing_env import LEORoutingEnv
from policy.routing_policy import GNNRoutingPolicy
from baselines.shortest_path import DijkstraRouter


def build_distance_labels(obs):
    """Build [N, N] shortest-path distance matrix from current topology.

    Returns:
        dist_matrix: [N, N] float32, shortest-path delay in ms. np.inf for unreachable.
        nexthop_slots: [N, N] int64, teacher next-hop slot indices. -100 for ignore.
    """
    ei = obs["edge_index"]
    ea = obs["edge_attr"]
    num_edges = int(obs["num_edges"])
    neighbor_table = obs["neighbor_table"]
    N = neighbor_table.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for idx in range(num_edges):
        u, v = int(ei[0, idx]), int(ei[1, idx])
        delay = float(ea[idx, 0])
        G.add_edge(u, v, weight=delay)

    dist_matrix = np.full((N, N), np.inf, dtype=np.float32)
    nexthop_slots = np.full((N, N), -100, dtype=np.int64)

    for src in range(N):
        try:
            lengths, paths = nx.single_source_dijkstra(G, src, weight='weight')
        except Exception:
            continue
        for dst, d in lengths.items():
            dist_matrix[src, dst] = d
            if src != dst and len(paths[dst]) >= 2:
                next_node = paths[dst][1]
                slots = np.where(neighbor_table[src] == next_node)[0]
                if len(slots) >= 1:
                    nexthop_slots[src, dst] = int(slots[0])

    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix, nexthop_slots


def collect_snapshots(env, n_snapshots, seed=42):
    """Collect diverse topology snapshots across orbital phases."""
    snapshots = []
    obs, _ = env.reset(seed=seed)

    for i in range(n_snapshots):
        dist_matrix, nexthop_slots = build_distance_labels(obs)
        snapshots.append((obs, dist_matrix, nexthop_slots))

        dummy = np.zeros((env.N, env.N), dtype=np.int64)
        obs, _, terminated, _, _ = env.step(dummy)
        if terminated:
            obs, _ = env.reset()

        if (i + 1) % 100 == 0:
            print(f"  Collected {i+1}/{n_snapshots} snapshots")

    return snapshots


def evaluate_policy(env, policy, device, n_steps=20, seed=999):
    """Evaluate policy in the env, return metrics."""
    obs, _ = env.reset(seed=seed)
    pdrs, delays = [], []

    for step in range(n_steps):
        data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, device)
        with torch.no_grad():
            slot_table, cost = policy.get_routing_table(data, nt, nd, mask)
            slot_table_np = slot_table.cpu().numpy()

        obs, reward, terminated, _, info = env.step(slot_table_np)
        pdrs.append(info["pdr"])
        delays.append(info["mean_delay_ms"])
        if terminated:
            break

    return {
        "pdr": float(np.mean(pdrs)),
        "delay": float(np.mean(delays)),
    }


def pretrain(cfg_path, output_dir, n_epochs=200, n_snapshots=500, lr=1e-3, seed=42):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = LEORoutingEnv(
        num_planes=cfg["constellation"]["num_planes"],
        sats_per_plane=cfg["constellation"]["sats_per_plane"],
        altitude_km=cfg["constellation"]["altitude_km"],
        inclination_deg=cfg["constellation"]["inclination_deg"],
        max_isl_range_km=cfg["isl"]["max_range_km"],
        polar_lat_threshold_deg=cfg["isl"]["polar_lat_threshold_deg"],
        scenario="uniform",
        seed=seed,
    )

    hidden_dim = cfg["policy"].get("hidden_dim", 128)
    rank = 64  # Higher rank for MLP scorer
    policy = GNNRoutingPolicy(
        node_feat_dim=env.node_feat_dim,
        edge_feat_dim=env.edge_feat_dim,
        hidden_dim=hidden_dim,
        rank=rank,
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy params: {n_params:,} (hidden_dim={hidden_dim}, rank={rank})")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine schedule
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Collect snapshots
    print(f"Collecting {n_snapshots} topology snapshots...")
    snapshots = collect_snapshots(env, n_snapshots, seed)

    # Stats
    reachable = sum((d < np.inf).sum() - d.shape[0] for _, d, _ in snapshots)
    total = sum(d.shape[0] * d.shape[1] - d.shape[0] for _, d, _ in snapshots)
    print(f"Reachable pairs: {reachable:,}/{total:,} ({reachable/total*100:.1f}%)")

    all_dists = np.concatenate([d[(d < np.inf) & (d > 0)].flatten() for _, d, _ in snapshots])
    dist_mean = float(all_dists.mean())
    dist_std = float(all_dists.std())
    print(f"Distance stats: mean={dist_mean:.2f}ms, std={dist_std:.2f}ms, "
          f"max={all_dists.max():.2f}ms, min={all_dists.min():.2f}ms")

    # Training
    t0 = time.time()
    best_acc = 0.0
    best_pdr = 0.0
    stale_count = 0

    for epoch in range(n_epochs):
        policy.train()
        epoch_cost_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_nexthop_acc = 0.0
        epoch_count = 0
        epoch_grad_norm = 0.0

        np.random.shuffle(snapshots)

        for obs, dist_matrix, nexthop_slots in snapshots:
            data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, device)

            cost_pred = policy(data)  # [N, N]

            # Target: raw distances in ms
            target = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
            diag_mask = torch.eye(target.shape[0], dtype=torch.bool, device=device)
            valid_mask = (target < 1e6) & ~diag_mask & (target > 0)

            if valid_mask.sum() == 0:
                continue

            # MSE loss (simpler, stronger gradient than Huber for this scale)
            cost_loss = F.mse_loss(cost_pred[valid_mask], target[valid_mask])

            # Ranking loss: teacher next-hop should have lowest candidate cost
            nbr_idx = nt.clamp(min=0)  # [N, K]
            nbr_cost = cost_pred[nbr_idx]  # [N, K, N]
            candidate = nd.unsqueeze(-1) + nbr_cost  # [N, K, N]
            inv_mask = (mask == 0).unsqueeze(-1)
            candidate = candidate.masked_fill(inv_mask, float('inf'))

            teacher = torch.tensor(nexthop_slots, dtype=torch.long, device=device)
            has_label = teacher >= 0  # [N, N]

            if has_label.sum() > 0:
                pred_slots = candidate.argmin(dim=1)  # [N, N]
                acc = (pred_slots[has_label] == teacher[has_label]).float().mean()
                epoch_nexthop_acc += acc.item()

                src_idx, dst_idx = torch.where(has_label)
                pair_candidates = candidate[src_idx, :, dst_idx]  # [P, K]
                pair_teacher = teacher[src_idx, dst_idx]  # [P]

                # Temperature-scaled cross-entropy (lower temp = sharper)
                temperature = max(0.5, 2.0 * (1.0 - epoch / n_epochs))
                rank_loss = F.cross_entropy(-pair_candidates / temperature, pair_teacher)
            else:
                rank_loss = torch.tensor(0.0, device=device)
                epoch_nexthop_acc += 0.0

            # Phased loss: cost-only warmup, then ramp in ranking
            rank_phase_start = 0.2  # start ranking at 20% of training
            rank_phase_end = 0.4    # full ranking weight at 40%
            progress = epoch / max(1, n_epochs)
            if progress < rank_phase_start:
                rank_w = 0.0
            elif progress < rank_phase_end:
                rank_w = (progress - rank_phase_start) / (rank_phase_end - rank_phase_start)
            else:
                rank_w = 1.0
            loss = cost_loss + rank_w * 2.0 * rank_loss

            optimizer.zero_grad()
            loss.backward()

            # Monitor gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            epoch_grad_norm += grad_norm.item()

            optimizer.step()

            epoch_cost_loss += cost_loss.item()
            epoch_rank_loss += rank_loss.item()
            epoch_count += 1

        scheduler.step()

        avg_cost = epoch_cost_loss / max(epoch_count, 1)
        avg_rank = epoch_rank_loss / max(epoch_count, 1)
        avg_acc = epoch_nexthop_acc / max(epoch_count, 1)
        avg_grad = epoch_grad_norm / max(epoch_count, 1)
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            policy.eval()
            metrics = evaluate_policy(env, policy, device)
            pdr = metrics["pdr"]
            delay = metrics["delay"]
            print(f"Epoch {epoch:3d}/{n_epochs} | cost={avg_cost:.1f} rank={avg_rank:.4f} "
                  f"acc={avg_acc:.3f} | PDR={pdr:.3f} delay={delay:.1f}ms | "
                  f"grad={avg_grad:.2f} lr={cur_lr:.1e} | {elapsed:.0f}s")

            improved = False
            if avg_acc > best_acc + 0.01 or pdr > best_pdr + 0.01:
                improved = True
                if avg_acc > best_acc:
                    best_acc = avg_acc
                if pdr > best_pdr:
                    best_pdr = pdr
                stale_count = 0
                torch.save({
                    "policy_state_dict": policy.state_dict(),
                    "epoch": epoch,
                    "pdr": pdr,
                    "acc": avg_acc,
                    "rank": rank,
                    "hidden_dim": hidden_dim,
                }, os.path.join(output_dir, "best.pt"))
                print(f"  -> New best (acc={avg_acc:.3f}, PDR={pdr:.3f})")
            else:
                stale_count += 1

            # Early stopping if no improvement for 40 eval cycles (200 epochs)
            if stale_count > 40:
                print(f"  Early stopping at epoch {epoch} (no improvement for {stale_count} evals)")
                break

    # Final save
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "epoch": n_epochs,
        "pdr": best_pdr,
        "acc": best_acc,
    }, os.path.join(output_dir, "final.pt"))
    print(f"\nTraining complete. Best acc: {best_acc:.3f}, Best PDR: {best_pdr:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="output/pretrain_ctg")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--snapshots", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pretrain(args.config, args.output_dir, args.epochs, args.snapshots, args.lr, args.seed)


if __name__ == "__main__":
    main()
