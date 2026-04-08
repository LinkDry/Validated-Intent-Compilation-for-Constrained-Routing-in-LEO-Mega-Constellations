"""Fine-grained routing metrics: path stretch, hop count, loop rate, exact-match."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import yaml
import json
import torch
import numpy as np
import networkx as nx

from env.routing_env import LEORoutingEnv
from policy.routing_policy import GNNRoutingPolicy


def compute_detailed_metrics(obs, gnn_slots, dijkstra_slots, neighbor_table):
    """Compare GNN routing table against Dijkstra on a single snapshot.

    Returns dict with:
    - exact_match_rate: fraction of (src,dst) pairs with identical next-hop
    - hop_stretch: mean ratio of GNN hops / Dijkstra hops
    - delay_stretch: mean ratio of GNN path delay / Dijkstra path delay
    - loop_rate: fraction of (src,dst) pairs where GNN routing has a loop
    - mean_hops_gnn, mean_hops_dijkstra
    """
    ei = obs["edge_index"]
    ea = obs["edge_attr"]
    num_edges = int(obs["num_edges"])
    nt = neighbor_table
    N = nt.shape[0]

    # Build edge delay lookup
    edge_delays = {}
    for idx in range(num_edges):
        u, v = int(ei[0, idx]), int(ei[1, idx])
        edge_delays[(u, v)] = float(ea[idx, 0])

    exact_matches = 0
    total_pairs = 0
    loops_gnn = 0
    hop_stretches = []
    delay_stretches = []
    hops_gnn_list = []
    hops_dij_list = []

    for src in range(N):
        for dst in range(N):
            if src == dst:
                continue

            # Trace GNN path
            gnn_path = _trace_path(src, dst, gnn_slots, nt, edge_delays, N)
            dij_path = _trace_path(src, dst, dijkstra_slots, nt, edge_delays, N)

            if dij_path is None or dij_path["hops"] == 0:
                continue

            total_pairs += 1

            # Exact match
            if gnn_slots[src, dst] == dijkstra_slots[src, dst]:
                exact_matches += 1

            # Loop detection
            if gnn_path is None or gnn_path["loop"]:
                loops_gnn += 1
                continue

            # Hop stretch
            if dij_path["hops"] > 0:
                hop_stretches.append(gnn_path["hops"] / dij_path["hops"])
                hops_gnn_list.append(gnn_path["hops"])
                hops_dij_list.append(dij_path["hops"])

            # Delay stretch
            if dij_path["delay"] > 0:
                delay_stretches.append(gnn_path["delay"] / dij_path["delay"])

    return {
        "exact_match_rate": exact_matches / max(total_pairs, 1),
        "loop_rate": loops_gnn / max(total_pairs, 1),
        "mean_hop_stretch": float(np.mean(hop_stretches)) if hop_stretches else 0,
        "mean_delay_stretch": float(np.mean(delay_stretches)) if delay_stretches else 0,
        "p95_delay_stretch": float(np.percentile(delay_stretches, 95)) if delay_stretches else 0,
        "p99_delay_stretch": float(np.percentile(delay_stretches, 99)) if delay_stretches else 0,
        "mean_hops_gnn": float(np.mean(hops_gnn_list)) if hops_gnn_list else 0,
        "mean_hops_dijkstra": float(np.mean(hops_dij_list)) if hops_dij_list else 0,
        "total_pairs": total_pairs,
    }


def _trace_path(src, dst, slot_table, neighbor_table, edge_delays, N, max_hops=30):
    """Trace a packet path using the slot table. Returns path info or None."""
    current = src
    visited = set()
    hops = 0
    delay = 0.0
    has_loop = False

    for _ in range(max_hops):
        if current == dst:
            return {"hops": hops, "delay": delay, "loop": False}
        if current in visited:
            return {"hops": hops, "delay": delay, "loop": True}
        visited.add(current)

        slot = int(slot_table[current, dst])
        next_node = int(neighbor_table[current, slot])
        if next_node < 0:
            return None  # dead end

        d = edge_delays.get((current, next_node))
        if d is None:
            return None
        delay += d
        hops += 1
        current = next_node

    # Exceeded max hops
    return {"hops": hops, "delay": delay, "loop": True}


def build_dijkstra_slots(obs):
    """Build Dijkstra next-hop slot table from observation."""
    ei = obs["edge_index"]
    ea = obs["edge_attr"]
    num_edges = int(obs["num_edges"])
    nt = obs["neighbor_table"]
    N = nt.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for idx in range(num_edges):
        u, v = int(ei[0, idx]), int(ei[1, idx])
        delay = float(ea[idx, 0])
        G.add_edge(u, v, weight=delay)

    slot_table = np.zeros((N, N), dtype=np.int64)
    for src in range(N):
        try:
            _, paths = nx.single_source_dijkstra(G, src, weight='weight')
        except Exception:
            continue
        for dst, path in paths.items():
            if src != dst and len(path) >= 2:
                next_node = path[1]
                slots = np.where(nt[src] == next_node)[0]
                if len(slots) >= 1:
                    slot_table[src, dst] = int(slots[0])

    return slot_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="output/pretrain_ctg3/best.pt")
    parser.add_argument("--n-snapshots", type=int, default=10)
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

    obs, _ = env.reset(seed=args.seed)
    all_metrics = []

    for i in range(args.n_snapshots):
        # GNN routing table
        data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, device)
        with torch.no_grad():
            gnn_slots, _ = policy.get_routing_table(data, nt, nd, mask)
        gnn_slots_np = gnn_slots.cpu().numpy()

        # Dijkstra routing table
        dij_slots = build_dijkstra_slots(obs)

        # Compare
        metrics = compute_detailed_metrics(
            obs, gnn_slots_np, dij_slots, obs["neighbor_table"]
        )
        all_metrics.append(metrics)

        print(f"Snapshot {i}: exact_match={metrics['exact_match_rate']:.4f} "
              f"loop_rate={metrics['loop_rate']:.4f} "
              f"hop_stretch={metrics['mean_hop_stretch']:.4f} "
              f"delay_stretch={metrics['mean_delay_stretch']:.4f} "
              f"p99_stretch={metrics['p99_delay_stretch']:.4f}")

        # Advance topology
        dummy = np.zeros((env.N, env.N), dtype=np.int64)
        obs, _, terminated, _, _ = env.step(dummy)
        if terminated:
            obs, _ = env.reset()

    # Aggregate
    print(f"\n{'='*60}")
    print(f"Aggregate over {len(all_metrics)} snapshots:")
    for key in ["exact_match_rate", "loop_rate", "mean_hop_stretch",
                "mean_delay_stretch", "p95_delay_stretch", "p99_delay_stretch",
                "mean_hops_gnn", "mean_hops_dijkstra"]:
        vals = [m[key] for m in all_metrics]
        print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Save
    out_path = "output/detailed_metrics.json"
    os.makedirs("output", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "snapshots": all_metrics,
            "aggregate": {
                key: {"mean": float(np.mean([m[key] for m in all_metrics])),
                      "std": float(np.std([m[key] for m in all_metrics]))}
                for key in all_metrics[0].keys()
            }
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
