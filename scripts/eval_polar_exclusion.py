"""Polar exclusion zone GNN robustness test.

Tests GNN routing performance under varying polar latitude thresholds.
Inter-plane ISLs are disabled when either endpoint exceeds the threshold.
With 53deg inclination, satellites reach up to ~53deg latitude, so
thresholds must be below 53deg to have effect.

Thresholds: 30, 40, 45, 50 degrees latitude.
"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.routing_env import LEORoutingEnv
from policy.routing_policy import GNNRoutingPolicy
from baselines.shortest_path import DijkstraRouter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "output/pretrain_ctg3/best.pt"
N_STEPS = 50
SEEDS = [42, 123, 456]
THRESHOLDS = [30.0, 40.0, 45.0, 50.0]


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    hidden = ckpt.get("hidden_dim", 128)
    rank = ckpt["policy_state_dict"]["cost_head.src_proj.0.weight"].shape[0]
    policy = GNNRoutingPolicy(node_feat_dim=8, edge_feat_dim=4,
                              hidden_dim=hidden, rank=rank)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(DEVICE)
    policy.eval()
    return policy


def node_table_to_slots(node_table, neighbor_table):
    N, K = neighbor_table.shape
    slot_table = np.zeros((N, N), dtype=np.int64)
    for node in range(N):
        for dst in range(N):
            nexthop = node_table[node, dst]
            if nexthop < 0:
                continue
            for k in range(K):
                if neighbor_table[node, k] == nexthop:
                    slot_table[node, dst] = k
                    break
    return slot_table


def evaluate_threshold(threshold, policy):
    """Run GNN and Dijkstra with a given polar exclusion threshold."""
    gnn_results = []
    dijk_results = []

    for seed in SEEDS:
        env = LEORoutingEnv(
            num_planes=20, sats_per_plane=20,
            altitude_km=550.0, inclination_deg=53.0,
            polar_lat_threshold_deg=threshold,
            scenario="uniform", seed=seed,
        )

        # --- GNN ---
        obs, _ = env.reset(seed=seed)
        num_edges = int(obs["num_edges"])
        total_possible = 400 * 4  # N * max_degree
        gnn_pdrs = []
        for step in range(N_STEPS):
            data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, DEVICE)
            with torch.no_grad():
                st, _ = policy.get_routing_table(data, nt, nd, mask)
            obs, _, terminated, _, info = env.step(st.cpu().numpy())
            gnn_pdrs.append(info["pdr"])
            if terminated:
                break

        # --- Dijkstra ---
        obs, _ = env.reset(seed=seed)
        dijkstra = DijkstraRouter()
        dijk_pdrs = []
        for step in range(N_STEPS):
            node_table = dijkstra.build_nexthop_table(obs)
            slot_table = node_table_to_slots(node_table, obs["neighbor_table"])
            obs, _, terminated, _, info = env.step(slot_table)
            dijk_pdrs.append(info["pdr"])
            if terminated:
                break

        gnn_results.append({"seed": seed, "mean_pdr": float(np.mean(gnn_pdrs))})
        dijk_results.append({"seed": seed, "mean_pdr": float(np.mean(dijk_pdrs))})

    gnn_avg = float(np.mean([r["mean_pdr"] for r in gnn_results]))
    dijk_avg = float(np.mean([r["mean_pdr"] for r in dijk_results]))
    gnn_std = float(np.std([r["mean_pdr"] for r in gnn_results]))
    dijk_std = float(np.std([r["mean_pdr"] for r in dijk_results]))

    return {
        "threshold_deg": threshold,
        "num_edges": num_edges,
        "edges_removed": total_possible - num_edges,
        "edge_removal_pct": round(100.0 * (total_possible - num_edges) / total_possible, 1),
        "gnn_pdr_mean": round(gnn_avg * 100, 2),
        "gnn_pdr_std": round(gnn_std * 100, 2),
        "dijkstra_pdr_mean": round(dijk_avg * 100, 2),
        "dijkstra_pdr_std": round(dijk_std * 100, 2),
        "gap": round((gnn_avg - dijk_avg) * 100, 2),
        "gnn_seeds": gnn_results,
        "dijkstra_seeds": dijk_results,
    }


def main():
    print("Loading GNN model (trained with polar_threshold=75deg)...")
    policy = load_model()
    print(f"Model loaded on {DEVICE}")
    print("Note: 53deg inclination => max satellite latitude ~53deg")
    print("Thresholds below 53deg will remove inter-plane ISLs near orbit edges")

    results = []
    for threshold in THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"Polar exclusion threshold: {threshold} deg")
        print(f"{'='*60}")
        t0 = time.time()
        r = evaluate_threshold(threshold, policy)
        elapsed = time.time() - t0
        results.append(r)
        print(f"  Active edges: {r['num_edges']} (removed {r['edges_removed']}, {r['edge_removal_pct']}%)")
        print(f"  GNN PDR:      {r['gnn_pdr_mean']:.2f}% (+/- {r['gnn_pdr_std']:.2f}%)")
        print(f"  Dijkstra PDR: {r['dijkstra_pdr_mean']:.2f}% (+/- {r['dijkstra_pdr_std']:.2f}%)")
        print(f"  Gap (GNN-Dij): {r['gap']:+.2f}%")
        print(f"  Time: {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("POLAR EXCLUSION ZONE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Threshold':>10s} {'Edges':>7s} {'Removed':>9s} {'GNN PDR':>10s} {'Dijkstra':>10s} {'Gap':>8s}")
    print(f"{'-'*10} {'-'*7} {'-'*9} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        print(f"{r['threshold_deg']:>9.0f}° {r['num_edges']:>7d} {r['edge_removal_pct']:>8.1f}% "
              f"{r['gnn_pdr_mean']:>9.2f}% {r['dijkstra_pdr_mean']:>9.2f}% "
              f"{r['gap']:>+7.2f}%")

    os.makedirs("output", exist_ok=True)
    with open("output/polar_exclusion_gnn.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to output/polar_exclusion_gnn.json")


if __name__ == "__main__":
    main()
