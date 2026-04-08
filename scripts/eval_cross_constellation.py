"""Zero-shot cross-constellation GNN generalization test.

Tests the GNN trained on Walker Delta 20x20/550km/53deg on two OOD configs:
  1. SSO: 20x20, 550km, 97deg inclination (near-polar)
  2. High-alt: 20x20, 1200km, 53deg inclination (higher orbit)

Reports PDR for GNN vs Dijkstra on each config + training config baseline.
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

CONFIGS = {
    "training": {
        "altitude_km": 550.0,
        "inclination_deg": 53.0,
        "max_isl_range_km": 5016.0,
        "label": "Walker-Delta 550km/53deg (training)",
    },
    "sso_97": {
        "altitude_km": 550.0,
        "inclination_deg": 97.0,
        "max_isl_range_km": 5016.0,
        "label": "SSO 550km/97deg (OOD)",
    },
    "high_alt": {
        "altitude_km": 1200.0,
        "inclination_deg": 53.0,
        "max_isl_range_km": 6000.0,
        "label": "High-alt 1200km/53deg (OOD)",
    },
}


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


def evaluate_config(config_name, config, policy):
    """Run GNN and Dijkstra on a constellation config across seeds."""
    gnn_results = []
    dijk_results = []

    for seed in SEEDS:
        env = LEORoutingEnv(
            num_planes=20, sats_per_plane=20,
            altitude_km=config["altitude_km"],
            inclination_deg=config["inclination_deg"],
            max_isl_range_km=config["max_isl_range_km"],
            scenario="uniform", seed=seed,
        )

        # --- GNN ---
        obs, _ = env.reset(seed=seed)
        gnn_pdrs, gnn_lats = [], []
        for step in range(N_STEPS):
            data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, DEVICE)
            with torch.no_grad():
                st, _ = policy.get_routing_table(data, nt, nd, mask)
            obs, _, terminated, _, info = env.step(st.cpu().numpy())
            gnn_pdrs.append(info["pdr"])
            if "avg_latency_ms" in info:
                gnn_lats.append(info["avg_latency_ms"])
            if terminated:
                break

        # --- Dijkstra ---
        obs, _ = env.reset(seed=seed)
        dijkstra = DijkstraRouter()
        dijk_pdrs, dijk_lats = [], []
        for step in range(N_STEPS):
            node_table = dijkstra.build_nexthop_table(obs)
            slot_table = node_table_to_slots(node_table, obs["neighbor_table"])
            obs, _, terminated, _, info = env.step(slot_table)
            dijk_pdrs.append(info["pdr"])
            if "avg_latency_ms" in info:
                dijk_lats.append(info["avg_latency_ms"])
            if terminated:
                break

        gnn_results.append({
            "seed": seed,
            "mean_pdr": float(np.mean(gnn_pdrs)),
            "mean_latency": float(np.mean(gnn_lats)) if gnn_lats else None,
        })
        dijk_results.append({
            "seed": seed,
            "mean_pdr": float(np.mean(dijk_pdrs)),
            "mean_latency": float(np.mean(dijk_lats)) if dijk_lats else None,
        })

    gnn_pdr_avg = float(np.mean([r["mean_pdr"] for r in gnn_results]))
    dijk_pdr_avg = float(np.mean([r["mean_pdr"] for r in dijk_results]))
    gnn_pdr_std = float(np.std([r["mean_pdr"] for r in gnn_results]))
    dijk_pdr_std = float(np.std([r["mean_pdr"] for r in dijk_results]))

    return {
        "config": config_name,
        "label": config["label"],
        "altitude_km": config["altitude_km"],
        "inclination_deg": config["inclination_deg"],
        "gnn_pdr_mean": round(gnn_pdr_avg * 100, 2),
        "gnn_pdr_std": round(gnn_pdr_std * 100, 2),
        "dijkstra_pdr_mean": round(dijk_pdr_avg * 100, 2),
        "dijkstra_pdr_std": round(dijk_pdr_std * 100, 2),
        "gnn_vs_dijkstra_gap": round((gnn_pdr_avg - dijk_pdr_avg) * 100, 2),
        "gnn_seeds": gnn_results,
        "dijkstra_seeds": dijk_results,
    }


def main():
    print("Loading GNN model (trained on 550km/53deg)...")
    policy = load_model()
    print(f"Model loaded on {DEVICE}")

    results = {}
    for name, config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {config['label']}")
        print(f"{'='*60}")
        t0 = time.time()
        r = evaluate_config(name, config, policy)
        elapsed = time.time() - t0
        results[name] = r
        print(f"  GNN PDR:      {r['gnn_pdr_mean']:.2f}% (+/- {r['gnn_pdr_std']:.2f}%)")
        print(f"  Dijkstra PDR: {r['dijkstra_pdr_mean']:.2f}% (+/- {r['dijkstra_pdr_std']:.2f}%)")
        print(f"  Gap (GNN-Dij): {r['gnn_vs_dijkstra_gap']:+.2f}%")
        print(f"  Time: {elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*60}")
    print("CROSS-CONSTELLATION GENERALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<35s} {'GNN PDR':>10s} {'Dijkstra':>10s} {'Gap':>8s}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*8}")
    for name in ["training", "sso_97", "high_alt"]:
        r = results[name]
        print(f"{r['label']:<35s} {r['gnn_pdr_mean']:>9.2f}% {r['dijkstra_pdr_mean']:>9.2f}% {r['gnn_vs_dijkstra_gap']:>+7.2f}%")

    os.makedirs("output", exist_ok=True)
    with open("output/cross_constellation_gnn.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to output/cross_constellation_gnn.json")


if __name__ == "__main__":
    main()
