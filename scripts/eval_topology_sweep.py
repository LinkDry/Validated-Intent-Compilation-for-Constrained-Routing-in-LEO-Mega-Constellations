"""Topology severity sweep: PDR vs fraction of topology removed.

Progressively disables more planes and measures routing performance.
Shows the crossover point where Dijkstra starts beating GNN.

Usage:
  python scripts/eval_topology_sweep.py
"""

import sys, os, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.routing_env import LEORoutingEnv
from policy.routing_policy import GNNRoutingPolicy
from baselines.shortest_path import DijkstraRouter
from intent.compiler import IntentCompiler
from intent.verifier import ConstraintVerifier
from intent.constrained_router import ConstrainedRouter
from intent.schema import ConstraintProgram
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEPS = 50
SEEDS = [42, 123, 456]
MODEL_PATH = "output/pretrain_ctg3/best.pt"


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


def build_verifier(env):
    obs, _ = env.reset(seed=0)
    ne = int(obs["num_edges"])
    latlon = env.constellation.get_latlon(t_seconds=0.0).astype(np.float32)
    return ConstraintVerifier(
        num_planes=env.constellation.num_planes,
        sats_per_plane=env.constellation.sats_per_plane,
        edge_index=obs["edge_index"][:, :ne],
        edge_delays=obs["edge_attr"][:ne, 0],
        neighbor_table=obs["neighbor_table"],
        latlon=latlon,
    )


def node_table_to_slots(node_table, neighbor_table):
    N, K = neighbor_table.shape
    slot_table = np.zeros((N, N), dtype=np.int64)
    for node in range(N):
        for k in range(K):
            nid = neighbor_table[node, k]
            if nid >= 0:
                mask = node_table[node] == nid
                slot_table[node, mask] = k
    return slot_table


# Sweep levels: disable 1, 2, 3, 5, 7, 10, 13, 15, 17 planes
SWEEP_LEVELS = [1, 2, 3, 5, 7, 10, 13, 15, 17]


def make_disable_intent(planes_to_disable):
    """Create intent text for disabling multiple planes.
    Uses range notation for >5 planes to keep intent short."""
    if len(planes_to_disable) <= 5:
        plane_str = ", ".join(str(p) for p in planes_to_disable)
        return f"Disable planes {plane_str} for maintenance"
    # Group consecutive planes into ranges
    ranges = []
    start = planes_to_disable[0]
    end = start
    for p in planes_to_disable[1:]:
        if p == end + 1:
            end = p
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = p
            end = p
    ranges.append(f"{start}-{end}" if start != end else str(start))
    range_str = ", ".join(ranges)
    return f"Disable planes {range_str} for maintenance"


def run_sweep_level(n_planes_disabled, env, policy, verifier, compiler, seed):
    """Run one sweep level: disable n_planes_disabled planes, measure PDR."""
    np.random.seed(seed + n_planes_disabled * 1000)
    planes = sorted(np.random.choice(20, n_planes_disabled, replace=False).tolist())
    disabled_nodes = set()
    for p in planes:
        for s in range(20):
            disabled_nodes.add(p * 20 + s)

    fraction_removed = n_planes_disabled / 20.0

    # Compile intent
    intent = make_disable_intent(planes)
    cr = compiler.compile(intent)
    if not cr.success:
        return {
            "n_planes": n_planes_disabled,
            "fraction_removed": fraction_removed,
            "planes": planes,
            "compile_success": False,
            "compile_errors": cr.errors,
        }

    program = cr.program

    # Run constrained GNN
    obs, _ = env.reset(seed=seed)
    router = ConstrainedRouter(policy=policy, verifier=verifier, device=DEVICE)
    gnn_pdrs = []
    for step in range(N_STEPS):
        result = router.route(obs, program)
        obs, _, terminated, _, info = env.step(result.slot_table)
        gnn_pdrs.append(info["pdr"])
        if terminated:
            break
    gnn_pdr = float(np.mean(gnn_pdrs))

    # Run constrained Dijkstra
    obs, _ = env.reset(seed=seed)
    dijkstra = DijkstraRouter()
    dijk_router = ConstrainedRouter(policy=None, verifier=verifier, device="cpu")
    dijk_pdrs = []
    for step in range(N_STEPS):
        node_table = dijkstra.build_nexthop_table(obs)
        slot_table = node_table_to_slots(node_table, obs["neighbor_table"])

        # Apply constraint masks
        edge_mask, node_mask, _, _ = dijk_router._ground_constraints(obs, program)
        N = slot_table.shape[0]
        nt = obs["neighbor_table"]
        ne = int(obs["num_edges"])
        ei = obs["edge_index"][:, :ne]

        disabled_edges = set()
        for idx in range(ne):
            if edge_mask[idx] == 0:
                disabled_edges.add((int(ei[0, idx]), int(ei[1, idx])))

        for src in range(N):
            if node_mask[src] == 0:
                slot_table[src, :] = 0
                continue
            for dst in range(N):
                if src == dst:
                    continue
                if node_mask[dst] == 0:
                    slot_table[src, dst] = 0
                    continue
                cur_slot = slot_table[src, dst]
                cur_nbr = int(nt[src, cur_slot])
                if node_mask[cur_nbr] == 0 or (src, cur_nbr) in disabled_edges:
                    K = nt.shape[1]
                    for k in range(K):
                        if obs["action_mask"][src, k] == 0:
                            continue
                        nbr = int(nt[src, k])
                        if nbr < 0 or node_mask[nbr] == 0:
                            continue
                        if (src, nbr) in disabled_edges:
                            continue
                        slot_table[src, dst] = k
                        break

        obs, _, terminated, _, info = env.step(slot_table)
        dijk_pdrs.append(info["pdr"])
        if terminated:
            break
    dijk_pdr = float(np.mean(dijk_pdrs))

    # Run unconstrained GNN (to show violation rate)
    obs, _ = env.reset(seed=seed)
    unc_pdrs = []
    for step in range(N_STEPS):
        data, nt_t, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, DEVICE)
        with torch.no_grad():
            st, _ = policy.get_routing_table(data, nt_t, nd, mask)
        obs, _, terminated, _, info = env.step(st.cpu().numpy())
        unc_pdrs.append(info["pdr"])
        if terminated:
            break
    unc_pdr = float(np.mean(unc_pdrs))

    return {
        "n_planes": n_planes_disabled,
        "fraction_removed": fraction_removed,
        "planes": planes,
        "compile_success": True,
        "compile_ms": cr.latency_ms,
        "compile_attempts": cr.attempts,
        "gnn_constrained_pdr": gnn_pdr,
        "dijkstra_constrained_pdr": dijk_pdr,
        "gnn_unconstrained_pdr": unc_pdr,
        "gnn_advantage": gnn_pdr - dijk_pdr,
    }


def main():
    print("Loading GNN model...")
    policy = load_model()
    print(f"Model loaded on {DEVICE}")

    print("Creating environment...")
    env = LEORoutingEnv(num_planes=20, sats_per_plane=20, scenario="uniform")

    print("Building verifier...")
    verifier = build_verifier(env)

    print("Creating compiler...")
    compiler = IntentCompiler(verifier=verifier, max_retries=3, timeout=300)

    # Resume support
    out_path = Path("output/topology_sweep.json")
    done_keys = set()
    all_results = []
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        # Only keep successful results; retry failed compilations
        all_results = [r for r in prev.get("results", []) if r.get("compile_success", False)]
        done_keys = {(r["n_planes"], r["seed"]) for r in all_results}
        print(f"Resuming: {len(done_keys)} successful, retrying failed")

    total_runs = len(SWEEP_LEVELS) * len(SEEDS)
    print(f"\nRunning topology severity sweep: {len(SWEEP_LEVELS)} levels x {len(SEEDS)} seeds = {total_runs} runs")

    for n_planes in SWEEP_LEVELS:
        print(f"\n{'='*60}")
        print(f"Disabling {n_planes}/20 planes ({n_planes/20*100:.0f}% topology removed)")
        print(f"{'='*60}")

        for seed in SEEDS:
            if (n_planes, seed) in done_keys:
                print(f"  Seed {seed}: already done, skipping")
                continue

            print(f"  Seed {seed}:", end=" ", flush=True)
            t0 = time.time()
            result = run_sweep_level(n_planes, env, policy, verifier, compiler, seed)
            result["seed"] = seed
            elapsed = time.time() - t0

            if result["compile_success"]:
                print(f"GNN={result['gnn_constrained_pdr']:.4f} "
                      f"Dijk={result['dijkstra_constrained_pdr']:.4f} "
                      f"Unc={result['gnn_unconstrained_pdr']:.4f} "
                      f"Δ={result['gnn_advantage']:+.4f} ({elapsed:.1f}s)")
            else:
                print(f"COMPILE FAILED: {result['compile_errors'][:2]} ({elapsed:.1f}s)")

            all_results.append(result)

            # Save after each run
            _save(all_results, out_path)

    _save(all_results, out_path)
    _print_summary(all_results)


def _save(results, out_path):
    # Aggregate by n_planes
    by_level = {}
    for r in results:
        n = r["n_planes"]
        if n not in by_level:
            by_level[n] = []
        by_level[n].append(r)

    summary = []
    for n in sorted(by_level.keys()):
        runs = [r for r in by_level[n] if r["compile_success"]]
        if not runs:
            continue
        summary.append({
            "n_planes_disabled": n,
            "fraction_removed": n / 20.0,
            "n_seeds": len(runs),
            "gnn_pdr_mean": float(np.mean([r["gnn_constrained_pdr"] for r in runs])),
            "gnn_pdr_std": float(np.std([r["gnn_constrained_pdr"] for r in runs])),
            "dijk_pdr_mean": float(np.mean([r["dijkstra_constrained_pdr"] for r in runs])),
            "dijk_pdr_std": float(np.std([r["dijkstra_constrained_pdr"] for r in runs])),
            "unc_pdr_mean": float(np.mean([r["gnn_unconstrained_pdr"] for r in runs])),
            "gnn_advantage_mean": float(np.mean([r["gnn_advantage"] for r in runs])),
        })

    final = {
        "sweep_levels": SWEEP_LEVELS,
        "seeds": SEEDS,
        "n_steps": N_STEPS,
        "summary": summary,
        "results": results,
    }

    Path("output").mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)


def _print_summary(results):
    by_level = {}
    for r in results:
        n = r["n_planes"]
        if n not in by_level:
            by_level[n] = []
        by_level[n].append(r)

    print(f"\n{'='*70}")
    print("TOPOLOGY SEVERITY SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Planes off':>10s} {'Removed':>8s} {'GNN PDR':>10s} {'Dijk PDR':>10s} {'Δ(GNN-Dijk)':>12s} {'Unc PDR':>10s}")
    print("-" * 65)

    for n in sorted(by_level.keys()):
        runs = [r for r in by_level[n] if r["compile_success"]]
        if not runs:
            print(f"{n:>10d} {n/20*100:>7.0f}%  COMPILE FAILED")
            continue
        gnn = np.mean([r["gnn_constrained_pdr"] for r in runs])
        dijk = np.mean([r["dijkstra_constrained_pdr"] for r in runs])
        unc = np.mean([r["gnn_unconstrained_pdr"] for r in runs])
        delta = gnn - dijk
        marker = " <-- crossover" if delta < 0 and n > 1 else ""
        print(f"{n:>10d} {n/20*100:>7.0f}% {gnn:>9.4f} {dijk:>9.4f} {delta:>+11.4f} {unc:>9.4f}{marker}")


if __name__ == "__main__":
    main()
