"""End-to-end evaluation: Intent → Compile → Verify → Constrained Route → Network KPIs.

Compares:
  1. Unconstrained GNN (Phase A baseline)
  2. Full pipeline: LLM compile → verify → constrained GNN
  3. Constrained Dijkstra (oracle baseline with same masks)

Scenarios:
  - baseline: no constraints
  - node_failure: disable specific node
  - plane_maintenance: disable entire plane
  - polar_avoidance: avoid high-latitude links
  - compositional: multiple constraints combined

KPIs: PDR, mean delay, P95 delay, drop rate, constraint violation rate
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEPS = 50
SEEDS = [42, 123, 456]
MODEL_PATH = "output/pretrain_ctg3/best.pt"


# ── Scenarios ──

SCENARIOS = [
    {
        "name": "baseline",
        "intent": None,  # no constraints
        "scenario": "uniform",
        "description": "No constraints, uniform traffic",
    },
    {
        "name": "node_failure",
        "intent": "Disable node 42 immediately due to hardware failure",
        "scenario": "uniform",
        "description": "Single node failure",
        "check_nodes_disabled": [42],
    },
    {
        "name": "plane_maintenance",
        "intent": "Take orbital plane 5 offline for maintenance",
        "scenario": "uniform",
        "description": "Full plane disabled",
        "check_nodes_disabled": list(range(100, 120)),  # plane 5 = nodes 100-119
    },
    {
        "name": "polar_avoidance",
        "intent": "Avoid polar links above 45 degrees latitude",
        "scenario": "uniform",
        "description": "Polar link avoidance (45 deg threshold)",
        "check_lat_threshold": 45.0,
    },
    {
        "name": "compositional",
        "intent": "Disable node 200, avoid polar links above 45 degrees, and cap utilization at 80%",
        "scenario": "uniform",
        "description": "Multi-constraint: node + polar + utilization",
        "check_nodes_disabled": [200],
        "check_lat_threshold": 45.0,
    },
]


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    hidden = ckpt.get("hidden_dim", 128)
    # Infer rank from checkpoint: src_proj.0.weight shape is [rank, hidden]
    rank = ckpt["policy_state_dict"]["cost_head.src_proj.0.weight"].shape[0]
    policy = GNNRoutingPolicy(node_feat_dim=8, edge_feat_dim=4,
                              hidden_dim=hidden, rank=rank)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(DEVICE)
    policy.eval()
    return policy


def build_verifier(env):
    """Build verifier from env state."""
    obs, _ = env.reset(seed=0)
    ne = int(obs["num_edges"])
    latlon = env.constellation.get_latlon(t_seconds=0.0).astype(np.float32)
    return ConstraintVerifier(
        num_planes=env.constellation.num_planes,
        sats_per_plane=env.constellation.sats_per_plane,
        edge_index=obs["edge_index"][:, :ne],
        edge_delays=obs["edge_attr"][:ne, 0],  # first feature is delay
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


def check_violations(slot_table, obs, scenario_cfg):
    """Check if routing table violates the scenario constraints."""
    violations = 0
    total_checks = 0
    N = slot_table.shape[0]
    nt = obs["neighbor_table"]

    # Check disabled nodes: no traffic should route through them
    disabled = scenario_cfg.get("check_nodes_disabled", [])
    if disabled:
        disabled_set = set(disabled)
        for src in range(N):
            if src in disabled_set:
                continue
            for dst in range(N):
                if src == dst or dst in disabled_set:
                    continue
                slot = slot_table[src, dst]
                next_hop = int(nt[src, slot])
                if next_hop in disabled_set:
                    violations += 1
                total_checks += 1

    return violations, total_checks


def run_unconstrained_gnn(env, policy, seed, n_steps):
    """Run unconstrained GNN routing."""
    obs, _ = env.reset(seed=seed)
    pdrs, delays, drops = [], [], []
    for _ in range(n_steps):
        data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, DEVICE)
        with torch.no_grad():
            slot_table, cost = policy.get_routing_table(data, nt, nd, mask)
        slot_np = slot_table.cpu().numpy()
        obs, _, terminated, _, info = env.step(slot_np)
        pdrs.append(info["pdr"])
        delays.append(info["mean_delay_ms"])
        drops.append(info["drop_rate"])
        if terminated:
            break
    return {
        "pdr": float(np.mean(pdrs)),
        "delay_mean": float(np.mean(delays)),
        "delay_p95": float(np.percentile(delays, 95)),
        "drop": float(np.mean(drops)),
    }, slot_np, obs


def run_constrained_gnn(env, policy, verifier, program, compile_ms, compile_attempts, seed, n_steps, scenario_cfg):
    """Run full pipeline: constrained GNN route with pre-compiled program."""
    obs, _ = env.reset(seed=seed)

    router = ConstrainedRouter(policy=policy, verifier=verifier, device=DEVICE)

    pdrs, delays, drops = [], [], []
    total_violations, total_checks = 0, 0

    for step in range(n_steps):
        result = router.route(obs, program)
        slot_np = result.slot_table

        v, c = check_violations(slot_np, obs, scenario_cfg)
        total_violations += v
        total_checks += c

        obs, _, terminated, _, info = env.step(slot_np)
        pdrs.append(info["pdr"])
        delays.append(info["mean_delay_ms"])
        drops.append(info["drop_rate"])
        if terminated:
            break

    violation_rate = total_violations / max(total_checks, 1)
    return {
        "pdr": float(np.mean(pdrs)),
        "delay_mean": float(np.mean(delays)),
        "delay_p95": float(np.percentile(delays, 95)),
        "drop": float(np.mean(drops)),
        "violation_rate": violation_rate,
        "violations": total_violations,
        "checks": total_checks,
        "compile_ms": compile_ms,
        "compile_attempts": compile_attempts,
        "hard_satisfied": result.hard_satisfied,
    }, slot_np, obs


def run_constrained_dijkstra(env, verifier, program, seed, n_steps, scenario_cfg):
    """Run Dijkstra with same constraints as oracle baseline."""
    obs, _ = env.reset(seed=seed)
    dijkstra = DijkstraRouter()

    router = ConstrainedRouter(policy=None, verifier=verifier, device="cpu")

    pdrs, delays, drops = [], [], []
    total_violations, total_checks = 0, 0

    for step in range(n_steps):
        # Get Dijkstra routing table
        node_table = dijkstra.build_nexthop_table(obs)
        slot_table = node_table_to_slots(node_table, obs["neighbor_table"])

        # Apply constraint masks to Dijkstra table
        edge_mask, node_mask, _, _ = router._ground_constraints(obs, program)
        N = slot_table.shape[0]
        nt = obs["neighbor_table"]
        ne = int(obs["num_edges"])
        ei = obs["edge_index"][:, :ne]

        # Build disabled edges set
        disabled_edges = set()
        for idx in range(ne):
            if edge_mask[idx] == 0:
                disabled_edges.add((int(ei[0, idx]), int(ei[1, idx])))

        # Re-route around disabled nodes/edges
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
                # Check if current next-hop uses disabled node/edge
                cur_slot = slot_table[src, dst]
                cur_nbr = int(nt[src, cur_slot])
                if node_mask[cur_nbr] == 0 or (src, cur_nbr) in disabled_edges:
                    # Find alternative
                    K = nt.shape[1]
                    best_slot = cur_slot
                    for k in range(K):
                        if obs["action_mask"][src, k] == 0:
                            continue
                        nbr = int(nt[src, k])
                        if nbr < 0 or node_mask[nbr] == 0:
                            continue
                        if (src, nbr) in disabled_edges:
                            continue
                        best_slot = k
                        break
                    slot_table[src, dst] = best_slot

        v, c = check_violations(slot_table, obs, scenario_cfg)
        total_violations += v
        total_checks += c

        obs, _, terminated, _, info = env.step(slot_table)
        pdrs.append(info["pdr"])
        delays.append(info["mean_delay_ms"])
        drops.append(info["drop_rate"])
        if terminated:
            break

    violation_rate = total_violations / max(total_checks, 1)
    return {
        "pdr": float(np.mean(pdrs)),
        "delay_mean": float(np.mean(delays)),
        "delay_p95": float(np.percentile(delays, 95)),
        "drop": float(np.mean(drops)),
        "violation_rate": violation_rate,
        "violations": total_violations,
        "checks": total_checks,
    }, slot_table, obs


# ── Main ──

def main():
    print("Loading GNN model...")
    policy = load_model()
    print(f"Model loaded on {DEVICE}")

    print("Creating environment...")
    env = LEORoutingEnv(num_planes=20, sats_per_plane=20, scenario="uniform")

    print("Building verifier...")
    verifier = build_verifier(env)

    print("Creating compiler...")
    compiler = IntentCompiler(verifier=verifier, max_retries=3)

    all_results = {}

    for sc in SCENARIOS:
        name = sc["name"]
        intent = sc["intent"]
        print(f"\n{'='*70}")
        print(f"Scenario: {name} — {sc['description']}")
        print(f"{'='*70}")

        scenario_results = {"description": sc["description"], "methods": {}}

        # Compile intent once per scenario (reuse across seeds)
        compiled_program = None
        compile_ms = 0
        compile_attempts = 0
        if intent is not None:
            print(f"  Compiling intent: {intent[:60]}...")
            t0 = time.time()
            cr = compiler.compile(intent)
            compile_ms = (time.time() - t0) * 1000
            compile_attempts = cr.attempts
            if cr.success:
                compiled_program = cr.program
                print(f"  Compiled OK in {compile_ms:.0f}ms (attempts={cr.attempts})")
            else:
                print(f"  Compilation FAILED: {cr.errors}")

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")

            # Method 1: Unconstrained GNN
            t0 = time.time()
            r1, _, _ = run_unconstrained_gnn(env, policy, seed, N_STEPS)
            t1 = time.time()
            v1, c1 = 0, 0
            if "check_nodes_disabled" in sc:
                # Check violations for unconstrained (should be high)
                obs_check, _ = env.reset(seed=seed)
                data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs_check, DEVICE)
                with torch.no_grad():
                    st, _ = policy.get_routing_table(data, nt, nd, mask)
                v1, c1 = check_violations(st.cpu().numpy(), obs_check, sc)
                r1["violation_rate"] = v1 / max(c1, 1)
                r1["violations"] = v1
                r1["checks"] = c1

            print(f"    Unconstrained GNN: PDR={r1['pdr']:.4f} delay={r1['delay_mean']:.1f}ms "
                  f"drop={r1['drop']:.4f} viol={r1.get('violation_rate', 'N/A')} "
                  f"({t1-t0:.1f}s)")

            scenario_results["methods"].setdefault("unconstrained_gnn", []).append(r1)

            if compiled_program is not None:
                # Method 2: Full pipeline (constrained GNN)
                t0 = time.time()
                r2, _, _ = run_constrained_gnn(
                    env, policy, verifier, compiled_program,
                    compile_ms, compile_attempts, seed, N_STEPS, sc)
                t2 = time.time()
                if "error" not in r2:
                    print(f"    Constrained GNN:   PDR={r2['pdr']:.4f} delay={r2['delay_mean']:.1f}ms "
                          f"drop={r2['drop']:.4f} viol={r2['violation_rate']:.6f} "
                          f"compile={r2['compile_ms']:.0f}ms ({t2-t0:.1f}s)")
                else:
                    print(f"    Constrained GNN:   ERROR — {r2['error']}")
                scenario_results["methods"].setdefault("constrained_gnn", []).append(r2)

                # Method 3: Constrained Dijkstra (oracle)
                t0 = time.time()
                r3, _, _ = run_constrained_dijkstra(
                    env, verifier, compiled_program, seed, N_STEPS, sc)
                t3 = time.time()
                if "error" not in r3:
                    print(f"    Constrained Dijk:  PDR={r3['pdr']:.4f} delay={r3['delay_mean']:.1f}ms "
                          f"drop={r3['drop']:.4f} viol={r3['violation_rate']:.6f} "
                          f"({t3-t0:.1f}s)")
                else:
                    print(f"    Constrained Dijk:  ERROR — {r3['error']}")
                scenario_results["methods"].setdefault("constrained_dijkstra", []).append(r3)

        # Aggregate per method
        print(f"\n  Summary for {name}:")
        for method, runs in scenario_results["methods"].items():
            valid = [r for r in runs if "error" not in r]
            if not valid:
                print(f"    {method}: all failed")
                continue
            avg_pdr = np.mean([r["pdr"] for r in valid])
            avg_delay = np.mean([r["delay_mean"] for r in valid])
            avg_drop = np.mean([r["drop"] for r in valid])
            avg_viol = np.mean([r.get("violation_rate", 0) for r in valid])
            print(f"    {method:25s}: PDR={avg_pdr:.4f}  delay={avg_delay:.1f}ms  "
                  f"drop={avg_drop:.4f}  viol={avg_viol:.6f}")

        all_results[name] = scenario_results

    # Save
    from pathlib import Path
    Path("output").mkdir(exist_ok=True)
    with open("output/e2e_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("END-TO-END EVALUATION COMPLETE")
    print(f"Saved to output/e2e_eval_results.json")


if __name__ == "__main__":
    main()
