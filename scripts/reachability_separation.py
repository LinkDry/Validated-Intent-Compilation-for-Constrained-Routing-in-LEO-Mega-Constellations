"""Reachability separation analysis for constrained routing scenarios.

Decomposes raw PDR into:
  1. Reachability ceiling: fraction of OD pairs that CAN reach each other
  2. Reachable PDR: delivered / sent_to_reachable (routing quality metric)

This separates topology-imposed limits from routing algorithm quality.
"""

import sys, os, json
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.routing_env import LEORoutingEnv


def build_adjacency(edge_index, num_edges, node_mask, edge_mask):
    """Build adjacency list from masked graph."""
    N = len(node_mask)
    adj = {i: set() for i in range(N) if node_mask[i] > 0}
    for idx in range(num_edges):
        if edge_mask[idx] == 0:
            continue
        u = int(edge_index[0, idx])
        v = int(edge_index[1, idx])
        if node_mask[u] > 0 and node_mask[v] > 0:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    return adj


def compute_reachability(adj, N, node_mask):
    """BFS from every active node. Returns reachable pair count and fraction."""
    active_nodes = [i for i in range(N) if node_mask[i] > 0]
    total_pairs = 0
    reachable_pairs = 0

    for src in active_nodes:
        visited = set()
        queue = deque([src])
        visited.add(src)
        while queue:
            node = queue.popleft()
            for nbr in adj.get(node, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        for dst in active_nodes:
            if dst == src:
                continue
            total_pairs += 1
            if dst in visited:
                reachable_pairs += 1

    return reachable_pairs, total_pairs


def apply_scenario_constraints(env, obs, scenario_name):
    """Apply scenario constraints and return masks."""
    N = env.N
    num_edges = int(obs["num_edges"])
    ei = obs["edge_index"][:, :num_edges]
    latlon = env.constellation.get_latlon(t_seconds=0.0)

    node_mask = np.ones(N, dtype=np.float32)
    edge_mask = np.ones(num_edges, dtype=np.float32)

    if scenario_name == "baseline":
        pass

    elif scenario_name == "node_failure":
        node_mask[42] = 0

    elif scenario_name == "plane_maintenance":
        for s in range(env.constellation.sats_per_plane):
            nid = 5 * env.constellation.sats_per_plane + s
            if nid < N:
                node_mask[nid] = 0

    elif scenario_name == "polar_avoidance":
        lat_thresh = 45.0
        for idx in range(num_edges):
            u, v = int(ei[0, idx]), int(ei[1, idx])
            if abs(latlon[u, 0]) > lat_thresh or abs(latlon[v, 0]) > lat_thresh:
                edge_mask[idx] = 0

    elif scenario_name == "compositional":
        node_mask[200] = 0
        lat_thresh = 45.0
        for idx in range(num_edges):
            u, v = int(ei[0, idx]), int(ei[1, idx])
            if abs(latlon[u, 0]) > lat_thresh or abs(latlon[v, 0]) > lat_thresh:
                edge_mask[idx] = 0

    # Propagate node mask to edges
    for idx in range(num_edges):
        u, v = int(ei[0, idx]), int(ei[1, idx])
        if node_mask[u] == 0 or node_mask[v] == 0:
            edge_mask[idx] = 0

    return node_mask, edge_mask


def main():
    with open("output/e2e_eval_results.json") as f:
        e2e = json.load(f)

    env = LEORoutingEnv(num_planes=20, sats_per_plane=20, seed=42)
    obs, _ = env.reset(seed=42)
    N = env.N

    scenarios = ["baseline", "node_failure", "plane_maintenance",
                 "polar_avoidance", "compositional"]

    results = {}

    for scenario in scenarios:
        print("")
        print("=" * 60)
        print("Scenario: %s" % scenario)
        print("=" * 60)

        node_mask, edge_mask = apply_scenario_constraints(env, obs, scenario)

        active_nodes = int(node_mask.sum())
        active_edges = int(edge_mask.sum())
        num_edges = int(obs["num_edges"])

        print("Active nodes: %d/%d" % (active_nodes, N))
        print("Active edges: %d/%d" % (active_edges, num_edges))

        adj = build_adjacency(obs["edge_index"], num_edges, node_mask, edge_mask)
        reachable, total = compute_reachability(adj, N, node_mask)

        reachability_frac = reachable / max(total, 1)
        print("Reachable OD pairs: %d/%d (%.4f)" % (reachable, total, reachability_frac))

        scenario_data = e2e.get(scenario, {})
        methods = scenario_data.get("methods", {})

        result = {
            "active_nodes": active_nodes,
            "active_edges": active_edges,
            "total_edges": num_edges,
            "reachable_pairs": reachable,
            "total_pairs": total,
            "reachability_fraction": round(reachability_frac, 6),
            "methods": {}
        }

        for method_name in ["unconstrained_gnn", "constrained_gnn", "constrained_dijkstra"]:
            mdata = methods.get(method_name, [])
            if not mdata:
                continue
            avg_pdr = np.mean([r["pdr"] for r in mdata])
            reachable_pdr = avg_pdr / reachability_frac if reachability_frac > 0 else 0

            result["methods"][method_name] = {
                "raw_pdr": round(float(avg_pdr), 6),
                "reachable_pdr": round(min(float(reachable_pdr), 1.0), 6),
            }
            print("  %s:" % method_name)
            print("    Raw PDR:       %.4f" % avg_pdr)
            print("    Reachable PDR: %.4f" % min(reachable_pdr, 1.0))

        gnn_data = result["methods"].get("constrained_gnn", {})
        dijk_data = result["methods"].get("constrained_dijkstra", {})
        if gnn_data and dijk_data:
            raw_gap = dijk_data["raw_pdr"] - gnn_data["raw_pdr"]
            reachable_gap = dijk_data["reachable_pdr"] - gnn_data["reachable_pdr"]
            unreachable_drop = 1.0 - reachability_frac

            gnn_raw = gnn_data["raw_pdr"]
            reachability_explains = 0.0
            if gnn_raw < 1.0:
                reachability_explains = unreachable_drop / (1.0 - gnn_raw) * 100

            result["gap_decomposition"] = {
                "raw_pdr_gap": round(raw_gap, 6),
                "reachable_pdr_gap": round(reachable_gap, 6),
                "unreachable_fraction": round(unreachable_drop, 6),
                "reachability_explains_pct": round(reachability_explains, 1),
            }
            print("")
            print("  Gap decomposition:")
            print("    Raw PDR gap (Dijk - GNN):       %.4f" % raw_gap)
            print("    Reachable PDR gap (Dijk - GNN): %.4f" % reachable_gap)
            print("    Unreachable fraction:           %.4f" % unreachable_drop)

        results[scenario] = result

    out_path = "output/reachability_separation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("")
    print("Results saved to " + out_path)


if __name__ == "__main__":
    main()
