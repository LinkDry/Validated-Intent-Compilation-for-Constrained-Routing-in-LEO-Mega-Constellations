"""Baseline routing policies: Dijkstra shortest path and random."""

import numpy as np
import networkx as nx
from typing import Dict


class DijkstraRouter:
    """Shortest-path routing using Dijkstra on current topology.

    Builds a per-destination next-hop table [N, N] each epoch.
    Must be used with env.set_nexthop_table() for correct forwarding.
    """

    def build_nexthop_table(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Build per-destination next-hop table from current topology.

        Returns: table [N, N] where table[node][dst] = next-hop node id, or -1.
        """
        N = obs["node_features"].shape[0]
        num_edges = int(obs["num_edges"])
        edge_index = obs["edge_index"][:, :num_edges]
        edge_attr = obs["edge_attr"][:num_edges]

        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        for idx in range(num_edges):
            src, dst = int(edge_index[0, idx]), int(edge_index[1, idx])
            delay = float(edge_attr[idx, 0])
            G.add_edge(src, dst, weight=delay)

        table = np.full((N, N), -1, dtype=np.int64)

        try:
            paths = dict(nx.all_pairs_dijkstra_path(G, weight="weight"))
        except nx.NetworkXError:
            return table

        for src in range(N):
            if src not in paths:
                continue
            for dst in range(N):
                if dst == src or dst not in paths[src]:
                    continue
                path = paths[src][dst]
                if len(path) >= 2:
                    table[src, dst] = path[1]

        return table

    def select_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Fallback: return a dummy action (env uses nexthop table instead)."""
        N = obs["node_features"].shape[0]
        return np.zeros(N, dtype=np.int64)


class RandomRouter:
    """Random next-hop selection (lower bound baseline)."""

    def select_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        N = obs["node_features"].shape[0]
        action_mask = obs["action_mask"]
        action = np.zeros(N, dtype=np.int64)
        for i in range(N):
            valid = np.where(action_mask[i] > 0)[0]
            if len(valid) > 0:
                action[i] = np.random.choice(valid)
        return action
