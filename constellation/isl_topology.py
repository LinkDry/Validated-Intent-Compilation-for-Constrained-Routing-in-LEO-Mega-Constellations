"""ISL topology computation: snapshot graphs with polar dropout."""

import numpy as np
import torch
from typing import Tuple, Dict, List

from .walker import WalkerDeltaConstellation
from .link_budget import propagation_delay_ms, isl_capacity_gbps


class ISLTopologyManager:
    """Manages time-varying ISL topology for a Walker Delta constellation.

    Each satellite has up to 4 ISLs:
    - 2 intra-plane (permanent): forward/backward in same orbital plane
    - 2 inter-plane (conditional): left/right to adjacent planes
      Disabled when either endpoint is above polar_lat_threshold.
    """

    def __init__(
        self,
        constellation: WalkerDeltaConstellation,
        max_isl_range_km: float = 5016.0,
        polar_lat_threshold_deg: float = 75.0,
        frequency_ghz: float = 23.0,
    ):
        self.constellation = constellation
        self.max_isl_range_km = max_isl_range_km
        self.polar_lat_threshold = polar_lat_threshold_deg
        self.frequency_ghz = frequency_ghz
        self.max_degree = 4  # max neighbors per node

        # Precompute static intra-plane edges (never change)
        self._intra_src = []
        self._intra_dst = []
        N = constellation.num_sats
        S = constellation.sats_per_plane
        P = constellation.num_planes
        for p in range(P):
            for s in range(S):
                node = p * S + s
                fwd = p * S + (s + 1) % S
                # Only add one direction; we'll symmetrize later
                self._intra_src.append(node)
                self._intra_dst.append(fwd)
        self._intra_src = np.array(self._intra_src)
        self._intra_dst = np.array(self._intra_dst)

        # Precompute candidate inter-plane edges
        self._inter_src = []
        self._inter_dst = []
        for p in range(P):
            right_p = (p + 1) % P
            for s in range(S):
                node = p * S + s
                right_node = right_p * S + s
                self._inter_src.append(node)
                self._inter_dst.append(right_node)
        self._inter_src = np.array(self._inter_src)
        self._inter_dst = np.array(self._inter_dst)

    def compute_topology(
        self, t_seconds: float
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, np.ndarray]:
        """Compute ISL topology at time t.

        Returns:
            edge_index: [2, E] long tensor (bidirectional)
            edge_attr: [E, 4] float tensor
                [delay_ms, capacity_gbps, link_type (0=intra,1=inter), is_active]
            adjacency_list: dict mapping node_id -> list of (neighbor_id, edge_idx)
        """
        c = self.constellation
        positions = c.get_positions_eci(t_seconds)  # [N, 3]
        latlon = c.get_latlon(t_seconds)  # [N, 2]
        lats = np.abs(latlon[:, 0])  # absolute latitude

        # --- Intra-plane edges (always active, bidirectional) ---
        intra_dists = np.sqrt(
            ((positions[self._intra_src] - positions[self._intra_dst]) ** 2).sum(axis=1)
        )
        intra_delays = propagation_delay_ms(intra_dists)
        intra_caps = np.array([
            isl_capacity_gbps(d, self.frequency_ghz) for d in intra_dists
        ])
        n_intra = len(self._intra_src)

        # --- Inter-plane edges (conditional on distance + polar) ---
        inter_dists = np.sqrt(
            ((positions[self._inter_src] - positions[self._inter_dst]) ** 2).sum(axis=1)
        )
        # Active if: distance < max AND neither endpoint is polar
        src_polar = lats[self._inter_src] > self.polar_lat_threshold
        dst_polar = lats[self._inter_dst] > self.polar_lat_threshold
        inter_active = (
            (inter_dists < self.max_isl_range_km)
            & ~src_polar
            & ~dst_polar
        )

        active_inter_idx = np.where(inter_active)[0]
        inter_src_active = self._inter_src[active_inter_idx]
        inter_dst_active = self._inter_dst[active_inter_idx]
        inter_dists_active = inter_dists[active_inter_idx]
        inter_delays = propagation_delay_ms(inter_dists_active)
        inter_caps = np.array([
            isl_capacity_gbps(d, self.frequency_ghz) for d in inter_dists_active
        ])
        n_inter = len(active_inter_idx)

        # --- Combine and symmetrize ---
        # Forward edges
        all_src = np.concatenate([self._intra_src, inter_src_active])
        all_dst = np.concatenate([self._intra_dst, inter_dst_active])
        all_delays = np.concatenate([intra_delays, inter_delays])
        all_caps = np.concatenate([intra_caps, inter_caps])
        all_types = np.concatenate([
            np.zeros(n_intra),
            np.ones(n_inter),
        ])

        # Symmetrize (add reverse edges)
        src = np.concatenate([all_src, all_dst])
        dst = np.concatenate([all_dst, all_src])
        delays = np.tile(all_delays, 2)
        caps = np.tile(all_caps, 2)
        types = np.tile(all_types, 2)
        active = np.ones(len(src))

        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_attr = torch.tensor(
            np.stack([delays, caps, types, active], axis=1),
            dtype=torch.float32,
        )

        # Build adjacency list
        adjacency = self._build_adjacency_list(src, dst)

        return edge_index, edge_attr, adjacency

    def _build_adjacency_list(
        self, src: np.ndarray, dst: np.ndarray
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Build adjacency list: node -> [(neighbor, edge_idx), ...]."""
        adj: Dict[int, List[Tuple[int, int]]] = {
            i: [] for i in range(self.constellation.num_sats)
        }
        for edge_idx in range(len(src)):
            adj[int(src[edge_idx])].append((int(dst[edge_idx]), edge_idx))
        return adj

    def get_neighbor_mask(
        self, adjacency: Dict[int, List[Tuple[int, int]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build padded neighbor table and action mask.

        Returns:
            neighbor_table: [N, max_degree] — neighbor node IDs (padded with -1)
            action_mask: [N, max_degree] — 1.0 for valid, 0.0 for invalid
        """
        N = self.constellation.num_sats
        K = self.max_degree
        neighbor_table = np.full((N, K), -1, dtype=np.int64)
        action_mask = np.zeros((N, K), dtype=np.float32)

        for node_id, neighbors in adjacency.items():
            for idx, (neigh_id, _) in enumerate(neighbors[:K]):
                neighbor_table[node_id, idx] = neigh_id
                action_mask[node_id, idx] = 1.0

        return neighbor_table, action_mask
