"""Constrained routing layer: applies verified constraints to GNN routing.

Takes a verified ConstraintProgram and modifies the GNN's cost-to-go
predictions to produce routing tables that satisfy hard constraints.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

from intent.schema import (
    ConstraintProgram, HardConstraintType, SoftConstraintType,
    KNOWN_REGIONS,
)
from intent.verifier import ConstraintVerifier


@dataclass
class ConstrainedRoutingResult:
    """Output of constrained routing."""
    slot_table: np.ndarray          # [N, N] next-hop slot table
    cost_matrix: np.ndarray         # [N, N] predicted cost-to-go
    edge_mask: np.ndarray           # [E] binary mask (1=active, 0=disabled)
    node_mask: np.ndarray           # [N] binary mask (1=active, 0=disabled)
    utilization_caps: np.ndarray    # [E] per-edge utilization cap
    flow_deadlines: Dict[str, float] = field(default_factory=dict)
    hard_satisfied: bool = True
    violations: List[str] = field(default_factory=list)


class ConstrainedRouter:
    """Applies ConstraintProgram to GNN routing decisions.

    This is the optimization layer that connects the semantic compiler
    output to the GNN's fast cost-to-go predictions.
    """

    def __init__(self, policy, verifier: ConstraintVerifier, device="cuda"):
        """
        Args:
            policy: trained GNNRoutingPolicy with get_routing_table()
            verifier: ConstraintVerifier with constellation metadata
            device: torch device
        """
        self.policy = policy
        self.verifier = verifier
        self.device = device
        self.N = verifier.N

    def route(self, obs: dict, program: ConstraintProgram,
              ) -> ConstrainedRoutingResult:
        """Produce constrained routing table from observation + program.

        Steps:
        1. Ground constraints to edge/node masks
        2. Run GNN to get cost-to-go predictions
        3. Apply masks to cost matrix
        4. Derive next-hop table under constraints
        5. Validate hard constraints
        """
        from policy.routing_policy import GNNRoutingPolicy

        # Step 1: Ground constraints
        edge_mask, node_mask, util_caps, flow_deadlines = (
            self._ground_constraints(obs, program)
        )

        # Step 2: GNN forward pass
        data, nt, nd, mask = GNNRoutingPolicy.obs_to_pyg(obs, self.device)
        with torch.no_grad():
            cost_pred = self.policy(data)  # [N, N]

        cost_np = cost_pred.cpu().numpy()

        # Step 3: Apply node mask — disabled nodes get infinite cost
        for n in range(self.N):
            if node_mask[n] == 0:
                cost_np[n, :] = 1e9  # from disabled node
                cost_np[:, n] = 1e9  # to disabled node

        # Step 4: Derive next-hop under edge mask
        nt_np = obs["neighbor_table"]  # [N, K]
        nd_np = obs["neighbor_delays"]  # [N, K]
        nm_np = obs.get("neighbor_mask", obs.get("action_mask"))  # [N, K]
        num_edges = int(obs["num_edges"])
        ei = obs["edge_index"][:, :num_edges]

        # Build edge-level mask lookup
        edge_disabled = set()
        for idx in range(num_edges):
            if edge_mask[idx] == 0:
                u, v = int(ei[0, idx]), int(ei[1, idx])
                edge_disabled.add((u, v))

        # Derive slot table with masked edges
        slot_table = np.zeros((self.N, self.N), dtype=np.int64)
        K = nt_np.shape[1]

        for src in range(self.N):
            if node_mask[src] == 0:
                continue
            for dst in range(self.N):
                if src == dst or node_mask[dst] == 0:
                    continue

                best_slot = 0
                best_cost = 1e9

                for k in range(K):
                    if nm_np[src, k] == 0:
                        continue
                    nbr = int(nt_np[src, k])
                    if nbr < 0 or node_mask[nbr] == 0:
                        continue
                    if (src, nbr) in edge_disabled:
                        continue

                    candidate_cost = nd_np[src, k] + cost_np[nbr, dst]
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_slot = k

                slot_table[src, dst] = best_slot

        # Step 5: Validate hard constraints
        violations = []
        hard_ok = True

        # Check latency deadlines (approximate via cost-to-go)
        for key, deadline in flow_deadlines.items():
            parts = key.split(":")
            if len(parts) == 2:
                idx = int(parts[1])
                if idx < len(program.flow_selectors):
                    fs = program.flow_selectors[idx]
                    src_nodes = self._resolve_flow_sources(fs, obs)
                    dst_nodes = self._resolve_flow_dests(fs, obs)
                    for s in src_nodes:
                        for d in dst_nodes:
                            if cost_np[s, d] > deadline:
                                violations.append(
                                    f"Latency {cost_np[s,d]:.1f}ms > "
                                    f"deadline {deadline}ms for "
                                    f"({s}->{d})")
                                hard_ok = False

        result = ConstrainedRoutingResult(
            slot_table=slot_table,
            cost_matrix=cost_np,
            edge_mask=edge_mask,
            node_mask=node_mask,
            utilization_caps=util_caps,
            flow_deadlines=flow_deadlines,
            hard_satisfied=hard_ok,
            violations=violations,
        )
        return result

    def _ground_constraints(self, obs: dict, program: ConstraintProgram,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Convert ConstraintProgram to numeric masks."""
        num_edges = int(obs["num_edges"])
        ei = obs["edge_index"][:, :num_edges]

        edge_mask = np.ones(num_edges, dtype=np.float32)
        node_mask = np.ones(self.N, dtype=np.float32)
        util_caps = np.ones(num_edges, dtype=np.float32)
        flow_deadlines = {}

        for hc in program.hard_constraints:
            # Skip conditional constraints if event not active
            if hc.condition and not hc.condition.active:
                continue

            if hc.type == HardConstraintType.DISABLE_NODE.value:
                if hc.target.startswith("node:"):
                    nid = int(hc.target.split(":")[1])
                    if 0 <= nid < self.N:
                        node_mask[nid] = 0

            elif hc.type == HardConstraintType.DISABLE_PLANE.value:
                if hc.target.startswith("plane:"):
                    pid = int(hc.target.split(":")[1])
                    for s in range(self.verifier.spp):
                        nid = pid * self.verifier.spp + s
                        if nid < self.N:
                            node_mask[nid] = 0

            elif hc.type == HardConstraintType.REROUTE_AWAY.value:
                if hc.target.startswith("node:"):
                    nid = int(hc.target.split(":")[1])
                    if 0 <= nid < self.N:
                        node_mask[nid] = 0

            elif hc.type == HardConstraintType.AVOID_LATITUDE.value:
                lat_thresh = float(hc.value) if hc.value else 75.0
                latlon = self.verifier.latlon
                for idx in range(num_edges):
                    u, v = int(ei[0, idx]), int(ei[1, idx])
                    if (abs(latlon[u, 0]) > lat_thresh or
                            abs(latlon[v, 0]) > lat_thresh):
                        edge_mask[idx] = 0

            elif hc.type == HardConstraintType.AVOID_REGION.value:
                region = str(hc.value) if hc.value else hc.target
                nodes = self.verifier.ground_region_to_nodes(region)
                for idx in range(num_edges):
                    u, v = int(ei[0, idx]), int(ei[1, idx])
                    if u in nodes or v in nodes:
                        edge_mask[idx] = 0

            elif hc.type == HardConstraintType.MAX_LATENCY_MS.value:
                flow_deadlines[hc.target] = float(hc.value)

        # Soft constraints: utilization caps
        for sc in program.soft_constraints:
            if sc.condition and not sc.condition.active:
                continue
            if sc.type == SoftConstraintType.MAX_UTILIZATION.value:
                cap = float(sc.value) if sc.value else 1.0
                util_caps[:] = min(util_caps.min(), cap)

        # Propagate node mask to edges
        for idx in range(num_edges):
            u, v = int(ei[0, idx]), int(ei[1, idx])
            if node_mask[u] == 0 or node_mask[v] == 0:
                edge_mask[idx] = 0

        return edge_mask, node_mask, util_caps, flow_deadlines

    def _resolve_flow_sources(self, fs, obs) -> List[int]:
        """Resolve flow selector to source node IDs."""
        if fs.src_node is not None:
            return [fs.src_node]
        if fs.src_region:
            return list(self.verifier.ground_region_to_nodes(fs.src_region))
        if fs.src_plane is not None:
            return list(range(
                fs.src_plane * self.verifier.spp,
                (fs.src_plane + 1) * self.verifier.spp))
        return list(range(self.N))

    def _resolve_flow_dests(self, fs, obs) -> List[int]:
        """Resolve flow selector to destination node IDs."""
        if fs.dst_node is not None:
            return [fs.dst_node]
        if fs.dst_region:
            return list(self.verifier.ground_region_to_nodes(fs.dst_region))
        if fs.dst_plane is not None:
            return list(range(
                fs.dst_plane * self.verifier.spp,
                (fs.dst_plane + 1) * self.verifier.spp))
        return list(range(self.N))
