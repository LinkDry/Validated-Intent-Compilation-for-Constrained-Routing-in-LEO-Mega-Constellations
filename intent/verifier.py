"""Constraint verifier: deterministic checks between compiler and optimizer.

Guarantees:
- Soundness: accepted programs are well-typed, grounded, physically admissible
- Safety: verified hard constraints hold if optimizer returns feasible solution
- Transparency: every issue is explicit and logged
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import heapq
from collections import deque
import numpy as np

from intent.schema import (
    ConstraintProgram, HardConstraint, SoftConstraint,
    FlowSelector, EdgeSelector, NodeSelector,
    HardConstraintType, SoftConstraintType,
    KNOWN_REGIONS, KNOWN_TRAFFIC_CLASSES, KNOWN_CORRIDORS,
)


@dataclass
class VerificationResult:
    """Result of constraint verification."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    grounded_edges: Dict[str, Set[int]] = field(default_factory=dict)
    grounded_nodes: Dict[str, Set[int]] = field(default_factory=dict)
    grounded_flows: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    masked_edges: Set[int] = field(default_factory=set)
    masked_nodes: Set[int] = field(default_factory=set)
    unsat_core: Optional[List[str]] = None
    # Pass 8: Feasibility certification
    certification_status: str = "pending"  # accepted / rejected / abstain / pending
    witnesses: Dict[str, Any] = field(default_factory=dict)
    feasibility_details: List[str] = field(default_factory=list)


class ConstraintVerifier:
    """Verifies ConstraintProgram against constellation state.

    Args:
        num_planes: number of orbital planes
        sats_per_plane: satellites per plane
        num_sats: total satellites (num_planes * sats_per_plane)
        edge_index: [2, E] array of directed edges
        edge_delays: [E] array of propagation delays in ms
        neighbor_table: [N, K] neighbor lookup
        latlon: [N, 2] satellite positions (lat, lon)
    """

    def __init__(self, num_planes: int, sats_per_plane: int,
                 edge_index: np.ndarray, edge_delays: np.ndarray,
                 neighbor_table: np.ndarray, latlon: np.ndarray,
                 polar_threshold: float = 75.0):
        self.num_planes = num_planes
        self.spp = sats_per_plane
        self.N = num_planes * sats_per_plane
        self.edge_index = edge_index
        self.edge_delays = edge_delays
        self.neighbor_table = neighbor_table
        self.latlon = latlon
        self.polar_threshold = polar_threshold

        # Build edge lookup
        self._edge_set = set()
        self._edge_id_map = {}
        for idx in range(edge_index.shape[1]):
            u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
            self._edge_set.add((u, v))
            self._edge_id_map[(u, v)] = idx

    def verify(self, cp: ConstraintProgram) -> VerificationResult:
        """Run all verification checks."""
        result = VerificationResult()

        self._check_schema(cp, result)
        if result.errors:
            result.valid = False
            return result

        self._check_entity_grounding(cp, result)
        self._check_type_safety(cp, result)
        self._check_value_ranges(cp, result)
        self._check_conflicts(cp, result)
        self._check_physical_admissibility(cp, result)
        self._check_reachability(cp, result)

        # Pass 8: Feasibility certification (only if structurally valid)
        if not result.errors:
            self._check_feasibility(cp, result)

        result.valid = len(result.errors) == 0

        # Write back to program
        cp.verified = result.valid
        cp.verification_errors = result.errors
        cp.verification_warnings = result.warnings
        return result

    def _check_schema(self, cp: ConstraintProgram, r: VerificationResult):
        """Basic schema validity."""
        if not cp.intent_id:
            r.errors.append("Missing intent_id")
        if not cp.source_text:
            r.warnings.append("Empty source_text")

        valid_priorities = {"critical", "high", "medium", "low"}
        if cp.priority not in valid_priorities:
            r.errors.append(f"Invalid priority '{cp.priority}', must be one of {valid_priorities}")

        for i, hc in enumerate(cp.hard_constraints):
            if not hc.type:
                r.errors.append(f"hard_constraints[{i}]: missing type")
            if not hc.target:
                r.errors.append(f"hard_constraints[{i}]: missing target")

        for i, sc in enumerate(cp.soft_constraints):
            if not sc.type:
                r.errors.append(f"soft_constraints[{i}]: missing type")
            if sc.penalty < 0:
                r.errors.append(f"soft_constraints[{i}]: negative penalty")

    def _check_entity_grounding(self, cp: ConstraintProgram, r: VerificationResult):
        """Verify all referenced entities exist."""
        for i, fs in enumerate(cp.flow_selectors):
            if fs.traffic_class and fs.traffic_class not in KNOWN_TRAFFIC_CLASSES:
                r.errors.append(
                    f"flow_selectors[{i}]: unknown traffic_class '{fs.traffic_class}'")
            if fs.src_region and fs.src_region.upper() not in KNOWN_REGIONS:
                r.errors.append(
                    f"flow_selectors[{i}]: unknown src_region '{fs.src_region}'")
            if fs.dst_region and fs.dst_region.upper() not in KNOWN_REGIONS:
                r.errors.append(
                    f"flow_selectors[{i}]: unknown dst_region '{fs.dst_region}'")
            if fs.src_node is not None and not (0 <= fs.src_node < self.N):
                r.errors.append(
                    f"flow_selectors[{i}]: src_node {fs.src_node} out of range [0,{self.N})")
            if fs.dst_node is not None and not (0 <= fs.dst_node < self.N):
                r.errors.append(
                    f"flow_selectors[{i}]: dst_node {fs.dst_node} out of range [0,{self.N})")
            if fs.src_plane is not None and not (0 <= fs.src_plane < self.num_planes):
                r.errors.append(
                    f"flow_selectors[{i}]: src_plane {fs.src_plane} out of range")
            if fs.dst_plane is not None and not (0 <= fs.dst_plane < self.num_planes):
                r.errors.append(
                    f"flow_selectors[{i}]: dst_plane {fs.dst_plane} out of range")

        for i, ns in enumerate(cp.node_selectors):
            if ns.node_ids:
                for nid in ns.node_ids:
                    if not (0 <= nid < self.N):
                        r.errors.append(
                            f"node_selectors[{i}]: node {nid} out of range")
            if ns.plane is not None and not (0 <= ns.plane < self.num_planes):
                r.errors.append(
                    f"node_selectors[{i}]: plane {ns.plane} out of range")
            if ns.region and ns.region.upper() not in KNOWN_REGIONS:
                r.errors.append(
                    f"node_selectors[{i}]: unknown region '{ns.region}'")

    def _check_type_safety(self, cp: ConstraintProgram, r: VerificationResult):
        """Ensure constraints attach to correct entity types."""
        flow_types = {
            HardConstraintType.MAX_LATENCY_MS.value,
            HardConstraintType.K_EDGE_DISJOINT.value,
            HardConstraintType.MAX_HOPS.value,
        }
        node_types = {
            HardConstraintType.DISABLE_NODE.value,
            HardConstraintType.REROUTE_AWAY.value,
        }
        edge_types = {
            HardConstraintType.DISABLE_EDGE.value,
            HardConstraintType.AVOID_REGION.value,
            HardConstraintType.AVOID_LATITUDE.value,
        }
        plane_types = {
            HardConstraintType.DISABLE_PLANE.value,
        }

        for i, hc in enumerate(cp.hard_constraints):
            target = hc.target
            if hc.type in flow_types and not target.startswith("flow_selector"):
                r.errors.append(
                    f"hard_constraints[{i}]: {hc.type} must target a flow_selector, got '{target}'")
            if hc.type in node_types and not (target.startswith("node") or target.startswith("node_selector")):
                r.errors.append(
                    f"hard_constraints[{i}]: {hc.type} must target a node, got '{target}'")
            if hc.type in plane_types and not target.startswith("plane"):
                r.errors.append(
                    f"hard_constraints[{i}]: {hc.type} must target a plane, got '{target}'")

    def _check_value_ranges(self, cp: ConstraintProgram, r: VerificationResult):
        """Validate numeric values are in reasonable ranges."""
        for i, hc in enumerate(cp.hard_constraints):
            if hc.type == HardConstraintType.MAX_LATENCY_MS.value:
                if hc.value is not None and (hc.value <= 0 or hc.value > 10000):
                    r.errors.append(
                        f"hard_constraints[{i}]: latency {hc.value}ms out of range (0, 10000]")
            if hc.type == HardConstraintType.K_EDGE_DISJOINT.value:
                if hc.value is not None and (hc.value < 1 or hc.value > 4):
                    r.errors.append(
                        f"hard_constraints[{i}]: k_disjoint={hc.value} out of range [1, 4]")
            if hc.type == HardConstraintType.MAX_HOPS.value:
                if hc.value is not None and (hc.value < 1 or hc.value > 40):
                    r.errors.append(
                        f"hard_constraints[{i}]: max_hops={hc.value} out of range [1, 40]")

        for i, sc in enumerate(cp.soft_constraints):
            if sc.type == SoftConstraintType.MAX_UTILIZATION.value:
                if sc.value is not None and (sc.value <= 0 or sc.value > 1.0):
                    r.errors.append(
                        f"soft_constraints[{i}]: utilization {sc.value} must be in (0, 1]")

    def _check_conflicts(self, cp: ConstraintProgram, r: VerificationResult):
        """Detect contradictory constraints on same scope."""
        # Check for conflicting latency bounds on same flow
        latency_bounds = {}
        for i, hc in enumerate(cp.hard_constraints):
            if hc.type == HardConstraintType.MAX_LATENCY_MS.value:
                key = hc.target
                if key in latency_bounds:
                    prev_i, prev_val = latency_bounds[key]
                    if prev_val != hc.value:
                        r.warnings.append(
                            f"Conflicting latency bounds on {key}: "
                            f"constraint[{prev_i}]={prev_val}ms vs [{i}]={hc.value}ms. "
                            f"Using stricter bound.")
                latency_bounds[key] = (i, hc.value)

        # Check disable + require on same entity
        disabled_nodes = set()
        required_nodes = set()
        for hc in cp.hard_constraints:
            if hc.type == HardConstraintType.DISABLE_NODE.value:
                disabled_nodes.add(hc.target)
            if hc.type == HardConstraintType.REROUTE_AWAY.value:
                required_nodes.add(hc.target)

        overlap = disabled_nodes & required_nodes
        if overlap:
            r.errors.append(
                f"Contradictory constraints: nodes both disabled and "
                f"reroute target: {overlap}")

    def _check_physical_admissibility(self, cp: ConstraintProgram, r: VerificationResult):
        """Check constraints are physically possible."""
        for i, hc in enumerate(cp.hard_constraints):
            if hc.type == HardConstraintType.DISABLE_NODE.value:
                target = hc.target
                if target.startswith("node:"):
                    nid = int(target.split(":")[1])
                    if not (0 <= nid < self.N):
                        r.errors.append(
                            f"hard_constraints[{i}]: node {nid} does not exist")

            if hc.type == HardConstraintType.DISABLE_PLANE.value:
                target = hc.target
                if target.startswith("plane:"):
                    pid = int(target.split(":")[1])
                    if not (0 <= pid < self.num_planes):
                        r.errors.append(
                            f"hard_constraints[{i}]: plane {pid} does not exist")

            if hc.type == HardConstraintType.MAX_LATENCY_MS.value:
                # Check if deadline is above minimum possible latency
                if hc.value is not None and hc.value < 2.0:
                    r.errors.append(
                        f"hard_constraints[{i}]: latency {hc.value}ms below "
                        f"physical minimum (~2.5ms single hop)")

            if hc.type == HardConstraintType.AVOID_LATITUDE.value:
                threshold = float(hc.value) if hc.value is not None else self.polar_threshold
                total_edges = self.edge_index.shape[1]
                removed = 0
                for idx in range(total_edges):
                    u = int(self.edge_index[0, idx])
                    v = int(self.edge_index[1, idx])
                    if abs(self.latlon[u, 0]) > threshold or abs(self.latlon[v, 0]) > threshold:
                        removed += 1
                if removed > total_edges * 0.5:
                    pct = 100.0 * removed / total_edges
                    r.warnings.append(
                        f"avoid_latitude threshold {threshold}° removes "
                        f"{removed}/{total_edges} edges ({pct:.0f}%)")

    def _check_reachability(self, cp: ConstraintProgram, r: VerificationResult):
        """Check that masked graph is still connected for required flows."""
        # Collect all disabled nodes and edges from hard constraints
        disabled_nodes = set()
        disabled_planes = set()

        for hc in cp.hard_constraints:
            if hc.type == HardConstraintType.DISABLE_NODE.value:
                if hc.target.startswith("node:"):
                    disabled_nodes.add(int(hc.target.split(":")[1]))
            if hc.type == HardConstraintType.DISABLE_PLANE.value:
                if hc.target.startswith("plane:"):
                    pid = int(hc.target.split(":")[1])
                    disabled_planes.add(pid)
                    for s in range(self.spp):
                        disabled_nodes.add(pid * self.spp + s)

        if not disabled_nodes:
            return

        # Build adjacency with disabled nodes removed
        active_nodes = set(range(self.N)) - disabled_nodes
        if len(active_nodes) == 0:
            r.errors.append("All nodes disabled — network unreachable")
            return

        # Capacity threshold warning (heuristic, not in soundness path)
        disabled_frac = len(disabled_nodes) / self.N
        if disabled_frac >= 0.75:
            pct = 100.0 * disabled_frac
            r.warnings.append(
                f"Severe capacity loss: {len(disabled_nodes)}/{self.N} "
                f"nodes ({pct:.0f}%) disabled — network likely non-functional")
        elif disabled_frac >= 0.5:
            pct = 100.0 * disabled_frac
            r.warnings.append(
                f"Massive capacity loss: {len(disabled_nodes)}/{self.N} "
                f"nodes ({pct:.0f}%) disabled")

        # BFS from an arbitrary active node
        adj = {n: [] for n in active_nodes}
        for idx in range(self.edge_index.shape[1]):
            u, v = int(self.edge_index[0, idx]), int(self.edge_index[1, idx])
            if u in active_nodes and v in active_nodes:
                adj[u].append(v)

        start = next(iter(active_nodes))
        visited = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for nb in adj.get(node, []):
                if nb not in visited:
                    queue.append(nb)

        unreachable = active_nodes - visited
        if unreachable:
            r.warnings.append(
                f"After disabling nodes/planes, {len(unreachable)} nodes "
                f"are unreachable from the main component")

    # --- Pass 8: Feasibility certification ---

    def _collect_topology_constraints(self, cp):
        """Collect all topology-modifying constraints into node/edge sets."""
        disabled_nodes = set()
        reroute_nodes = set()
        disabled_edges = set()

        for hc in cp.hard_constraints:
            htype = hc.type
            if htype == HardConstraintType.DISABLE_NODE.value:
                if hc.target.startswith("node:"):
                    nid = int(hc.target.split(":")[1])
                    if 0 <= nid < self.N:
                        disabled_nodes.add(nid)

            elif htype == HardConstraintType.DISABLE_PLANE.value:
                if hc.target.startswith("plane:"):
                    pid = int(hc.target.split(":")[1])
                    if 0 <= pid < self.num_planes:
                        for s in range(self.spp):
                            disabled_nodes.add(pid * self.spp + s)

            elif htype == HardConstraintType.REROUTE_AWAY.value:
                if hc.target.startswith("node:"):
                    nid = int(hc.target.split(":")[1])
                    if 0 <= nid < self.N:
                        reroute_nodes.add(nid)

            elif htype == HardConstraintType.AVOID_LATITUDE.value:
                threshold = float(hc.value) if hc.value is not None else self.polar_threshold
                for idx in range(self.edge_index.shape[1]):
                    u = int(self.edge_index[0, idx])
                    v = int(self.edge_index[1, idx])
                    if abs(self.latlon[u, 0]) > threshold or abs(self.latlon[v, 0]) > threshold:
                        disabled_edges.add((u, v))

            elif htype == HardConstraintType.AVOID_REGION.value:
                region = hc.value if isinstance(hc.value, str) else ""
                region_nodes = self.ground_region_to_nodes(region)
                for idx in range(self.edge_index.shape[1]):
                    u = int(self.edge_index[0, idx])
                    v = int(self.edge_index[1, idx])
                    if u in region_nodes or v in region_nodes:
                        disabled_edges.add((u, v))

            elif htype == HardConstraintType.DISABLE_EDGE.value:
                # Parse edge target if specific
                if hc.target.startswith("edge:"):
                    parts = hc.target.split(":")[1].split(",")
                    if len(parts) == 2:
                        u, v = int(parts[0]), int(parts[1])
                        disabled_edges.add((u, v))

        return disabled_nodes, reroute_nodes, disabled_edges

    def _build_constrained_adj(self, disabled_nodes, reroute_nodes,
                                disabled_edges, src=None, dst=None):
        """Build adjacency list with delays for the constrained graph."""
        active_nodes = set(range(self.N)) - disabled_nodes
        transit_nodes = active_nodes - reroute_nodes
        endpoints = set()
        if src is not None and src in active_nodes:
            endpoints.add(src)
        if dst is not None and dst in active_nodes:
            endpoints.add(dst)
        usable = transit_nodes | endpoints

        adj = {n: [] for n in usable}
        for idx in range(self.edge_index.shape[1]):
            u = int(self.edge_index[0, idx])
            v = int(self.edge_index[1, idx])
            if u in usable and v in usable and (u, v) not in disabled_edges:
                adj[u].append((v, float(self.edge_delays[idx])))
        return adj, usable

    def _ground_flow_endpoints(self, fs, active_nodes):
        """Ground a FlowSelector to concrete (src, dst) node IDs."""
        src = dst = None
        if fs.src_node is not None:
            src = fs.src_node
        elif getattr(fs, 'src_plane', None) is not None:
            for s in range(self.spp):
                nid = fs.src_plane * self.spp + s
                if nid in active_nodes:
                    src = nid
                    break
        elif fs.src_region:
            cands = self.ground_region_to_nodes(fs.src_region) & active_nodes
            if cands:
                src = min(cands)

        if fs.dst_node is not None:
            dst = fs.dst_node
        elif getattr(fs, 'dst_plane', None) is not None:
            for s in range(self.spp):
                nid = fs.dst_plane * self.spp + s
                if nid in active_nodes:
                    dst = nid
                    break
        elif fs.dst_region:
            cands = self.ground_region_to_nodes(fs.dst_region) & active_nodes
            if cands:
                dst = min(cands)
        return src, dst

    def _collect_flow_constraints(self, cp, flow_idx):
        """Collect hard constraints targeting a specific flow selector."""
        target = f"flow_selector:{flow_idx}"
        out = {}
        for hc in cp.hard_constraints:
            if hc.target == target:
                out[hc.type] = hc.value
        return out

    def _classify_fragment(self, fc):
        """Classify which certified fragment a flow belongs to."""
        lat = HardConstraintType.MAX_LATENCY_MS.value in fc
        hop = HardConstraintType.MAX_HOPS.value in fc
        dis = HardConstraintType.K_EDGE_DISJOINT.value in fc
        cap = HardConstraintType.MIN_CAPACITY_RESERVE.value in fc
        if cap or (dis and (lat or hop)):
            return "UNSUPPORTED"
        if dis:
            return "F5"
        if lat and hop:
            return "F4"
        if lat:
            return "F2"
        if hop:
            return "F3"
        return "F1"

    def _certify_f1(self, adj, src, dst):
        """F1: BFS reachability. Returns path list or None."""
        if src not in adj or dst not in adj:
            return None
        prev = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                path = []
                n = dst
                while n is not None:
                    path.append(n)
                    n = prev[n]
                return list(reversed(path))
            for v, _ in adj.get(u, []):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        return None

    def _certify_f2(self, adj, src, dst, deadline):
        """F2: Dijkstra with latency deadline. Returns (path, delay) or None."""
        if src not in adj or dst not in adj:
            return None
        dist = {src: 0.0}
        prev = {src: None}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float('inf')):
                continue
            if u == dst:
                if d <= deadline:
                    path = []
                    n = dst
                    while n is not None:
                        path.append(n)
                        n = prev[n]
                    return (list(reversed(path)), d)
                return None
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return None

    def _certify_f3(self, adj, src, dst, max_hops):
        """F3: BFS with hop limit. Returns (path, hops) or None."""
        if src not in adj or dst not in adj:
            return None
        visited = {src: (None, 0)}
        q = deque([(src, 0)])
        while q:
            u, h = q.popleft()
            if u == dst:
                path = []
                n = dst
                while n is not None:
                    path.append(n)
                    n = visited[n][0]
                return (list(reversed(path)), h)
            if h >= max_hops:
                continue
            for v, _ in adj.get(u, []):
                if v not in visited:
                    visited[v] = (u, h + 1)
                    q.append((v, h + 1))
        return None

    def _certify_f4(self, adj, src, dst, deadline, max_hops):
        """F4: Hop-layered Dijkstra. Returns (path, delay, hops) or None."""
        if src not in adj or dst not in adj:
            return None
        dist = {(src, 0): 0.0}
        prev = {(src, 0): None}
        heap = [(0.0, src, 0)]
        best = None
        while heap:
            d, u, h = heapq.heappop(heap)
            if d > dist.get((u, h), float('inf')):
                continue
            if u == dst and d <= deadline:
                if best is None or d < best[0]:
                    best = (d, h, (u, h))
                continue
            if h >= max_hops:
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                st = (v, h + 1)
                if nd < dist.get(st, float('inf')):
                    dist[st] = nd
                    prev[st] = (u, h)
                    heapq.heappush(heap, (nd, v, h + 1))
        if best is None:
            return None
        path = []
        st = best[2]
        while st is not None:
            path.append(st[0])
            st = prev.get(st)
        return (list(reversed(path)), best[0], best[1])

    def _certify_f5(self, adj, src, dst, k):
        """F5: k edge-disjoint paths via Edmonds-Karp. Returns paths or None."""
        if src not in adj or dst not in adj:
            return None
        cap = {}
        for u in adj:
            if u not in cap:
                cap[u] = {}
            for v, _ in adj[u]:
                if v not in cap:
                    cap[v] = {}
                cap[u][v] = cap[u].get(v, 0) + 1
                if u not in cap[v]:
                    cap[v][u] = 0
        paths = []
        while len(paths) < k:
            parent = {src: None}
            q = deque([src])
            found = False
            while q and not found:
                u = q.popleft()
                for v in cap.get(u, {}):
                    if v not in parent and cap[u][v] > 0:
                        parent[v] = u
                        if v == dst:
                            found = True
                            break
                        q.append(v)
            if not found:
                break
            path = []
            v = dst
            while v is not None:
                path.append(v)
                v = parent[v]
            path = list(reversed(path))
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                cap[u][v] -= 1
                cap[v][u] += 1
            paths.append(path)
        return paths if len(paths) >= k else None

    def _check_feasibility(self, cp, r):
        """Pass 8: Constructive feasibility certification.

        For each demanded flow, certifies that a routing witness exists
        satisfying all hard constraints simultaneously on the constrained
        topology. Three outcomes: accepted / rejected / abstain.
        """
        disabled_nodes, reroute_nodes, disabled_edges = \
            self._collect_topology_constraints(cp)
        active_nodes = set(range(self.N)) - disabled_nodes

        # No flow selectors → topology-only program
        if not cp.flow_selectors:
            # Check if disabling everything leaves an empty graph
            if not active_nodes:
                r.errors.append(
                    "Feasibility: all nodes disabled, no routing possible")
                r.certification_status = "rejected"
            else:
                # Cannot constructively certify routing feasibility
                # without demanded flows — abstain
                r.certification_status = "abstain"
                r.feasibility_details.append(
                    "topology-only program: no flow selectors to certify against")
            return

        all_ok = True
        any_abstain = False

        for i, fs in enumerate(cp.flow_selectors):
            src, dst = self._ground_flow_endpoints(fs, active_nodes)

            # Endpoint grounding failures
            if src is None and dst is None and not fs.src_region and not fs.dst_region:
                any_abstain = True
                r.feasibility_details.append(
                    f"flow[{i}]: abstain — no concrete endpoints")
                continue
            if src is not None and src in disabled_nodes:
                r.errors.append(f"flow[{i}]: src node {src} is disabled")
                all_ok = False
                continue
            if dst is not None and dst in disabled_nodes:
                r.errors.append(f"flow[{i}]: dst node {dst} is disabled")
                all_ok = False
                continue
            if src is None:
                label = getattr(fs, 'src_region', None) or 'unspecified'
                r.errors.append(
                    f"flow[{i}]: no reachable node for src ({label})")
                all_ok = False
                continue
            if dst is None:
                label = getattr(fs, 'dst_region', None) or 'unspecified'
                r.errors.append(
                    f"flow[{i}]: no reachable node for dst ({label})")
                all_ok = False
                continue

            adj, _ = self._build_constrained_adj(
                disabled_nodes, reroute_nodes, disabled_edges, src, dst)

            fc = self._collect_flow_constraints(cp, i)
            frag = self._classify_fragment(fc)

            if frag == "UNSUPPORTED":
                any_abstain = True
                r.feasibility_details.append(
                    f"flow[{i}]: abstain — unsupported combination")
                continue

            witness = None
            if frag == "F1":
                witness = self._certify_f1(adj, src, dst)
            elif frag == "F2":
                witness = self._certify_f2(
                    adj, src, dst, float(fc[HardConstraintType.MAX_LATENCY_MS.value]))
            elif frag == "F3":
                witness = self._certify_f3(
                    adj, src, dst, int(fc[HardConstraintType.MAX_HOPS.value]))
            elif frag == "F4":
                witness = self._certify_f4(
                    adj, src, dst,
                    float(fc[HardConstraintType.MAX_LATENCY_MS.value]),
                    int(fc[HardConstraintType.MAX_HOPS.value]))
            elif frag == "F5":
                witness = self._certify_f5(
                    adj, src, dst, int(fc[HardConstraintType.K_EDGE_DISJOINT.value]))

            if witness is not None:
                r.witnesses[f"flow_selector:{i}"] = {
                    "fragment": frag, "witness": witness}
                r.feasibility_details.append(
                    f"flow[{i}]: certified feasible ({frag})")
            else:
                r.errors.append(
                    f"flow[{i}]: infeasible — no routing satisfies "
                    f"{frag} constraints on constrained graph "
                    f"(src={src}, dst={dst})")
                all_ok = False

        if not all_ok:
            r.certification_status = "rejected"
        elif any_abstain:
            r.certification_status = "abstain"
        else:
            r.certification_status = "accepted"

    # --- Grounding helpers ---

    def ground_region_to_nodes(self, region: str, radius_deg: float = 15.0) -> Set[int]:
        """Map a region name to nearby satellite node IDs."""
        region_upper = region.upper()
        if region_upper not in KNOWN_REGIONS:
            return set()
        lat, lon = KNOWN_REGIONS[region_upper]
        dlat = self.latlon[:, 0] - lat
        dlon = self.latlon[:, 1] - lon
        dlon = np.where(dlon > 180, dlon - 360, dlon)
        dlon = np.where(dlon < -180, dlon + 360, dlon)
        dist = np.sqrt(dlat**2 + dlon**2)
        return set(np.where(dist < radius_deg)[0].tolist())

    def ground_polar_edges(self, lat_threshold: float = None) -> Set[int]:
        """Return edge indices that cross the polar region."""
        if lat_threshold is None:
            lat_threshold = self.polar_threshold
        polar_edges = set()
        for idx in range(self.edge_index.shape[1]):
            u, v = int(self.edge_index[0, idx]), int(self.edge_index[1, idx])
            if (abs(self.latlon[u, 0]) > lat_threshold or
                    abs(self.latlon[v, 0]) > lat_threshold):
                polar_edges.add(idx)
        return polar_edges
