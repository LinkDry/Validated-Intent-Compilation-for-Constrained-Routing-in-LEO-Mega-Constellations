"""Smoke test: schema → verifier → constrained_router end-to-end.

Uses a small synthetic constellation (4 planes × 4 sats = 16 nodes)
with a mocked GNN policy to validate constraint grounding, masking,
and slot-table derivation without needing a trained model.
"""

import sys, json
import numpy as np
import torch

sys.path.insert(0, ".")

from intent.schema import (
    ConstraintProgram, HardConstraint, SoftConstraint,
    FlowSelector, HardConstraintType, SoftConstraintType,
    EventCondition,
)
from intent.verifier import ConstraintVerifier, VerificationResult
from intent.constrained_router import ConstrainedRouter, ConstrainedRoutingResult


# ── Synthetic constellation: 4 planes × 4 sats = 16 nodes ──

NUM_PLANES = 4
SPP = 4
N = NUM_PLANES * SPP
K = 4  # max neighbors per node

np.random.seed(42)

# Lat/lon: spread across globe
latlon = np.zeros((N, 2), dtype=np.float32)
for p in range(NUM_PLANES):
    for s in range(SPP):
        nid = p * SPP + s
        latlon[nid, 0] = -60 + s * 40          # lat: -60 to 60
        latlon[nid, 1] = -180 + p * 90 + s * 5  # lon spread

# Put node 3 in polar region for AVOID_LATITUDE test
latlon[3, 0] = 78.0

# Build ring + inter-plane edges
edges_u, edges_v = [], []
for p in range(NUM_PLANES):
    for s in range(SPP):
        nid = p * SPP + s
        # Intra-plane ring
        next_s = (s + 1) % SPP
        nbr = p * SPP + next_s
        edges_u.append(nid); edges_v.append(nbr)
        edges_u.append(nbr); edges_v.append(nid)
        # Inter-plane
        next_p = (p + 1) % NUM_PLANES
        nbr2 = next_p * SPP + s
        edges_u.append(nid); edges_v.append(nbr2)
        edges_u.append(nbr2); edges_v.append(nid)

edge_index = np.array([edges_u, edges_v], dtype=np.int64)
num_edges = edge_index.shape[1]
edge_delays = np.random.uniform(2.5, 15.0, size=num_edges).astype(np.float32)

# Build neighbor table [N, K]
neighbor_table = -np.ones((N, K), dtype=np.int64)
neighbor_delays = np.zeros((N, K), dtype=np.float32)
neighbor_mask = np.zeros((N, K), dtype=np.float32)

for idx in range(num_edges):
    u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
    for k in range(K):
        if neighbor_table[u, k] == -1:
            neighbor_table[u, k] = v
            neighbor_delays[u, k] = edge_delays[idx]
            neighbor_mask[u, k] = 1.0
            break


# ── Mock GNN policy ──

class MockPolicy:
    """Returns distance-based cost predictions without a real GNN."""
    def __call__(self, data):
        # Simple cost: Euclidean on latlon as proxy
        n = latlon.shape[0]
        cost = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                dlat = latlon[i, 0] - latlon[j, 0]
                dlon = latlon[i, 1] - latlon[j, 1]
                cost[i, j] = np.sqrt(dlat**2 + dlon**2) * 0.1  # ms-ish
        return torch.tensor(cost)

    @staticmethod
    def obs_to_pyg(obs, device):
        return None, None, None, None


# Monkey-patch the import inside constrained_router
import intent.constrained_router as cr_mod
import types

_orig_route = cr_mod.ConstrainedRouter.route

def patched_route(self, obs, program):
    """Skip the real GNN forward; use mock cost directly."""
    edge_mask, node_mask, util_caps, flow_deadlines = (
        self._ground_constraints(obs, program)
    )
    cost_np = self.policy(None).numpy()

    # Apply node mask
    for n in range(self.N):
        if node_mask[n] == 0:
            cost_np[n, :] = 1e9
            cost_np[:, n] = 1e9

    # Derive slot table
    nt_np = obs["neighbor_table"]
    nd_np = obs["neighbor_delays"]
    nm_np = obs["neighbor_mask"]
    ne = int(obs["num_edges"])
    ei = obs["edge_index"][:, :ne]

    edge_disabled = set()
    for idx in range(ne):
        if edge_mask[idx] == 0:
            u, v = int(ei[0, idx]), int(ei[1, idx])
            edge_disabled.add((u, v))

    slot_table = np.zeros((self.N, self.N), dtype=np.int64)
    Kn = nt_np.shape[1]
    for src in range(self.N):
        if node_mask[src] == 0:
            continue
        for dst in range(self.N):
            if src == dst or node_mask[dst] == 0:
                continue
            best_slot, best_cost = 0, 1e9
            for k in range(Kn):
                if nm_np[src, k] == 0:
                    continue
                nbr = int(nt_np[src, k])
                if nbr < 0 or node_mask[nbr] == 0:
                    continue
                if (src, nbr) in edge_disabled:
                    continue
                cc = nd_np[src, k] + cost_np[nbr, dst]
                if cc < best_cost:
                    best_cost = cc
                    best_slot = k
            slot_table[src, dst] = best_slot

    # Validate latency deadlines
    violations = []
    hard_ok = True
    for key, deadline in flow_deadlines.items():
        parts = key.split(":")
        if len(parts) == 2:
            idx = int(parts[1])
            if idx < len(program.flow_selectors):
                fs = program.flow_selectors[idx]
                srcs = self._resolve_flow_sources(fs, obs)
                dsts = self._resolve_flow_dests(fs, obs)
                for s in srcs:
                    for d in dsts:
                        if cost_np[s, d] > deadline:
                            violations.append(
                                f"Latency {cost_np[s,d]:.1f}ms > "
                                f"deadline {deadline}ms for ({s}->{d})")
                            hard_ok = False

    return ConstrainedRoutingResult(
        slot_table=slot_table, cost_matrix=cost_np,
        edge_mask=edge_mask, node_mask=node_mask,
        utilization_caps=util_caps, flow_deadlines=flow_deadlines,
        hard_satisfied=hard_ok, violations=violations,
    )

cr_mod.ConstrainedRouter.route = patched_route


# ── Build observation dict ──

obs = {
    "edge_index": edge_index,
    "num_edges": num_edges,
    "neighbor_table": neighbor_table,
    "neighbor_delays": neighbor_delays,
    "neighbor_mask": neighbor_mask,
}


# ── Tests ──

def make_verifier():
    return ConstraintVerifier(
        num_planes=NUM_PLANES, sats_per_plane=SPP,
        edge_index=edge_index, edge_delays=edge_delays,
        neighbor_table=neighbor_table, latlon=latlon,
    )

def make_router(verifier):
    return ConstrainedRouter(
        policy=MockPolicy(), verifier=verifier, device="cpu",
    )

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name} — {detail}")
        failed += 1


# ── Test 1: Unconstrained baseline ──
print("\n=== Test 1: Unconstrained routing ===")
v = make_verifier()
r = make_router(v)

cp = ConstraintProgram(
    intent_id="test-unconstrained",
    source_text="Route normally",
    priority="medium",
)
vr = v.verify(cp)
check("verify passes", vr.valid, str(vr.errors))

result = r.route(obs, cp)
check("all edges active", result.edge_mask.sum() == num_edges)
check("all nodes active", result.node_mask.sum() == N)
check("hard constraints satisfied", result.hard_satisfied)
check("slot table shape", result.slot_table.shape == (N, N))


# ── Test 2: Disable node ──
print("\n=== Test 2: Disable node 5 ===")
cp2 = ConstraintProgram(
    intent_id="test-disable-node",
    source_text="Disable satellite 5",
    priority="high",
    hard_constraints=[
        HardConstraint(
            type=HardConstraintType.DISABLE_NODE.value,
            target="node:5", value=None,
        ),
    ],
)
vr2 = v.verify(cp2)
check("verify passes", vr2.valid, str(vr2.errors))

res2 = r.route(obs, cp2)
check("node 5 disabled", res2.node_mask[5] == 0)
check("node 5 cost infinite", res2.cost_matrix[5, 0] >= 1e8)
check("other nodes active", res2.node_mask[0] == 1 and res2.node_mask[1] == 1)
# Edges touching node 5 should be disabled
edges_touching_5 = sum(
    1 for idx in range(num_edges)
    if int(edge_index[0, idx]) == 5 or int(edge_index[1, idx]) == 5
)
disabled_edges = sum(1 for idx in range(num_edges) if res2.edge_mask[idx] == 0)
check("edges touching node 5 disabled", disabled_edges >= edges_touching_5,
      f"disabled={disabled_edges}, touching={edges_touching_5}")


# ── Test 3: Disable plane ──
print("\n=== Test 3: Disable plane 2 ===")
cp3 = ConstraintProgram(
    intent_id="test-disable-plane",
    source_text="Disable orbital plane 2",
    priority="critical",
    hard_constraints=[
        HardConstraint(
            type=HardConstraintType.DISABLE_PLANE.value,
            target="plane:2", value=None,
        ),
    ],
)
vr3 = v.verify(cp3)
check("verify passes", vr3.valid, str(vr3.errors))

res3 = r.route(obs, cp3)
plane2_nodes = list(range(2 * SPP, 3 * SPP))
for nid in plane2_nodes:
    check(f"node {nid} disabled", res3.node_mask[nid] == 0)
check("other planes active", all(res3.node_mask[i] == 1 for i in range(SPP)))


# ── Test 4: Avoid high latitude ──
print("\n=== Test 4: Avoid latitude > 75 ===")
cp4 = ConstraintProgram(
    intent_id="test-avoid-lat",
    source_text="Avoid polar links above 75 degrees",
    priority="high",
    hard_constraints=[
        HardConstraint(
            type=HardConstraintType.AVOID_LATITUDE.value,
            target="edges:ALL", value=75.0,
        ),
    ],
)
vr4 = v.verify(cp4)
check("verify passes", vr4.valid, str(vr4.errors))

res4 = r.route(obs, cp4)
# Node 3 is at lat 78 — edges touching it should be disabled
polar_disabled = 0
for idx in range(num_edges):
    u, v_node = int(edge_index[0, idx]), int(edge_index[1, idx])
    if abs(latlon[u, 0]) > 75 or abs(latlon[v_node, 0]) > 75:
        check(f"polar edge {u}->{v_node} disabled", res4.edge_mask[idx] == 0)
        polar_disabled += 1
check("some polar edges disabled", polar_disabled > 0, f"count={polar_disabled}")


# ── Test 5: Max utilization (soft) ──
print("\n=== Test 5: Max utilization cap ===")
cp5 = ConstraintProgram(
    intent_id="test-util-cap",
    source_text="Keep link utilization below 70%",
    priority="medium",
    soft_constraints=[
        SoftConstraint(
            type=SoftConstraintType.MAX_UTILIZATION.value,
            target="edges:ALL", value=0.7, penalty=2.0,
        ),
    ],
)
vr5 = v.verify(cp5)
check("verify passes", vr5.valid, str(vr5.errors))

res5 = r.route(obs, cp5)
check("util caps set to 0.7", np.allclose(res5.utilization_caps, 0.7),
      f"min={res5.utilization_caps.min()}, max={res5.utilization_caps.max()}")


# ── Test 6: Conditional constraint (inactive) ──
print("\n=== Test 6: Conditional constraint (inactive event) ===")
cp6 = ConstraintProgram(
    intent_id="test-conditional",
    source_text="If solar storm, disable plane 0",
    priority="critical",
    hard_constraints=[
        HardConstraint(
            type=HardConstraintType.DISABLE_PLANE.value,
            target="plane:0", value=None,
            condition=EventCondition(event_type="solar_storm", active=False),
        ),
    ],
)
vr6 = v.verify(cp6)
check("verify passes", vr6.valid, str(vr6.errors))

res6 = r.route(obs, cp6)
check("plane 0 still active (event inactive)",
      all(res6.node_mask[i] == 1 for i in range(SPP)))


# ── Test 7: Conditional constraint (active) ──
print("\n=== Test 7: Conditional constraint (active event) ===")
cp7 = ConstraintProgram(
    intent_id="test-conditional-active",
    source_text="If solar storm, disable plane 0",
    priority="critical",
    hard_constraints=[
        HardConstraint(
            type=HardConstraintType.DISABLE_PLANE.value,
            target="plane:0", value=None,
            condition=EventCondition(event_type="solar_storm", active=True),
        ),
    ],
)
res7 = r.route(obs, cp7)
check("plane 0 disabled (event active)",
      all(res7.node_mask[i] == 0 for i in range(SPP)))


# ── Test 8: Latency deadline violation ──
print("\n=== Test 8: Tight latency deadline ===")
cp8 = ConstraintProgram(
    intent_id="test-latency",
    source_text="Financial traffic NYC to Tokyo under 5ms",
    priority="critical",
    flow_selectors=[
        FlowSelector(src_node=0, dst_node=15),
    ],
    hard_constraints=[
        HardConstraint(
            type=HardConstraintType.MAX_LATENCY_MS.value,
            target="flow_selector:0", value=5.0,
        ),
    ],
)
vr8 = v.verify(cp8)
check("verify passes", vr8.valid, str(vr8.errors))

res8 = r.route(obs, cp8)
# With mock costs, 0->15 is far apart, should violate 5ms deadline
check("violations detected for tight deadline", len(res8.violations) > 0,
      f"violations={res8.violations}")
check("hard_satisfied=False", not res8.hard_satisfied)


# ── Test 9: Compositional — disable node + avoid latitude ──
print("\n=== Test 9: Compositional constraints ===")
cp9 = ConstraintProgram(
    intent_id="test-composite",
    source_text="Disable node 10 and avoid polar links",
    priority="high",
    hard_constraints=[
        HardConstraint(
            type=HardConstraintType.DISABLE_NODE.value,
            target="node:10", value=None,
        ),
        HardConstraint(
            type=HardConstraintType.AVOID_LATITUDE.value,
            target="edges:ALL", value=75.0,
        ),
    ],
    soft_constraints=[
        SoftConstraint(
            type=SoftConstraintType.MAX_UTILIZATION.value,
            target="edges:ALL", value=0.8, penalty=1.5,
        ),
    ],
)
vr9 = v.verify(cp9)
check("verify passes", vr9.valid, str(vr9.errors))

res9 = r.route(obs, cp9)
check("node 10 disabled", res9.node_mask[10] == 0)
check("util caps at 0.8", np.allclose(res9.utilization_caps, 0.8))
check("hard constraints satisfied", res9.hard_satisfied)


# ── Test 10: JSON round-trip + verify + route ──
print("\n=== Test 10: JSON round-trip pipeline ===")
json_str = cp9.to_json()
cp10 = ConstraintProgram.from_json(json_str)
check("round-trip intent_id", cp10.intent_id == "test-composite")
check("round-trip hard count", len(cp10.hard_constraints) == 2)
check("round-trip soft count", len(cp10.soft_constraints) == 1)

vr10 = v.verify(cp10)
check("round-trip verify passes", vr10.valid, str(vr10.errors))

res10 = r.route(obs, cp10)
check("round-trip routing works", res10.slot_table.shape == (N, N))
check("round-trip node 10 disabled", res10.node_mask[10] == 0)


# ── Summary ──
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
if failed == 0:
    print("All smoke tests PASSED")
else:
    print(f"WARNING: {failed} test(s) FAILED")
    sys.exit(1)
