"""Smoke test: IntentCompiler via LM Studio API."""

import sys, json, time
sys.path.insert(0, ".")

import numpy as np
from intent.compiler import IntentCompiler
from intent.verifier import ConstraintVerifier

# ── Build a realistic verifier (20 planes × 20 sats) ──
NUM_PLANES, SPP = 20, 20
N = NUM_PLANES * SPP

np.random.seed(42)
latlon = np.zeros((N, 2), dtype=np.float32)
for p in range(NUM_PLANES):
    for s in range(SPP):
        nid = p * SPP + s
        latlon[nid, 0] = -90 + (s / (SPP - 1)) * 180
        latlon[nid, 1] = -180 + (p / (NUM_PLANES - 1)) * 360

# Build edges: intra-plane ring + inter-plane
eu, ev = [], []
for p in range(NUM_PLANES):
    for s in range(SPP):
        nid = p * SPP + s
        eu.append(nid); ev.append(p * SPP + (s + 1) % SPP)
        eu.append(p * SPP + (s + 1) % SPP); ev.append(nid)
        nbr_p = (p + 1) % NUM_PLANES
        eu.append(nid); ev.append(nbr_p * SPP + s)
        eu.append(nbr_p * SPP + s); ev.append(nid)

edge_index = np.array([eu, ev], dtype=np.int64)
edge_delays = np.random.uniform(2.5, 15.0, size=edge_index.shape[1]).astype(np.float32)
neighbor_table = -np.ones((N, 4), dtype=np.int64)
for idx in range(edge_index.shape[1]):
    u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
    for k in range(4):
        if neighbor_table[u, k] == -1:
            neighbor_table[u, k] = v
            break

verifier = ConstraintVerifier(
    num_planes=NUM_PLANES, sats_per_plane=SPP,
    edge_index=edge_index, edge_delays=edge_delays,
    neighbor_table=neighbor_table, latlon=latlon,
)

compiler = IntentCompiler(verifier=verifier, max_retries=3)

# ── Test intents ──
test_intents = [
    ("simple", "Disable node 42"),
    ("latency", "Ensure financial traffic from NYC to LONDON stays under 60ms"),
    ("compositional", "Disable plane 3 and avoid polar links above 75 degrees"),
    ("conditional", "If a solar storm occurs, reroute away from node 100"),
    ("complex", "Disable node 200, avoid TOKYO region, and cap utilization at 70%"),
]

passed = 0
for label, intent in test_intents:
    print(f"\n{'='*60}")
    print(f"[{label}] {intent}")
    print(f"{'='*60}")

    result = compiler.compile(intent)

    print(f"  Success: {result.success}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Latency: {result.latency_ms:.0f}ms")

    if result.success and result.program:
        cp = result.program
        print(f"  Intent ID: {cp.intent_id}")
        print(f"  Priority: {cp.priority}")
        print(f"  Hard constraints: {len(cp.hard_constraints)}")
        print(f"  Soft constraints: {len(cp.soft_constraints)}")
        print(f"  Flow selectors: {len(cp.flow_selectors)}")
        if result.verification:
            print(f"  Warnings: {result.verification.warnings}")
        passed += 1
    else:
        print(f"  Errors: {result.errors}")
        if result.raw_json:
            print(f"  Raw JSON (first 200): {result.raw_json[:200]}")

print(f"\n{'='*60}")
print(f"Results: {passed}/{len(test_intents)} compiled successfully")
