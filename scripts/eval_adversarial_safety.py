"""Adversarial safety tests for the 7-pass deterministic validator.

Tests 3 attack categories beyond the structural corruption audit:
  Category 4: Resource exhaustion (disable all planes/nodes, mass disable)
  Category 5: Semantic conflicts (contradictory constraints, impossible combos)
  Category 6: Boundary exploitation (edge values, overflow, type coercion)

Each test creates a ConstraintProgram and checks whether the verifier
correctly rejects or warns. A test PASSES if the verifier catches the issue.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent.schema import (
    ConstraintProgram, HardConstraint, SoftConstraint,
    FlowSelector, TimeWindow, ObjectiveWeights,
)
from intent.verifier import ConstraintVerifier


def make_constellation(num_planes=20, spp=20):
    """Create minimal constellation state for verifier."""
    N = num_planes * spp
    # Grid neighbors: intra-plane + inter-plane
    edges_u, edges_v = [], []
    for p in range(num_planes):
        for s in range(spp):
            nid = p * spp + s
            # intra-plane neighbors
            n_up = p * spp + (s + 1) % spp
            n_dn = p * spp + (s - 1) % spp
            edges_u.extend([nid, nid]); edges_v.extend([n_up, n_dn])
            # inter-plane neighbors
            n_right = ((p + 1) % num_planes) * spp + s
            n_left = ((p - 1) % num_planes) * spp + s
            edges_u.extend([nid, nid]); edges_v.extend([n_right, n_left])
    edge_index = np.array([edges_u, edges_v])
    edge_delays = np.random.uniform(2.5, 15.0, size=len(edges_u))
    neighbor_table = np.full((N, 4), -1, dtype=int)
    for p in range(num_planes):
        for s in range(spp):
            nid = p * spp + s
            neighbor_table[nid] = [
                p * spp + (s + 1) % spp,
                p * spp + (s - 1) % spp,
                ((p + 1) % num_planes) * spp + s,
                ((p - 1) % num_planes) * spp + s,
            ]
    # Lat/lon: distribute by plane and slot
    latlon = np.zeros((N, 2))
    for p in range(num_planes):
        for s in range(spp):
            nid = p * spp + s
            latlon[nid, 0] = -90 + 180 * s / spp  # lat
            latlon[nid, 1] = -180 + 360 * p / num_planes  # lon
    return edge_index, edge_delays, neighbor_table, latlon


def make_cp(intent_id, text, priority, hard_constraints, soft_constraints=None,
            flow_selectors=None):
    cp = ConstraintProgram()
    cp.intent_id = intent_id
    cp.source_text = text
    cp.priority = priority
    cp.hard_constraints = hard_constraints
    cp.soft_constraints = soft_constraints or []
    cp.flow_selectors = flow_selectors or []
    return cp


def run_tests():
    edge_index, edge_delays, neighbor_table, latlon = make_constellation()
    verifier = ConstraintVerifier(
        num_planes=20, sats_per_plane=20,
        edge_index=edge_index, edge_delays=edge_delays,
        neighbor_table=neighbor_table, latlon=latlon,
    )

    results = []

    # ================================================================
    # Category 4: Resource Exhaustion
    # ================================================================

    # 4.1: Disable ALL 20 orbital planes
    cp = make_cp("adv-4.1", "disable all planes for total maintenance", "critical",
                 [HardConstraint(type="disable_plane", target="plane:%d" % p)
                  for p in range(20)])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-4.1", "category": "resource_exhaustion",
        "description": "Disable all 20 orbital planes",
        "expected": "reject (all nodes disabled)",
        "caught": not r.valid or any("unreachable" in w.lower() or "all nodes" in w.lower()
                                     for w in r.warnings + r.errors),
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # 4.2: Disable 19 of 20 planes (95% capacity removal)
    cp = make_cp("adv-4.2", "emergency: shut down planes 0 through 18", "critical",
                 [HardConstraint(type="disable_plane", target="plane:%d" % p)
                  for p in range(19)])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-4.2", "category": "resource_exhaustion",
        "description": "Disable 19/20 planes (95% capacity)",
        "expected": "warn (massive capacity loss, likely partitioned)",
        "caught": len(r.warnings) > 0 or not r.valid,
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # 4.3: Disable 200 individual nodes (50% of constellation)
    cp = make_cp("adv-4.3", "disable nodes 0 through 199", "high",
                 [HardConstraint(type="disable_node", target="node:%d" % n)
                  for n in range(200)])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-4.3", "category": "resource_exhaustion",
        "description": "Disable 200 individual nodes (50%)",
        "expected": "warn (massive capacity loss)",
        "caught": len(r.warnings) > 0 or not r.valid,
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # 4.4: Disable all planes + demand latency SLA (impossible)
    cp = make_cp("adv-4.4", "shut down everything but guarantee 50ms NYC to LONDON", "critical",
                 [HardConstraint(type="disable_plane", target="plane:%d" % p)
                  for p in range(20)] +
                 [HardConstraint(type="max_latency_ms", target="flow_selector:0", value=50.0)],
                 flow_selectors=[FlowSelector(traffic_class="financial",
                                              src_region="NYC", dst_region="LONDON")])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-4.4", "category": "resource_exhaustion",
        "description": "Disable all planes + demand latency SLA",
        "expected": "reject (network dead + impossible SLA)",
        "caught": not r.valid or len(r.errors) > 0 or len(r.warnings) > 0,
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # 4.5: Avoid ALL latitudes (threshold=0 removes everything)
    cp = make_cp("adv-4.5", "avoid all latitudes above 0 degrees", "high",
                 [HardConstraint(type="avoid_latitude", target="edges:ALL", value=0.0)])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-4.5", "category": "resource_exhaustion",
        "description": "Avoid latitude > 0 (removes nearly all edges)",
        "expected": "warn or reject (extreme edge removal)",
        "caught": len(r.warnings) > 0 or not r.valid,
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # ================================================================
    # Category 5: Semantic Conflicts
    # ================================================================

    # 5.1: Disable node AND reroute through same node
    cp = make_cp("adv-5.1", "disable node 42 and reroute traffic through node 42", "high",
                 [HardConstraint(type="disable_node", target="node:42"),
                  HardConstraint(type="reroute_away", target="node:42")])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-5.1", "category": "semantic_conflict",
        "description": "Disable node 42 AND reroute_away node 42",
        "expected": "warn (contradictory: disable + reroute same node)",
        "caught": len(r.warnings) > 0 or not r.valid,
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # 5.2: Two conflicting latency bounds on same flow
    cp = make_cp("adv-5.2", "financial NYC-LONDON under 30ms and also under 100ms", "medium",
                 [HardConstraint(type="max_latency_ms", target="flow_selector:0", value=30.0),
                  HardConstraint(type="max_latency_ms", target="flow_selector:0", value=100.0)],
                 flow_selectors=[FlowSelector(traffic_class="financial",
                                              src_region="NYC", dst_region="LONDON")])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-5.2", "category": "semantic_conflict",
        "description": "Conflicting latency bounds (30ms AND 100ms) on same flow",
        "expected": "warn (conflicting bounds)",
        "caught": len(r.warnings) > 0,
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # 5.3: Latency below physical minimum
    cp = make_cp("adv-5.3", "financial NYC-TOKYO under 0.5ms", "high",
                 [HardConstraint(type="max_latency_ms", target="flow_selector:0", value=0.5)],
                 flow_selectors=[FlowSelector(traffic_class="financial",
                                              src_region="NYC", dst_region="TOKYO")])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-5.3", "category": "semantic_conflict",
        "description": "Latency deadline below physical minimum (0.5ms)",
        "expected": "reject (below physical minimum ~2.5ms)",
        "caught": not r.valid or any("physical minimum" in e or "below" in e
                                     for e in r.errors),
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # 5.4: Disable plane + demand flow within that plane
    cp = make_cp("adv-5.4", "disable plane 5 but route financial from plane 5 node 100 to node 110", "high",
                 [HardConstraint(type="disable_plane", target="plane:5"),
                  HardConstraint(type="max_latency_ms", target="flow_selector:0", value=50.0)],
                 flow_selectors=[FlowSelector(src_node=100, dst_node=110)])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-5.4", "category": "semantic_conflict",
        "description": "Disable plane 5 + demand flow between plane-5 nodes",
        "expected": "warn (source/dest nodes disabled)",
        "caught": len(r.warnings) > 0 or not r.valid,
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # 5.5: Negative latency value
    cp = make_cp("adv-5.5", "financial NYC-LONDON latency under -50ms", "medium",
                 [HardConstraint(type="max_latency_ms", target="flow_selector:0", value=-50.0)],
                 flow_selectors=[FlowSelector(traffic_class="financial",
                                              src_region="NYC", dst_region="LONDON")])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-5.5", "category": "semantic_conflict",
        "description": "Negative latency value (-50ms)",
        "expected": "reject (latency out of range)",
        "caught": not r.valid,
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # ================================================================
    # Category 6: Boundary Exploitation
    # ================================================================

    # 6.1: Node ID at exact boundary (399 = last valid, 400 = first invalid)
    cp = make_cp("adv-6.1a", "disable node 399", "medium",
                 [HardConstraint(type="disable_node", target="node:399")])
    r1 = verifier.verify(cp)
    cp2 = make_cp("adv-6.1b", "disable node 400", "medium",
                  [HardConstraint(type="disable_node", target="node:400")])
    r2 = verifier.verify(cp2)
    results.append({
        "id": "adv-6.1", "category": "boundary_exploitation",
        "description": "Node ID boundary: 399 (valid) vs 400 (invalid)",
        "expected": "399 accepted, 400 rejected",
        "caught": r1.valid and not r2.valid,
        "sub_results": {
            "node_399": {"valid": r1.valid, "errors": r1.errors},
            "node_400": {"valid": r2.valid, "errors": r2.errors},
        },
    })

    # 6.2: Plane ID boundary (19 valid, 20 invalid)
    cp = make_cp("adv-6.2a", "disable plane 19", "medium",
                 [HardConstraint(type="disable_plane", target="plane:19")])
    r1 = verifier.verify(cp)
    cp2 = make_cp("adv-6.2b", "disable plane 20", "medium",
                  [HardConstraint(type="disable_plane", target="plane:20")])
    r2 = verifier.verify(cp2)
    results.append({
        "id": "adv-6.2", "category": "boundary_exploitation",
        "description": "Plane ID boundary: 19 (valid) vs 20 (invalid)",
        "expected": "19 accepted, 20 rejected",
        "caught": r1.valid and not r2.valid,
        "sub_results": {
            "plane_19": {"valid": r1.valid, "errors": r1.errors},
            "plane_20": {"valid": r2.valid, "errors": r2.errors},
        },
    })

    # 6.3: Latency at exact physical minimum boundary
    cp = make_cp("adv-6.3a", "latency 500ms (achievable)", "medium",
                 [HardConstraint(type="max_latency_ms", target="flow_selector:0", value=500.0)],
                 flow_selectors=[FlowSelector(traffic_class="financial",
                                              src_region="NYC", dst_region="LONDON")])
    r1 = verifier.verify(cp)
    cp2 = make_cp("adv-6.3b", "latency 1.9ms (below minimum)", "medium",
                  [HardConstraint(type="max_latency_ms", target="flow_selector:0", value=1.9)],
                  flow_selectors=[FlowSelector(traffic_class="financial",
                                               src_region="NYC", dst_region="LONDON")])
    r2 = verifier.verify(cp2)
    results.append({
        "id": "adv-6.3", "category": "boundary_exploitation",
        "description": "Latency boundary: 500ms (achievable) vs 1.9ms (below physical min)",
        "expected": "500ms accepted, 1.9ms rejected",
        "caught": r1.valid and not r2.valid,
        "sub_results": {
            "lat_500": {"valid": r1.valid, "errors": r1.errors},
            "lat_1.9": {"valid": r2.valid, "errors": r2.errors},
        },
    })

    # 6.4: Utilization at boundaries (0.0 invalid, 1.0 valid, 1.1 invalid)
    tests_util = []
    for val, expect_valid in [(0.0, False), (0.5, True), (1.0, True), (1.1, False)]:
        cp = make_cp("adv-6.4-%.1f" % val, "cap util at %.1f" % val, "medium", [],
                     [SoftConstraint(type="max_utilization", target="edges:ALL",
                                     value=val, penalty=1.0)])
        r = verifier.verify(cp)
        tests_util.append({"value": val, "expected_valid": expect_valid,
                           "actual_valid": r.valid, "errors": r.errors})
    all_correct = all(t["expected_valid"] == t["actual_valid"] for t in tests_util)
    results.append({
        "id": "adv-6.4", "category": "boundary_exploitation",
        "description": "Utilization boundaries: 0.0/0.5/1.0/1.1",
        "expected": "0.0 rejected, 0.5 accepted, 1.0 accepted, 1.1 rejected",
        "caught": all_correct,
        "sub_results": tests_util,
    })

    # 6.5: Hallucinated region name
    cp = make_cp("adv-6.5", "avoid routing through ATLANTIS region", "medium",
                 [HardConstraint(type="avoid_region", target="region:ATLANTIS", value="ATLANTIS")],
                 flow_selectors=[FlowSelector(src_region="ATLANTIS", dst_region="NYC")])
    r = verifier.verify(cp)
    results.append({
        "id": "adv-6.5", "category": "boundary_exploitation",
        "description": "Hallucinated region name (ATLANTIS)",
        "expected": "reject (unknown region)",
        "caught": not r.valid or any("unknown" in e.lower() for e in r.errors),
        "errors": r.errors, "warnings": r.warnings, "valid": r.valid,
    })

    # ================================================================
    # Summary
    # ================================================================
    total = len(results)
    caught = sum(1 for r in results if r["caught"])
    by_cat = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "caught": 0}
        by_cat[cat]["total"] += 1
        by_cat[cat]["caught"] += 1 if r["caught"] else 0

    summary = {
        "total_tests": total,
        "total_caught": caught,
        "detection_rate": "%.1f%%" % (100.0 * caught / total) if total > 0 else "N/A",
        "by_category": by_cat,
        "tests": results,
    }

    os.makedirs("output", exist_ok=True)
    with open("output/adversarial_safety.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("=" * 60)
    print("ADVERSARIAL SAFETY TEST RESULTS")
    print("=" * 60)
    print("Total tests: %d" % total)
    print("Caught: %d / %d (%.1f%%)" % (caught, total, 100.0 * caught / total))
    print()
    for cat, stats in sorted(by_cat.items()):
        print("  %s: %d/%d caught" % (cat, stats["caught"], stats["total"]))
    print()
    for r in results:
        status = "PASS" if r["caught"] else "FAIL"
        print("[%s] %s: %s" % (status, r["id"], r["description"]))
        if not r["caught"]:
            print("       Expected: %s" % r.get("expected", ""))
            print("       Errors: %s" % r.get("errors", []))
            print("       Warnings: %s" % r.get("warnings", []))


if __name__ == "__main__":
    run_tests()
