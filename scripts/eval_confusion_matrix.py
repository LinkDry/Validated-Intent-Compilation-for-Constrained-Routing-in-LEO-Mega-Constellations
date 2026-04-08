"""Re-calculate verifier confusion matrix with Pass 8 feasibility certifier.

Verifies all 240 ground-truth constraint programs from the benchmark
and builds a confusion matrix comparing verifier accept/reject with
the feasibility label (category == "infeasible").
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent.schema import (
    ConstraintProgram, HardConstraint, SoftConstraint,
    FlowSelector, TimeWindow, ObjectiveWeights,
)
from intent.verifier import ConstraintVerifier


def haversine_km(lat1, lon1, lat2, lon2, R=6921.0):
    """Great-circle distance at orbital altitude (R = Earth_R + 550km)."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def make_constellation(num_planes=20, spp=20):
    """Create constellation with distance-based ISL delays."""
    N = num_planes * spp
    edges_u, edges_v = [], []
    for p in range(num_planes):
        for s in range(spp):
            nid = p * spp + s
            n_up = p * spp + (s + 1) % spp
            n_dn = p * spp + (s - 1) % spp
            edges_u.extend([nid, nid]); edges_v.extend([n_up, n_dn])
            n_right = ((p + 1) % num_planes) * spp + s
            n_left = ((p - 1) % num_planes) * spp + s
            edges_u.extend([nid, nid]); edges_v.extend([n_right, n_left])
    edge_index = np.array([edges_u, edges_v])
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
    latlon = np.zeros((N, 2))
    for p in range(num_planes):
        for s in range(spp):
            nid = p * spp + s
            latlon[nid, 0] = -90 + 180 * s / spp
            latlon[nid, 1] = -180 + 360 * p / num_planes

    # Distance-based delays: ISL speed ≈ 300 km/ms (free-space optical)
    edge_delays = np.zeros(len(edges_u))
    for idx in range(len(edges_u)):
        u, v = edges_u[idx], edges_v[idx]
        dist = haversine_km(latlon[u, 0], latlon[u, 1],
                            latlon[v, 0], latlon[v, 1])
        edge_delays[idx] = dist / 300.0  # ms

    return edge_index, edge_delays, neighbor_table, latlon


def parse_cp(entry):
    """Parse benchmark entry into ConstraintProgram."""
    cpd = entry["constraint_program"]
    cp = ConstraintProgram()
    cp.intent_id = cpd.get("intent_id", entry["id"])
    cp.source_text = cpd.get("source_text", entry.get("intent_text", ""))
    cp.priority = cpd.get("priority", "medium")

    cp.hard_constraints = []
    for hc in cpd.get("hard_constraints", []):
        cp.hard_constraints.append(HardConstraint(
            type=hc["type"], target=hc["target"],
            value=hc.get("value"), condition=hc.get("condition")))

    cp.soft_constraints = []
    for sc in cpd.get("soft_constraints", []):
        cp.soft_constraints.append(SoftConstraint(
            type=sc["type"], target=sc.get("target", "edges:ALL"),
            value=sc.get("value"), penalty=sc.get("penalty", 1.0)))

    cp.flow_selectors = []
    for fs in cpd.get("flow_selectors", []):
        cp.flow_selectors.append(FlowSelector(
            traffic_class=fs.get("traffic_class"),
            src_region=fs.get("src_region"),
            dst_region=fs.get("dst_region"),
            src_node=fs.get("src_node"),
            dst_node=fs.get("dst_node"),
            src_plane=fs.get("src_plane"),
            dst_plane=fs.get("dst_plane"),
        ))

    return cp


def main():
    edge_index, edge_delays, neighbor_table, latlon = make_constellation()
    verifier = ConstraintVerifier(
        num_planes=20, sats_per_plane=20,
        edge_index=edge_index, edge_delays=edge_delays,
        neighbor_table=neighbor_table, latlon=latlon,
    )

    with open("intent/benchmark/benchmark_240.json") as f:
        benchmark = json.load(f)

    # Category-level tracking
    cats = {"single": [], "compositional": [], "conditional": [], "infeasible": []}
    all_results = []

    for entry in benchmark:
        cat = entry["category"]
        is_infeasible = (cat == "infeasible")
        cp = parse_cp(entry)
        r = verifier.verify(cp)

        accepted = r.valid
        cert = r.certification_status

        result = {
            "id": entry["id"],
            "category": cat,
            "intent": entry["intent_text"],
            "is_infeasible": is_infeasible,
            "verifier_accepted": accepted,
            "certification_status": cert,
            "errors": r.errors,
            "warnings": r.warnings,
            "feasibility_details": r.feasibility_details,
        }
        all_results.append(result)
        if cat in cats:
            cats[cat].append(result)

    # 3-way classification: ACCEPT / REJECT / ABSTAIN
    for result in all_results:
        cert = result["certification_status"]
        has_errors = len(result["errors"]) > 0
        if has_errors or cert == "rejected":
            result["decision"] = "REJECT"
        elif cert == "accepted":
            result["decision"] = "ACCEPT"
        elif cert == "abstain":
            result["decision"] = "ABSTAIN"
        else:
            # pending (Pass 8 didn't run due to earlier errors)
            result["decision"] = "REJECT"

    # Build 3-way matrix
    def build_3way(results):
        m = {"ACCEPT": 0, "REJECT": 0, "ABSTAIN": 0}
        for r in results:
            m[r["decision"]] += 1
        return m

    overall_3way = build_3way(all_results)
    by_cat_3way = {}
    for cat, results in cats.items():
        m = build_3way(results)
        m["total"] = len(results)
        by_cat_3way[cat] = m

    feasible_results = [r for r in all_results if not r["is_infeasible"]]
    infeasible_results = [r for r in all_results if r["is_infeasible"]]

    f3 = build_3way(feasible_results)
    i3 = build_3way(infeasible_results)

    # Safety: unsafe acceptance = infeasible intents that are ACCEPT
    unsafe_accept_feasible = sum(1 for r in feasible_results
                                 if r["decision"] == "ACCEPT" and r["is_infeasible"])
    unsafe_accept_infeasible = i3["ACCEPT"]
    total_unsafe = unsafe_accept_infeasible

    # Coverage: decided (ACCEPT+REJECT) vs undecided (ABSTAIN)
    total_decided = overall_3way["ACCEPT"] + overall_3way["REJECT"]
    total_abstain = overall_3way["ABSTAIN"]
    coverage_rate = round(100.0 * total_decided / len(all_results), 1)

    # Fragment breakdown for certified programs
    fragment_counts = {}
    for r in all_results:
        if r["decision"] == "ACCEPT":
            for detail in r.get("feasibility_details", []):
                for frag in ["F1", "F2", "F3", "F4", "F5"]:
                    if frag in detail:
                        fragment_counts[frag] = fragment_counts.get(frag, 0) + 1

    # Certification breakdown for infeasible
    cert_counts = {}
    for r in infeasible_results:
        cs = r["certification_status"]
        cert_counts[cs] = cert_counts.get(cs, 0) + 1

    output = {
        "three_way_matrix": {
            "overall": overall_3way,
            "by_category": by_cat_3way,
            "feasible": f3,
            "infeasible": i3,
        },
        "safety": {
            "infeasible_unsafe_accept": unsafe_accept_infeasible,
            "infeasible_unsafe_accept_rate": round(
                100.0 * unsafe_accept_infeasible / len(infeasible_results), 1
            ) if infeasible_results else 0,
            "feasible_unsafe_accept": 0,
            "note": "ABSTAIN counts as non-accept (safe)",
        },
        "coverage": {
            "decided": total_decided,
            "abstain": total_abstain,
            "total": len(all_results),
            "coverage_rate": coverage_rate,
        },
        "fragment_breakdown": fragment_counts,
        "infeasible_certification": cert_counts,
        "overall_total": len(all_results),
        "pass_8_note": "8-pass pipeline with constructive feasibility certification (F1-F5)",
        "details": all_results,
    }

    os.makedirs("output", exist_ok=True)
    with open("output/verifier_confusion_matrix.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print("=" * 60)
    print("VERIFIER 3-WAY CONFUSION MATRIX (with Pass 8)")
    print("=" * 60)
    print(f"Total intents: {len(all_results)}")
    print(f"\n{'':15s} {'ACCEPT':>8s} {'REJECT':>8s} {'ABSTAIN':>8s}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    for cat in ["single", "compositional", "conditional", "infeasible"]:
        m = by_cat_3way[cat]
        print(f"{cat:15s} {m['ACCEPT']:8d} {m['REJECT']:8d} {m['ABSTAIN']:8d}  (n={m['total']})")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'TOTAL':15s} {overall_3way['ACCEPT']:8d} {overall_3way['REJECT']:8d} {overall_3way['ABSTAIN']:8d}  (n={len(all_results)})")

    print(f"\n--- Safety Analysis ---")
    print(f"Infeasible ACCEPT (unsafe): {i3['ACCEPT']}/{len(infeasible_results)} = "
          f"{100.0 * i3['ACCEPT'] / len(infeasible_results):.1f}%")
    print(f"Infeasible REJECT (safe):   {i3['REJECT']}/{len(infeasible_results)}")
    print(f"Infeasible ABSTAIN (safe):  {i3['ABSTAIN']}/{len(infeasible_results)}")

    print(f"\n--- Coverage Analysis ---")
    print(f"Decided (ACCEPT+REJECT): {total_decided}/{len(all_results)} ({coverage_rate}%)")
    print(f"Abstain: {total_abstain}/{len(all_results)} ({100-coverage_rate:.1f}%)")

    print(f"\n--- Feasible Intents ({len(feasible_results)}) ---")
    print(f"  ACCEPT: {f3['ACCEPT']}  REJECT: {f3['REJECT']}  ABSTAIN: {f3['ABSTAIN']}")
    print(f"  Routing-infeasible (REJECT by Pass 8): {f3['REJECT']}")

    print(f"\n--- Infeasible Intents ({len(infeasible_results)}) ---")
    print(f"  ACCEPT: {i3['ACCEPT']}  REJECT: {i3['REJECT']}  ABSTAIN: {i3['ABSTAIN']}")
    print(f"  Certification breakdown: {cert_counts}")

    if fragment_counts:
        print(f"\n--- Fragment Breakdown (certified programs) ---")
        for frag, count in sorted(fragment_counts.items()):
            print(f"  {frag}: {count}")

    # Show ABSTAIN details
    abstain_list = [r for r in all_results if r["decision"] == "ABSTAIN"]
    if abstain_list:
        print(f"\n--- ABSTAIN cases ({len(abstain_list)}) ---")
        for r in abstain_list:
            print(f"  {r['id']}: {r['intent'][:60]}...")


if __name__ == "__main__":
    main()
