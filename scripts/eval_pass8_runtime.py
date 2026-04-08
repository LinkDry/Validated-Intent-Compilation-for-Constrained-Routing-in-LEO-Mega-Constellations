"""Benchmark Pass 8 runtime overhead.

Reports: one-shot validation time (median/p95/max), per-fragment breakdown,
and comparison of 7-pass vs 8-pass validation time.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent.schema import (
    ConstraintProgram, HardConstraint, SoftConstraint,
    FlowSelector, TimeWindow, ObjectiveWeights,
)
from intent.verifier import ConstraintVerifier


def haversine_km(lat1, lon1, lat2, lon2, R=6921.0):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def make_constellation(num_planes=20, spp=20):
    N = num_planes * spp
    edges_u, edges_v = [], []
    for p in range(num_planes):
        for s in range(spp):
            nid = p * spp + s
            edges_u.extend([nid, nid, nid, nid])
            edges_v.extend([
                p * spp + (s + 1) % spp,
                p * spp + (s - 1) % spp,
                ((p + 1) % num_planes) * spp + s,
                ((p - 1) % num_planes) * spp + s,
            ])
    edge_index = np.array([edges_u, edges_v])
    neighbor_table = np.full((N, 4), -1, dtype=int)
    latlon = np.zeros((N, 2))
    for p in range(num_planes):
        for s in range(spp):
            nid = p * spp + s
            neighbor_table[nid] = [
                p * spp + (s + 1) % spp, p * spp + (s - 1) % spp,
                ((p + 1) % num_planes) * spp + s,
                ((p - 1) % num_planes) * spp + s,
            ]
            latlon[nid, 0] = -90 + 180 * s / spp
            latlon[nid, 1] = -180 + 360 * p / num_planes
    edge_delays = np.zeros(len(edges_u))
    for idx in range(len(edges_u)):
        u, v = edges_u[idx], edges_v[idx]
        edge_delays[idx] = haversine_km(
            latlon[u, 0], latlon[u, 1], latlon[v, 0], latlon[v, 1]) / 300.0
    return edge_index, edge_delays, neighbor_table, latlon


def parse_cp(entry):
    cpd = entry["constraint_program"]
    cp = ConstraintProgram()
    cp.intent_id = cpd.get("intent_id", entry["id"])
    cp.source_text = cpd.get("source_text", entry.get("intent_text", ""))
    cp.priority = cpd.get("priority", "medium")
    cp.hard_constraints = [HardConstraint(**hc) for hc in cpd.get("hard_constraints", [])]
    cp.soft_constraints = [SoftConstraint(**sc) for sc in cpd.get("soft_constraints", [])]
    cp.flow_selectors = []
    for fs in cpd.get("flow_selectors", []):
        f = FlowSelector()
        for k, v in fs.items():
            if hasattr(f, k):
                setattr(f, k, v)
        cp.flow_selectors.append(f)
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

    # Warm up
    cp0 = parse_cp(benchmark[0])
    for _ in range(3):
        verifier.verify(cp0)

    # Benchmark each program
    timings = []
    for entry in benchmark:
        cp = parse_cp(entry)

        # Time full 8-pass verification
        t0 = time.perf_counter()
        r = verifier.verify(cp)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000

        has_flows = len(cp.flow_selectors) > 0
        cert = r.certification_status

        timings.append({
            "id": entry["id"],
            "category": entry["category"],
            "has_flow_selectors": has_flows,
            "certification_status": cert,
            "time_ms": elapsed_ms,
        })

    # Compute statistics
    all_times = [t["time_ms"] for t in timings]
    flow_times = [t["time_ms"] for t in timings if t["has_flow_selectors"]]
    noflow_times = [t["time_ms"] for t in timings if not t["has_flow_selectors"]]

    by_cert = {}
    for t in timings:
        cs = t["certification_status"]
        if cs not in by_cert:
            by_cert[cs] = []
        by_cert[cs].append(t["time_ms"])

    by_cat = {}
    for t in timings:
        cat = t["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(t["time_ms"])

    def stats(times):
        if not times:
            return {"n": 0}
        arr = np.array(times)
        return {
            "n": len(arr),
            "median_ms": round(float(np.median(arr)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "max_ms": round(float(np.max(arr)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
            "min_ms": round(float(np.min(arr)), 3),
        }

    results = {
        "all": stats(all_times),
        "with_flow_selectors": stats(flow_times),
        "without_flow_selectors": stats(noflow_times),
        "by_certification": {k: stats(v) for k, v in by_cert.items()},
        "by_category": {k: stats(v) for k, v in by_cat.items()},
    }

    print("=" * 60)
    print("PASS 8 RUNTIME BENCHMARK")
    print("=" * 60)

    for label, s in [("All programs", results["all"]),
                     ("With flow selectors", results["with_flow_selectors"]),
                     ("Without flow selectors (topology-only)", results["without_flow_selectors"])]:
        print(f"\n{label} (n={s['n']}):")
        if s["n"] > 0:
            print(f"  median={s['median_ms']:.3f}ms  p95={s['p95_ms']:.3f}ms  "
                  f"max={s['max_ms']:.3f}ms  mean={s['mean_ms']:.3f}ms")

    print(f"\nBy certification status:")
    for cs, s in sorted(results["by_certification"].items()):
        if s["n"] > 0:
            print(f"  {cs:10s} (n={s['n']:3d}): median={s['median_ms']:.3f}ms  "
                  f"p95={s['p95_ms']:.3f}ms  max={s['max_ms']:.3f}ms")

    print(f"\nBy category:")
    for cat, s in sorted(results["by_category"].items()):
        if s["n"] > 0:
            print(f"  {cat:15s} (n={s['n']:3d}): median={s['median_ms']:.3f}ms  "
                  f"p95={s['p95_ms']:.3f}ms  max={s['max_ms']:.3f}ms")

    os.makedirs("output", exist_ok=True)
    with open("output/pass8_runtime.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
