"""Independent oracle for verifying the 32 routing-infeasible cases.

Uses a completely separate Dijkstra implementation (not Pass 8 internals)
to compute exact shortest-path latency on the constrained topology.
If min_latency > deadline, the case is independently confirmed infeasible.
"""
import sys, os, json, heapq
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent.schema import ConstraintProgram, HardConstraint, SoftConstraint, FlowSelector


# Known regions (independent copy, not imported from verifier)
REGIONS = {
    "NYC": (40.7, -74.0), "LONDON": (51.5, -0.1), "TOKYO": (35.7, 139.7),
    "SYDNEY": (-33.9, 151.2), "SAO_PAULO": (-23.5, -46.6),
    "MUMBAI": (19.1, 72.9), "SHANGHAI": (31.2, 121.5),
    "DUBAI": (25.3, 55.3), "FRANKFURT": (50.1, 8.7), "PARIS": (48.9, 2.3),
    "SINGAPORE": (1.3, 103.8), "SEOUL": (37.6, 127.0),
    "MOSCOW": (55.8, 37.6), "BEIJING": (39.9, 116.4),
    "JOHANNESBURG": (-26.2, 28.0), "CAIRO": (30.0, 31.2),
}


def haversine_km(lat1, lon1, lat2, lon2, R=6921.0):
    """Great-circle distance at orbital altitude."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_constellation(num_planes=20, spp=20):
    """Build constellation with distance-based delays (independent impl)."""
    N = num_planes * spp
    latlon = np.zeros((N, 2))
    for p in range(num_planes):
        for s in range(spp):
            nid = p * spp + s
            latlon[nid, 0] = -90 + 180 * s / spp
            latlon[nid, 1] = -180 + 360 * p / num_planes

    # Build adjacency with delays
    adj = {i: [] for i in range(N)}
    for p in range(num_planes):
        for s in range(spp):
            nid = p * spp + s
            neighbors = [
                p * spp + (s + 1) % spp,
                p * spp + (s - 1) % spp,
                ((p + 1) % num_planes) * spp + s,
                ((p - 1) % num_planes) * spp + s,
            ]
            for nb in neighbors:
                dist = haversine_km(latlon[nid, 0], latlon[nid, 1],
                                    latlon[nb, 0], latlon[nb, 1])
                delay = dist / 300.0  # ms, speed of light
                adj[nid].append((nb, delay))

    return N, latlon, adj


def ground_region(region_name, latlon, radius_deg=15.0):
    """Map region name to nearest satellite node (independent impl)."""
    region_upper = region_name.upper()
    if region_upper not in REGIONS:
        return None
    rlat, rlon = REGIONS[region_upper]
    dlat = latlon[:, 0] - rlat
    dlon = latlon[:, 1] - rlon
    dlon = np.where(dlon > 180, dlon - 360, dlon)
    dlon = np.where(dlon < -180, dlon + 360, dlon)
    dist = np.sqrt(dlat**2 + dlon**2)
    candidates = np.where(dist < radius_deg)[0]
    if len(candidates) == 0:
        return None
    # Return the closest node
    return int(candidates[np.argmin(dist[candidates])])


def apply_constraints(adj, N, latlon, hard_constraints, num_planes=20, spp=20):
    """Apply topology constraints and return constrained adjacency."""
    disabled_nodes = set()
    disabled_edges = set()

    for hc in hard_constraints:
        htype = hc.get("type", "")
        target = hc.get("target", "")
        value = hc.get("value")

        if htype == "disable_node" and target.startswith("node:"):
            nid = int(target.split(":")[1])
            if 0 <= nid < N:
                disabled_nodes.add(nid)

        elif htype == "disable_plane" and target.startswith("plane:"):
            pid = int(target.split(":")[1])
            if 0 <= pid < num_planes:
                for s in range(spp):
                    disabled_nodes.add(pid * spp + s)

        elif htype == "avoid_latitude":
            threshold = float(value) if value is not None else 75.0
            for u in range(N):
                if abs(latlon[u, 0]) > threshold:
                    disabled_nodes.add(u)

        elif htype == "avoid_region":
            region = value if isinstance(value, str) else ""
            region_upper = region.upper()
            if region_upper in REGIONS:
                rlat, rlon = REGIONS[region_upper]
                for u in range(N):
                    dlat = latlon[u, 0] - rlat
                    dlon = latlon[u, 1] - rlon
                    if dlon > 180: dlon -= 360
                    if dlon < -180: dlon += 360
                    if np.sqrt(dlat**2 + dlon**2) < 15.0:
                        disabled_nodes.add(u)

        elif htype == "reroute_away" and target.startswith("node:"):
            nid = int(target.split(":")[1])
            if 0 <= nid < N:
                disabled_nodes.add(nid)

    # Build constrained adjacency
    constrained = {i: [] for i in range(N) if i not in disabled_nodes}
    for u in constrained:
        for (v, delay) in adj[u]:
            if v not in disabled_nodes and (u, v) not in disabled_edges:
                constrained[u].append((v, delay))

    return constrained, disabled_nodes


def independent_dijkstra(adj, src, dst):
    """Standard Dijkstra — completely independent from Pass 8."""
    if src not in adj or dst not in adj:
        return float('inf'), []

    dist = {src: 0.0}
    prev = {src: None}
    heap = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue
        if u == dst:
            # Reconstruct path
            path = []
            node = dst
            while node is not None:
                path.append(node)
                node = prev[node]
            return d, list(reversed(path))
        for (v, w) in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return float('inf'), []


def main():
    N, latlon, adj = build_constellation()

    with open("intent/benchmark/benchmark_240.json") as f:
        benchmark = json.load(f)

    # Load the confusion matrix to find the 32 REJECT feasible cases
    with open("output/verifier_confusion_matrix.json") as f:
        cm = json.load(f)

    reject_ids = set()
    for d in cm["details"]:
        if not d["is_infeasible"] and d["decision"] == "REJECT":
            reject_ids.add(d["id"])

    print(f"Independently verifying {len(reject_ids)} routing-infeasible cases")
    print("=" * 70)

    results = []
    all_confirmed = True

    for entry in benchmark:
        if entry["id"] not in reject_ids:
            continue

        cpd = entry["constraint_program"]
        hcs = cpd.get("hard_constraints", [])
        fss = cpd.get("flow_selectors", [])

        # Apply topology constraints
        constrained_adj, disabled = apply_constraints(
            adj, N, latlon, hcs)

        # Find the flow selector and deadline
        deadline = None
        src_region = dst_region = None
        for hc in hcs:
            if hc["type"] == "max_latency_ms":
                deadline = float(hc["value"])
        for fs in fss:
            src_region = fs.get("src_region")
            dst_region = fs.get("dst_region")

        if deadline is None or src_region is None or dst_region is None:
            results.append({
                "id": entry["id"], "confirmed": False,
                "reason": "missing deadline or regions"})
            all_confirmed = False
            continue

        # Ground regions independently
        src = ground_region(src_region, latlon)
        dst = ground_region(dst_region, latlon)

        if src is None or dst is None:
            results.append({
                "id": entry["id"], "confirmed": False,
                "reason": f"cannot ground regions: {src_region}->{dst_region}"})
            all_confirmed = False
            continue

        if src in disabled or dst in disabled:
            min_latency = float('inf')
            path = []
        else:
            min_latency, path = independent_dijkstra(constrained_adj, src, dst)

        gap = min_latency - deadline
        confirmed = min_latency > deadline

        results.append({
            "id": entry["id"],
            "intent": entry["intent_text"][:70],
            "src_region": src_region, "dst_region": dst_region,
            "src_node": src, "dst_node": dst,
            "deadline_ms": deadline,
            "min_latency_ms": round(min_latency, 2) if min_latency < float('inf') else "unreachable",
            "gap_ms": round(gap, 2) if gap < float('inf') else "inf",
            "hops": len(path) - 1 if path else "N/A",
            "confirmed_infeasible": confirmed,
        })

        if not confirmed:
            all_confirmed = False

        status = "CONFIRMED" if confirmed else "FAILED"
        print(f"[{status}] {entry['id']}: {src_region}->{dst_region} "
              f"deadline={deadline}ms min_path={round(min_latency, 2) if min_latency < float('inf') else 'inf'}ms "
              f"gap={round(gap, 2) if gap < float('inf') else 'inf'}ms")

    confirmed_count = sum(1 for r in results if r.get("confirmed_infeasible"))
    print(f"\n{'=' * 70}")
    print(f"INDEPENDENT VERIFICATION: {confirmed_count}/{len(results)} confirmed infeasible")
    if all_confirmed:
        print("ALL 32 cases independently verified as routing-infeasible.")
    else:
        failed = [r for r in results if not r.get("confirmed_infeasible")]
        print(f"FAILED cases ({len(failed)}):")
        for r in failed:
            print(f"  {r['id']}: {r.get('reason', 'min_latency <= deadline')}")

    # Save results
    output = {
        "total_cases": len(results),
        "confirmed_infeasible": confirmed_count,
        "all_confirmed": all_confirmed,
        "method": "Independent Dijkstra (separate code path from Pass 8)",
        "oracle_type": "exact shortest-path latency on constrained topology",
        "results": results,
    }
    os.makedirs("output", exist_ok=True)
    with open("output/independent_oracle_32.json", "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
