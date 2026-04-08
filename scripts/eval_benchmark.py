"""Evaluate IntentCompiler on the 240-intent benchmark.

Metrics:
  - Compilation rate: passes verification
  - Structural accuracy: correct constraint types
  - Full match: types + targets + values
  - Per-category breakdown
  - Repair rate: needed retries
  - Latency stats
"""

import sys, json, time, math
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
from intent.compiler import IntentCompiler
from intent.verifier import ConstraintVerifier


def numeric_eq(a, b, tol=1e-6):
    """Compare two values with numeric tolerance."""
    if a is None and b is None:
        return True
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return str(a).strip().upper() == str(b).strip().upper()


def semantic_values_match(expected_constraints, got_constraints, key="value"):
    """Compare constraint values with numeric tolerance, order-insensitive."""
    exp_vals = sorted([c.get(key) if isinstance(c, dict) else getattr(c, key, None)
                       for c in expected_constraints], key=lambda x: str(x))
    got_vals = sorted([c.get(key) if isinstance(c, dict) else getattr(c, key, None)
                       for c in got_constraints], key=lambda x: str(x))
    if len(exp_vals) != len(got_vals):
        return False
    return all(numeric_eq(a, b) for a, b in zip(exp_vals, got_vals))

# ── Build verifier (20×20 constellation, distance-based delays) ──
NUM_PLANES, SPP = 20, 20
N = NUM_PLANES * SPP


def haversine_km(lat1, lon1, lat2, lon2, R=6921.0):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


latlon = np.zeros((N, 2), dtype=np.float32)
for p in range(NUM_PLANES):
    for s in range(SPP):
        nid = p * SPP + s
        latlon[nid, 0] = -90 + 180 * s / SPP
        latlon[nid, 1] = -180 + 360 * p / NUM_PLANES

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
# Distance-based ISL delays: haversine distance / 300 km/ms (speed of light)
edge_delays = np.zeros(edge_index.shape[1], dtype=np.float32)
for idx in range(edge_index.shape[1]):
    u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
    edge_delays[idx] = haversine_km(
        latlon[u, 0], latlon[u, 1], latlon[v, 0], latlon[v, 1]) / 300.0

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

# ── Load benchmark ──
with open("intent/benchmark/benchmark_240.json") as f:
    benchmark = json.load(f)

print(f"Loaded {len(benchmark)} benchmark intents")

# ── Evaluate ──
results = []
cat_stats = {}
total_t0 = time.time()

for i, entry in enumerate(benchmark):
    cat = entry["category"]
    intent = entry["intent_text"]
    expected = entry["constraint_program"]

    print(f"\n[{i+1}/240] ({cat}) {intent[:70]}...")

    cr = compiler.compile(intent)

    # Compare with expected
    match_info = {"types_match": False, "targets_match": False, "values_match": False}

    if cr.success and cr.program:
        cp = cr.program
        # Check structural match: constraint types
        exp_hard_types = sorted([h["type"] for h in expected["hard_constraints"]])
        got_hard_types = sorted([h.type for h in cp.hard_constraints])
        exp_soft_types = sorted([s["type"] for s in expected["soft_constraints"]])
        got_soft_types = sorted([s.type for s in cp.soft_constraints])

        match_info["types_match"] = (exp_hard_types == got_hard_types and
                                      exp_soft_types == got_soft_types)

        # Check targets
        exp_hard_targets = sorted([h["target"] for h in expected["hard_constraints"]])
        got_hard_targets = sorted([h.target for h in cp.hard_constraints])
        exp_soft_targets = sorted([s["target"] for s in expected["soft_constraints"]])
        got_soft_targets = sorted([s.target for s in cp.soft_constraints])

        match_info["targets_match"] = (exp_hard_targets == got_hard_targets and
                                        exp_soft_targets == got_soft_targets)

        # Check values (semantic: numeric tolerance, order-insensitive)
        match_info["values_match"] = (
            semantic_values_match(expected["hard_constraints"], cp.hard_constraints) and
            semantic_values_match(expected["soft_constraints"], cp.soft_constraints)
        )

    full_match = all(match_info.values()) and cr.success

    status = "PASS" if cr.success else "FAIL"
    match_str = "FULL" if full_match else ("TYPE" if match_info["types_match"] else "MISS")
    print(f"  {status} | match={match_str} | attempts={cr.attempts} | {cr.latency_ms:.0f}ms")

    result_entry = {
        "id": entry["id"],
        "category": cat,
        "intent": intent,
        "compiled": cr.success,
        "attempts": cr.attempts,
        "latency_ms": cr.latency_ms,
        "types_match": match_info["types_match"],
        "targets_match": match_info["targets_match"],
        "values_match": match_info["values_match"],
        "full_match": full_match,
        "errors": cr.errors,
    }
    results.append(result_entry)

    # Per-category stats
    if cat not in cat_stats:
        cat_stats[cat] = {"total": 0, "compiled": 0, "types": 0,
                          "targets": 0, "full": 0, "attempts": [],
                          "latencies": [], "first_try": 0}
    cs = cat_stats[cat]
    cs["total"] += 1
    cs["compiled"] += int(cr.success)
    cs["types"] += int(match_info["types_match"])
    cs["targets"] += int(match_info["targets_match"])
    cs["full"] += int(full_match)
    cs["attempts"].append(cr.attempts)
    cs["latencies"].append(cr.latency_ms)
    if cr.success and cr.attempts == 1:
        cs["first_try"] += 1

    # Save incrementally every 10 intents
    if (i + 1) % 10 == 0:
        with open("output/benchmark_eval_progress.json", "w") as f:
            json.dump({"done": i + 1, "results": results}, f, indent=2)

total_time = time.time() - total_t0

# ── Summary ──
print(f"\n{'='*70}")
print(f"BENCHMARK EVALUATION COMPLETE — {total_time/60:.1f} minutes")
print(f"{'='*70}")

print(f"\n{'Category':<16} {'N':>4} {'Compiled':>9} {'1st-try':>8} "
      f"{'Types':>7} {'Targets':>8} {'Full':>6} {'Avg ms':>8}")
print("-" * 70)

for cat in ["single", "compositional", "conditional", "infeasible"]:
    if cat not in cat_stats:
        continue
    cs = cat_stats[cat]
    n = cs["total"]
    print(f"{cat:<16} {n:>4} {cs['compiled']/n*100:>8.1f}% "
          f"{cs['first_try']/n*100:>7.1f}% {cs['types']/n*100:>6.1f}% "
          f"{cs['targets']/n*100:>7.1f}% {cs['full']/n*100:>5.1f}% "
          f"{np.mean(cs['latencies']):>7.0f}")

# Overall
total = len(results)
compiled = sum(1 for r in results if r["compiled"])
first_try = sum(1 for r in results if r["compiled"] and r["attempts"] == 1)
types_ok = sum(1 for r in results if r["types_match"])
targets_ok = sum(1 for r in results if r["targets_match"])
full_ok = sum(1 for r in results if r["full_match"])
avg_lat = np.mean([r["latency_ms"] for r in results])

print("-" * 70)
print(f"{'OVERALL':<16} {total:>4} {compiled/total*100:>8.1f}% "
      f"{first_try/total*100:>7.1f}% {types_ok/total*100:>6.1f}% "
      f"{targets_ok/total*100:>7.1f}% {full_ok/total*100:>5.1f}% "
      f"{avg_lat:>7.0f}")

# Save final results
final = {
    "total_time_min": total_time / 60,
    "total_intents": total,
    "compiled_rate": compiled / total,
    "first_try_rate": first_try / total,
    "types_match_rate": types_ok / total,
    "targets_match_rate": targets_ok / total,
    "full_match_rate": full_ok / total,
    "avg_latency_ms": avg_lat,
    "per_category": {
        cat: {
            "total": cs["total"],
            "compiled_rate": cs["compiled"] / cs["total"],
            "first_try_rate": cs["first_try"] / cs["total"],
            "types_match_rate": cs["types"] / cs["total"],
            "targets_match_rate": cs["targets"] / cs["total"],
            "full_match_rate": cs["full"] / cs["total"],
            "avg_latency_ms": float(np.mean(cs["latencies"])),
            "avg_attempts": float(np.mean(cs["attempts"])),
        }
        for cat, cs in cat_stats.items()
    },
    "results": results,
}

Path("output").mkdir(exist_ok=True)
with open("output/benchmark_eval_240.json", "w") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

print(f"\nSaved to output/benchmark_eval_240.json")
