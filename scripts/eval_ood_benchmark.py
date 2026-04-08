"""Evaluate intent compiler on OOD paraphrase benchmark.

Tests robustness to non-template language: synonyms, informal phrasing,
abbreviations, ambiguity, ASR-like noise.

Usage:
  python scripts/eval_ood_benchmark.py
"""

import sys, os, json, time, math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")

import numpy as np
from intent.compiler import IntentCompiler
from intent.verifier import ConstraintVerifier
from intent.schema import ConstraintProgram

NUM_PLANES, SPP = 20, 20
N = NUM_PLANES * SPP


def build_verifier():
    np.random.seed(42)
    latlon = np.zeros((N, 2), dtype=np.float32)
    for p in range(NUM_PLANES):
        for s in range(SPP):
            nid = p * SPP + s
            latlon[nid, 0] = -90 + (s / (SPP - 1)) * 180
            latlon[nid, 1] = -180 + (p / (NUM_PLANES - 1)) * 360

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

    return ConstraintVerifier(
        num_planes=NUM_PLANES, sats_per_plane=SPP,
        edge_index=edge_index, edge_delays=edge_delays,
        neighbor_table=neighbor_table, latlon=latlon,
    )


def numeric_eq(a, b, tol=1e-6):
    if a is None and b is None:
        return True
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return str(a).strip().upper() == str(b).strip().upper()


def evaluate_result(cr, expected):
    match_info = {"types_match": False, "targets_match": False, "values_match": False}
    if not (cr.success and cr.program):
        return match_info

    cp = cr.program
    exp_hard_types = sorted([h["type"] for h in expected["hard_constraints"]])
    got_hard_types = sorted([h.type for h in cp.hard_constraints])
    exp_soft_types = sorted([s["type"] for s in expected["soft_constraints"]])
    got_soft_types = sorted([s.type for s in cp.soft_constraints])
    match_info["types_match"] = (exp_hard_types == got_hard_types and
                                  exp_soft_types == got_soft_types)

    exp_hard_targets = sorted([h["target"] for h in expected["hard_constraints"]])
    got_hard_targets = sorted([h.target for h in cp.hard_constraints])
    exp_soft_targets = sorted([s["target"] for s in expected["soft_constraints"]])
    got_soft_targets = sorted([s.target for s in cp.soft_constraints])
    match_info["targets_match"] = (exp_hard_targets == got_hard_targets and
                                    exp_soft_targets == got_soft_targets)

    exp_vals = sorted([h.get("value") for h in expected["hard_constraints"]], key=str)
    got_vals = sorted([h.value for h in cp.hard_constraints], key=str)
    exp_svals = sorted([s.get("value") for s in expected["soft_constraints"]], key=str)
    got_svals = sorted([s.value for s in cp.soft_constraints], key=str)

    vals_ok = (len(exp_vals) == len(got_vals) and
               all(numeric_eq(a, b) for a, b in zip(exp_vals, got_vals)))
    svals_ok = (len(exp_svals) == len(got_svals) and
                all(numeric_eq(a, b) for a, b in zip(exp_svals, got_svals)))
    match_info["values_match"] = vals_ok and svals_ok

    return match_info


def evaluate_ambiguous(cr):
    """For ambiguous intents, check if it produces a valid, verified program."""
    if not cr.success:
        return {"compiled": False, "verified": False, "has_constraints": False}
    cp = cr.program
    return {
        "compiled": True,
        "verified": True,
        "has_constraints": len(cp.hard_constraints) + len(cp.soft_constraints) > 0,
        "num_hard": len(cp.hard_constraints),
        "num_soft": len(cp.soft_constraints),
    }


def main():
    print("Building verifier...")
    verifier = build_verifier()

    print("Creating compiler...")
    compiler = IntentCompiler(verifier=verifier, max_retries=3)

    print("Loading OOD benchmark...")
    with open("intent/benchmark/benchmark_ood_paraphrases.json") as f:
        benchmark = json.load(f)
    print(f"Loaded {len(benchmark)} OOD intents")

    # Resume support
    out_path = Path("output/ood_eval_results.json")
    done_ids = set()
    results = []
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        results = prev.get("results", [])
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} already done")

    cat_stats = defaultdict(lambda: {
        "total": 0, "compiled": 0, "types_ok": 0, "full_ok": 0, "ambiguous_ok": 0
    })

    for i, entry in enumerate(benchmark):
        if entry["id"] in done_ids:
            # Count existing results in stats
            for r in results:
                if r["id"] == entry["id"]:
                    cat = r["category"]
                    cat_stats[cat]["total"] += 1
                    if r.get("compiled"):
                        cat_stats[cat]["compiled"] += 1
                    if r.get("types_match"):
                        cat_stats[cat]["types_ok"] += 1
                    if r.get("full_match"):
                        cat_stats[cat]["full_ok"] += 1
                    if r.get("ambiguous_ok"):
                        cat_stats[cat]["ambiguous_ok"] += 1
            continue

        cat = entry["category"]
        intent = entry["intent_text"]
        expected = entry.get("constraint_program")

        print(f"  [{len(results)+1}/{len(benchmark)}] ({cat}) {intent[:70]}...")

        cr = compiler.compile(intent)

        result_entry = {
            "id": entry["id"],
            "category": cat,
            "subcategory": entry.get("subcategory", ""),
            "intent": intent,
            "compiled": cr.success,
            "attempts": cr.attempts,
            "latency_ms": cr.latency_ms,
            "errors": cr.errors,
        }

        if cat == "ambiguous" or expected is None:
            amb = evaluate_ambiguous(cr)
            result_entry.update(amb)
            result_entry["ambiguous_ok"] = amb.get("has_constraints", False)
            cat_stats[cat]["total"] += 1
            if cr.success:
                cat_stats[cat]["compiled"] += 1
            if amb.get("has_constraints"):
                cat_stats[cat]["ambiguous_ok"] += 1
            status = "OK" if amb.get("has_constraints") else "WEAK"
            print(f"    {status} | compiled={cr.success} constraints={amb.get('num_hard',0)}h+{amb.get('num_soft',0)}s | {cr.latency_ms:.0f}ms")
        else:
            match_info = evaluate_result(cr, expected)
            full_match = all(match_info.values()) and cr.success
            result_entry.update(match_info)
            result_entry["full_match"] = full_match

            cat_stats[cat]["total"] += 1
            if cr.success:
                cat_stats[cat]["compiled"] += 1
            if match_info["types_match"]:
                cat_stats[cat]["types_ok"] += 1
            if full_match:
                cat_stats[cat]["full_ok"] += 1

            status = "PASS" if cr.success else "FAIL"
            match_str = "FULL" if full_match else ("TYPE" if match_info["types_match"] else "MISS")
            print(f"    {status} | {match_str} | attempts={cr.attempts} | {cr.latency_ms:.0f}ms")

        results.append(result_entry)

        # Save every 5
        if len(results) % 5 == 0:
            _save(results, out_path)

    _save(results, out_path)
    _print_summary(results, cat_stats)


def _save(results, out_path):
    total = len(results)
    if total == 0:
        return
    compiled = sum(1 for r in results if r.get("compiled"))
    non_ambig = [r for r in results if r["category"] != "ambiguous"]
    types_ok = sum(1 for r in non_ambig if r.get("types_match"))
    full_ok = sum(1 for r in non_ambig if r.get("full_match"))

    final = {
        "total_intents": total,
        "compiled_rate": compiled / total,
        "types_match_rate": types_ok / max(len(non_ambig), 1),
        "full_match_rate": full_ok / max(len(non_ambig), 1),
        "avg_latency_ms": float(np.mean([r["latency_ms"] for r in results])),
        "avg_attempts": float(np.mean([r["attempts"] for r in results])),
        "results": results,
    }

    Path("output").mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)


def _print_summary(results, cat_stats):
    total = len(results)
    if total == 0:
        print("No results")
        return

    print(f"\n{'='*70}")
    print("OOD PARAPHRASE EVALUATION SUMMARY")
    print(f"{'='*70}")

    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        n = s["total"]
        if n == 0:
            continue
        comp = s["compiled"] / n * 100
        if cat == "ambiguous":
            ok = s["ambiguous_ok"] / n * 100
            print(f"  {cat:<15s}: {n} intents, compiled={comp:.0f}%, reasonable_output={ok:.0f}%")
        else:
            types = s["types_ok"] / n * 100
            full = s["full_ok"] / n * 100
            print(f"  {cat:<15s}: {n} intents, compiled={comp:.0f}%, types={types:.0f}%, full={full:.0f}%")

    non_ambig = [r for r in results if r["category"] != "ambiguous"]
    if non_ambig:
        comp = sum(1 for r in non_ambig if r.get("compiled")) / len(non_ambig) * 100
        types = sum(1 for r in non_ambig if r.get("types_match")) / len(non_ambig) * 100
        full = sum(1 for r in non_ambig if r.get("full_match")) / len(non_ambig) * 100
        print(f"\n  Overall (non-ambiguous): compiled={comp:.1f}%, types={types:.1f}%, full={full:.1f}%")
        print(f"  vs template benchmark:   compiled=97.9%, types=91.7%, full=86.2%")


if __name__ == "__main__":
    main()
