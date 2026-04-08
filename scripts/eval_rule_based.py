"""Evaluate rule-based parser baseline on the 240-intent benchmark.

Same scoring as eval_benchmark.py but using RuleBasedParser instead of LLM.
"""

import sys, json, time, math
sys.path.insert(0, ".")

import numpy as np
from intent.rule_based_parser import RuleBasedParser


def numeric_eq(a, b, tol=1e-6):
    if a is None and b is None:
        return True
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return str(a).strip().upper() == str(b).strip().upper()


def semantic_values_match(expected_constraints, got_constraints, key="value"):
    exp_vals = sorted([c.get(key) if isinstance(c, dict) else getattr(c, key, None)
                       for c in expected_constraints], key=lambda x: str(x))
    got_vals = sorted([c.get(key) if isinstance(c, dict) else getattr(c, key, None)
                       for c in got_constraints], key=lambda x: str(x))
    if len(exp_vals) != len(got_vals):
        return False
    return all(numeric_eq(a, b) for a, b in zip(exp_vals, got_vals))


def main():
    with open("intent/benchmark/benchmark_240.json") as f:
        benchmark = json.load(f)
    print("Loaded %d benchmark intents" % len(benchmark))

    parser = RuleBasedParser()

    results = []
    cat_stats = {}

    for i, entry in enumerate(benchmark):
        cat = entry["category"]
        intent = entry["intent_text"]
        expected = entry["constraint_program"]

        cr = parser.compile(intent)

        match_info = {"types_match": False, "targets_match": False, "values_match": False}

        if cr.success and cr.program:
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

            match_info["values_match"] = (
                semantic_values_match(expected["hard_constraints"], cp.hard_constraints) and
                semantic_values_match(expected["soft_constraints"], cp.soft_constraints)
            )

        full_match = all(match_info.values()) and cr.success

        status = "PASS" if cr.success else "FAIL"
        match_str = "FULL" if full_match else ("TYPE" if match_info["types_match"] else "MISS")

        if (i + 1) % 40 == 0 or not cr.success:
            print("[%d/240] (%s) %s... %s|%s" % (
                i + 1, cat, intent[:50], status, match_str))

        results.append({
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
        })

        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "compiled": 0, "types": 0,
                              "targets": 0, "full": 0, "latencies": []}
        cs = cat_stats[cat]
        cs["total"] += 1
        cs["compiled"] += int(cr.success)
        cs["types"] += int(match_info["types_match"])
        cs["targets"] += int(match_info["targets_match"])
        cs["full"] += int(full_match)
        cs["latencies"].append(cr.latency_ms)

    # Print summary
    print("\n" + "=" * 80)
    print("RULE-BASED PARSER BASELINE RESULTS")
    print("=" * 80)
    print("%-16s %4s %8s %8s %8s %8s" % (
        "Category", "N", "Compiled", "Types", "Targets", "Full"))
    print("-" * 60)

    for cat in ["single", "compositional", "conditional", "infeasible"]:
        cs = cat_stats.get(cat, {"total": 0, "compiled": 0, "types": 0, "targets": 0, "full": 0})
        n = cs["total"]
        if n == 0:
            continue
        print("%-16s %4d %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (
            cat, n,
            cs["compiled"] / n * 100,
            cs["types"] / n * 100,
            cs["targets"] / n * 100,
            cs["full"] / n * 100,
        ))

    total = len(results)
    compiled = sum(1 for r in results if r["compiled"])
    types_ok = sum(1 for r in results if r["types_match"])
    targets_ok = sum(1 for r in results if r["targets_match"])
    full_ok = sum(1 for r in results if r["full_match"])

    print("-" * 60)
    print("%-16s %4d %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (
        "OVERALL", total,
        compiled / total * 100,
        types_ok / total * 100,
        targets_ok / total * 100,
        full_ok / total * 100,
    ))

    avg_latency = np.mean([r["latency_ms"] for r in results])
    print("\nAvg latency: %.2f ms" % avg_latency)

    # Save results
    summary = {
        "method": "rule_based_parser",
        "total_intents": total,
        "compiled_rate": compiled / total,
        "types_match_rate": types_ok / total,
        "targets_match_rate": targets_ok / total,
        "full_match_rate": full_ok / total,
        "avg_latency_ms": avg_latency,
        "per_category": {},
        "results": results,
    }
    for cat, cs in cat_stats.items():
        n = cs["total"]
        summary["per_category"][cat] = {
            "total": n,
            "compiled_rate": cs["compiled"] / n,
            "types_match_rate": cs["types"] / n,
            "targets_match_rate": cs["targets"] / n,
            "full_match_rate": cs["full"] / n,
            "avg_latency_ms": float(np.mean(cs["latencies"])),
        }

    with open("output/ablation_rule_based.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved to output/ablation_rule_based.json")

    # Comparison with LLM results
    print("\n" + "=" * 80)
    print("COMPARISON: Rule-Based vs LLM (9B) vs LLM (4B)")
    print("=" * 80)
    print("%-16s %12s %12s %12s" % ("Metric", "Rule-Based", "LLM 9B", "LLM 4B"))
    print("-" * 56)
    print("%-16s %11.1f%% %11.1f%% %11.1f%%" % (
        "Compiled", compiled / total * 100, 97.9, 59.6))
    print("%-16s %11.1f%% %11.1f%% %11.1f%%" % (
        "Types Match", types_ok / total * 100, 91.7, 55.4))
    print("%-16s %11.1f%% %11.1f%% %11.1f%%" % (
        "Full Match", full_ok / total * 100, 86.2, 54.2))
    print("%-16s %10.1fms %10.1fms %10.1fms" % (
        "Avg Latency", avg_latency, 15700, 204400))


if __name__ == "__main__":
    main()
