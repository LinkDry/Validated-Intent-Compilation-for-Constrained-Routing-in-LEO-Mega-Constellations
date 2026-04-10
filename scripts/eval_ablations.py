"""Ablation study: evaluate compiler variants on the 240-intent benchmark.

Configs:
  full          — 5-shot + verifier + 3 total attempts (baseline)
  no_verifier   — 5-shot + NO verification (raw LLM output)
  no_repair     — 5-shot + verifier + 1 attempt (no retry)
  zero_shot     — 0-shot + verifier + 3 total attempts
  retry_1       — 5-shot + verifier + 1 retry
  retry_2       — 5-shot + verifier + 2 retries
  model_4b      — 4B model + 5-shot + verifier + 3 total attempts

Usage:
  python scripts/eval_ablations.py full
  python scripts/eval_ablations.py no_verifier
  python scripts/eval_ablations.py --all
"""

import sys, json, time, math, argparse
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
from intent.compiler import IntentCompiler, FEW_SHOT_EXAMPLES, SCHEMA_DESCRIPTION, CompilationResult
from intent.verifier import ConstraintVerifier
from intent.schema import ConstraintProgram


# ── Reuse verifier setup from eval_benchmark ──

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


# ── Evaluation helpers (same as eval_benchmark) ──

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


def evaluate_result(cr, expected):
    """Evaluate a compilation result against expected program."""
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

    return match_info


# ── Compiler variants ──

class NoVerifierCompiler(IntentCompiler):
    """Skip verification — accept any parseable ConstraintProgram."""

    def compile(self, intent_text):
        result = CompilationResult()
        t0 = time.time()
        result.attempts = 1

        messages = [
            {"role": "system", "content": SCHEMA_DESCRIPTION},
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": intent_text},
        ]

        raw = self._call_llm(messages)
        if raw is None:
            result.errors.append("LLM API call failed")
            result.latency_ms = (time.time() - t0) * 1000
            return result

        json_str = self._extract_json(raw)
        if json_str is None:
            result.errors.append("No valid JSON in response")
            result.latency_ms = (time.time() - t0) * 1000
            return result

        result.raw_json = json_str
        try:
            cp = ConstraintProgram.from_json(json_str)
            result.success = True
            result.program = cp
        except Exception as e:
            result.errors.append(f"JSON parse error: {e}")

        result.latency_ms = (time.time() - t0) * 1000
        return result


class ZeroShotCompiler(IntentCompiler):
    """No few-shot examples — system prompt + intent only."""

    def compile(self, intent_text):
        result = CompilationResult()
        t0 = time.time()

        # No few-shot examples
        messages = [
            {"role": "system", "content": SCHEMA_DESCRIPTION},
            {"role": "user", "content": intent_text},
        ]

        for attempt in range(1, self.max_retries + 1):
            result.attempts = attempt

            raw = self._call_llm(messages)
            if raw is None:
                result.errors.append(f"Attempt {attempt}: LLM API call failed")
                continue

            json_str = self._extract_json(raw)
            if json_str is None:
                result.errors.append(f"Attempt {attempt}: No valid JSON")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    "Your response was not valid JSON. Output ONLY the ConstraintProgram JSON."})
                continue

            result.raw_json = json_str
            try:
                cp = ConstraintProgram.from_json(json_str)
            except Exception as e:
                result.errors.append(f"Attempt {attempt}: Parse error: {e}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    f"JSON parsing failed: {e}. Fix and output again."})
                continue

            vr = self.verifier.verify(cp)
            result.verification = vr
            if vr.valid:
                result.success = True
                result.program = cp
                break
            else:
                err_msg = "; ".join(vr.errors[:5])
                result.errors.append(f"Attempt {attempt}: Verification failed: {err_msg}")
                if attempt < self.max_retries:
                    messages.append({"role": "assistant", "content": json_str})
                    messages.append({"role": "user", "content":
                        f"Verifier errors: {err_msg}. Fix and output corrected JSON only."})

        result.latency_ms = (time.time() - t0) * 1000
        return result


def make_compiler(config_name, verifier):
    """Create compiler for a given ablation config."""
    MODEL_9B = "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2"
    MODEL_4B = "qwen3.5-4b-uncensored-hauhaucs-aggressive"

    if config_name == "full":
        return IntentCompiler(verifier=verifier, max_retries=3, model=MODEL_9B)
    elif config_name == "no_verifier":
        return NoVerifierCompiler(verifier=verifier, max_retries=1, model=MODEL_9B)
    elif config_name == "no_repair":
        return IntentCompiler(verifier=verifier, max_retries=1, model=MODEL_9B)
    elif config_name == "zero_shot":
        return ZeroShotCompiler(verifier=verifier, max_retries=3, model=MODEL_9B)
    elif config_name == "retry_1":
        return IntentCompiler(verifier=verifier, max_retries=1, model=MODEL_9B)
    elif config_name == "retry_2":
        return IntentCompiler(verifier=verifier, max_retries=2, model=MODEL_9B)
    elif config_name == "model_4b":
        return IntentCompiler(verifier=verifier, max_retries=3, model=MODEL_4B, timeout=300)
    else:
        raise ValueError(f"Unknown config: {config_name}")


ALL_CONFIGS = ["full", "no_verifier", "no_repair", "zero_shot", "retry_1", "retry_2", "model_4b"]


# ── Main evaluation loop ──

def run_config(config_name, benchmark, verifier):
    """Run one ablation config on the full benchmark."""
    compiler = make_compiler(config_name, verifier)
    out_path = Path(f"output/ablation_{config_name}.json")

    # Resume support
    done_ids = set()
    results = []
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        results = prev.get("results", [])
        done_ids = {r["id"] for r in results}
        print(f"  Resuming: {len(done_ids)} already done")

    cat_stats = {}
    t0_all = time.time()

    for i, entry in enumerate(benchmark):
        if entry["id"] in done_ids:
            continue

        cat = entry["category"]
        intent = entry["intent_text"]
        expected = entry["constraint_program"]

        print(f"  [{len(results)+1}/240] ({cat}) {intent[:60]}...")

        cr = compiler.compile(intent)
        match_info = evaluate_result(cr, expected)
        full_match = all(match_info.values()) and cr.success

        status = "PASS" if cr.success else "FAIL"
        match_str = "FULL" if full_match else ("TYPE" if match_info["types_match"] else "MISS")
        print(f"    {status} | {match_str} | attempts={cr.attempts} | {cr.latency_ms:.0f}ms")

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

        # Save every 10
        if len(results) % 10 == 0:
            _save_results(config_name, results, time.time() - t0_all, out_path)

    total_time = time.time() - t0_all
    _save_results(config_name, results, total_time, out_path)
    _print_summary(config_name, results, total_time)
    return results


def _save_results(config_name, results, total_time, out_path):
    total = len(results)
    if total == 0:
        return
    compiled = sum(1 for r in results if r["compiled"])
    first_try = sum(1 for r in results if r["compiled"] and r["attempts"] == 1)
    types_ok = sum(1 for r in results if r["types_match"])
    targets_ok = sum(1 for r in results if r["targets_match"])
    full_ok = sum(1 for r in results if r["full_match"])

    final = {
        "config": config_name,
        "total_time_min": total_time / 60,
        "total_intents": total,
        "compiled_rate": compiled / total,
        "first_try_rate": first_try / total,
        "types_match_rate": types_ok / total,
        "targets_match_rate": targets_ok / total,
        "full_match_rate": full_ok / total,
        "avg_latency_ms": float(np.mean([r["latency_ms"] for r in results])),
        "avg_attempts": float(np.mean([r["attempts"] for r in results])),
        "results": results,
    }

    Path("output").mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)


def _print_summary(config_name, results, total_time):
    total = len(results)
    if total == 0:
        print(f"  No results for {config_name}")
        return

    compiled = sum(1 for r in results if r["compiled"])
    first_try = sum(1 for r in results if r["compiled"] and r["attempts"] == 1)
    types_ok = sum(1 for r in results if r["types_match"])
    full_ok = sum(1 for r in results if r["full_match"])
    avg_lat = np.mean([r["latency_ms"] for r in results])
    avg_att = np.mean([r["attempts"] for r in results])

    print(f"\n  === {config_name} ({total_time/60:.1f} min) ===")
    print(f"  Compiled:   {compiled}/{total} ({compiled/total*100:.1f}%)")
    print(f"  First-try:  {first_try}/{total} ({first_try/total*100:.1f}%)")
    print(f"  Types OK:   {types_ok}/{total} ({types_ok/total*100:.1f}%)")
    print(f"  Full match: {full_ok}/{total} ({full_ok/total*100:.1f}%)")
    print(f"  Avg latency: {avg_lat:.0f}ms  Avg attempts: {avg_att:.2f}")


def print_comparison_table(configs_done):
    """Print side-by-side comparison of all completed configs."""
    print(f"\n{'='*80}")
    print("ABLATION COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Config':<16} {'Compiled':>9} {'1st-try':>8} {'Types':>7} "
          f"{'Full':>6} {'Avg ms':>8} {'Avg att':>8}")
    print("-" * 80)

    for cfg in ALL_CONFIGS:
        path = Path(f"output/ablation_{cfg}.json")
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        print(f"{cfg:<16} {data['compiled_rate']*100:>8.1f}% "
              f"{data['first_try_rate']*100:>7.1f}% "
              f"{data['types_match_rate']*100:>6.1f}% "
              f"{data['full_match_rate']*100:>5.1f}% "
              f"{data['avg_latency_ms']:>7.0f} "
              f"{data['avg_attempts']:>7.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study on 240-intent benchmark")
    parser.add_argument("configs", nargs="*", default=["full"],
                        help="Config names to run, or --all")
    parser.add_argument("--all", action="store_true", help="Run all configs")
    parser.add_argument("--compare", action="store_true", help="Print comparison table only")
    args = parser.parse_args()

    if args.compare:
        print_comparison_table(ALL_CONFIGS)
        sys.exit(0)

    configs = ALL_CONFIGS if args.all else args.configs

    print("Building verifier...")
    verifier = build_verifier()

    print("Loading benchmark...")
    with open("intent/benchmark/benchmark_240.json") as f:
        benchmark = json.load(f)
    print(f"Loaded {len(benchmark)} intents")

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"Running ablation: {cfg}")
        print(f"{'='*70}")
        run_config(cfg, benchmark, verifier)

    if len(configs) > 1:
        print_comparison_table(configs)
