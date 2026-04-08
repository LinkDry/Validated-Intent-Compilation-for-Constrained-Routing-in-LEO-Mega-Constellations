"""Generate expanded compositional OOD benchmark (N=5 -> 30+)."""
import json, random

random.seed(2026)

with open("intent/benchmark/benchmark_ood_paraphrases.json") as f:
    ood = json.load(f)

REGIONS = ["NYC", "LONDON", "TOKYO", "SHANGHAI", "MUMBAI", "SAO_PAULO", "SYDNEY", "DUBAI", "FRANKFURT", "PARIS"]
TRAFFIC = ["financial", "emergency", "telemetry", "video", "bulk", "control_plane", "maritime", "military"]
LATENCIES = [30, 40, 50, 60, 80, 100, 150]
NODES = list(range(0, 400, 13))
PLANES = list(range(0, 20))
LAT_THRESHOLDS = [60, 65, 70, 75, 80]
UTIL_CAPS = [50, 60, 70, 80, 90]

def make_cp(iid, text, pri, fsel, hcs, scs, evts=None):
    return {
        "intent_id": iid, "source_text": text, "priority": pri,
        "time_window": {"start": "now", "duration_min": None},
        "flow_selectors": fsel, "edge_selectors": [], "node_selectors": [],
        "hard_constraints": hcs, "soft_constraints": scs,
        "objective_weights": {}, "fallback_policy": "reject_if_hard_infeasible",
        "event_conditions": evts or [],
    }

def fs(tc=None, src=None, dst=None):
    return {"traffic_class": tc, "src_region": src, "dst_region": dst,
            "src_node": None, "dst_node": None, "src_plane": None, "dst_plane": None, "corridor": None}

def hc(typ, target, value=None, cond=None):
    return {"type": typ, "target": target, "value": value, "condition": cond}

def sc(typ, target, value=None, penalty=1.0, cond=None):
    return {"type": typ, "target": target, "value": value, "penalty": penalty, "condition": cond}

new_intents = []
idx = 5

# Type 1: node_disable + avoid_region (5 intents)
t1 = [
    "shut down sat {n} and keep all traffic away from {r}, this is urgent",
    "please disable node {n} and also avoid routing through the {r} area",
    "node {n} offline. additionally bypass {r} region entirely.",
    "take satellite {n} out of the mesh and reroute everything around {r}",
    "kill {n}, avoid {r}",
]
for tmpl in t1:
    n = random.choice(NODES); r = random.choice(REGIONS)
    text = tmpl.format(n=n, r=r)
    cp = make_cp("ood-comp-%d" % idx, text, "high", [],
                 [hc("disable_node", "node:%d" % n), hc("avoid_region", "region:%s" % r, r)], [])
    new_intents.append({"id": "ood_comp_%d" % idx, "category": "compositional",
                        "subcategory": "ood_paraphrase", "intent_text": text,
                        "original_pattern": "node_disable+avoid_region", "constraint_program": cp})
    idx += 1

# Type 2: avoid_latitude + max_latency (5 intents)
t2 = [
    "cut all polar links above {lat} degrees and guarantee {tc} from {s} to {d} stays under {ms}ms",
    "no ISLs above {lat}deg latitude. also {tc} {s} to {d} must be below {ms} milliseconds",
    "disable high-latitude links (>{lat} degrees) and ensure {tc} traffic from {s} to {d} remains within {ms}ms",
    "polar avoidance at {lat} degrees plus {tc} SLA {s} to {d} under {ms}ms please",
    "hey, we need to avoid links above {lat} degrees and also keep {tc} from {s} to {d} under {ms}ms",
]
for tmpl in t2:
    lat = random.choice(LAT_THRESHOLDS); tc = random.choice(TRAFFIC)
    s, d = random.sample(REGIONS, 2); ms = random.choice(LATENCIES)
    text = tmpl.format(lat=lat, tc=tc, s=s, d=d, ms=ms)
    cp = make_cp("ood-comp-%d" % idx, text, "medium", [fs(tc, s, d)],
                 [hc("avoid_latitude", "edges:ALL", float(lat)),
                  hc("max_latency_ms", "flow_selector:0", float(ms))], [])
    new_intents.append({"id": "ood_comp_%d" % idx, "category": "compositional",
                        "subcategory": "ood_paraphrase", "intent_text": text,
                        "original_pattern": "avoid_latitude+latency", "constraint_program": cp})
    idx += 1

# Type 3: plane_disable + node_disable + utilization (5 intents)
t3 = [
    "maintenance on plane {p}, also node {n} is failing, cap utilization at {u}%",
    "take plane {p} offline and disable node {n}. oh and limit util to {u} percent",
    "plane {p} down for maintenance. node {n} also offline. max utilization {u}%.",
    "we need to pull plane {p} and node {n} out of service, and keep link utilization under {u}%",
    "disable orbital plane {p}, shut down satellite {n}, utilization ceiling {u}%",
]
for tmpl in t3:
    p = random.choice(PLANES)
    n = random.choice([x for x in NODES if x // 20 != p])
    u = random.choice(UTIL_CAPS)
    text = tmpl.format(p=p, n=n, u=u)
    cp = make_cp("ood-comp-%d" % idx, text, "medium", [],
                 [hc("disable_plane", "plane:%d" % p), hc("disable_node", "node:%d" % n)],
                 [sc("max_utilization", "edges:ALL", u / 100.0)])
    new_intents.append({"id": "ood_comp_%d" % idx, "category": "compositional",
                        "subcategory": "ood_paraphrase", "intent_text": text,
                        "original_pattern": "plane+node+utilization", "constraint_program": cp})
    idx += 1

# Type 4: reroute_away + avoid_latitude + latency (5 intents)
t4 = [
    "reroute around node {n}, avoid polar above {lat} degrees, and {tc} {s} to {d} under {ms}ms",
    "keep traffic away from satellite {n}. also no links above {lat} degrees. {tc} from {s} to {d} must stay below {ms}ms.",
    "node {n} is compromised - reroute away. cut polar links above {lat} degrees. {tc} SLA: {s} to {d} under {ms}ms",
    "three things: avoid node {n}, disable polar ISLs above {lat} degrees, guarantee {tc} {s} to {d} within {ms}ms",
    "bypass sat {n}, polar cutoff {lat} degrees, {tc} latency {s} to {d} max {ms}ms",
]
for tmpl in t4:
    n = random.choice(NODES); lat = random.choice(LAT_THRESHOLDS)
    tc = random.choice(TRAFFIC); s, d = random.sample(REGIONS, 2); ms = random.choice(LATENCIES)
    text = tmpl.format(n=n, lat=lat, tc=tc, s=s, d=d, ms=ms)
    cp = make_cp("ood-comp-%d" % idx, text, "high", [fs(tc, s, d)],
                 [hc("reroute_away", "node:%d" % n), hc("avoid_latitude", "edges:ALL", float(lat)),
                  hc("max_latency_ms", "flow_selector:0", float(ms))], [])
    new_intents.append({"id": "ood_comp_%d" % idx, "category": "compositional",
                        "subcategory": "ood_paraphrase", "intent_text": text,
                        "original_pattern": "reroute+latitude+latency", "constraint_program": cp})
    idx += 1

# Type 5: dual latency (5 intents)
t5 = [
    "{tc1} from {s1} to {d1} under {ms1}ms and {tc2} from {s2} to {d2} under {ms2}ms",
    "two SLAs: {tc1} {s1} to {d1} under {ms1}ms, {tc2} {s2} to {d2} under {ms2}ms",
    "ensure both {tc1} traffic {s1} to {d1} stays below {ms1}ms and {tc2} traffic {s2} to {d2} stays below {ms2}ms",
    "we need {tc1} {s1} to {d1} under {ms1} milliseconds plus {tc2} {s2} to {d2} under {ms2} milliseconds",
    "latency requirements: {tc1} from {s1} to {d1} max {ms1}ms, {tc2} from {s2} to {d2} max {ms2}ms",
]
for tmpl in t5:
    tc1, tc2 = random.sample(TRAFFIC, 2)
    s1, d1, s2, d2 = random.sample(REGIONS, 4)
    ms1, ms2 = random.sample(LATENCIES, 2)
    text = tmpl.format(tc1=tc1, s1=s1, d1=d1, ms1=ms1, tc2=tc2, s2=s2, d2=d2, ms2=ms2)
    cp = make_cp("ood-comp-%d" % idx, text, "medium", [fs(tc1, s1, d1), fs(tc2, s2, d2)],
                 [hc("max_latency_ms", "flow_selector:0", float(ms1)),
                  hc("max_latency_ms", "flow_selector:1", float(ms2))], [])
    new_intents.append({"id": "ood_comp_%d" % idx, "category": "compositional",
                        "subcategory": "ood_paraphrase", "intent_text": text,
                        "original_pattern": "dual_latency", "constraint_program": cp})
    idx += 1

ood.extend(new_intents)

with open("intent/benchmark/benchmark_ood_expanded.json", "w") as f:
    json.dump(ood, f, indent=2, ensure_ascii=False)

print("Added %d new compositional OOD intents" % len(new_intents))
print("Total OOD benchmark: %d intents" % len(ood))
cats = {}
for item in ood:
    cats[item["category"]] = cats.get(item["category"], 0) + 1
for cat, n in sorted(cats.items()):
    print("  %s: %d" % (cat, n))
pats = {}
for item in new_intents:
    pats[item["original_pattern"]] = pats.get(item["original_pattern"], 0) + 1
print("Pattern distribution:")
for p, n in sorted(pats.items()):
    print("  %s: %d" % (p, n))
