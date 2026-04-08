"""Programmatic intent-JSON pair generator for QLoRA fine-tuning.

Generates 15k-20k diverse (natural_language, ConstraintProgram_JSON) pairs.
Distribution:
  - 45% single-clause
  - 20% two-constraint compositional
  - 15% three+ constraint compositional
  - 10% conditional (event-triggered)
  - 10% infeasible / conflicting

Each pair is: {"instruction": <intent_text>, "response": <JSON string>}
"""

import json, random, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ── Constants ──

REGIONS = ["NYC", "LONDON", "TOKYO", "SHANGHAI", "MUMBAI",
           "SAO_PAULO", "SYDNEY", "DUBAI", "FRANKFURT", "PARIS",
           "MADRID", "HONOLULU"]
TRAFFIC = ["financial", "emergency", "telemetry", "video",
           "bulk", "control_plane", "maritime", "military"]
CORRIDORS = ["transatlantic", "transpacific", "europe_us", "asia_europe"]
EVENTS = [
    ("solar_storm", ["solar storm", "geomagnetic storm", "solar event",
                     "solar flare", "space weather event"]),
    ("node_failure", ["node failure", "satellite failure", "sat malfunction",
                      "hardware failure on a satellite", "node going down"]),
    ("gateway_failure", ["gateway failure", "ground station outage",
                         "gateway going offline", "ground link failure"]),
    ("maintenance", ["scheduled maintenance", "maintenance window",
                     "planned downtime", "maintenance operation"]),
    ("overload", ["traffic overload", "congestion event", "capacity overload",
                  "bandwidth saturation", "link congestion"]),
]

NUM_PLANES = 20
SPP = 20
N = NUM_PLANES * SPP

# ── Natural language templates ──
# Each template list has many phrasings per constraint type.

TPL_DISABLE_NODE = [
    "Take satellite {nid} offline",
    "Disable node {nid}",
    "Remove sat {nid} from the routing mesh",
    "Pull satellite {nid} out of service",
    "Shut down node {nid}",
    "Node {nid} needs to go offline now",
    "Deactivate satellite {nid}",
    "Exclude node {nid} from all routing paths",
    "Mark satellite {nid} as unavailable",
    "Stop routing through node {nid}",
    "Satellite {nid} is faulty, take it down",
    "Kill routing for sat {nid}",
    "Node {nid} offline please",
    "Remove node {nid} from the constellation mesh",
    "Isolate satellite {nid} from the network",
]

TPL_DISABLE_PLANE = [
    "Disable orbital plane {pid}",
    "Take plane {pid} offline",
    "Remove all satellites in plane {pid} from routing",
    "Shut down the entire plane {pid}",
    "Plane {pid} is undergoing maintenance, disable it",
    "Pull plane {pid} out of the mesh",
    "Deactivate all sats in orbital plane {pid}",
    "Plane {pid} offline for orbit correction",
    "Exclude plane {pid} from routing decisions",
    "Ground plane {pid}",
    "Orbital plane {pid} needs to be disabled",
    "Take the whole plane {pid} down",
    "Plane {pid} out of service",
    "Disable every satellite on plane {pid}",
    "No traffic through plane {pid}",
]

TPL_AVOID_REGION = [
    "Avoid routing through the {reg} region",
    "Do not use links near {reg}",
    "Keep traffic away from {reg}",
    "Reroute all flows around {reg}",
    "Bypass the {reg} area",
    "No paths through {reg}",
    "Exclude {reg} from all routing",
    "Stay away from {reg} coverage zone",
    "Avoid {reg} airspace for all traffic",
    "Block routing near {reg}",
    "Route around {reg}",
    "Skip {reg} region entirely",
    "{reg} is restricted, avoid it",
    "Do not transit through {reg}",
    "Keep all flows clear of {reg}",
]

TPL_AVOID_LAT = [
    "Avoid polar links above {lat} degrees latitude",
    "Disable ISLs in polar regions above {lat}°",
    "No routing through latitudes higher than {lat}",
    "Keep traffic below {lat}° latitude",
    "Polar avoidance: cut links above {lat} degrees",
    "Avoid high-latitude links beyond {lat}°",
    "Disable polar crossings above {lat} degrees",
    "No inter-satellite links above {lat}° lat",
    "Stay below {lat} degrees latitude for all paths",
    "Polar link shutdown above {lat}°",
    "Cut all ISLs above {lat} degrees",
    "Avoid the polar cap above {lat}°",
    "No routing in polar zone above latitude {lat}",
    "Disable links crossing {lat}° latitude",
    "Polar dropout threshold at {lat} degrees",
]

TPL_MAX_LATENCY = [
    "Ensure {tc} traffic from {src} to {dst} stays under {ms}ms",
    "Max latency {ms}ms for {tc} flows between {src} and {dst}",
    "{tc} traffic {src} to {dst} must not exceed {ms}ms",
    "SLA: {ms}ms latency cap on {tc} from {src} to {dst}",
    "Keep {src}-{dst} {tc} latency below {ms} milliseconds",
    "Guarantee {ms}ms or less for {tc} between {src} and {dst}",
    "{tc} from {src} to {dst}: hard limit {ms}ms",
    "Latency budget for {tc} {src}-{dst}: {ms}ms max",
    "Cap end-to-end delay at {ms}ms for {tc} traffic from {src} to {dst}",
    "No more than {ms}ms for {tc} flows {src} to {dst}",
    "{ms}ms deadline on {tc} between {src} and {dst}",
    "Enforce {ms}ms latency SLA for {tc} from {src} to {dst}",
    "{tc} {src}-{dst} path must be under {ms}ms",
    "Hard latency constraint: {tc} {src} to {dst} <= {ms}ms",
    "Route {tc} from {src} to {dst} within {ms}ms",
]

TPL_REROUTE = [
    "Reroute all traffic away from node {nid}",
    "Node {nid} is degraded, shift traffic elsewhere",
    "Avoid using satellite {nid} as a relay",
    "Divert flows around node {nid}",
    "Node {nid} showing errors, reroute away",
    "Move all paths off node {nid}",
    "Redirect traffic that goes through node {nid}",
    "Node {nid} is overloaded, reroute",
    "Shift all flows away from sat {nid}",
    "Bypass node {nid} for all traffic",
    "Stop using node {nid} as transit",
    "Node {nid} degraded performance, avoid it",
    "Clear node {nid} of all transit traffic",
    "No more relay through satellite {nid}",
    "Evacuate traffic from node {nid}",
]

TPL_UTIL = [
    "Cap link utilization at {pct}%",
    "Keep all ISL utilization below {pct} percent",
    "No link should exceed {pct}% capacity",
    "Set a {pct}% utilization ceiling on all edges",
    "Limit bandwidth usage to {pct}% across the mesh",
    "Max {pct}% utilization on every link",
    "Utilization cap: {pct}% for all ISLs",
    "Keep link loads under {pct}%",
    "Do not let any link go above {pct}% usage",
    "Global utilization limit: {pct}%",
    "Cap all edges at {pct}% bandwidth",
    "Enforce {pct}% max on link utilization",
    "No ISL above {pct}% load",
    "Throttle to {pct}% utilization everywhere",
    "Set utilization threshold at {pct}%",
]

TPL_BALANCE = [
    "Balance load evenly across all inter-plane links",
    "Minimize traffic concentration on any single link",
    "Spread traffic as uniformly as possible",
    "Enable load balancing across the constellation",
    "Equalize utilization across orbital planes",
    "Distribute traffic evenly across the mesh",
    "Avoid hotspots, balance the load",
    "Even out link utilization across the network",
    "Load-balance all flows",
    "Smooth out traffic distribution",
]

# ── Compositional connectors ──

CONNECTORS_2 = [
    "{a} and {b}",
    "{a}, and also {b}",
    "{a}. Additionally, {b}",
    "{a}. At the same time, {b}",
    "{a} while ensuring {b}",
    "{a}; also {b}",
    "I need two things: {a}, and {b}",
    "{a}. On top of that, {b}",
]

CONNECTORS_3 = [
    "{a}, {b}, and {c}",
    "{a}. Also {b}. Finally, {c}",
    "Three requirements: {a}; {b}; {c}",
    "{a}, plus {b}, and {c}",
    "First, {a}. Second, {b}. Third, {c}",
    "{a}. Additionally, {b} and {c}",
]

# ── Event preambles ──

TPL_EVENT_PRE = [
    "If a {evt} occurs, ",
    "During a {evt}, ",
    "When {evt} is detected, ",
    "In case of {evt}, ",
    "Should a {evt} happen, ",
    "Upon {evt} detection, ",
    "If we detect a {evt}, ",
    "When there is a {evt}, ",
    "In the event of a {evt}, ",
    "Triggered by {evt}: ",
]


# ── Generator functions ──

def rand_node():
    return random.randint(0, N - 1)

def rand_plane():
    return random.randint(0, NUM_PLANES - 1)

def rand_region():
    return random.choice(REGIONS)

def rand_tc():
    return random.choice(TRAFFIC)

def rand_latency():
    return random.choice([10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200])

def rand_lat_thresh():
    return random.choice([60, 65, 70, 75, 80, 85])

def rand_util():
    return random.choice([0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

def rand_priority():
    return random.choice(["critical", "high", "medium", "low"])


def base_cp(iid, text, priority="medium"):
    return {
        "intent_id": iid,
        "source_text": text,
        "priority": priority,
        "time_window": {"start": "now", "duration_min": None},
        "flow_selectors": [],
        "edge_selectors": [],
        "node_selectors": [],
        "hard_constraints": [],
        "soft_constraints": [],
        "objective_weights": {},
        "fallback_policy": "reject_if_hard_infeasible",
        "event_conditions": [],
    }


# Each gen_* returns (text_fragment, cp_updates_dict)
# cp_updates_dict has keys: hard_constraints, soft_constraints, flow_selectors, etc.

def gen_disable_node() -> Tuple[str, Dict]:
    nid = rand_node()
    text = random.choice(TPL_DISABLE_NODE).format(nid=nid)
    upd = {"hard_constraints": [{"type": "disable_node", "target": f"node:{nid}",
                                  "value": None, "condition": None}]}
    return text, upd

def gen_disable_plane() -> Tuple[str, Dict]:
    pid = rand_plane()
    text = random.choice(TPL_DISABLE_PLANE).format(pid=pid)
    upd = {"hard_constraints": [{"type": "disable_plane", "target": f"plane:{pid}",
                                  "value": None, "condition": None}]}
    return text, upd

def gen_avoid_region() -> Tuple[str, Dict]:
    reg = rand_region()
    text = random.choice(TPL_AVOID_REGION).format(reg=reg)
    upd = {"hard_constraints": [{"type": "avoid_region", "target": "edges:ALL",
                                  "value": reg, "condition": None}]}
    return text, upd

def gen_avoid_lat() -> Tuple[str, Dict]:
    lat = rand_lat_thresh()
    text = random.choice(TPL_AVOID_LAT).format(lat=lat)
    upd = {"hard_constraints": [{"type": "avoid_latitude", "target": "edges:ALL",
                                  "value": lat, "condition": None}]}
    return text, upd

def gen_max_latency() -> Tuple[str, Dict]:
    src, dst = random.sample(REGIONS, 2)
    tc = rand_tc()
    ms = rand_latency()
    text = random.choice(TPL_MAX_LATENCY).format(tc=tc, src=src, dst=dst, ms=ms)
    upd = {
        "flow_selectors": [{"traffic_class": tc, "src_region": src, "dst_region": dst}],
        "hard_constraints": [{"type": "max_latency_ms", "target": "flow_selector:0",
                               "value": ms, "condition": None}],
    }
    return text, upd

def gen_reroute() -> Tuple[str, Dict]:
    nid = rand_node()
    text = random.choice(TPL_REROUTE).format(nid=nid)
    upd = {"hard_constraints": [{"type": "reroute_away", "target": f"node:{nid}",
                                  "value": None, "condition": None}]}
    return text, upd

def gen_util() -> Tuple[str, Dict]:
    cap = rand_util()
    pct = int(cap * 100)
    text = random.choice(TPL_UTIL).format(pct=pct)
    upd = {"soft_constraints": [{"type": "max_utilization", "target": "edges:ALL",
                                  "value": cap, "penalty": round(random.uniform(1.0, 3.0), 1),
                                  "condition": None}]}
    return text, upd

def gen_balance() -> Tuple[str, Dict]:
    text = random.choice(TPL_BALANCE)
    upd = {"soft_constraints": [{"type": "load_balance", "target": "edges:ALL",
                                  "value": None, "penalty": 1.0, "condition": None}]}
    return text, upd


SINGLE_GENERATORS = [
    (gen_disable_node, 0.15),
    (gen_disable_plane, 0.12),
    (gen_avoid_region, 0.12),
    (gen_avoid_lat, 0.10),
    (gen_max_latency, 0.20),
    (gen_reroute, 0.12),
    (gen_util, 0.12),
    (gen_balance, 0.07),
]

def pick_generator():
    r = random.random()
    cum = 0
    for gen, w in SINGLE_GENERATORS:
        cum += w
        if r < cum:
            return gen
    return SINGLE_GENERATORS[0][0]


def merge_updates(cp: Dict, upd: Dict):
    """Merge constraint updates into a ConstraintProgram dict."""
    for key in ["hard_constraints", "soft_constraints", "flow_selectors",
                "edge_selectors", "node_selectors", "event_conditions"]:
        if key in upd:
            # Fix flow_selector indices for additional flows
            if key == "hard_constraints" and "flow_selectors" in upd:
                existing_flows = len(cp.get("flow_selectors", []))
                for hc in upd[key]:
                    if hc["target"].startswith("flow_selector:"):
                        idx = int(hc["target"].split(":")[1])
                        hc["target"] = f"flow_selector:{existing_flows + idx}"
            cp.setdefault(key, []).extend(upd[key])


def apply_condition(upd: Dict, cond: Dict):
    """Add event condition to all constraints in an update."""
    for hc in upd.get("hard_constraints", []):
        hc["condition"] = cond
    for sc in upd.get("soft_constraints", []):
        sc["condition"] = cond


def generate_single(idx: int) -> Dict:
    gen = pick_generator()
    text, upd = gen()
    pri = rand_priority()
    cp = base_cp(f"train_single_{idx}", text, pri)
    merge_updates(cp, upd)
    return {"instruction": text, "response": json.dumps(cp, ensure_ascii=False)}


def generate_comp2(idx: int) -> Dict:
    g1, g2 = random.sample([g for g, _ in SINGLE_GENERATORS], 2)
    t1, u1 = g1()
    t2, u2 = g2()
    connector = random.choice(CONNECTORS_2)
    text = connector.format(a=t1[0].lower() + t1[1:], b=t2[0].lower() + t2[1:])
    # Capitalize first letter
    text = text[0].upper() + text[1:]
    pri = random.choice(["critical", "high", "high", "medium"])
    cp = base_cp(f"train_comp2_{idx}", text, pri)
    merge_updates(cp, u1)
    merge_updates(cp, u2)
    return {"instruction": text, "response": json.dumps(cp, ensure_ascii=False)}


def generate_comp3plus(idx: int) -> Dict:
    k = random.choice([3, 3, 3, 4])
    gens = random.sample([g for g, _ in SINGLE_GENERATORS], min(k, len(SINGLE_GENERATORS)))
    parts = []
    updates = []
    for g in gens:
        t, u = g()
        parts.append(t[0].lower() + t[1:])
        updates.append(u)

    if len(parts) == 3:
        connector = random.choice(CONNECTORS_3)
        text = connector.format(a=parts[0], b=parts[1], c=parts[2])
    else:
        text = ", ".join(parts[:-1]) + ", and " + parts[-1]
    text = text[0].upper() + text[1:]

    pri = random.choice(["critical", "critical", "high"])
    cp = base_cp(f"train_comp{k}_{idx}", text, pri)
    for u in updates:
        merge_updates(cp, u)
    return {"instruction": text, "response": json.dumps(cp, ensure_ascii=False)}


def generate_conditional(idx: int) -> Dict:
    evt_type, evt_names = random.choice(EVENTS)
    evt_name = random.choice(evt_names)
    preamble = random.choice(TPL_EVENT_PRE).format(evt=evt_name)

    # 1-2 constraints under the condition
    k = random.choice([1, 1, 2])
    gens = random.sample([g for g, _ in SINGLE_GENERATORS], k)
    parts = []
    updates = []
    cond = {"event_type": evt_type, "active": False}

    for g in gens:
        t, u = g()
        apply_condition(u, cond)
        parts.append(t[0].lower() + t[1:])
        updates.append(u)

    body = " and ".join(parts) if len(parts) > 1 else parts[0]
    text = preamble + body
    text = text[0].upper() + text[1:]

    pri = random.choice(["critical", "critical", "high"])
    cp = base_cp(f"train_cond_{idx}", text, pri)
    cp["event_conditions"] = [{"event_type": evt_type, "active": False}]
    for u in updates:
        merge_updates(cp, u)
    return {"instruction": text, "response": json.dumps(cp, ensure_ascii=False)}


def generate_infeasible(idx: int) -> Dict:
    kind = random.choice(["impossible_latency", "too_many_disabled",
                          "conflict", "out_of_range"])

    if kind == "impossible_latency":
        src, dst = random.sample(REGIONS, 2)
        tc = rand_tc()
        ms = random.choice([0.1, 0.5, 1.0, 1.5])
        text = random.choice(TPL_MAX_LATENCY).format(tc=tc, src=src, dst=dst, ms=ms)
        cp = base_cp(f"train_infeasible_{idx}", text, "critical")
        cp["flow_selectors"] = [{"traffic_class": tc, "src_region": src, "dst_region": dst}]
        cp["hard_constraints"] = [{"type": "max_latency_ms", "target": "flow_selector:0",
                                    "value": ms, "condition": None}]

    elif kind == "too_many_disabled":
        num = random.randint(14, 19)
        planes = random.sample(list(range(NUM_PLANES)), num)
        plane_str = ", ".join(str(p) for p in planes[:4]) + f" and {num-4} others"
        text = f"Disable planes {plane_str} for emergency maintenance"
        cp = base_cp(f"train_infeasible_{idx}", text, "critical")
        cp["hard_constraints"] = [
            {"type": "disable_plane", "target": f"plane:{p}", "value": None, "condition": None}
            for p in planes
        ]

    elif kind == "conflict":
        nid = rand_node()
        templates = [
            f"Disable node {nid} but also ensure all traffic routes through node {nid}",
            f"Take satellite {nid} offline and reroute traffic to use node {nid} as primary relay",
            f"Shut down node {nid} while keeping it as a backup path",
        ]
        text = random.choice(templates)
        cp = base_cp(f"train_infeasible_{idx}", text, "high")
        cp["hard_constraints"] = [
            {"type": "disable_node", "target": f"node:{nid}", "value": None, "condition": None},
            {"type": "reroute_away", "target": f"node:{nid}", "value": None, "condition": None},
        ]

    else:  # out_of_range
        if random.random() < 0.5:
            nid = random.randint(400, 600)
            text = f"Disable satellite {nid}"
            cp = base_cp(f"train_infeasible_{idx}", text, "high")
            cp["hard_constraints"] = [{"type": "disable_node", "target": f"node:{nid}",
                                        "value": None, "condition": None}]
        else:
            pid = random.randint(20, 30)
            text = f"Take plane {pid} offline"
            cp = base_cp(f"train_infeasible_{idx}", text, "high")
            cp["hard_constraints"] = [{"type": "disable_plane", "target": f"plane:{pid}",
                                        "value": None, "condition": None}]

    return {"instruction": text, "response": json.dumps(cp, ensure_ascii=False)}


def generate_dataset(total: int = 18000, seed: int = 2024) -> List[Dict]:
    random.seed(seed)

    # Distribution
    n_single = int(total * 0.45)
    n_comp2 = int(total * 0.20)
    n_comp3 = int(total * 0.15)
    n_cond = int(total * 0.10)
    n_infeasible = total - n_single - n_comp2 - n_comp3 - n_cond

    data = []
    for i in range(n_single):
        data.append(generate_single(i))
    for i in range(n_comp2):
        data.append(generate_comp2(i))
    for i in range(n_comp3):
        data.append(generate_comp3plus(i))
    for i in range(n_cond):
        data.append(generate_conditional(i))
    for i in range(n_infeasible):
        data.append(generate_infeasible(i))

    random.shuffle(data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output", type=str, default="intent/benchmark/train_18k.jsonl")
    args = parser.parse_args()

    data = generate_dataset(args.total, args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    print(f"Generated {len(data)} training pairs -> {out_path}")

    # Count categories by intent_id prefix
    cats = {}
    for item in data:
        cp = json.loads(item["response"])
        iid = cp["intent_id"]
        cat = iid.split("_")[1]  # single, comp2, comp3/4, cond, infeasible
        cats[cat] = cats.get(cat, 0) + 1

    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")

    # Sample
    print("\nSample entries:")
    for item in data[:3]:
        print(f"  IN:  {item['instruction'][:80]}...")
        cp = json.loads(item["response"])
        print(f"  OUT: {len(cp['hard_constraints'])} hard, {len(cp['soft_constraints'])} soft")
        print()
