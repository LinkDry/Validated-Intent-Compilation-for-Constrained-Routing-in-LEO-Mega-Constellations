"""Build the 240-intent evaluation benchmark for the LLM compiler.

Categories:
  - 80 single-clause (one constraint each)
  - 100 compositional (2-4 constraints combined)
  - 30 conditional (event-triggered)
  - 30 infeasible / conflicting

Each entry: {id, category, intent_text, constraint_program, difficulty, tags}
"""

import json, random, itertools
from pathlib import Path

random.seed(2024)

# ── Building blocks ──

REGIONS = ["NYC", "LONDON", "TOKYO", "SHANGHAI", "MUMBAI",
           "SAO_PAULO", "SYDNEY", "DUBAI", "FRANKFURT", "PARIS"]
TRAFFIC = ["financial", "emergency", "telemetry", "video",
           "bulk", "control_plane", "maritime", "military"]
CORRIDORS = ["transatlantic", "transpacific", "europe_us", "asia_europe"]
LATENCIES = [20, 30, 40, 50, 60, 80, 100, 150]
UTIL_CAPS = [0.5, 0.6, 0.7, 0.8, 0.9]
PLANES = list(range(20))
NODES = list(range(400))
LAT_THRESHOLDS = [65, 70, 75, 80]

entries = []

def add(cat, text, cp_dict, diff, tags):
    entries.append({
        "id": f"{cat}_{len([e for e in entries if e['category']==cat])+1:03d}",
        "category": cat,
        "intent_text": text,
        "constraint_program": cp_dict,
        "difficulty": diff,
        "tags": tags,
    })

def base_cp(intent_id, text, priority="medium"):
    return {
        "intent_id": intent_id,
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


# ═══════════════════════════════════════════════
# SINGLE-CLAUSE (80)
# ═══════════════════════════════════════════════

# --- Disable node (10) ---
for i in range(10):
    nid = random.choice(NODES)
    text = random.choice([
        f"Take satellite {nid} offline for maintenance",
        f"Disable node {nid} immediately",
        f"Remove sat {nid} from the routing mesh",
        f"Pull satellite {nid} out of service",
        f"Shut down node {nid}",
    ])
    cp = base_cp(f"single_disable_node_{i}", text, "high")
    cp["hard_constraints"] = [{"type": "disable_node", "target": f"node:{nid}", "value": None, "condition": None}]
    add("single", text, cp, "easy", ["disable_node"])

# --- Disable plane (10) ---
for i in range(10):
    pid = random.choice(PLANES)
    text = random.choice([
        f"Disable orbital plane {pid}",
        f"Take plane {pid} offline for orbit adjustment",
        f"Remove all satellites in plane {pid} from routing",
        f"Shut down the entire plane {pid}",
        f"Plane {pid} is undergoing maintenance, disable it",
    ])
    cp = base_cp(f"single_disable_plane_{i}", text, "critical")
    cp["hard_constraints"] = [{"type": "disable_plane", "target": f"plane:{pid}", "value": None, "condition": None}]
    add("single", text, cp, "easy", ["disable_plane"])

# --- Avoid region (10) ---
for i in range(10):
    reg = random.choice(REGIONS)
    text = random.choice([
        f"Avoid routing through the {reg} region",
        f"Do not use links near {reg}",
        f"Keep traffic away from {reg} airspace",
        f"Reroute all flows around {reg}",
        f"Bypass the {reg} area completely",
    ])
    cp = base_cp(f"single_avoid_region_{i}", text, "high")
    cp["hard_constraints"] = [{"type": "avoid_region", "target": "edges:ALL", "value": reg, "condition": None}]
    add("single", text, cp, "easy", ["avoid_region"])

# --- Avoid latitude / polar (10) ---
for i in range(10):
    lat = random.choice(LAT_THRESHOLDS)
    text = random.choice([
        f"Avoid polar links above {lat} degrees latitude",
        f"Disable inter-satellite links in polar regions above {lat}°",
        f"No routing through latitudes higher than {lat}",
        f"Keep traffic below {lat}° latitude",
        f"Polar avoidance: cut links above {lat} degrees",
    ])
    cp = base_cp(f"single_avoid_lat_{i}", text, "high")
    cp["hard_constraints"] = [{"type": "avoid_latitude", "target": "edges:ALL", "value": lat, "condition": None}]
    add("single", text, cp, "easy", ["avoid_latitude"])

# --- Max latency (15) ---
for i in range(15):
    src_r, dst_r = random.sample(REGIONS, 2)
    tc = random.choice(TRAFFIC)
    lat_ms = random.choice(LATENCIES)
    text = random.choice([
        f"Ensure {tc} traffic from {src_r} to {dst_r} stays under {lat_ms}ms",
        f"Max latency {lat_ms}ms for {tc} flows between {src_r} and {dst_r}",
        f"{tc.capitalize()} traffic {src_r} to {dst_r} must not exceed {lat_ms}ms end-to-end",
        f"SLA: {lat_ms}ms latency cap on {tc} from {src_r} to {dst_r}",
        f"Keep {src_r}-{dst_r} {tc} latency below {lat_ms} milliseconds",
    ])
    cp = base_cp(f"single_max_latency_{i}", text, "high")
    cp["flow_selectors"] = [{"traffic_class": tc, "src_region": src_r, "dst_region": dst_r}]
    cp["hard_constraints"] = [{"type": "max_latency_ms", "target": "flow_selector:0", "value": lat_ms, "condition": None}]
    add("single", text, cp, "medium", ["max_latency_ms"])

# --- Reroute away (10) ---
for i in range(10):
    nid = random.choice(NODES)
    text = random.choice([
        f"Reroute all traffic away from node {nid}",
        f"Node {nid} is degraded, shift traffic elsewhere",
        f"Avoid using satellite {nid} as a relay",
        f"Divert flows around node {nid}",
        f"Node {nid} showing errors, reroute away",
    ])
    cp = base_cp(f"single_reroute_{i}", text, "high")
    cp["hard_constraints"] = [{"type": "reroute_away", "target": f"node:{nid}", "value": None, "condition": None}]
    add("single", text, cp, "easy", ["reroute_away"])

# --- Max utilization (10) ---
for i in range(10):
    cap = random.choice(UTIL_CAPS)
    pct = int(cap * 100)
    text = random.choice([
        f"Cap link utilization at {pct}%",
        f"Keep all ISL utilization below {pct} percent",
        f"No link should exceed {pct}% capacity",
        f"Set a {pct}% utilization ceiling on all edges",
        f"Limit bandwidth usage to {pct}% across the mesh",
    ])
    cp = base_cp(f"single_util_{i}", text, "medium")
    cp["soft_constraints"] = [{"type": "max_utilization", "target": "edges:ALL", "value": cap, "penalty": 2.0, "condition": None}]
    add("single", text, cp, "easy", ["max_utilization"])

# --- Load balance / misc soft (5) ---
for i in range(5):
    text = random.choice([
        "Balance load evenly across all inter-plane links",
        "Minimize traffic concentration on any single link",
        "Spread traffic as uniformly as possible",
        "Enable load balancing across the constellation",
        "Equalize utilization across orbital planes",
    ])
    cp = base_cp(f"single_balance_{i}", text, "low")
    cp["soft_constraints"] = [{"type": "load_balance", "target": "edges:ALL", "value": None, "penalty": 1.0, "condition": None}]
    add("single", text, cp, "easy", ["load_balance"])

assert len([e for e in entries if e["category"] == "single"]) == 80


# ═══════════════════════════════════════════════
# COMPOSITIONAL (100)
# ═══════════════════════════════════════════════

# --- 2-constraint combos (50) ---
for i in range(15):
    nid = random.choice(NODES)
    src_r, dst_r = random.sample(REGIONS, 2)
    tc = random.choice(TRAFFIC)
    lat_ms = random.choice(LATENCIES)
    text = f"Disable node {nid} and ensure {tc} traffic from {src_r} to {dst_r} stays under {lat_ms}ms"
    cp = base_cp(f"comp2_node_lat_{i}", text, "high")
    cp["flow_selectors"] = [{"traffic_class": tc, "src_region": src_r, "dst_region": dst_r}]
    cp["hard_constraints"] = [
        {"type": "disable_node", "target": f"node:{nid}", "value": None, "condition": None},
        {"type": "max_latency_ms", "target": "flow_selector:0", "value": lat_ms, "condition": None},
    ]
    add("compositional", text, cp, "medium", ["disable_node", "max_latency_ms"])

for i in range(10):
    reg = random.choice(REGIONS)
    cap = random.choice(UTIL_CAPS)
    pct = int(cap * 100)
    text = f"Avoid the {reg} region and keep utilization below {pct}%"
    cp = base_cp(f"comp2_region_util_{i}", text, "high")
    cp["hard_constraints"] = [
        {"type": "avoid_region", "target": "edges:ALL", "value": reg, "condition": None},
    ]
    cp["soft_constraints"] = [
        {"type": "max_utilization", "target": "edges:ALL", "value": cap, "penalty": 2.0, "condition": None},
    ]
    add("compositional", text, cp, "medium", ["avoid_region", "max_utilization"])

for i in range(10):
    pid = random.choice(PLANES)
    lat = random.choice(LAT_THRESHOLDS)
    text = f"Disable plane {pid} and avoid polar links above {lat}°"
    cp = base_cp(f"comp2_plane_polar_{i}", text, "critical")
    cp["hard_constraints"] = [
        {"type": "disable_plane", "target": f"plane:{pid}", "value": None, "condition": None},
        {"type": "avoid_latitude", "target": "edges:ALL", "value": lat, "condition": None},
    ]
    add("compositional", text, cp, "medium", ["disable_plane", "avoid_latitude"])

for i in range(15):
    src_r, dst_r = random.sample(REGIONS, 2)
    tc = random.choice(TRAFFIC)
    lat_ms = random.choice(LATENCIES)
    reg = random.choice([r for r in REGIONS if r not in (src_r, dst_r)])
    text = f"Route {tc} from {src_r} to {dst_r} under {lat_ms}ms while avoiding {reg}"
    cp = base_cp(f"comp2_lat_region_{i}", text, "high")
    cp["flow_selectors"] = [{"traffic_class": tc, "src_region": src_r, "dst_region": dst_r}]
    cp["hard_constraints"] = [
        {"type": "max_latency_ms", "target": "flow_selector:0", "value": lat_ms, "condition": None},
        {"type": "avoid_region", "target": "edges:ALL", "value": reg, "condition": None},
    ]
    add("compositional", text, cp, "medium", ["max_latency_ms", "avoid_region"])

# --- 3-constraint combos (30) ---
for i in range(15):
    nid = random.choice(NODES)
    reg = random.choice(REGIONS)
    src_r, dst_r = random.sample([r for r in REGIONS if r != reg], 2)
    tc = random.choice(TRAFFIC)
    lat_ms = random.choice(LATENCIES)
    text = (f"Disable node {nid}, avoid {reg}, and guarantee {tc} "
            f"from {src_r} to {dst_r} under {lat_ms}ms")
    cp = base_cp(f"comp3_full_{i}", text, "critical")
    cp["flow_selectors"] = [{"traffic_class": tc, "src_region": src_r, "dst_region": dst_r}]
    cp["hard_constraints"] = [
        {"type": "disable_node", "target": f"node:{nid}", "value": None, "condition": None},
        {"type": "avoid_region", "target": "edges:ALL", "value": reg, "condition": None},
        {"type": "max_latency_ms", "target": "flow_selector:0", "value": lat_ms, "condition": None},
    ]
    add("compositional", text, cp, "hard", ["disable_node", "avoid_region", "max_latency_ms"])

for i in range(15):
    pid = random.choice(PLANES)
    lat = random.choice(LAT_THRESHOLDS)
    cap = random.choice(UTIL_CAPS)
    pct = int(cap * 100)
    text = (f"Take plane {pid} offline, avoid polar links above {lat}°, "
            f"and cap utilization at {pct}%")
    cp = base_cp(f"comp3_plane_{i}", text, "critical")
    cp["hard_constraints"] = [
        {"type": "disable_plane", "target": f"plane:{pid}", "value": None, "condition": None},
        {"type": "avoid_latitude", "target": "edges:ALL", "value": lat, "condition": None},
    ]
    cp["soft_constraints"] = [
        {"type": "max_utilization", "target": "edges:ALL", "value": cap, "penalty": 2.0, "condition": None},
    ]
    add("compositional", text, cp, "hard", ["disable_plane", "avoid_latitude", "max_utilization"])

# --- 4-constraint combos (20) ---
for i in range(20):
    nid = random.choice(NODES)
    pid = random.choice(PLANES)
    while nid // 20 == pid:  # avoid disabling same plane as node
        pid = random.choice(PLANES)
    src_r, dst_r = random.sample(REGIONS, 2)
    tc = random.choice(TRAFFIC)
    lat_ms = random.choice(LATENCIES)
    cap = random.choice(UTIL_CAPS)
    pct = int(cap * 100)
    text = (f"Disable node {nid} and plane {pid}, ensure {tc} from {src_r} "
            f"to {dst_r} under {lat_ms}ms, cap utilization at {pct}%")
    cp = base_cp(f"comp4_full_{i}", text, "critical")
    cp["flow_selectors"] = [{"traffic_class": tc, "src_region": src_r, "dst_region": dst_r}]
    cp["hard_constraints"] = [
        {"type": "disable_node", "target": f"node:{nid}", "value": None, "condition": None},
        {"type": "disable_plane", "target": f"plane:{pid}", "value": None, "condition": None},
        {"type": "max_latency_ms", "target": "flow_selector:0", "value": lat_ms, "condition": None},
    ]
    cp["soft_constraints"] = [
        {"type": "max_utilization", "target": "edges:ALL", "value": cap, "penalty": 2.0, "condition": None},
    ]
    add("compositional", text, cp, "hard",
        ["disable_node", "disable_plane", "max_latency_ms", "max_utilization"])

assert len([e for e in entries if e["category"] == "compositional"]) == 100


# ═══════════════════════════════════════════════
# CONDITIONAL (30)
# ═══════════════════════════════════════════════

events = [
    ("solar_storm", "solar storm"),
    ("node_failure", "node failure"),
    ("gateway_failure", "gateway failure"),
    ("maintenance", "scheduled maintenance"),
    ("overload", "traffic overload"),
]

for i in range(10):
    evt_type, evt_name = random.choice(events[:2])  # solar/node failure
    pid = random.choice(PLANES)
    text = f"If a {evt_name} occurs, disable plane {pid}"
    cp = base_cp(f"cond_plane_{i}", text, "critical")
    cp["hard_constraints"] = [{
        "type": "disable_plane", "target": f"plane:{pid}", "value": None,
        "condition": {"event_type": evt_type, "active": False},
    }]
    cp["event_conditions"] = [{"event_type": evt_type, "active": False}]
    add("conditional", text, cp, "medium", ["disable_plane", evt_type])

for i in range(10):
    evt_type, evt_name = random.choice(events)
    nid = random.choice(NODES)
    lat = random.choice(LAT_THRESHOLDS)
    text = (f"During a {evt_name}, reroute away from node {nid} "
            f"and avoid polar links above {lat}°")
    cp = base_cp(f"cond_reroute_{i}", text, "critical")
    cond = {"event_type": evt_type, "active": False}
    cp["hard_constraints"] = [
        {"type": "reroute_away", "target": f"node:{nid}", "value": None, "condition": cond},
        {"type": "avoid_latitude", "target": "edges:ALL", "value": lat, "condition": cond},
    ]
    cp["event_conditions"] = [{"event_type": evt_type, "active": False}]
    add("conditional", text, cp, "hard", ["reroute_away", "avoid_latitude", evt_type])

for i in range(10):
    evt_type, evt_name = random.choice(events)
    src_r, dst_r = random.sample(REGIONS, 2)
    tc = random.choice(TRAFFIC)
    lat_ms = random.choice(LATENCIES)
    cap = random.choice(UTIL_CAPS)
    pct = int(cap * 100)
    text = (f"When {evt_name} is detected, enforce {lat_ms}ms latency on "
            f"{tc} from {src_r} to {dst_r} and cap utilization at {pct}%")
    cp = base_cp(f"cond_sla_{i}", text, "high")
    cond = {"event_type": evt_type, "active": False}
    cp["flow_selectors"] = [{"traffic_class": tc, "src_region": src_r, "dst_region": dst_r}]
    cp["hard_constraints"] = [
        {"type": "max_latency_ms", "target": "flow_selector:0", "value": lat_ms, "condition": cond},
    ]
    cp["soft_constraints"] = [
        {"type": "max_utilization", "target": "edges:ALL", "value": cap, "penalty": 2.0, "condition": cond},
    ]
    cp["event_conditions"] = [{"event_type": evt_type, "active": False}]
    add("conditional", text, cp, "hard", ["max_latency_ms", "max_utilization", evt_type])

assert len([e for e in entries if e["category"] == "conditional"]) == 30


# ═══════════════════════════════════════════════
# INFEASIBLE / CONFLICTING (30)
# ═══════════════════════════════════════════════

# Impossible latency (8)
for i in range(8):
    src_r, dst_r = random.sample(REGIONS, 2)
    tc = random.choice(TRAFFIC)
    impossible_ms = random.choice([0.5, 1.0, 1.5])
    text = f"Guarantee {tc} from {src_r} to {dst_r} under {impossible_ms}ms"
    cp = base_cp(f"infeasible_latency_{i}", text, "critical")
    cp["flow_selectors"] = [{"traffic_class": tc, "src_region": src_r, "dst_region": dst_r}]
    cp["hard_constraints"] = [
        {"type": "max_latency_ms", "target": "flow_selector:0", "value": impossible_ms, "condition": None},
    ]
    add("infeasible", text, cp, "hard", ["max_latency_ms", "infeasible"])

# Disable too many planes (8)
for i in range(8):
    num_disable = random.randint(15, 19)
    planes_to_disable = random.sample(PLANES, num_disable)
    plane_list = ", ".join(str(p) for p in planes_to_disable[:5]) + f" and {num_disable-5} more"
    text = f"Disable planes {plane_list} for maintenance"
    cp = base_cp(f"infeasible_planes_{i}", text, "critical")
    cp["hard_constraints"] = [
        {"type": "disable_plane", "target": f"plane:{p}", "value": None, "condition": None}
        for p in planes_to_disable
    ]
    add("infeasible", text, cp, "hard", ["disable_plane", "infeasible"])

# Conflicting: disable + reroute same node (7)
for i in range(7):
    nid = random.choice(NODES)
    text = f"Disable node {nid} but also reroute critical traffic through node {nid}"
    cp = base_cp(f"infeasible_conflict_{i}", text, "high")
    cp["hard_constraints"] = [
        {"type": "disable_node", "target": f"node:{nid}", "value": None, "condition": None},
        {"type": "reroute_away", "target": f"node:{nid}", "value": None, "condition": None},
    ]
    add("infeasible", text, cp, "hard", ["disable_node", "reroute_away", "conflicting"])

# Out-of-range values (7)
for i in range(4):
    nid = random.choice(range(400, 500))  # node doesn't exist
    text = f"Disable satellite {nid}"
    cp = base_cp(f"infeasible_range_{i}", text, "high")
    cp["hard_constraints"] = [
        {"type": "disable_node", "target": f"node:{nid}", "value": None, "condition": None},
    ]
    add("infeasible", text, cp, "medium", ["disable_node", "out_of_range"])

for i in range(3):
    pid = random.choice(range(20, 30))  # plane doesn't exist
    text = f"Take plane {pid} offline"
    cp = base_cp(f"infeasible_plane_range_{i}", text, "high")
    cp["hard_constraints"] = [
        {"type": "disable_plane", "target": f"plane:{pid}", "value": None, "condition": None},
    ]
    add("infeasible", text, cp, "medium", ["disable_plane", "out_of_range"])

assert len([e for e in entries if e["category"] == "infeasible"]) == 30


# ═══════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════

out_dir = Path("intent/benchmark")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "benchmark_240.json", "w") as f:
    json.dump(entries, f, indent=2, ensure_ascii=False)

# Stats
cats = {}
for e in entries:
    cats[e["category"]] = cats.get(e["category"], 0) + 1

print(f"Total intents: {len(entries)}")
for c, n in sorted(cats.items()):
    print(f"  {c}: {n}")

# Tag distribution
tags = {}
for e in entries:
    for t in e["tags"]:
        tags[t] = tags.get(t, 0) + 1
print("\nTag distribution:")
for t, n in sorted(tags.items(), key=lambda x: -x[1]):
    print(f"  {t}: {n}")

print(f"\nSaved to {out_dir / 'benchmark_240.json'}")
