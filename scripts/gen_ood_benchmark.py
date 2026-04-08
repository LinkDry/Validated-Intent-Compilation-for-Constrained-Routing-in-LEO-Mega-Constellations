"""Generate OOD paraphrase benchmark for intent compiler robustness testing.

Creates paraphrased versions of existing benchmark intents with:
- Synonym substitution (disable → kill, shut down, take offline, pull)
- Informal/colloquial phrasing
- Abbreviations and shorthand
- Ambiguous wording
- ASR-like noise (minor typos, run-on sentences)
- Varied sentence structure (passive voice, questions, imperatives)

Each paraphrase maps to the same expected ConstraintProgram as the original.
"""

import json, random, copy
from pathlib import Path

random.seed(42)


def make_single_paraphrases():
    """OOD paraphrases for single-constraint intents."""
    intents = []

    # Node disable variants
    node_templates = [
        # Informal/colloquial
        ("Kill sat {n}, it's broken", "disable_node", "node:{n}"),
        ("node {n} is toast, yank it from the mesh", "disable_node", "node:{n}"),
        ("pls take {n} offline asap", "disable_node", "node:{n}"),
        ("satellite number {n} needs to go dark right now", "disable_node", "node:{n}"),
        ("can we pull node {n} out? hardware issue", "disable_node", "node:{n}"),
        # Passive voice
        ("Node {n} should be removed from active routing", "disable_node", "node:{n}"),
        ("Satellite {n} must be isolated from the network", "disable_node", "node:{n}"),
        # ASR-like noise
        ("disable note {n} immediately", "disable_node", "node:{n}"),  # "note" typo
        ("shut down know {n}", "disable_node", "node:{n}"),  # "know" typo for "node"
        # Verbose/roundabout
        ("We've detected an anomaly on satellite {n}, please ensure no traffic is routed through it", "disable_node", "node:{n}"),
    ]

    for template, ctype, target_tmpl in node_templates:
        n = random.randint(0, 399)
        intent_text = template.format(n=n)
        target = target_tmpl.format(n=n)
        intents.append({
            "id": f"ood_single_node_{len(intents)}",
            "category": "single",
            "subcategory": "ood_paraphrase",
            "intent_text": intent_text,
            "original_pattern": "disable_node",
            "constraint_program": {
                "intent_id": f"ood-node-{n}",
                "source_text": intent_text,
                "priority": "high",
                "time_window": {"start": "now", "duration_min": None},
                "flow_selectors": [],
                "edge_selectors": [],
                "node_selectors": [],
                "hard_constraints": [
                    {"type": "disable_node", "target": target, "value": None, "condition": None}
                ],
                "soft_constraints": [],
                "objective_weights": {},
                "fallback_policy": "reject_if_hard_infeasible",
                "event_conditions": []
            }
        })

    # Plane disable variants
    plane_templates = [
        ("orbital plane {p} is having issues, take the whole thing down", "disable_plane", "plane:{p}"),
        ("ground all sats in plane {p}", "disable_plane", "plane:{p}"),
        ("plane {p} maintenance window starting now, disable everything", "disable_plane", "plane:{p}"),
        ("pull plane number {p} for servicing", "disable_plane", "plane:{p}"),
        ("we need plane {p} offline, orbit correction maneuver", "disable_plane", "plane:{p}"),
    ]

    for template, ctype, target_tmpl in plane_templates:
        p = random.randint(0, 19)
        intent_text = template.format(p=p)
        target = target_tmpl.format(p=p)
        intents.append({
            "id": f"ood_single_plane_{len(intents)}",
            "category": "single",
            "subcategory": "ood_paraphrase",
            "intent_text": intent_text,
            "original_pattern": "disable_plane",
            "constraint_program": {
                "intent_id": f"ood-plane-{p}",
                "source_text": intent_text,
                "priority": "high",
                "time_window": {"start": "now", "duration_min": None},
                "flow_selectors": [],
                "edge_selectors": [],
                "node_selectors": [],
                "hard_constraints": [
                    {"type": "disable_plane", "target": target, "value": None, "condition": None}
                ],
                "soft_constraints": [],
                "objective_weights": {},
                "fallback_policy": "reject_if_hard_infeasible",
                "event_conditions": []
            }
        })

    # Region avoidance variants
    regions = ["NYC", "LONDON", "TOKYO", "SHANGHAI", "MUMBAI", "SAO_PAULO", "SYDNEY", "FRANKFURT"]
    region_templates = [
        ("stay away from {r}, there's interference", "avoid_region", "edges:ALL", "{r}"),
        ("no links near {r} please", "avoid_region", "edges:ALL", "{r}"),
        ("route around the {r} area, we have a ground station issue", "avoid_region", "edges:ALL", "{r}"),
        ("avoid {r} airspace for the next hour", "avoid_region", "edges:ALL", "{r}"),
        ("{r} is a no-go zone right now", "avoid_region", "edges:ALL", "{r}"),
    ]

    for template, ctype, target, val_tmpl in region_templates:
        r = random.choice(regions)
        intent_text = template.format(r=r)
        intents.append({
            "id": f"ood_single_region_{len(intents)}",
            "category": "single",
            "subcategory": "ood_paraphrase",
            "intent_text": intent_text,
            "original_pattern": "avoid_region",
            "constraint_program": {
                "intent_id": f"ood-avoid-{r.lower()}",
                "source_text": intent_text,
                "priority": "high",
                "time_window": {"start": "now", "duration_min": None},
                "flow_selectors": [],
                "edge_selectors": [],
                "node_selectors": [],
                "hard_constraints": [
                    {"type": "avoid_region", "target": target, "value": r, "condition": None}
                ],
                "soft_constraints": [],
                "objective_weights": {},
                "fallback_policy": "reject_if_hard_infeasible",
                "event_conditions": []
            }
        })

    return intents


def make_compositional_paraphrases():
    """OOD paraphrases for multi-constraint intents."""
    intents = []
    regions = ["NYC", "LONDON", "TOKYO", "SHANGHAI", "MUMBAI"]
    traffic_classes = ["financial", "emergency", "video", "telemetry"]

    templates = [
        # Informal multi-constraint
        {
            "text": "node {n} is down and we need {tc} traffic from {r1} to {r2} under {ms}ms, make it happen",
            "priority": "critical",
        },
        # Run-on sentence
        {
            "text": "ok so {tc} from {r1} to {r2} needs to be under {ms} milliseconds and also kill node {n} and keep utilization below 80 percent",
            "priority": "high",
        },
        # Terse operator shorthand
        {
            "text": "n{n} down. {tc} {r1}->{r2} <{ms}ms. util cap 80%.",
            "priority": "high",
        },
        # Question form
        {
            "text": "Can you disable node {n} and make sure {tc} traffic between {r1} and {r2} stays under {ms}ms?",
            "priority": "high",
        },
        # Verbose explanation
        {
            "text": "We have a situation: satellite {n} has failed and we need to ensure that {tc} class traffic originating from the {r1} region destined for {r2} maintains latency below {ms} milliseconds",
            "priority": "critical",
        },
    ]

    for tmpl in templates:
        n = random.randint(0, 399)
        tc = random.choice(traffic_classes)
        r1, r2 = random.sample(regions, 2)
        ms = random.choice([40, 50, 60, 80, 100])

        intent_text = tmpl["text"].format(n=n, tc=tc, r1=r1, r2=r2, ms=ms)
        intents.append({
            "id": f"ood_comp_{len(intents)}",
            "category": "compositional",
            "subcategory": "ood_paraphrase",
            "intent_text": intent_text,
            "original_pattern": "node_disable+latency",
            "constraint_program": {
                "intent_id": f"ood-comp-{n}-{tc}-{r1.lower()}-{r2.lower()}",
                "source_text": intent_text,
                "priority": tmpl["priority"],
                "time_window": {"start": "now", "duration_min": None},
                "flow_selectors": [
                    {"traffic_class": tc, "src_region": r1, "dst_region": r2,
                     "src_node": None, "dst_node": None, "src_plane": None, "dst_plane": None, "corridor": None}
                ],
                "edge_selectors": [],
                "node_selectors": [],
                "hard_constraints": [
                    {"type": "disable_node", "target": f"node:{n}", "value": None, "condition": None},
                    {"type": "max_latency_ms", "target": "flow_selector:0", "value": float(ms), "condition": None},
                ],
                "soft_constraints": [],
                "objective_weights": {},
                "fallback_policy": "reject_if_hard_infeasible",
                "event_conditions": []
            }
        })

    return intents


def make_conditional_paraphrases():
    """OOD paraphrases for event-triggered intents."""
    intents = []

    templates = [
        ("when solar activity spikes, shut down plane {p}", "solar_storm", "disable_plane", "plane:{p}"),
        ("if we get hit by a solar storm take plane {p} offline", "solar_storm", "disable_plane", "plane:{p}"),
        ("solar event contingency: ground plane {p}", "solar_storm", "disable_plane", "plane:{p}"),
        ("in case of node failure on {n}, reroute everything away from it", "node_failure", "reroute_away", "node:{n}"),
        ("if node {n} goes down redirect traffic around it", "node_failure", "reroute_away", "node:{n}"),
        ("maintenance alert: when we start work on plane {p}, disable it automatically", "maintenance", "disable_plane", "plane:{p}"),
        ("prepare contingency for gateway failure: avoid node {n}", "gateway_failure", "reroute_away", "node:{n}"),
        ("overload protection: if traffic spikes, cap utilization at 70%", "overload", "max_utilization", "edges:ALL"),
    ]

    for text_tmpl, event, ctype, target_tmpl in templates:
        n = random.randint(0, 399)
        p = random.randint(0, 19)
        intent_text = text_tmpl.format(n=n, p=p)
        target = target_tmpl.format(n=n, p=p)

        hc = {"type": ctype, "target": target, "value": None,
              "condition": {"event_type": event, "active": False}}

        if ctype == "max_utilization":
            hc = {"type": ctype, "target": target, "value": 0.7,
                  "penalty": 2.0, "condition": {"event_type": event, "active": False}}
            sc_list = [hc]
            hc_list = []
        else:
            hc_list = [hc]
            sc_list = []

        intents.append({
            "id": f"ood_cond_{len(intents)}",
            "category": "conditional",
            "subcategory": "ood_paraphrase",
            "intent_text": intent_text,
            "original_pattern": f"conditional_{ctype}",
            "constraint_program": {
                "intent_id": f"ood-cond-{event}-{ctype}",
                "source_text": intent_text,
                "priority": "critical",
                "time_window": {"start": "now", "duration_min": None},
                "flow_selectors": [],
                "edge_selectors": [],
                "node_selectors": [],
                "hard_constraints": hc_list,
                "soft_constraints": sc_list,
                "objective_weights": {},
                "fallback_policy": "reject_if_hard_infeasible",
                "event_conditions": [{"event_type": event, "active": False}]
            }
        })

    return intents


def make_ambiguous_intents():
    """Genuinely ambiguous intents that test interpretation robustness."""
    return [
        {
            "id": "ood_ambig_0",
            "category": "ambiguous",
            "subcategory": "ood_paraphrase",
            "intent_text": "Make the network faster",
            "original_pattern": "ambiguous_optimization",
            "constraint_program": None,  # Multiple valid interpretations
            "notes": "Could mean: minimize latency, increase throughput, reduce hops. Accept any reasonable interpretation.",
        },
        {
            "id": "ood_ambig_1",
            "category": "ambiguous",
            "subcategory": "ood_paraphrase",
            "intent_text": "Fix the routing around Tokyo",
            "original_pattern": "ambiguous_region",
            "constraint_program": None,
            "notes": "Could mean: avoid TOKYO, optimize TOKYO routes, or reroute away from TOKYO. Accept any.",
        },
        {
            "id": "ood_ambig_2",
            "category": "ambiguous",
            "subcategory": "ood_paraphrase",
            "intent_text": "Something is wrong with plane 3, handle it",
            "original_pattern": "ambiguous_plane",
            "constraint_program": None,
            "notes": "Could mean: disable plane 3, reroute away, or investigate. Accept disable_plane as reasonable.",
        },
        {
            "id": "ood_ambig_3",
            "category": "ambiguous",
            "subcategory": "ood_paraphrase",
            "intent_text": "Protect the financial links",
            "original_pattern": "ambiguous_traffic",
            "constraint_program": None,
            "notes": "Could mean: prioritize financial traffic, add redundancy, reduce latency. Accept any financial-related constraint.",
        },
        {
            "id": "ood_ambig_4",
            "category": "ambiguous",
            "subcategory": "ood_paraphrase",
            "intent_text": "We're expecting bad weather over the poles, prepare the network",
            "original_pattern": "ambiguous_polar",
            "constraint_program": None,
            "notes": "Could mean: avoid polar links, reduce polar utilization, add backup paths. Accept any polar-related constraint.",
        },
    ]


if __name__ == "__main__":
    all_intents = []
    all_intents.extend(make_single_paraphrases())
    all_intents.extend(make_compositional_paraphrases())
    all_intents.extend(make_conditional_paraphrases())
    all_intents.extend(make_ambiguous_intents())

    print(f"Generated {len(all_intents)} OOD paraphrase intents:")
    cats = {}
    for e in all_intents:
        c = e["category"]
        cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")

    Path("intent/benchmark").mkdir(parents=True, exist_ok=True)
    with open("intent/benchmark/benchmark_ood_paraphrases.json", "w") as f:
        json.dump(all_intents, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to intent/benchmark/benchmark_ood_paraphrases.json")
