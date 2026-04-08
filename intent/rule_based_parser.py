"""Rule-based intent parser baseline.

Regex + keyword matching approach to compile natural language intents
into ConstraintProgram IR. Used as a baseline to demonstrate the value
of LLM-based compilation on compositional/conditional intents.
"""

from __future__ import annotations
import re, time
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from intent.schema import (
    ConstraintProgram, HardConstraint, SoftConstraint,
    FlowSelector, TimeWindow, EventCondition, ObjectiveWeights,
    KNOWN_REGIONS, KNOWN_TRAFFIC_CLASSES,
    HardConstraintType, SoftConstraintType, EventType,
)


@dataclass
class CompilationResult:
    success: bool = False
    program: Optional[ConstraintProgram] = None
    attempts: int = 1
    errors: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


# Precompile patterns
_REGION_PATTERN = "|".join(sorted(KNOWN_REGIONS.keys(), key=len, reverse=True))
_TRAFFIC_PATTERN = "|".join(sorted(KNOWN_TRAFFIC_CLASSES, key=len, reverse=True))
_EVENT_MAP = {
    "solar storm": "solar_storm",
    "solar_storm": "solar_storm",
    "node failure": "node_failure",
    "node_failure": "node_failure",
    "gateway failure": "gateway_failure",
    "maintenance": "maintenance",
    "overload": "overload",
}

# Constraint extraction patterns
PATTERNS = [
    # disable_node (including satellite/sat aliases)
    (r"(?:disable|shut\s*down|deactivate|take\s+down)\s+(?:node|satellite|sat)\s+(\d+)",
     "disable_node", lambda m: ("node:" + m.group(1), None)),
    (r"(?:pull|remove)\s+(?:node|satellite|sat)\s+(\d+)\s+(?:out\s+of\s+service|from\s+(?:the\s+)?(?:routing\s+)?mesh|offline)",
     "disable_node", lambda m: ("node:" + m.group(1), None)),
    (r"(?:node|satellite|sat)\s+(\d+)\s+(?:is\s+)?(?:down|offline|failed|out)",
     "disable_node", lambda m: ("node:" + m.group(1), None)),

    # disable_plane
    (r"(?:disable|shut\s*down|deactivate)\s+(?:orbital\s+)?plane\s+(\d+)",
     "disable_plane", lambda m: ("plane:" + m.group(1), None)),
    (r"take\s+(?:orbital\s+)?plane\s+(\d+)\s+(?:offline|down|out)",
     "disable_plane", lambda m: ("plane:" + m.group(1), None)),

    # disable multiple planes
    (r"disable\s+planes?\s+([\d,\s]+(?:and\s+\d+)?)",
     "disable_plane_multi", lambda m: (m.group(1), None)),

    # reroute_away (including satellite/sat aliases)
    (r"reroute\s+(?:traffic\s+|all\s+flows\s+)?away\s+from\s+(?:node|satellite|sat)\s+(\d+)",
     "reroute_away", lambda m: ("node:" + m.group(1), None)),
    (r"avoid\s+(?:using\s+)?(?:node|satellite|sat)\s+(\d+)\s+(?:as\s+a\s+relay|for\s+routing|for\s+transit)",
     "reroute_away", lambda m: ("node:" + m.group(1), None)),

    # avoid_latitude (expanded patterns)
    (r"avoid\s+(?:polar\s+)?(?:links?|ISLs?)\s+(?:above|over|beyond)\s+(\d+\.?\d*)\s*(?:deg|degrees?|°)",
     "avoid_latitude", lambda m: ("edges:ALL", float(m.group(1)))),
    (r"avoid\s+(?:links?\s+)?(?:above|over)\s+(\d+\.?\d*)\s*(?:deg|degrees?|°)\s*latitude",
     "avoid_latitude", lambda m: ("edges:ALL", float(m.group(1)))),
    (r"(?:no|disable)\s+(?:routing|links?|ISLs?)\s+(?:through|in|above)\s+(?:latitudes?\s+)?(?:higher\s+than|above|over)\s+(\d+\.?\d*)",
     "avoid_latitude", lambda m: ("edges:ALL", float(m.group(1)))),
    (r"(?:keep|maintain)\s+(?:traffic|routing)\s+(?:below|under)\s+(\d+\.?\d*)\s*(?:°|deg|degrees?)\s*latitude",
     "avoid_latitude", lambda m: ("edges:ALL", float(m.group(1)))),
    (r"polar\s+avoidance:?\s+(?:cut|disable|remove)\s+(?:links?|ISLs?)\s+(?:above|over)\s+(\d+\.?\d*)\s*(?:deg|degrees?|°)?",
     "avoid_latitude", lambda m: ("edges:ALL", float(m.group(1)))),
    (r"disable\s+inter-satellite\s+links?\s+in\s+polar\s+regions?\s+(?:above|over)\s+(\d+\.?\d*)",
     "avoid_latitude", lambda m: ("edges:ALL", float(m.group(1)))),

    # avoid_region (expanded patterns)
    (r"avoid\s+(?:the\s+)?(?:routing\s+through\s+(?:the\s+)?)?(" + _REGION_PATTERN + r")\s*(?:region|area)?",
     "avoid_region", lambda m: ("region:" + m.group(1).upper(), m.group(1).upper())),
    (r"(?:bypass|skip)\s+(?:the\s+)?(" + _REGION_PATTERN + r")\s*(?:region|area|completely)?",
     "avoid_region", lambda m: ("region:" + m.group(1).upper(), m.group(1).upper())),
    (r"(?:keep\s+traffic\s+away\s+from|reroute\s+all\s+flows\s+around)\s+(?:the\s+)?(" + _REGION_PATTERN + r")\s*(?:airspace|region|area)?",
     "avoid_region", lambda m: ("region:" + m.group(1).upper(), m.group(1).upper())),
    (r"do\s+not\s+use\s+(?:links?|nodes?)\s+(?:near|in|around)\s+(" + _REGION_PATTERN + r")",
     "avoid_region", lambda m: ("region:" + m.group(1).upper(), m.group(1).upper())),

    # max_latency_ms (expanded patterns)
    (r"(?:ensure|guarantee|keep|maintain)\s+(?:that\s+)?(?:(" + _TRAFFIC_PATTERN +
     r")\s+)?(?:traffic\s+)?(?:from\s+(" + _REGION_PATTERN +
     r")\s+to\s+(" + _REGION_PATTERN +
     r")\s+)?(?:stays?\s+|is\s+|remains?\s+)?(?:under|below|within)\s+(\d+\.?\d*)\s*ms",
     "max_latency_ms_flow", None),
    (r"(?:max|maximum)\s+latency\s+(\d+\.?\d*)\s*ms\s+for\s+(" + _TRAFFIC_PATTERN +
     r")\s+(?:flows?\s+)?(?:between|from)\s+(" + _REGION_PATTERN +
     r")\s+(?:and|to)\s+(" + _REGION_PATTERN + r")",
     "max_latency_ms_flow2", None),
    (r"SLA:?\s+(\d+\.?\d*)\s*ms\s+latency\s+cap\s+on\s+(" + _TRAFFIC_PATTERN +
     r")\s+from\s+(" + _REGION_PATTERN + r")\s+to\s+(" + _REGION_PATTERN + r")",
     "max_latency_ms_sla", None),
    (r"(" + _TRAFFIC_PATTERN + r")\s+(?:traffic\s+)?(" + _REGION_PATTERN +
     r")\s+to\s+(" + _REGION_PATTERN +
     r")\s+must\s+not\s+exceed\s+(\d+\.?\d*)\s*ms",
     "max_latency_ms_must", None),
    (r"(?:keep|maintain)\s+(" + _REGION_PATTERN + r")-(" + _REGION_PATTERN +
     r")\s+(" + _TRAFFIC_PATTERN + r")\s+latency\s+(?:below|under)\s+(\d+\.?\d*)\s*(?:ms|milliseconds?)",
     "max_latency_ms_keep", None),
    (r"[Rr]oute\s+(" + _TRAFFIC_PATTERN + r")\s+from\s+(" + _REGION_PATTERN +
     r")\s+to\s+(" + _REGION_PATTERN +
     r")\s+(?:under|below|within)\s+(\d+\.?\d*)\s*ms",
     "max_latency_ms_route", None),

    # max_latency_ms (simple)
    (r"(?:latency|delay)\s+(?:under|below|within)\s+(\d+\.?\d*)\s*ms",
     "max_latency_ms_simple", lambda m: ("flow:ALL", float(m.group(1)))),

    # max_utilization (expanded patterns)
    (r"cap\s+(?:link\s+)?utilization\s+(?:at|to)\s+(\d+)\s*%",
     "max_utilization", lambda m: ("edges:ALL", float(m.group(1)) / 100)),
    (r"(?:limit|restrict)\s+(?:link\s+)?(?:utilization|bandwidth\s+usage)\s+(?:to|at)\s+(\d+)\s*%",
     "max_utilization", lambda m: ("edges:ALL", float(m.group(1)) / 100)),
    (r"(?:keep|maintain)\s+(?:all\s+)?(?:ISL\s+)?utilization\s+(?:below|under)\s+(\d+)\s*(?:%|percent)",
     "max_utilization", lambda m: ("edges:ALL", float(m.group(1)) / 100)),
    (r"set\s+a\s+(\d+)\s*%\s+utilization\s+ceiling",
     "max_utilization", lambda m: ("edges:ALL", float(m.group(1)) / 100)),

    # load_balance (expanded patterns)
    (r"(?:balance|distribute)\s+(?:the\s+)?(?:traffic\s+)?load",
     "load_balance", lambda m: ("edges:ALL", None)),
    (r"(?:spread|distribute)\s+traffic\s+(?:as\s+)?(?:uniformly|evenly)",
     "load_balance", lambda m: ("edges:ALL", None)),
    (r"equalize\s+(?:traffic\s+|link\s+)?utilization",
     "load_balance", lambda m: ("edges:ALL", None)),

    # disable_edge
    (r"disable\s+(?:the\s+)?(?:ISL\s+|link\s+)?(?:edge\s+)?between\s+(?:node\s+)?(\d+)\s+and\s+(?:node\s+)?(\d+)",
     "disable_edge", lambda m: ("edge:" + m.group(1) + "-" + m.group(2), None)),
]

# Event/condition patterns
EVENT_PATTERN = re.compile(
    r"(?:if|during|when|in\s+case\s+of)\s+(?:a\s+|an\s+)?(" +
    "|".join(re.escape(k) for k in sorted(_EVENT_MAP.keys(), key=len, reverse=True)) +
    r")(?:\s+occurs?|\s+happens?|\s+is\s+detected)?",
    re.IGNORECASE
)


class RuleBasedParser:
    """Rule-based intent parser using regex and keyword matching."""

    def compile(self, intent_text: str) -> CompilationResult:
        t0 = time.time()
        try:
            program = self._parse(intent_text)
            latency = (time.time() - t0) * 1000
            return CompilationResult(
                success=True, program=program,
                attempts=1, latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return CompilationResult(
                success=False, attempts=1,
                errors=[str(e)], latency_ms=latency,
            )

    def _parse(self, text: str) -> ConstraintProgram:
        cp = ConstraintProgram()
        cp.source_text = text
        cp.intent_id = "rule_" + re.sub(r"\W+", "_", text[:40]).lower()
        cp.priority = self._detect_priority(text)

        # Detect conditional
        event_cond = self._detect_event(text)
        if event_cond:
            cp.event_conditions.append(event_cond)

        # Split compositional intents
        clauses = self._split_clauses(text)

        for clause in clauses:
            self._extract_constraints(clause, cp, event_cond)

        if not cp.hard_constraints and not cp.soft_constraints:
            raise ValueError("No constraints extracted from: " + text)

        cp.compiler_confidence = 0.8
        return cp

    def _detect_priority(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["immediately", "emergency", "critical", "urgent"]):
            return "high"
        if any(w in t for w in ["maintenance", "scheduled"]):
            return "medium"
        return "medium"

    def _detect_event(self, text: str) -> Optional[EventCondition]:
        m = EVENT_PATTERN.search(text)
        if m:
            event_key = m.group(1).lower().strip()
            event_type = _EVENT_MAP.get(event_key, "none")
            return EventCondition(event_type=event_type, active=False)
        return None

    def _split_clauses(self, text: str) -> List[str]:
        # Remove conditional prefix
        cleaned = EVENT_PATTERN.sub("", text).strip()
        if cleaned.startswith(","):
            cleaned = cleaned[1:].strip()

        # Split on ", and " / " and " / ","
        # But be careful not to split "from X to Y and ..."
        parts = re.split(r",\s*and\s+|,\s+and\s+|\s+and\s+(?=(?:disable|avoid|ensure|guarantee|cap|limit|reroute|take|balance|restrict))", cleaned)

        # Also split on standalone commas if parts have constraint keywords
        result = []
        for part in parts:
            sub = re.split(r",\s*(?=(?:disable|avoid|ensure|guarantee|cap|limit|reroute|take|balance|restrict))", part)
            result.extend(sub)

        return [p.strip() for p in result if p.strip()]

    def _extract_constraints(self, clause: str, cp: ConstraintProgram,
                             event_cond: Optional[EventCondition]):
        matched = False

        for pattern, ctype, extractor in PATTERNS:
            m = re.search(pattern, clause, re.IGNORECASE)
            if not m:
                continue

            if ctype == "disable_node":
                target, value = extractor(m)
                cp.hard_constraints.append(HardConstraint(
                    type="disable_node", target=target, value=value,
                    condition=event_cond,
                ))
                matched = True

            elif ctype == "disable_plane":
                target, value = extractor(m)
                cp.hard_constraints.append(HardConstraint(
                    type="disable_plane", target=target, value=value,
                    condition=event_cond,
                ))
                matched = True

            elif ctype == "disable_plane_multi":
                nums_str = m.group(1)
                nums = re.findall(r"\d+", nums_str)
                for n in nums:
                    cp.hard_constraints.append(HardConstraint(
                        type="disable_plane", target="plane:" + n, value=None,
                        condition=event_cond,
                    ))
                matched = True

            elif ctype == "reroute_away":
                target, value = extractor(m)
                cp.hard_constraints.append(HardConstraint(
                    type="reroute_away", target=target, value=value,
                    condition=event_cond,
                ))
                matched = True

            elif ctype == "avoid_latitude":
                target, value = extractor(m)
                cp.hard_constraints.append(HardConstraint(
                    type="avoid_latitude", target=target, value=value,
                    condition=event_cond,
                ))
                matched = True

            elif ctype == "avoid_region":
                target, value = extractor(m)
                cp.hard_constraints.append(HardConstraint(
                    type="avoid_region", target=target, value=value,
                    condition=event_cond,
                ))
                matched = True

            elif ctype == "max_latency_ms_flow":
                traffic = m.group(1)
                src = m.group(2)
                dst = m.group(3)
                latency = float(m.group(4))

                fs = FlowSelector(
                    traffic_class=traffic.lower() if traffic else None,
                    src_region=src.upper() if src else None,
                    dst_region=dst.upper() if dst else None,
                )
                fs_idx = len(cp.flow_selectors)
                cp.flow_selectors.append(fs)

                cp.hard_constraints.append(HardConstraint(
                    type="max_latency_ms",
                    target="flow_selector:" + str(fs_idx),
                    value=latency,
                    condition=event_cond,
                ))
                matched = True

            elif ctype == "max_latency_ms_flow2":
                latency = float(m.group(1))
                traffic = m.group(2)
                src = m.group(3)
                dst = m.group(4)
                fs = FlowSelector(
                    traffic_class=traffic.lower() if traffic else None,
                    src_region=src.upper() if src else None,
                    dst_region=dst.upper() if dst else None,
                )
                fs_idx = len(cp.flow_selectors)
                cp.flow_selectors.append(fs)
                cp.hard_constraints.append(HardConstraint(
                    type="max_latency_ms",
                    target="flow_selector:" + str(fs_idx),
                    value=latency, condition=event_cond,
                ))
                matched = True

            elif ctype == "max_latency_ms_sla":
                latency = float(m.group(1))
                traffic = m.group(2)
                src = m.group(3)
                dst = m.group(4)
                fs = FlowSelector(
                    traffic_class=traffic.lower() if traffic else None,
                    src_region=src.upper() if src else None,
                    dst_region=dst.upper() if dst else None,
                )
                fs_idx = len(cp.flow_selectors)
                cp.flow_selectors.append(fs)
                cp.hard_constraints.append(HardConstraint(
                    type="max_latency_ms",
                    target="flow_selector:" + str(fs_idx),
                    value=latency, condition=event_cond,
                ))
                matched = True

            elif ctype in ("max_latency_ms_must", "max_latency_ms_route"):
                traffic = m.group(1)
                src = m.group(2)
                dst = m.group(3)
                latency = float(m.group(4))
                fs = FlowSelector(
                    traffic_class=traffic.lower() if traffic else None,
                    src_region=src.upper() if src else None,
                    dst_region=dst.upper() if dst else None,
                )
                fs_idx = len(cp.flow_selectors)
                cp.flow_selectors.append(fs)
                cp.hard_constraints.append(HardConstraint(
                    type="max_latency_ms",
                    target="flow_selector:" + str(fs_idx),
                    value=latency, condition=event_cond,
                ))
                matched = True

            elif ctype == "max_latency_ms_keep":
                src = m.group(1)
                dst = m.group(2)
                traffic = m.group(3)
                latency = float(m.group(4))
                fs = FlowSelector(
                    traffic_class=traffic.lower() if traffic else None,
                    src_region=src.upper() if src else None,
                    dst_region=dst.upper() if dst else None,
                )
                fs_idx = len(cp.flow_selectors)
                cp.flow_selectors.append(fs)
                cp.hard_constraints.append(HardConstraint(
                    type="max_latency_ms",
                    target="flow_selector:" + str(fs_idx),
                    value=latency, condition=event_cond,
                ))
                matched = True

            elif ctype == "max_latency_ms_simple":
                target, value = extractor(m)
                cp.hard_constraints.append(HardConstraint(
                    type="max_latency_ms", target=target, value=value,
                    condition=event_cond,
                ))
                matched = True

            elif ctype == "max_utilization":
                target, value = extractor(m)
                cp.soft_constraints.append(SoftConstraint(
                    type="max_utilization", target=target, value=value,
                    penalty=1.0, condition=event_cond,
                ))
                matched = True

            elif ctype == "load_balance":
                target, _ = extractor(m)
                cp.soft_constraints.append(SoftConstraint(
                    type="load_balance", target=target, value=None,
                    penalty=1.0, condition=event_cond,
                ))
                matched = True

            elif ctype == "disable_edge":
                target, value = extractor(m)
                cp.hard_constraints.append(HardConstraint(
                    type="disable_edge", target=target, value=value,
                    condition=event_cond,
                ))
                matched = True

            if matched:
                break

        if not matched:
            # Try to extract something from unmatched clause
            # Look for any node reference
            node_m = re.search(r"node\s+(\d+)", clause, re.IGNORECASE)
            if node_m:
                cp.hard_constraints.append(HardConstraint(
                    type="disable_node",
                    target="node:" + node_m.group(1),
                    condition=event_cond,
                ))
                return

            # Look for any plane reference
            plane_m = re.search(r"plane\s+(\d+)", clause, re.IGNORECASE)
            if plane_m:
                cp.hard_constraints.append(HardConstraint(
                    type="disable_plane",
                    target="plane:" + plane_m.group(1),
                    condition=event_cond,
                ))
                return

            # If nothing matched at all, this clause is unparseable
            # Don't raise - just skip (partial parse is better than failure)
