"""Formal constraint representation for intent-driven LEO routing.

ConstraintProgram is the verified output of the LLM intent compiler.
It captures hard/soft constraints, objective weights, and selectors
that the constrained GNN optimizer uses for topology control + routing.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


# --- Enums ---

class Priority(str, Enum):
    CRITICAL = "critical"   # safety, physical
    HIGH = "high"           # SLA, resilience
    MEDIUM = "medium"       # load balancing
    LOW = "low"             # efficiency preferences

class HardConstraintType(str, Enum):
    MAX_LATENCY_MS = "max_latency_ms"
    K_EDGE_DISJOINT = "k_edge_disjoint_paths"
    AVOID_REGION = "avoid_region"
    AVOID_LATITUDE = "avoid_latitude"
    DISABLE_NODE = "disable_node"
    DISABLE_PLANE = "disable_plane"
    DISABLE_EDGE = "disable_edge"
    REROUTE_AWAY = "reroute_away"
    MIN_CAPACITY_RESERVE = "min_capacity_reserve"
    MAX_HOPS = "max_hops"

class SoftConstraintType(str, Enum):
    MAX_UTILIZATION = "max_utilization"
    LOAD_BALANCE = "load_balance"
    MIN_BACKUP_PATHS = "min_backup_paths"
    MINIMIZE_SWITCHING = "minimize_switching"
    PREFER_LOW_LATENCY = "prefer_low_latency"

class FallbackPolicy(str, Enum):
    REJECT_IF_HARD_INFEASIBLE = "reject_if_hard_infeasible"
    RELAX_SOFT_FIRST = "relax_soft_first"
    REPORT_UNSAT_CORE = "report_unsat_core"

class EventType(str, Enum):
    SOLAR_STORM = "solar_storm"
    NODE_FAILURE = "node_failure"
    GATEWAY_FAILURE = "gateway_failure"
    MAINTENANCE = "maintenance"
    OVERLOAD = "overload"
    NONE = "none"


# --- Known entities for grounding ---

KNOWN_REGIONS = {
    "NYC": (40.7, -74.0), "LONDON": (51.5, -0.1),
    "TOKYO": (35.7, 139.7), "SHANGHAI": (31.2, 121.5),
    "MUMBAI": (19.1, 72.9), "SAO_PAULO": (-23.5, -46.6),
    "SYDNEY": (-33.9, 151.2), "DUBAI": (25.2, 55.3),
    "FRANKFURT": (50.1, 8.7), "PARIS": (48.9, 2.3),
    "MADRID": (40.4, -3.7), "HONOLULU": (21.3, -157.8),
    "ARCTIC": (75.0, 0.0), "POLAR_NORTH": (80.0, 0.0),
}

KNOWN_TRAFFIC_CLASSES = [
    "financial", "emergency", "telemetry", "video",
    "bulk", "control_plane", "maritime", "military",
]

KNOWN_CORRIDORS = [
    "transatlantic", "transpacific", "europe_us",
    "asia_europe", "north_south", "polar",
]


# --- Dataclasses ---

@dataclass
class FlowSelector:
    """Identifies a set of traffic flows."""
    traffic_class: Optional[str] = None
    src_region: Optional[str] = None
    dst_region: Optional[str] = None
    src_node: Optional[int] = None
    dst_node: Optional[int] = None
    src_plane: Optional[int] = None
    dst_plane: Optional[int] = None
    corridor: Optional[str] = None

@dataclass
class EdgeSelector:
    """Identifies a set of ISL edges."""
    scope: str = "ALL"  # ALL, corridor name, or specific
    latitude_band: Optional[str] = None  # "polar", "mid", "equatorial"
    plane: Optional[int] = None
    edge_ids: Optional[List[tuple]] = None

@dataclass
class NodeSelector:
    """Identifies a set of satellite nodes."""
    scope: str = "ALL"
    node_ids: Optional[List[int]] = None
    plane: Optional[int] = None
    region: Optional[str] = None

@dataclass
class TimeWindow:
    """Temporal scope for a constraint."""
    start: str = "now"  # "now" or ISO timestamp
    duration_min: Optional[float] = None  # None = indefinite
    condition: Optional[str] = None  # event condition

@dataclass
class EventCondition:
    """Conditional trigger for a constraint."""
    event_type: str = "none"
    active: bool = False

@dataclass
class HardConstraint:
    """Must be satisfied; violation = infeasible."""
    type: str = ""
    target: str = ""  # "flow_selector:0", "edges:ALL", "node:142"
    value: Any = None
    condition: Optional[EventCondition] = None

@dataclass
class SoftConstraint:
    """Penalized if violated; can be relaxed."""
    type: str = ""
    target: str = ""
    value: Any = None
    penalty: float = 1.0
    condition: Optional[EventCondition] = None

@dataclass
class ObjectiveWeights:
    """Relative importance of optimization objectives."""
    priority_traffic: float = 1.0
    latency: float = 1.0
    congestion: float = 1.0
    switching: float = 1.0
    throughput: float = 1.0
    drop_penalty: float = 5.0

@dataclass
class ConstraintProgram:
    """Verified output of the LLM intent compiler.

    This is the formal interface between the semantic layer (LLM)
    and the optimization layer (GNN + constrained solver).
    """
    intent_id: str = ""
    source_text: str = ""
    time_window: TimeWindow = field(default_factory=TimeWindow)
    priority: str = "medium"
    flow_selectors: List[FlowSelector] = field(default_factory=list)
    edge_selectors: List[EdgeSelector] = field(default_factory=list)
    node_selectors: List[NodeSelector] = field(default_factory=list)
    hard_constraints: List[HardConstraint] = field(default_factory=list)
    soft_constraints: List[SoftConstraint] = field(default_factory=list)
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    fallback_policy: str = "reject_if_hard_infeasible"
    compiler_confidence: float = 0.0
    event_conditions: List[EventCondition] = field(default_factory=list)

    # --- Verification status (set by verifier, not compiler) ---
    verified: bool = False
    verification_errors: List[str] = field(default_factory=list)
    verification_warnings: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> "ConstraintProgram":
        d = json.loads(s)
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> "ConstraintProgram":
        cp = cls()
        cp.intent_id = d.get("intent_id", "")
        cp.source_text = d.get("source_text", "")
        cp.priority = d.get("priority", "medium")
        cp.fallback_policy = d.get("fallback_policy", "reject_if_hard_infeasible")
        cp.compiler_confidence = d.get("compiler_confidence", 0.0)

        tw = d.get("time_window", {})
        cp.time_window = TimeWindow(
            start=tw.get("start", "now"),
            duration_min=tw.get("duration_min"),
            condition=tw.get("condition"),
        )

        cp.flow_selectors = [FlowSelector(**f) for f in d.get("flow_selectors", [])]
        cp.edge_selectors = [EdgeSelector(**e) for e in d.get("edge_selectors", [])]
        cp.node_selectors = [NodeSelector(**n) for n in d.get("node_selectors", [])]

        cp.hard_constraints = []
        for hc in d.get("hard_constraints", []):
            cond = None
            if hc.get("condition"):
                cond = EventCondition(**hc["condition"])
            cp.hard_constraints.append(HardConstraint(
                type=hc["type"], target=hc["target"],
                value=hc.get("value"), condition=cond,
            ))

        cp.soft_constraints = []
        for sc in d.get("soft_constraints", []):
            cond = None
            if sc.get("condition"):
                cond = EventCondition(**sc["condition"])
            cp.soft_constraints.append(SoftConstraint(
                type=sc["type"], target=sc["target"],
                value=sc.get("value"), penalty=sc.get("penalty", 1.0),
                condition=cond,
            ))

        ow = d.get("objective_weights", {})
        cp.objective_weights = ObjectiveWeights(**ow) if ow else ObjectiveWeights()

        cp.event_conditions = [
            EventCondition(**e) for e in d.get("event_conditions", [])
        ]
        return cp
