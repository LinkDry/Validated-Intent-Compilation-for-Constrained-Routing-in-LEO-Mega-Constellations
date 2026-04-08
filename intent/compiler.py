"""LLM Intent Compiler: natural language → verified ConstraintProgram.

Uses LM Studio API (OpenAI-compatible) with few-shot prompting
and verifier-feedback repair loop.
"""

from __future__ import annotations
import json, re, time, logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

import requests
import numpy as np

from intent.schema import ConstraintProgram
from intent.verifier import ConstraintVerifier, VerificationResult

logger = logging.getLogger(__name__)

# ── Schema description for system prompt ──

SCHEMA_DESCRIPTION = """\
You are a LEO satellite constellation routing compiler. Your job is to convert
natural language operator intents into a formal ConstraintProgram JSON.

## Constellation
- 20 orbital planes × 20 satellites/plane = 400 nodes (IDs 0-399)
- Node ID = plane * 20 + sat_in_plane
- 4 ISL links per satellite (2 intra-plane, 2 inter-plane)
- Polar link dropout above ~75° latitude

## ConstraintProgram JSON Schema

```json
{
  "intent_id": "<unique string>",
  "source_text": "<original operator intent>",
  "priority": "critical|high|medium|low",
  "time_window": {"start": "now", "duration_min": null},
  "flow_selectors": [
    {"traffic_class": "<str>", "src_region": "<str>", "dst_region": "<str>",
     "src_node": <int|null>, "dst_node": <int|null>,
     "src_plane": <int|null>, "dst_plane": <int|null>, "corridor": "<str|null>"}
  ],
  "edge_selectors": [],
  "node_selectors": [],
  "hard_constraints": [
    {"type": "<HardConstraintType>", "target": "<str>", "value": <any>,
     "condition": {"event_type": "<str>", "active": false} | null}
  ],
  "soft_constraints": [
    {"type": "<SoftConstraintType>", "target": "<str>", "value": <any>,
     "penalty": <float>, "condition": null}
  ],
  "objective_weights": {},
  "fallback_policy": "reject_if_hard_infeasible",
  "event_conditions": []
}
```

## Valid Types

HardConstraintType: max_latency_ms, k_edge_disjoint_paths, avoid_region,
  avoid_latitude, disable_node, disable_plane, disable_edge, reroute_away,
  min_capacity_reserve, max_hops

SoftConstraintType: max_utilization, load_balance, min_backup_paths,
  minimize_switching, prefer_low_latency

## Valid Regions
NYC, LONDON, TOKYO, SHANGHAI, MUMBAI, SAO_PAULO, SYDNEY, DUBAI,
FRANKFURT, PARIS, MADRID, HONOLULU, ARCTIC, POLAR_NORTH

## Valid Traffic Classes
financial, emergency, telemetry, video, bulk, control_plane, maritime, military

## Valid Event Types
solar_storm, node_failure, gateway_failure, maintenance, overload, none

## Target Format
- Node constraints: "node:<id>" (e.g. "node:42")
- Plane constraints: "plane:<id>" (e.g. "plane:5")
- Flow constraints: "flow_selector:<index>" (e.g. "flow_selector:0")
- Edge constraints: "edges:ALL"

## Priority Guidelines
- critical: safety, physical constraints, emergency
- high: SLA, resilience, failures
- medium: load balancing, optimization
- low: efficiency preferences

## Rules
1. Output ONLY valid JSON. No explanation, no markdown fences.
2. Set intent_id to a descriptive slug derived from the intent.
3. Copy the original text into source_text verbatim.
4. For conditional/event-triggered intents, set condition on each constraint
   and add the event to event_conditions with active=false.
5. For infeasible requests (impossible latency, out-of-range nodes/planes),
   still produce the JSON faithfully — the verifier will catch errors.
6. flow_selectors are referenced by index: the first is "flow_selector:0", etc.
"""

# ── Few-shot examples ──

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Disable node 142 immediately"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent_id": "disable-node-142",
            "source_text": "Disable node 142 immediately",
            "priority": "high",
            "time_window": {"start": "now", "duration_min": None},
            "flow_selectors": [],
            "edge_selectors": [],
            "node_selectors": [],
            "hard_constraints": [
                {"type": "disable_node", "target": "node:142", "value": None, "condition": None}
            ],
            "soft_constraints": [],
            "objective_weights": {},
            "fallback_policy": "reject_if_hard_infeasible",
            "event_conditions": []
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": "Ensure financial traffic from NYC to TOKYO stays under 80ms"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent_id": "financial-nyc-tokyo-80ms",
            "source_text": "Ensure financial traffic from NYC to TOKYO stays under 80ms",
            "priority": "high",
            "time_window": {"start": "now", "duration_min": None},
            "flow_selectors": [
                {"traffic_class": "financial", "src_region": "NYC", "dst_region": "TOKYO",
                 "src_node": None, "dst_node": None, "src_plane": None, "dst_plane": None, "corridor": None}
            ],
            "edge_selectors": [],
            "node_selectors": [],
            "hard_constraints": [
                {"type": "max_latency_ms", "target": "flow_selector:0", "value": 80.0, "condition": None}
            ],
            "soft_constraints": [],
            "objective_weights": {},
            "fallback_policy": "reject_if_hard_infeasible",
            "event_conditions": []
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": "Disable plane 7 and avoid polar links above 75 degrees, cap utilization at 80%"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent_id": "disable-plane7-polar-util",
            "source_text": "Disable plane 7 and avoid polar links above 75 degrees, cap utilization at 80%",
            "priority": "critical",
            "time_window": {"start": "now", "duration_min": None},
            "flow_selectors": [],
            "edge_selectors": [],
            "node_selectors": [],
            "hard_constraints": [
                {"type": "disable_plane", "target": "plane:7", "value": None, "condition": None},
                {"type": "avoid_latitude", "target": "edges:ALL", "value": 75.0, "condition": None}
            ],
            "soft_constraints": [
                {"type": "max_utilization", "target": "edges:ALL", "value": 0.8, "penalty": 2.0, "condition": None}
            ],
            "objective_weights": {},
            "fallback_policy": "reject_if_hard_infeasible",
            "event_conditions": []
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": "If a solar storm occurs, reroute away from node 55 and avoid polar links above 70°"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent_id": "solar-storm-reroute-55-polar",
            "source_text": "If a solar storm occurs, reroute away from node 55 and avoid polar links above 70°",
            "priority": "critical",
            "time_window": {"start": "now", "duration_min": None},
            "flow_selectors": [],
            "edge_selectors": [],
            "node_selectors": [],
            "hard_constraints": [
                {"type": "reroute_away", "target": "node:55", "value": None,
                 "condition": {"event_type": "solar_storm", "active": False}},
                {"type": "avoid_latitude", "target": "edges:ALL", "value": 70.0,
                 "condition": {"event_type": "solar_storm", "active": False}}
            ],
            "soft_constraints": [],
            "objective_weights": {},
            "fallback_policy": "reject_if_hard_infeasible",
            "event_conditions": [{"event_type": "solar_storm", "active": False}]
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": "Route emergency traffic from LONDON to MUMBAI under 50ms while avoiding DUBAI"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent_id": "emergency-london-mumbai-50ms-avoid-dubai",
            "source_text": "Route emergency traffic from LONDON to MUMBAI under 50ms while avoiding DUBAI",
            "priority": "critical",
            "time_window": {"start": "now", "duration_min": None},
            "flow_selectors": [
                {"traffic_class": "emergency", "src_region": "LONDON", "dst_region": "MUMBAI",
                 "src_node": None, "dst_node": None, "src_plane": None, "dst_plane": None, "corridor": None}
            ],
            "edge_selectors": [],
            "node_selectors": [],
            "hard_constraints": [
                {"type": "max_latency_ms", "target": "flow_selector:0", "value": 50.0, "condition": None},
                {"type": "avoid_region", "target": "edges:ALL", "value": "DUBAI", "condition": None}
            ],
            "soft_constraints": [],
            "objective_weights": {},
            "fallback_policy": "reject_if_hard_infeasible",
            "event_conditions": []
        }, ensure_ascii=False)
    },
]


@dataclass
class CompilationResult:
    """Result of intent compilation."""
    success: bool = False
    program: Optional[ConstraintProgram] = None
    verification: Optional[VerificationResult] = None
    raw_json: str = ""
    attempts: int = 0
    errors: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


class IntentCompiler:
    """Compiles natural language intents to verified ConstraintPrograms.

    Uses LM Studio API with few-shot prompting and verifier-feedback repair.
    """

    def __init__(
        self,
        verifier: ConstraintVerifier,
        api_base: str = "http://localhost:1234/v1",
        model: str = "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        self.verifier = verifier
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout

    def compile(self, intent_text: str) -> CompilationResult:
        """Compile intent text to a verified ConstraintProgram."""
        result = CompilationResult()
        t0 = time.time()

        messages = [
            {"role": "system", "content": SCHEMA_DESCRIPTION},
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": intent_text},
        ]

        for attempt in range(1, self.max_retries + 1):
            result.attempts = attempt

            # Call LLM
            raw = self._call_llm(messages)
            if raw is None:
                result.errors.append(f"Attempt {attempt}: LLM API call failed")
                continue

            # Extract JSON
            json_str = self._extract_json(raw)
            if json_str is None:
                result.errors.append(f"Attempt {attempt}: No valid JSON in response")
                # Feed back to LLM
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    "Your response was not valid JSON. Please output ONLY "
                    "the ConstraintProgram JSON with no extra text."})
                continue

            result.raw_json = json_str

            # Parse to ConstraintProgram
            try:
                cp = ConstraintProgram.from_json(json_str)
            except Exception as e:
                result.errors.append(f"Attempt {attempt}: JSON parse error: {e}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    f"JSON parsing failed: {e}. Fix the JSON and output again."})
                continue

            # Verify
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
                        f"The verifier found errors: {err_msg}. "
                        f"Fix these issues and output the corrected JSON only."})

        result.latency_ms = (time.time() - t0) * 1000
        return result

    def _call_llm(self, messages: List[Dict]) -> Optional[str]:
        """Call LM Studio chat completion API."""
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return None

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from LLM response, handling markdown fences."""
        # Try direct parse first
        text = text.strip()

        # Remove thinking tags if present (Qwen reasoning model)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Try direct parse
        if text.startswith("{"):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass

        # Try extracting from markdown code fence
        m = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', text, re.DOTALL)
        if m:
            try:
                json.loads(m.group(1))
                return m.group(1)
            except json.JSONDecodeError:
                pass

        # Try finding the outermost { ... }
        depth = 0
        start = None
        for i, c in enumerate(text):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        start = None

        return None
