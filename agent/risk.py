"""
Risk assessment nodes — over-escalation trap check and urgency scoring.

Stubs for:
  - trap_check: keyword-based over-escalation guard (false alarm protection)
  - urgency_score: composite urgency calculation

Each node returns its expected state fields with None/empty defaults.
No logic implementation yet — those come in later tasks.
"""
from __future__ import annotations

from .config import URGENCY_WEIGHTS, GENUINE_EMERGENCY_SUBCATEGORIES, DANGER_WORDS
from .state import WorkflowState
from .utils import _compile_kw_regex, _DANGER_WORDS_RE


# ── Over-escalation trap patterns (placeholder) ──────────────────────────────

TRAP_PATTERNS = [
    {
        "dangerous": "flood_cleanup",
        "keywords": ["damp", "humid", "musty", "small leak", "drip", "dripping"],
        "correction": {"subcategory": "leak_detection", "max_risk": "MEDIUM"},
    },
    {
        "dangerous": "flood_cleanup",
        "keywords": ["mold spot", "small patch", "corner", "bathroom ceiling"],
        "exclude_keywords": ["spreading", "growing", "large", "entire", "wall-to-wall"],
        "correction": {"subcategory": "mold_remediation", "max_risk": "MEDIUM"},
    },
]

# Compile each pattern's keyword regexes once at module load.
for _trap in TRAP_PATTERNS:
    _trap["_kw_re"] = _compile_kw_regex(_trap["keywords"])
    if _trap.get("exclude_keywords"):
        _trap["_exc_re"] = _compile_kw_regex(_trap["exclude_keywords"])


# ── Nodes ────────────────────────────────────────────────────────────────────


def trap_check(state: WorkflowState):
    """Stub: keyword-driven over-escalation defense.

    TODO: implement full trap pattern matching:
      1. Check each TRAP_PATTERN against the classification subcategory
         and transcript for keyword matches
      2. If matched, rewrite subcategory/category and cap risk
      3. Clamp dispatched_emergency_services to only
         GENUINE_EMERGENCY_SUBCATEGORIES
      4. Flag needs_human_review when de-escalation is applied
    """
    classification = state.get("classification") or {}

    trap_check_result = {
        "trap_applied": False,
        "original_subcategory": classification.get("subcategory"),
        "correction": None,
    }

    return {
        "trap_check_result": trap_check_result,
    }


def urgency_score(state: WorkflowState):
    """Stub: composite urgency score calculation.

    TODO: implement full urgency scoring:
      1. Collect dimension signals (from Jev Score/Noul, keyword matches)
      2. Apply weights from URGENCY_WEIGHTS config
      3. Calculate Composite Urgency = weighted blend of signals
      4. Compute HITL Threshold = Composite Urgency × P(error) × Cost(error)
      5. Set needs_human_review when threshold >= HITL_THRESHOLD
    """
    classification = state.get("classification") or {}

    urgency_signals = {
        "water_spread_score": None,    # Jev Score (0-1) — how fast is water spreading
        "safety_hazard_noul": None,    # Jev Noul — is there a safety hazard?
        "contained_noul": None,        # Jev Noul — is the issue contained?
        "keyword_safety_signals": [],  # matched danger keywords from DANGER_WORDS
    }

    urgency_score = None  # composite = weighted blend

    return {
        "urgency_signals": urgency_signals,
        "urgency_score": urgency_score,
    }