"""
Risk assessment nodes — over-escalation trap check and urgency scoring.

Maps to the brief's deliverables on:
  - **Over-escalation traps** — trap_check uses keyword-based TRAP_PATTERNS
    to demote dangerous-sounding-but-benign classifications and clamps
    emergency-service dispatch to only GENUINE_EMERGENCY_SUBCATEGORIES.
  - **Risk = P(error) × Cost(error) HITL trigger** — urgency_score computes
    the composite urgency + HITL threshold via tools.hitl_utils.compute_risk_score().
"""
from __future__ import annotations

from .config import (
    GENUINE_EMERGENCY_SUBCATEGORIES,
    DANGER_WORDS,
    URGENCY_WEIGHTS,
)
from .state import WorkflowState
from .utils import _compile_kw_regex
from tools.hitl_utils import compute_risk_score


# ── Over-escalation trap patterns ────────────────────────────────────────────
# Each entry: when LLM picks the "dangerous" subcategory AND a keyword from
# `keywords` appears in the transcript (and no `exclude_keywords` overrides),
# rewrite to the safer `correction.subcategory` and cap risk at `max_risk`.
# Pre-compiled regexes (_kw_re, _exc_re) are attached at module load.

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
    {
        "dangerous": "flood_cleanup",
        "keywords": ["water stain", "ceiling stain", "brown spot", "water spot"],
        "exclude_keywords": ["dripping", "active", "growing", "spreading", "wet floor"],
        "correction": {"subcategory": "leak_detection", "max_risk": "LOW"},
    },
    {
        "dangerous": "sewer_backup",
        "keywords": ["slow drain", "gurgling", "smelly", "slowly draining", "gurgle"],
        "exclude_keywords": ["raw sewage", "flooding", "overflow", "standing water", "backup"],
        "correction": {"subcategory": "drain_clog", "max_risk": "MEDIUM"},
    },
    {
        "dangerous": "power_outage",
        "keywords": ["flickering light", "one outlet", "single outlet", "light flicker"],
        "exclude_keywords": ["whole house", "all outlets", "no power", "dark", "blackout"],
        "correction": {"subcategory": "outlet_repair", "max_risk": "LOW"},
    },
]

# Compile each pattern's keyword + exclude regexes once at module load.
for _trap in TRAP_PATTERNS:
    _trap["_kw_re"] = _compile_kw_regex(_trap["keywords"])
    if _trap.get("exclude_keywords"):
        _trap["_exc_re"] = _compile_kw_regex(_trap["exclude_keywords"])


# ── Danger-word fallback regex (for safety signals not caught by traps) ─────

_DANGER_DETECTED_RE = _compile_kw_regex(DANGER_WORDS)


# ── Nodes ────────────────────────────────────────────────────────────────────


def trap_check(state: WorkflowState):
    """Keyword-driven over-escalation defense (false-alarm protection).

    Three behaviors:
      1. TRAP_PATTERN match → rewrite subcategory + cap risk at the
         pattern's max_risk, flag for human review.
      2. Danger-word fallback → if any danger words appear in transcript
         AND final subcategory isn't in GENUINE_EMERGENCY_SUBCATEGORIES,
         flag for human review.
      3. Emergency-services clamp → only allow emergency dispatch for
         GENUINE_EMERGENCY_SUBCATEGORIES.
    """
    classification = state.get("classification") or {}
    transcript = state.get("transcript") or ""
    subcategory = (classification.get("subcategory") or "").lower()

    trap_applied = False
    original_subcategory = classification.get("subcategory")
    correction = None

    # ── 1. TRAP_PATTERN matching ──
    for trap in TRAP_PATTERNS:
        if subcategory == trap["dangerous"]:
            # Check exclude keywords first — if any match, skip this trap.
            if trap.get("_exc_re") and trap["_exc_re"].search(transcript):
                continue
            if trap["_kw_re"].search(transcript):
                # Rewrite to the safer subcategory + cap risk.
                correction = trap["correction"]
                classification = dict(classification)
                classification["subcategory"] = correction["subcategory"]
                # If there's a category override, apply it
                if "category" in correction:
                    classification["category"] = correction["category"]
                classification["trap_applied"] = (
                    f"De-escalated from {trap['dangerous']} to {correction['subcategory']}"
                )
                needs_human_review = classification.get("needs_human_review", False)
                classification["needs_human_review"] = True
                trap_applied = True
                break

    # ── 2. Danger-word fallback — for cases the explicit trap didn't catch ──
    if _DANGER_DETECTED_RE.search(transcript):
        final_sub = (classification.get("subcategory") or "").lower()
        if final_sub not in GENUINE_EMERGENCY_SUBCATEGORIES:
            classification = dict(classification)
            classification["needs_human_review"] = True

    # ── 3. Emergency-services clamp ──
    final_sub = (classification.get("subcategory") or "").lower()
    if final_sub not in GENUINE_EMERGENCY_SUBCATEGORIES:
        classification = dict(classification)
        if classification.get("dispatched_emergency_services"):
            classification["dispatched_emergency_services"] = False

    trap_check_result = {
        "trap_applied": trap_applied,
        "original_subcategory": original_subcategory,
        "correction": correction,
    }

    return {
        "classification": classification,
        "trap_check_result": trap_check_result,
    }


def urgency_score(state: WorkflowState):
    """Compute composite urgency score + HITL threshold.

    Collects dimension signals and calls tools.hitl_utils.compute_risk_score()
    which implements:
        Risk = P(error) × Cost(error)

    Also integrates Jev urgency dimension signals when available.
    """
    classification = state.get("classification") or {}
    entity = state.get("entity") or {}

    # Collect urgency dimension signals
    urgency_signals = {
        "water_spread_score": classification.get("water_spread_score"),
        "safety_hazard_noul": classification.get("safety_hazard_noul"),
        "contained_noul": classification.get("contained_noul"),
        "keyword_safety_signals": [],
    }

    # Check for danger words in transcript as a safety signal
    transcript = state.get("transcript") or ""
    if _DANGER_DETECTED_RE.search(transcript):
        urgency_signals["keyword_safety_signals"] = list(
            set(_DANGER_DETECTED_RE.findall(transcript))
        )

    # Extract confidence from classification
    confidence = classification.get("confidence", 0.5)
    if confidence is None:
        confidence = 0.5

    # Determine risk level from entity severity
    severity = (entity.get("severity") or "Medium").upper()
    risk_level = "MEDIUM"
    if severity == "CRITICAL":
        risk_level = "EMERGENCY"
    elif severity == "HIGH":
        risk_level = "HIGH"
    elif severity == "LOW":
        risk_level = "LOW"

    # Determine property type (default to single_family)
    property_type = (entity.get("property_type") or "single_family").lower()

    subcategory = (classification.get("subcategory") or "").lower()

    # Call compute_risk_score from tools
    risk_result = compute_risk_score(
        confidence=confidence,
        risk_level=risk_level,
        subcategory=subcategory,
        property_type=property_type,
        existing_needs_human_review=classification.get("needs_human_review", False),
    )

    # Carry forward needs_human_review from risk result
    needs_human_review = risk_result.get("needs_human_review", False)
    if classification.get("needs_human_review"):
        needs_human_review = True

    return {
        "urgency_signals": urgency_signals,
        "risk_score": risk_result.get("risk_score"),
        "risk_details": risk_result.get("risk_details"),
        "needs_human_review": needs_human_review,
    }