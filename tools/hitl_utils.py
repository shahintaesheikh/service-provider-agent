"""
HITL (Human-in-the-Loop) Utility Functions

Two distinct function groups with different natures:

1. **Clarification Gate (intake-side):** Deterministic heuristics that detect
   when the user's description is too vague, missing location info, or has
   ambiguous signals. If triggered, routes to a gather_followup loop. This is
   a lightweight quick-check before classification — NOT the same as the HITL
   validator.

2. **HITL Validator + Override + Trainer (end of pipeline):** Checks
   needs_human_review after classification + provider matching. If true, pauses
   for a human reviewer via interrupt(). Human corrections get applied via
   override, then the full trainer log is written.

Trainer logs write to a local JSON file for now (MVP placeholder).
In production this goes to a proper database.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.types import interrupt


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1: Clarification Gate (intake-side)
# ═══════════════════════════════════════════════════════════════════════════════

# — Heuristic thresholds ------------------------------------------------------

SPARSE_FIRST_UTTERANCE_WORDS = 6
SPARSE_TOTAL_WORDS = 35

# — Taxonomy surface forms for the generic-opening heuristic -------------------
# These are the home-service categories and subcategories the agent knows about.
# A caller who literally repeats a taxonomy label ("issue with plumbing") instead
# of describing a symptom triggers the generic-opening flag.
_HOME_SERVICE_LABELS = [
    # Categories
    "plumbing",
    "hvac",
    "electrical",
    "roofing",
    "water damage",
    "appliance repair",
    "pest control",
    "locksmith",
    "handyman",
    # Common subcategory surface forms
    "pipe leak",
    "drain clog",
    "water heater",
    "toilet repair",
    "faucet repair",
    "ac repair",
    "furnace repair",
    "thermostat",
    "no heat",
    "no cooling",
    "outlet repair",
    "breaker trip",
    "wiring issue",
    "leaky roof",
    "shingle repair",
    "gutter repair",
    "flood cleanup",
    "mold remediation",
    "drywall repair",
    "refrigerator repair",
    "oven repair",
    "washer repair",
    "dryer repair",
    "dishwasher repair",
    "rodent removal",
    "termite treatment",
    "bed bug treatment",
    "lock repair",
    "key replacement",
    "door lock",
    "window repair",
    "drywall patch",
    "painting",
    "caulking",
    "furniture assembly",
]

_GENERIC_OPENING_RX = re.compile(
    r"(?:there'?s? an?|i mean,?|you know,?)?\s*(?:issue|problem)\s+with\s+("
    + "|".join(re.escape(label) for label in _HOME_SERVICE_LABELS)
    + r")",
    re.IGNORECASE,
)


# — Internal helpers -----------------------------------------------------------


def _parse_transcript(transcript: str) -> list[dict]:
    """Parse a transcript string into a list of turns.

    Expects lines prefixed with ``[CALLER]`` or ``[AGENT]``, or treats each
    line as a caller utterance. Returns a list of ``{"speaker": ..., "text": ...}``
    dicts consistent with the reference project's ``turns_split`` format.
    """
    turns: list[dict] = []
    for line in transcript.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("[CALLER]"):
            turns.append({"speaker": "caller", "text": line[len("[CALLER]") :].strip()})
        elif line.startswith("[AGENT]"):
            turns.append({"speaker": "agent", "text": line[len("[AGENT]") :].strip()})
        else:
            # Unmarked lines are treated as caller speech
            turns.append({"speaker": "caller", "text": line})
    if not turns:
        # Fallback: treat entire transcript as one caller turn
        turns = [{"speaker": "caller", "text": transcript.strip() or transcript}]
    return turns


def _sparse_caller(turns: list[dict]) -> Optional[tuple[int, int]]:
    """Check if the caller's speech is abnormally sparse.

    Fires when the caller's first utterance is short (≤6 words) AND their
    total speech across all turns is short (≤35 words). Sparse calls on
    safety-sensitive symptoms often warrant follow-up.
    """
    caller_texts = [t["text"] for t in turns if t.get("speaker") == "caller"]
    if not caller_texts:
        return None
    first = len(caller_texts[0].split())
    total = sum(len(t.split()) for t in caller_texts)
    if first <= SPARSE_FIRST_UTTERANCE_WORDS and total <= SPARSE_TOTAL_WORDS:
        return (first, total)
    return None


def _generic_opening(turns: list[dict]) -> Optional[str]:
    """Check if the caller's first utterance repeats a taxonomy label.

    Fires when the caller's first utterance literally repeats a known label
    (e.g., "issue with plumbing"). This signals that an operator or the
    system's taxonomy is leading the caller rather than the caller volunteering
    a symptom.
    """
    first = next(
        (t["text"] for t in turns if t.get("speaker") == "caller"),
        "",
    )
    m = _GENERIC_OPENING_RX.search(first.lower())
    return m.group(1) if m else None


def _missing_location(entity: Optional[dict]) -> Optional[str]:
    """Check if the extracted entity lacks location information.

    Fires when no address, city, or zip code is present in the entity.
    Returns None when entity is None (pre-extraction stage) so the
    caller is not penalised for missing data that hasn't been collected yet.
    """
    if not entity:
        return None
    address = entity.get("address") or ""
    city = entity.get("city") or ""
    zip_code = entity.get("zip") or ""
    if not address and not city and not zip_code:
        return "missing_address_city_zip"
    if not address:
        return "missing_street_address"
    return None


def _vague_problem(entity: Optional[dict]) -> Optional[str]:
    """Check if the description is too vague to classify.

    Relies on the ``description_specific_enough`` flag set by the extraction
    step, or falls back to heuristic checks on the summary.
    Returns None when entity is None (pre-extraction stage) so the
    caller is not penalised for missing data that hasn't been collected yet.
    """
    if not entity:
        return None
    # If the extractor explicitly flagged it
    if entity.get("description_specific_enough") is False:
        return "description_not_specific_enough"
    # Fallback: check summary for vagueness markers
    summary = (entity.get("summary") or "").lower()
    vague_markers = ["issue", "problem", "something", "not working"]
    if summary in vague_markers or len(summary.split()) < 3:
        return "vague_summary"
    return None


def _build_clarification_question(*, sparse: Optional[tuple], generic: Optional[str]) -> str:
    """Pick the most specific clarifying question for whichever heuristic(s) fired."""
    if sparse or generic:
        return (
            "Could you tell me a bit more about what's going wrong — for example, "
            "what specifically isn't working and how long it's been like that?"
        )
    return (
        "Could you describe the issue in a bit more detail so I can route it correctly?"
    )


# — Public API -----------------------------------------------------------------


def check_clarification_needed(
    transcript: str,
    entity: Optional[dict] = None,
    user_profile: Optional[dict] = None,
) -> dict:
    """Apply deterministic heuristics to decide whether clarification is needed.

    This is a lightweight, deterministic quick-check that runs *before*
    classification. It is NOT the same as the HITL validator — this gate
    catches vague or under-specified user descriptions early so the agent
    can ask follow-up questions before committing to a classification path.

    Parameters
    ----------
    transcript : str
        The full conversation transcript, optionally with ``[CALLER]`` /
        ``[AGENT]`` markers.
    entity : dict or None
        Extracted entity fields (problem_category, summary, address, etc.).
    user_profile : dict or None
        Optional caller profile with known defaults (name, building, floor).

    Returns
    -------
    dict
        ``{"needs_clarification": bool, "question": str | None, "reasons": list[str]}``
    """
    reasons: list[str] = []
    turns = _parse_transcript(transcript)

    sparse = _sparse_caller(turns)
    generic = _generic_opening(turns)
    missing_loc = _missing_location(entity)
    vague = _vague_problem(entity)

    if sparse:
        reasons.append(f"sparse_caller:first{sparse[0]}_total{sparse[1]}")
    if generic:
        reasons.append(f"taxonomy_echo:{generic.replace(' ', '_')}")
    if missing_loc:
        reasons.append(f"missing_location:{missing_loc}")
    if vague:
        reasons.append(f"vague_problem:{vague}")

    needs = bool(reasons)
    question = (
        _build_clarification_question(sparse=sparse, generic=generic)
        if needs
        else None
    )

    return {
        "needs_clarification": needs,
        "question": question,
        "reasons": reasons,
    }


def gather_followup(state: dict) -> dict:
    """Interactively gather a clarifying follow-up answer from the user.

    Calls ``interrupt()`` with the first pending follow-up question, appends
    the user's answer to the transcript, and clears the follow-up queue.

    In eval mode the caller is expected to skip this node via a conditional
    edge rather than calling it.

    Parameters
    ----------
    state : dict
        Must contain ``follow_ups`` (list of question strings) and
        ``transcript`` (the conversation so far).

    Returns
    -------
    dict
        Updated state with the answer appended to transcript and follow_ups
        cleared.
    """
    follow_ups = state.get("follow_ups") or []
    if not follow_ups:
        return {"transcript": state.get("transcript", ""), "follow_ups": []}

    question = follow_ups[0]
    answer = interrupt(
        {
            "type": "followup_question",
            "question": question,
            "accumulated_so_far": state.get("transcript", ""),
        }
    )

    new_transcript = f"{state.get('transcript', '')}\n[CALLER] {answer}"
    return {
        "transcript": new_transcript,
        "follow_ups": follow_ups[1:],
    }


def build_query_from_entities(entity: dict) -> str:
    """Convert extracted entity fields into a structured query string.

    The query is used by the classification step to search for the right
    subcategory and provider. Fields present in the entity are assembled
    into a concise, structured text representation.

    Parameters
    ----------
    entity : dict
        Extracted entity with fields like ``problem_category``,
        ``building_name``, ``address``, ``floor``, ``location_detail``,
        ``severity``, ``summary``.

    Returns
    -------
    str
        A structured query string for the classification step.
    """
    parts: list[str] = []

    category = entity.get("problem_category") or entity.get("category")
    if category:
        parts.append(f"Category: {category}")

    summary = entity.get("summary")
    if summary:
        parts.append(f"Summary: {summary}")

    building = entity.get("building_name")
    address = entity.get("address")
    floor = entity.get("floor")
    location_detail = entity.get("location_detail")

    location_parts = [p for p in [building, address, floor, location_detail] if p]
    if location_parts:
        parts.append(f"Location: {', '.join(location_parts)}")

    severity = entity.get("severity")
    if severity:
        parts.append(f"Severity: {severity}")

    affected = entity.get("affected_system")
    if affected:
        parts.append(f"Affected system: {affected}")

    urgency_indicators = entity.get("urgency_indicators")
    if urgency_indicators:
        indicators = (
            ", ".join(urgency_indicators)
            if isinstance(urgency_indicators, list)
            else str(urgency_indicators)
        )
        parts.append(f"Urgency indicators: {indicators}")

    return "\n".join(parts) if parts else "No structured entity data available."


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2: HITL Validator + Override + Trainer (end of pipeline)
# ═══════════════════════════════════════════════════════════════════════════════


def validator_gate(
    needs_human_review: bool,
    classification: dict,
    risk_details: Optional[dict] = None,
    eval_mode: bool = False,
) -> dict:
    """HITL validator gate — decide whether to auto-route or pause for human review.

    In eval mode (batch grading) or when ``needs_human_review`` is False, the
    decision is auto-routed and returned as a routing decision string.

    In interactive mode, if ``needs_human_review`` is True, fires ``interrupt()``
    with the classification + risk context so a human reviewer can approve,
    override, or reject. The human's response is returned as ``human_override``.

    Parameters
    ----------
    needs_human_review : bool
        Whether the risk / HITL threshold calculation flagged this for review.
    classification : dict
        The AI classification (must contain at least ``category`` and
        ``subcategory``).
    risk_details : dict or None
        Optional risk context (reason, risk_score, etc.).
    eval_mode : bool
        If True, always auto-route (no interrupt).

    Returns
    -------
    dict
        ``{"routing_decision": str, "human_override": dict | None}``
    """
    risk_score = (risk_details or {}).get("risk_score", 0.0)
    cat = classification.get("category", "unknown")
    sub = classification.get("subcategory", "unknown")

    if eval_mode or not needs_human_review:
        if needs_human_review:
            routing = f"Flagged for review → {cat}/{sub} (risk {risk_score:.2f})"
        else:
            routing = f"Auto-routed → {cat}/{sub} (risk {risk_score:.2f})"
        return {"routing_decision": routing, "human_override": None}

    # Interactive mode — interrupt for human review
    decision = interrupt(
        {
            "type": "review_required",
            "reason": (risk_details or {}).get("reason", "elevated risk"),
            "risk_score": risk_score,
            "classification": classification,
            "risk_details": risk_details,
        }
    )

    if decision.get("override_code"):
        return {
            "human_override": decision,
            "routing_decision": f"Human override → {decision.get('override_code')}",
        }

    return {
        "routing_decision": f"Human approved → {sub}",
        "human_override": None,
    }


def apply_override(classification: dict, human_override: dict) -> dict:
    """Apply human corrections to a classification.

    Merges the human reviewer's corrections (category, subcategory, urgency)
    into the classification and marks it as human-reviewed.

    Parameters
    ----------
    classification : dict
        The original AI classification.
    human_override : dict
        Human corrections. May contain ``category``, ``subcategory``,
        ``risk_level`` (or ``urgency``), and any other fields to override.
        Pass ``None`` or an empty dict for no-op.

    Returns
    -------
    dict
        Corrected classification with ``human_reviewed=True``.
    """
    if not human_override:
        corrected = classification.copy()
        corrected["human_reviewed"] = True
        return corrected

    corrected = classification.copy()
    for field in ("category", "subcategory", "risk_level", "urgency"):
        if field in human_override and human_override[field] is not None:
            corrected[field] = human_override[field]
    corrected["human_reviewed"] = True
    return corrected


def write_trainer_log(
    transcript: str,
    prediction: dict,
    human_override: Optional[dict] = None,
    final_decision: Optional[dict] = None,
    log_dir: str = "data/leads",
) -> str:
    """Write the full trainer log to a local JSON file.

    The trainer log captures the complete per-call record for replayability
    in prompt tuning / fine-tuning. Writes to ``data/leads/lead_<timestamp>.json``.

    **Note:** The JSON file is an MVP placeholder. In production this goes to
    a proper database.

    Parameters
    ----------
    transcript : str
        The full conversation transcript.
    prediction : dict
        The AI prediction (category, subcategory, urgency, provider_match,
        and all provenance fields).
    human_override : dict or None
        Any human corrections applied.
    final_decision : dict or None
        The final decision after overrides are applied. If None, uses the
        AI prediction.
    log_dir : str
        Directory to write log files to. Defaults to ``data/leads``.

    Returns
    -------
    str
        The absolute path to the written log file.
    """
    if final_decision is None:
        final_decision = prediction

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    log_entry = {
        "timestamp": timestamp,
        "transcript": transcript,
        "ai_prediction": {
            "category": prediction.get("category"),
            "subcategory": prediction.get("subcategory"),
            "urgency": prediction.get("urgency"),
            "provider_match": prediction.get("provider_match"),
            # Include any extra provenance fields
            **{
                k: v
                for k, v in prediction.items()
                if k not in ("category", "subcategory", "urgency", "provider_match")
            },
        },
        "human_override": human_override,
        "final_decision": {
            "category": final_decision.get("category"),
            "subcategory": final_decision.get("subcategory"),
            "urgency": final_decision.get("urgency"),
            "provider_match": final_decision.get("provider_match"),
            **{
                k: v
                for k, v in final_decision.items()
                if k
                not in (
                    "category",
                    "subcategory",
                    "urgency",
                    "provider_match",
                )
            },
        },
        "conversion_status": "pending",
    }

    os.makedirs(log_dir, exist_ok=True)

    # Use a filename-safe timestamp
    file_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"lead_{file_stamp}.json"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)

    return os.path.abspath(filepath)


def assemble_lead_record(
    transcript: str,
    classification: dict,
    provider_matches: list[dict],
    urgency: dict,
    trainer_log_path: str,
) -> dict:
    """Assemble the full lead record for downstream analysis.

    Combines the conversation transcript, AI prediction, provider matches,
    urgency scores, HITL outcome, and lead quality signal into a single
    record suitable for persistence and analysis.

    Parameters
    ----------
    transcript : str
        The full conversation transcript.
    classification : dict
        The final classification (after any human overrides).
    provider_matches : list[dict]
        List of matched provider entries (name, phone, rating, etc.).
    urgency : dict
        Urgency dimension signals and composite score.
    trainer_log_path : str
        Path to the written trainer log JSON file.

    Returns
    -------
    dict
        A fully assembled lead record dict.
    """
    return {
        "conversation_transcript": transcript,
        "ai_prediction": {
            "category": classification.get("category"),
            "subcategory": classification.get("subcategory"),
            "urgency": classification.get("urgency"),
            "provider_match": classification.get("provider_match"),
        },
        "provider_matches": provider_matches,
        "urgency": urgency,
        "hitl_outcome": {
            "human_reviewed": classification.get("human_reviewed", False),
            "human_override": classification.get("human_override"),
            "routing_decision": classification.get("routing_decision"),
        },
        "trainer_log_path": trainer_log_path,
        "lead_quality_score": None,  # Filled in by downstream Jev Noul call
        "conversion_status": "pending",
    }