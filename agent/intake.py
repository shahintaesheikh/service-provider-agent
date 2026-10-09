"""
Intake nodes — first-contact processing of the homeowner's transcript.

Maps to the brief's deliverables on:
  - Entity extraction via ChatOpenAI with structured output (ExtractedEntity)
  - Clarification-gate heuristics (sparse caller, vague description)
  - Multi-turn / follow-up via interrupt()
  - Enriched query building from extracted entities
"""
from __future__ import annotations

from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from .config import SUBCATEGORY_TO_CATEGORY, SANTA_BARBARA_CITIES
from .prompts import EXTRACT_SYSTEM_PROMPT, EXTRACT_USER_TEMPLATE
from .state import ClarificationDecision, ExtractedEntity, WorkflowState
from .utils import _log_cache_usage


# ── Entity extraction via ChatOpenAI ─────────────────────────────────────────


def make_extract_and_check(llm: Optional[ChatOpenAI] = None):
    """Combined entity extraction + quality check in a single LLM call.

    Uses ChatOpenAI with structured output (ExtractedEntity) to pull
    structured fields from free-form homeowner speech, plus a quality
    check that emits a follow-up question when the description is too vague.

    If no LLM is provided, falls back to a minimal stub that returns
    default values so the graph can still compile and run in smoke-test mode.
    """
    if llm is not None:
        # include_raw=True returns {"raw": AIMessage, "parsed": ExtractedEntity, ...}
        # so we can both get the structured entity AND surface cache-hit telemetry
        extractor = llm.with_structured_output(ExtractedEntity, include_raw=True)
    else:
        extractor = None

    def extract_and_check(state: WorkflowState):
        if state.get("extraction_complete"):
            return {"follow_ups": [], "needs_clarification": False}

        description = state["transcript"]

        if extractor is not None:
            # Real LLM extraction
            out = extractor.invoke([
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACT_USER_TEMPLATE.format(description=description)},
            ])
            _log_cache_usage("extract_and_check", out.get("raw"))
            parsed = out.get("parsed")
            if parsed is None:
                # Parsing failed — fall back to permissive default
                return {"follow_ups": [], "extraction_complete": True, "needs_clarification": False}
            data = parsed.model_dump(mode="json")
        else:
            # Fallback stub when no LLM is available (smoke-test mode)
            data = {
                "problem_category": None,
                "problem_description": description[:200] if description else "",
                "address": None,
                "city": "Santa Barbara",
                "neighborhood": None,
                "urgency_indicators": [],
                "severity": "Medium",
                "summary": description[:150] if description else "",
                "description_specific_enough": True,
                "location_ok": True,
                "needs_clarification": False,
                "followup_question": "",
            }

        # Split into entity fields and quality fields
        entity = {
            "problem_category": data.get("problem_category"),
            "problem_description": data.get("problem_description", ""),
            "address": data.get("address"),
            "city": data.get("city", "Santa Barbara"),
            "neighborhood": data.get("neighborhood"),
            "urgency_indicators": data.get("urgency_indicators", []),
            "severity": data.get("severity", "Medium"),
            "summary": data.get("summary", ""),
        }

        needs_clarification = data.get("needs_clarification", False)
        follow_ups = [data["followup_question"]] if data.get("followup_question") else []
        extraction_complete = (
            data.get("description_specific_enough", True)
            and data.get("location_ok", True)
        )

        return {
            "entity": entity,
            "follow_ups": follow_ups,
            "extraction_complete": extraction_complete,
            "needs_clarification": needs_clarification,
        }

    return extract_and_check


# ── Clarification-gate heuristics ────────────────────────────────────────────

SPARSE_FIRST_TURN_WORDS = 6
SPARSE_TOTAL_WORDS = 35

# Surface forms for the home-service subcategory keywords.
SUBCATEGORY_SURFACE_FORMS = {
    sub: [sub.replace("_", " ")] for sub in SUBCATEGORY_TO_CATEGORY
}


def _missing_entities(entity: Optional[dict]) -> list[str]:
    """Return required entity fields that are None/empty. Pure deterministic check."""
    if not entity:
        return ["problem_description", "address", "city"]
    missing = []
    if not entity.get("problem_description"):
        missing.append("problem_description")
    if not entity.get("address") and not entity.get("city"):
        missing.append("address")
    return missing


def _sparse_caller(turns: list[dict]) -> Optional[Tuple[int, int]]:
    """Fires when the caller's first utterance is short (<=6 words) AND
    their total speech across all turns is short (<=35 words). Sparse calls
    on safety-sensitive symptoms often warrant follow-up.
    """
    caller = [t.get("text", "") for t in turns if t.get("speaker") == "caller"]
    if not caller:
        return None
    first = len(caller[0].split())
    total = sum(len(t.split()) for t in caller)
    if first <= SPARSE_FIRST_TURN_WORDS and total <= SPARSE_TOTAL_WORDS:
        return (first, total)
    return None


def _build_question(*, sparse=None, missing=None, generic=None) -> str:
    """Pick a clarifying question based on which heuristic(s) fired."""
    if sparse:
        return (
            "Could you tell me a bit more about what's going wrong — for example, "
            "what specifically isn't working and how long it's been like that?"
        )
    if missing:
        if "problem_description" in missing:
            return "What exactly is happening? Can you describe the issue?"
        if "address" in missing:
            return "Where in Santa Barbara is this located?"
    return (
        "Could you describe the issue in a bit more detail so I can connect you "
        "with the right provider?"
    )


def decide_clarification(turns: list[dict], entity: Optional[dict] = None) -> ClarificationDecision:
    """Aggregate deterministic heuristics into a single verdict + audit trail.

    Uses ONLY deterministic checks: sparse caller + missing entity fields.
    No LLM judgment involved.
    """
    reasons: list[str] = []

    sparse = _sparse_caller(turns)
    missing = _missing_entities(entity)

    if sparse:
        reasons.append(f"sparse_caller:first{sparse[0]}_total{sparse[1]}")
    if missing:
        reasons.append(f"missing_fields:{','.join(missing)}")

    needs = bool(reasons)
    question = (
        _build_question(sparse=sparse, missing=missing) if needs else None
    )

    return ClarificationDecision(
        needs_clarification=needs,
        question=question,
        reasons=tuple(reasons),
    )


# ── Nodes ────────────────────────────────────────────────────────────────────


def clarification_gate(state: WorkflowState):
    """Deterministic clarification-signal gate.

    Runs the sparse-caller heuristic against the turns plus the missing-entity
    check. Writes its verdict to state.gate_clarification_verdict — does NOT
    touch state.needs_clarification, so downstream nodes can override.

    In interactive mode the question is also appended to state.follow_ups so
    the downstream gather_followup node actually asks it. In eval mode the
    needs_followup conditional skips the gather loop, so the appended question
    is a no-op there.
    """
    entity = state.get("entity")
    decision = decide_clarification(
        state.get("turns_split") or [],
        entity=entity,
    )

    existing_follow_ups = list(state.get("follow_ups") or [])
    if decision.question and decision.question not in existing_follow_ups:
        existing_follow_ups.append(decision.question)

    return {
        "gate_clarification_verdict": decision.needs_clarification,
        "clarification_question": decision.question,
        "clarification_reasons": decision.reasons,
        "follow_ups": existing_follow_ups,
    }


def gather_followup(state: WorkflowState):
    """Pause for caller input via interrupt().

    Bypassed in eval mode by the needs_followup conditional edge.
    """
    question = state.get("follow_ups", [None])[0]
    if not question:
        return {"follow_ups": [], "extraction_complete": False}

    answer = interrupt({
        "type": "followup_question",
        "question": question,
        "accumulated_so_far": state["transcript"],
    })

    new_transcript = f"{state['transcript']}\n[CALLER] {answer}"
    return {
        "transcript": new_transcript,
        "follow_ups": state["follow_ups"][1:],
        "extraction_complete": False,
    }


def build_enriched_query(state: WorkflowState):
    """Build a structured query from extracted entities for use in provider search.

    Combines:
      - extracted problem description
      - category/subcategory hints from classification
      - location (Santa Barbara area)
      - urgency signals
    into a compact query string for downstream provider APIs.
    """
    entity = state.get("entity") or {}
    classification = state.get("classification") or {}

    problem = entity.get("problem_description", "") or ""
    category = classification.get("category", "") or ""
    subcategory = classification.get("subcategory", "") or ""
    city = entity.get("city", "Santa Barbara") or "Santa Barbara"
    neighborhood = entity.get("neighborhood") or ""

    # Build a descriptive label from category/subcategory
    type_label = subcategory.replace("_", " ") if subcategory else (
        category.replace("_", " ") if category else "home service"
    )

    # Build location string
    location_parts = [city]
    if neighborhood:
        location_parts = [neighborhood, city]
    location_str = ", ".join(location_parts)

    enriched = (
        f"{type_label} — {problem} "
        f"in {location_str}"
    ).strip()
    # Collapse whitespace
    enriched = " ".join(enriched.split())

    # Also prepare a vendor-type lookup key for provider search
    vendor_type = subcategory if subcategory else (
        category.lower() if category else "home_services"
    )

    return {
        "enriched_query": enriched,
        "_vendor_type": vendor_type,
        "_search_location": f"{city}, CA",
    }