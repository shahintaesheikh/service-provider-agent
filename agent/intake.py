"""
Intake nodes — first-contact processing of the homeowner's transcript.

Stubs for:
  - extract_and_check: NLU entity extraction via ChatOpenAI (placeholder)
  - clarification_gate: deterministic heuristics to detect vague/missing info
  - gather_followup: multi-turn follow-up via interrupt()
  - build_enriched_query: build a structured query from extracted entities

Each node returns its expected state fields with None/empty defaults.
No internal logic or API calling yet — those come in later tasks.
"""
from __future__ import annotations

from typing import Optional, Tuple

from langgraph.types import interrupt

from .config import SUBCATEGORY_TO_CATEGORY, SANTA_BARBARA_CITIES
from .state import ClarificationDecision, WorkflowState


# ── Stub: extract_and_check ──────────────────────────────────────────────────


def make_extract_and_check(llm=None):
    """Stub: entity extraction via ChatOpenAI (NLU).

    TODO: implement with real ChatOpenAI structured output calls using
    EXTRACT_SYSTEM_PROMPT and EXTRACT_USER_TEMPLATE from prompts.py.
    """
    def extract_and_check(state: WorkflowState):
        if state.get("extraction_complete"):
            return {"follow_ups": [], "needs_clarification": False}

        description = state["transcript"]

        # Stub: create a minimal entity dict from the raw transcript.
        # Real implementation will use llm.with_structured_output(ExtractedEntity).
        entity = {
            "problem_category": None,
            "problem_description": description[:200] if description else "",
            "address": None,
            "city": "Santa Barbara",  # default for MVP
            "neighborhood": None,
            "urgency_indicators": [],
            "severity": "Medium",
            "summary": description[:150] if description else "",
        }

        return {
            "entity": entity,
            "follow_ups": [],
            "extraction_complete": True,
            "needs_clarification": False,
        }

    return extract_and_check


# ── Clarification-gate heuristics (stub) ─────────────────────────────────────

SPARSE_FIRST_TURN_WORDS = 6
SPARSE_TOTAL_WORDS = 35


def _sparse_caller(turns: list[dict]) -> Optional[Tuple[int, int]]:
    """Stub: check if caller's speech is unusually short.

    TODO: implement actual word-count heuristic following reference project pattern.
    """
    return None


def _build_question(*, sparse=None, generic=None) -> str:
    """Pick a clarifying question based on which heuristic(s) fired."""
    if sparse:
        return (
            "Could you tell me a bit more about what's going wrong — for example, "
            "what specifically isn't working and how long it's been like that?"
        )
    return "Could you describe the issue in a bit more detail so I can connect you with the right provider?"


def decide_clarification(turns: list[dict]) -> ClarificationDecision:
    """Stub: aggregate heuristics into a single verdict.

    TODO: implement the full set of deterministic heuristics:
      - sparse_caller (first turn <= 6 words, total <= 35 words)
      - generic_opening (taxonomy label echo)
      - floor/location issues
    """
    reasons: list[str] = []
    sparse = _sparse_caller(turns)

    if sparse:
        reasons.append(f"sparse_caller:first{sparse[0]}_total{sparse[1]}")

    needs = bool(reasons)
    question = (
        _build_question(sparse=sparse) if needs else None
    )

    return ClarificationDecision(
        needs_clarification=needs,
        question=question,
        reasons=tuple(reasons),
    )


# ── Nodes ────────────────────────────────────────────────────────────────────


def clarification_gate(state: WorkflowState):
    """Stub: deterministic clarification-signal gate.

    TODO: implement full heuristic set (sparse_caller, generic_opening, etc.)
    following the reference project pattern.
    """
    existing_follow_ups = list(state.get("follow_ups") or [])

    decision = ClarificationDecision(
        needs_clarification=False,
        question=None,
        reasons=(),
    )

    if decision.question and decision.question not in existing_follow_ups:
        existing_follow_ups.append(decision.question)

    return {
        "gate_clarification_verdict": decision.needs_clarification,
        "clarification_question": decision.question,
        "clarification_reasons": decision.reasons,
        "follow_ups": existing_follow_ups,
    }


def gather_followup(state: WorkflowState):
    """Stub: pause for caller input via interrupt().

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
    """Stub: build a structured query from extracted entities.

    TODO: construct a rich query string combining:
      - extracted problem description
      - category/subcategory hints from classification
      - location (Santa Barbara area)
      - urgency signals
    for use in downstream provider search.
    """
    entity = state.get("entity") or {}
    enriched = (
        f"{entity.get('problem_description', '')} "
        f"in {entity.get('city', 'Santa Barbara')}"
    )

    return {
        "enriched_query": enriched.strip(),
    }