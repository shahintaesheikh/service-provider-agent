"""
LangGraph state schema + Pydantic models that flow through the Service
Provider Agent.

WorkflowState is the TypedDict that every node receives and partially
returns. It carries:
  - the original transcript + structured turns
  - the LLM-extracted entity (problem description, urgency, location)
  - clarification-gate verdict + reasons
  - enriched query built from extracted entities
  - category / subcategory classification outputs (Jev Choice placeholders)
  - API selection decision (Jev Choice placeholder)
  - provider matches from Google Places / Yelp (stub)
  - merged + ranked provider results
  - over-escalation trap check result
  - urgency dimensions + composite score
  - resolved location (address, city)
  - HITL gate outputs + trainer log
  - lead quality check (Jev Noul placeholder)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, TypedDict

from pydantic import BaseModel, Field


# ── Pydantic schemas for LLM structured output ──────────────────────────────


class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ExtractedEntity(BaseModel):
    """Entity extraction output from ChatOpenAI NLU pass."""
    problem_category: str = Field(
        description="General category of the problem (e.g., 'plumbing issue', 'electrical problem')"
    )
    problem_description: str = Field(
        description="Specific description of the problem (e.g., 'water leaking from pipe under sink')"
    )
    address: Optional[str] = Field(
        default=None,
        description="Street address mentioned by the user"
    )
    city: Optional[str] = Field(
        default=None,
        description="City mentioned (defaults to Santa Barbara if not specified)"
    )
    neighborhood: Optional[str] = Field(
        default=None,
        description="Santa Barbara neighborhood (e.g., 'Downtown', 'Montecito', 'Goleta')"
    )
    urgency_indicators: list[str] = Field(
        default_factory=list,
        description="Specific phrases indicating urgency (e.g., 'water rising', 'gas smell')"
    )
    severity: Severity = Field(
        description="Critical / High / Medium / Low based on safety and property damage risk"
    )
    summary: str = Field(description="One-sentence summary of the homeowner's issue")
    description_specific_enough: bool = Field(
        description="True if the problem describes a specific symptom, not vague"
    )
    location_ok: bool = Field(
        default=True,
        description="True if location info (address/city/neighborhood) is present"
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if description is too vague or location is missing"
    )
    followup_question: str = Field(
        default="",
        description="Question to get more specific info. Empty string if ok."
    )


# ── Deterministic clarification-gate decision ────────────────────────────────


@dataclass(frozen=True)
class ClarificationDecision:
    needs_clarification: bool
    question: Optional[str]
    reasons: Tuple[str, ...]


# ── LangGraph state ─────────────────────────────────────────────────────────


class WorkflowState(TypedDict):
    transcript: str
    turns_split: list[dict]
    caller_phone: Optional[str]
    caller_profile: Optional[dict]
    eval_mode: bool
    entity: Optional[dict]
    follow_ups: list
    extraction_complete: bool
    needs_clarification: bool
    gate_clarification_verdict: bool
    clarification_question: Optional[str]
    clarification_reasons: Tuple[str, ...]
    enriched_query: str
    classification: Optional[dict]
    api_selection: Optional[str]
    provider_matches: list
    merged_providers: list
    trap_check_result: Optional[dict]
    urgency_score: Optional[float]
    urgency_signals: Optional[dict]
    address: Optional[str]
    city: Optional[str]
    needs_human_review: bool
    lead_quality: Optional[dict]
    routing_decision: Optional[str]
    human_override: Optional[dict]
    trainer_log: Optional[dict]