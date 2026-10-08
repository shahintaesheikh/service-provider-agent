"""
LangGraph DAG wiring for the Service Provider Agent.

This module knows about every node and conditional edge; everything below
only knows about its own slice. The wiring order is:

  START
    → extract_and_check                    (NLU stub)
    → clarification_gate                   (heuristics stub)
    → (gather_followup loop OR build_enriched_query)
        gather_followup → extract_and_check
    → build_enriched_query                 (query builder stub)
    → classify_category                    (Jev Choice stub)
    → classify_subcategory                 (Jev Choice stub)
    → api_selection                        (Jev Choice stub)
    → call_google_places                   (stub — actual API in separate file)
    → call_yelp                            (stub — actual API in separate file)
    → merge_and_rank                       (stub)
    → trap_check                           (over-escalation stub)
    → urgency_score                        (composite calc stub)
    → resolve_location                     (address stub)
    → validator_gate                       (HITL stub)
    → (override_node → trainer_log) OR trainer_log
    → lead_quality_check                   (Jev Noul stub)
    → trainer_log                          (HITL stub)
    → END

Every node is a stub that just returns empty/skeleton state. The graph
compiles and runs without error.
"""
from __future__ import annotations

from typing import Literal, Optional

from langgraph.graph import END, START, StateGraph

from .config import SUBCATEGORY_TO_CATEGORY
from .dispatch import merge_and_rank, resolve_location
from .hitl import make_trainer_log, override_node, validator_gate
from .intake import (
    build_enriched_query,
    clarification_gate,
    gather_followup,
    make_extract_and_check,
)
from .risk import trap_check, urgency_score
from .state import WorkflowState
from .utils import timed_node


# ── Stub node factories ──────────────────────────────────────────────────────


def make_classify_category():
    """Stub: Jev Choice node for category classification.

    TODO: implement with Jev Choice to classify among the 10 home service
    categories (Plumbing, HVAC, Electrical, Roofing, Water Damage,
    Appliance Repair, Pest Control, Locksmith, Handyman, Not Applicable).
    """
    def classify_category(state: WorkflowState):
        classification = state.get("classification") or {}
        return {
            "classification": {
                **classification,
                "category": None,
                "category_confidence": None,
            }
        }
    return classify_category


def make_classify_subcategory():
    """Stub: Jev Choice node for subcategory classification.

    TODO: implement with Jev Choice to classify among 3–6 subcategories
    within the chosen category.
    """
    def classify_subcategory(state: WorkflowState):
        classification = state.get("classification") or {}
        return {
            "classification": {
                **classification,
                "subcategory": None,
                "subcategory_confidence": None,
            }
        }
    return classify_subcategory


def make_api_selection():
    """Stub: Jev Choice node for API selection.

    TODO: implement with Jev Choice to decide which provider API(s) to query:
    Google Places, Yelp, Both, or None (based on problem context).
    """
    def api_selection(state: WorkflowState):
        classification = state.get("classification") or {}
        return {
            "classification": classification,
            "api_selection": None,
        }
    return api_selection


def make_call_google_places():
    """Stub: Google Places API call.

    TODO: implement actual Google Places API query using the enriched query
    and selected category/subcategory. Results stored in provider_matches.
    """
    def call_google_places(state: WorkflowState):
        return {
            "provider_matches": [],
        }
    return call_google_places


def make_call_yelp():
    """Stub: Yelp Fusion API call.

    TODO: implement actual Yelp Fusion API query. Results stored in
    provider_matches (appended to Google Places results).
    """
    def call_yelp(state: WorkflowState):
        return {
            "provider_matches": state.get("provider_matches") or [],
        }
    return call_yelp


def make_lead_quality_check():
    """Stub: Jev Noul lead quality check.

    TODO: implement with Jev Noul to assess whether the matched provider
    lead is likely to convert. Returns calibrated probability.
    """
    def lead_quality_check(state: WorkflowState):
        return {
            "lead_quality": {
                "likely_to_convert": None,
                "confidence": None,
            }
        }
    return lead_quality_check


# ── Conditional edge functions ───────────────────────────────────────────────


def needs_followup(state: WorkflowState) -> Literal["gather_followup", "build_enriched_query"]:
    """Fired AFTER clarification_gate. Routes to gather_followup when
    follow_ups are pending, or to build_enriched_query otherwise.
    """
    if state.get("eval_mode"):
        return "build_enriched_query"
    if state.get("extraction_complete"):
        return "build_enriched_query"
    if state.get("follow_ups"):
        return "gather_followup"
    return "build_enriched_query"


def api_selection_route(state: WorkflowState) -> Literal["call_google_places", "call_yelp", "merge_and_rank"]:
    """Route based on api_selection decision.

    google_places → call_google_places → call_yelp → merge_and_rank
    yelp         → call_yelp → merge_and_rank
    both         → call_google_places → call_yelp → merge_and_rank
    none         → merge_and_rank (skip API calls)
    """
    selection = state.get("api_selection")
    if selection == "google_places":
        return "call_google_places"
    elif selection == "both":
        return "call_google_places"
    elif selection == "yelp":
        return "call_yelp"
    else:
        return "merge_and_rank"


def after_google_route(state: WorkflowState) -> Literal["call_yelp", "merge_and_rank"]:
    """After Google Places, route to Yelp if selected, or to merge_and_rank."""
    selection = state.get("api_selection")
    if selection == "both":
        return "call_yelp"
    return "merge_and_rank"


def override_route(state: WorkflowState) -> Literal["override_node", "lead_quality_check"]:
    """After validator_gate, route through override if human_override exists."""
    if state.get("human_override"):
        return "override_node"
    return "lead_quality_check"


# ── Graph builder ────────────────────────────────────────────────────────────


def create_graph(eval_mode: bool = True):
    """Construct and compile the LangGraph pipeline.

    Every node is a stub. No LLM clients, no vector stores, no API keys
    required. The compiled graph is ready for invocation and returns
    skeleton state.

    Args:
        eval_mode: When True, bypasses interactive interrupt() calls.

    Returns:
        A compiled LangGraph CompiledStateGraph.
    """
    workflow = StateGraph(WorkflowState)

    # Create all stub nodes
    workflow.add_node("extract_and_check", timed_node("extract_and_check", make_extract_and_check()))
    workflow.add_node("clarification_gate", timed_node("clarification_gate", clarification_gate))
    workflow.add_node("gather_followup", timed_node("gather_followup", gather_followup))
    workflow.add_node("build_enriched_query", timed_node("build_enriched_query", build_enriched_query))
    workflow.add_node("classify_category", timed_node("classify_category", make_classify_category()))
    workflow.add_node("classify_subcategory", timed_node("classify_subcategory", make_classify_subcategory()))
    workflow.add_node("api_selection", timed_node("api_selection", make_api_selection()))
    workflow.add_node("call_google_places", timed_node("call_google_places", make_call_google_places()))
    workflow.add_node("call_yelp", timed_node("call_yelp", make_call_yelp()))
    workflow.add_node("merge_and_rank", timed_node("merge_and_rank", merge_and_rank))
    workflow.add_node("trap_check", timed_node("trap_check", trap_check))
    workflow.add_node("urgency_score", timed_node("urgency_score", urgency_score))
    workflow.add_node("resolve_location", timed_node("resolve_location", resolve_location))
    workflow.add_node("validator_gate", timed_node("validator_gate", validator_gate))
    workflow.add_node("override_node", timed_node("override_node", override_node))
    workflow.add_node("lead_quality_check", timed_node("lead_quality_check", make_lead_quality_check()))
    workflow.add_node("trainer_log", timed_node("trainer_log", make_trainer_log))

    # ── Edges ──

    # Intake phase
    workflow.add_edge(START, "extract_and_check")
    workflow.add_edge("extract_and_check", "clarification_gate")
    workflow.add_conditional_edges("clarification_gate", needs_followup)
    workflow.add_edge("gather_followup", "extract_and_check")
    workflow.add_edge("build_enriched_query", "classify_category")

    # Classification phase (Jev Choice stubs)
    workflow.add_edge("classify_category", "classify_subcategory")
    workflow.add_edge("classify_subcategory", "api_selection")

    # API calling phase (conditional on selection)
    workflow.add_conditional_edges("api_selection", api_selection_route)
    workflow.add_conditional_edges("call_google_places", after_google_route)
    workflow.add_edge("call_yelp", "merge_and_rank")

    # Provider processing
    workflow.add_edge("merge_and_rank", "trap_check")
    workflow.add_edge("trap_check", "urgency_score")
    workflow.add_edge("urgency_score", "resolve_location")

    # HITL phase
    workflow.add_edge("resolve_location", "validator_gate")
    workflow.add_conditional_edges("validator_gate", override_route)
    workflow.add_edge("override_node", "lead_quality_check")
    workflow.add_edge("lead_quality_check", "trainer_log")
    workflow.add_edge("trainer_log", END)

    return workflow.compile()


# ── Entry point ──────────────────────────────────────────────────────────────


_graph = None


def get_graph():
    """Lazy singleton for the compiled graph. Built on first call."""
    global _graph
    if _graph is None:
        _graph = create_graph(eval_mode=True)
    return _graph