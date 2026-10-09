"""
LangGraph DAG wiring for the Service Provider Agent.

This module knows about every node and conditional edge; everything below
only knows about its own slice. The wiring order is:

  START
    → extract_and_check                    (ChatOpenAI NLU)
    → clarification_gate                   (heuristics)
    → (gather_followup loop OR build_enriched_query)
        gather_followup → extract_and_check
    → build_enriched_query                 (query builder)
    → classify_category                    (Jev Choice)
    → classify_subcategory                 (Jev Choice)
    → api_selection                        (Jev Choice)
    → urgency_dimension                    (Jev Score/Noul)
    → call_google_places                   (Google Places API)
    → call_yelp                            (stub — Yelp not yet integrated)
    → merge_and_rank                       (dedup + rating sort)
    → trap_check                           (over-escalation guard)
    → urgency_score                        (Risk = P × C via hitl_utils)
    → resolve_location                     (geocoding)
    → validator_gate                       (HITL gate)
    → (override_node → trainer_log) OR lead_quality_check
    → lead_quality_check                   (Jev Noul)
    → trainer_log                          (HITL log)
    → END

Uses real tool functions from tools/ modules for every node.
Gracefully handles missing API keys by falling back to sensible defaults.
"""
from __future__ import annotations

from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .config import SUBCATEGORY_TO_CATEGORY, SUBCATEGORY_TO_VENDOR_TYPE
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
from tools.jev_utils import (
    assess_lead_quality,
    assess_urgency_factor,
    classify_category,
    classify_subcategory,
    select_api,
)
from tools.provider_search import search_providers, get_place_details


# ── Jev-integrated node factories ───────────────────────────────────────────


def make_classify_category():
    """Jev Choice node for category classification.

    Classifies among the 10 home service categories using
    tools.jev_utils.classify_category().
    """
    def classify_category_node(state: WorkflowState):
        # Get the transcript or enriched query as input
        description = (
            state.get("enriched_query")
            or state.get("entity", {}).get("problem_description")
            or state["transcript"]
        )

        try:
            result = classify_category(description)
            category = result.get("category")
            confidence = result.get("confidence", 0.5)
            probabilities = result.get("probabilities", {})
        except (RuntimeError, Exception):
            # Graceful fallback when TYPESAFE_API_KEY is missing
            category = None
            confidence = None
            probabilities = {}

        classification = state.get("classification") or {}
        return {
            "classification": {
                **classification,
                "category": category,
                "category_confidence": confidence,
                "category_probabilities": probabilities,
                "confidence": confidence,  # shared confidence for downstream
            }
        }

    return classify_category_node


def make_classify_subcategory():
    """Jev Choice node for subcategory classification.

    Classifies among 3–6 subcategories within the chosen category using
    tools.jev_utils.classify_subcategory().
    """
    def classify_subcategory_node(state: WorkflowState):
        classification = state.get("classification") or {}
        category = classification.get("category")

        description = (
            state.get("enriched_query")
            or state.get("entity", {}).get("problem_description")
            or state["transcript"]
        )

        try:
            result = classify_subcategory(description, category or "")
            subcategory = result.get("subcategory")
            confidence = result.get("confidence", 0.5)
            probabilities = result.get("probabilities", {})
        except (RuntimeError, Exception):
            # Graceful fallback when TYPESAFE_API_KEY is missing
            subcategory = None
            confidence = None
            probabilities = {}

        return {
            "classification": {
                **classification,
                "subcategory": subcategory,
                "subcategory_confidence": confidence,
                "subcategory_probabilities": probabilities,
                "confidence": confidence,
            }
        }

    return classify_subcategory_node


def make_api_selection():
    """Jev Choice node for API selection.

    Uses tools.jev_utils.select_api() to decide which provider API(s)
    to query: Google Places, CSLB, Both, or None.
    """
    def api_selection_node(state: WorkflowState):
        classification = state.get("classification") or {}
        category = classification.get("category") or ""

        description = (
            state.get("enriched_query")
            or state.get("entity", {}).get("problem_description")
            or state["transcript"]
        )

        try:
            result = select_api(description, category)
            api_choice = result.get("api", "Google Places")
            api_confidence = result.get("confidence", 1.0)
        except (RuntimeError, Exception):
            # Graceful fallback — default to Google Places
            api_choice = "Google Places"
            api_confidence = 1.0

        # Map Jev API names to LangGraph router values
        api_map = {
            "Google Places": "google_places",
            "Google Places + CSLB": "both",
            "CSLB only": "google_places",  # treat as google_places for now
            "None": "none",
        }
        mapped = api_map.get(api_choice, "google_places")

        return {
            "classification": classification,
            "api_selection": mapped,
        }

    return api_selection_node


def make_call_google_places():
    """Google Places API call using tools.provider_search.

    Uses search_providers() with the subcategory's vendor type and
    the resolved location. Gracefully handles missing API key.
    """
    def call_google_places(state: WorkflowState):
        classification = state.get("classification") or {}
        subcategory = (classification.get("subcategory") or "").lower()
        category = (classification.get("category") or "").lower()

        # Determine the vendor/service type for the API query
        vendor_type = SUBCATEGORY_TO_VENDOR_TYPE.get(subcategory, category)
        if not vendor_type or vendor_type == "other":
            vendor_type = "home_services"

        # Use location from resolve_location if available, else fallback
        city = state.get("city") or "Santa Barbara"
        latitude = state.get("latitude")
        longitude = state.get("longitude")

        # Try text search first
        location_str = f"{city}, CA"
        results = search_providers(
            service_type=vendor_type,
            location=location_str,
            max_results=10,
        )

        # If we have coordinates and text search returned nothing, try nearby search
        if not results and latitude and longitude:
            from tools.provider_search import search_providers_nearby
            results = search_providers_nearby(
                latitude=latitude,
                longitude=longitude,
                radius_meters=80467,  # ~50 miles in meters
                max_results=10,
            )

        # Enhance results with place details for the top matches
        enhanced = []
        for place in results[:5]:
            place_id = place.get("id") or place.get("place_id", "")
            if place_id:
                details = get_place_details(place_id)
                if details:
                    # Merge details with search result
                    merged = {**place, **details}
                    enhanced.append(merged)
                else:
                    enhanced.append(place)
            else:
                enhanced.append(place)

        existing = state.get("provider_matches") or []
        return {
            "provider_matches": existing + enhanced,
        }

    return call_google_places


def make_call_yelp():
    """Yelp Fusion API call — stub.

    Yelp integration is not yet implemented in the tools module.
    Passes through existing provider_matches unchanged.
    """
    def call_yelp(state: WorkflowState):
        return {
            "provider_matches": state.get("provider_matches") or [],
        }

    return call_yelp


def make_urgency_dimension():
    """Assess urgency dimension signals via Jev Score/Noul.

    Calls tools.jev_utils.assess_urgency_factor() for each dimension
    (water_damage_spreading, safety_hazard, user_contained).
    Gracefully handles missing API key.
    """
    def urgency_dimension(state: WorkflowState):
        description = (
            state.get("enriched_query")
            or state.get("entity", {}).get("problem_description")
            or state["transcript"]
        )

        result = {}

        # Try to assess each dimension independently
        for factor in ["water_damage_spreading", "safety_hazard", "user_contained"]:
            try:
                signal = assess_urgency_factor(description, factor)
                result[signal["dimension"]] = signal["value"]
            except (RuntimeError, ValueError, Exception):
                pass  # Skip if Jev is unavailable

        return {
            "water_spread_score": result.get("water_damage_spreading"),
            "safety_hazard_noul": result.get("safety_hazard"),
            "contained_noul": result.get("user_contained"),
        }

    return urgency_dimension


def make_lead_quality_check():
    """Jev Noul lead quality check.

    Uses tools.jev_utils.assess_lead_quality() to assess whether the
    matched provider lead is likely to convert.
    Returns calibrated probability.
    """
    def lead_quality_check_node(state: WorkflowState):
        merged = state.get("merged_providers") or []
        if not merged:
            return {
                "lead_quality": {
                    "likely_to_convert": None,
                    "confidence": None,
                }
            }

        # Use the top-ranked provider
        top_provider = merged[0]
        provider_name = (
            top_provider.get("displayName")
            or top_provider.get("name")
            or "Unknown Provider"
        )

        # Build a conversation summary from state
        entity = state.get("entity") or {}
        classification = state.get("classification") or {}
        summary_lines = [
            f"Problem: {entity.get('problem_description', '')}",
            f"Category: {classification.get('category', '')}",
            f"Subcategory: {classification.get('subcategory', '')}",
            f"City: {state.get('city', 'Santa Barbara')}",
        ]
        conversation_summary = "\n".join(summary_lines)

        try:
            result = assess_lead_quality(conversation_summary, provider_name)
            will_convert = result.get("will_convert", False)
            confidence = result.get("confidence", 0.5)
        except (RuntimeError, Exception):
            will_convert = None
            confidence = None

        return {
            "lead_quality": {
                "likely_to_convert": will_convert,
                "confidence": confidence,
            }
        }

    return lead_quality_check_node


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
    """Route based on api_selection decision."""
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

    Uses real tool functions from tools/ modules for every node.
    Gracefully handles missing API keys (returns None/defaults).

    Args:
        eval_mode: When True, bypasses interactive interrupt() calls.

    Returns:
        A compiled LangGraph CompiledStateGraph.
    """
    workflow = StateGraph(WorkflowState)

    # Create all nodes
    # LLM for entity extraction (GPT-6 Luna — smaller/faster model for structured extraction)
    llm_extract = ChatOpenAI(model="gpt-6-luna", temperature=0, max_retries=3, request_timeout=20)

    workflow.add_node("extract_and_check", timed_node("extract_and_check", make_extract_and_check(llm_extract)))
    workflow.add_node("clarification_gate", timed_node("clarification_gate", clarification_gate))
    workflow.add_node("gather_followup", timed_node("gather_followup", gather_followup))
    workflow.add_node("build_enriched_query", timed_node("build_enriched_query", build_enriched_query))
    workflow.add_node("classify_category", timed_node("classify_category", make_classify_category()))
    workflow.add_node("classify_subcategory", timed_node("classify_subcategory", make_classify_subcategory()))
    workflow.add_node("api_selection", timed_node("api_selection", make_api_selection()))
    workflow.add_node("urgency_dimension", timed_node("urgency_dimension", make_urgency_dimension()))
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

    # Classification phase (Jev Choice)
    workflow.add_edge("classify_category", "classify_subcategory")
    workflow.add_edge("classify_subcategory", "api_selection")

    # Urgency dimension assessment (Jev Score/Noul) — runs before API routing
    workflow.add_edge("api_selection", "urgency_dimension")

    # API calling phase (conditional on selection, triggered after urgency_dimension)
    workflow.add_conditional_edges("urgency_dimension", api_selection_route)
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