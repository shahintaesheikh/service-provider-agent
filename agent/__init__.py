"""
Service Provider Agent — Home Service Provider Matching for Santa Barbara.

A conversational agent that helps Santa Barbara homeowners find the right
home service provider. Uses LangGraph for orchestration, ChatOpenAI for NLU,
Jev for typed decisions, and Google Places / Yelp for provider matching.

Package organization:

    config.py       - Home service taxonomy, Santa Barbara constants, thresholds
    state.py        - WorkflowState + Pydantic schemas
    utils.py        - Shared helpers (transcript flatten, timing, regex)
    prompts.py      - Prompt templates (EXTRACT, CLASSIFY)
    intake.py       - NLU extraction, clarification gate, follow-up loop
    dispatch.py     - Location resolution, provider merge/rank
    risk.py         - Over-escalation traps, urgency scoring
    hitl.py         - HITL trinity (Validator + Override + Trainer)
    graph.py        - LangGraph DAG wiring + entry points

Public API:
    create_graph() -> CompiledStateGraph
    get_graph()    -> CompiledStateGraph (cached singleton)
"""
from .graph import create_graph, get_graph
from .utils import _flatten, _log_cache_usage, timed_node, print_timing_report

__all__ = [
    "create_graph",
    "get_graph",
    "_flatten",
    "_log_cache_usage",
    "timed_node",
    "print_timing_report",
]