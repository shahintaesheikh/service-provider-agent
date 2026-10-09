"""
Human-in-the-loop trinity nodes.

Maps to the brief's three required HITL patterns (carried forward verbatim
from the reference project):
  - **Pattern 1: Validator** — validator_gate evaluates `needs_human_review`
    and either auto-routes (eval mode / not flagged) or pauses for a human
    reviewer via interrupt() (interactive mode, flagged).
  - **Pattern 2: Override** — override_node applies human corrections (from
    the validator_gate interrupt payload) to the classification before it
    proceeds to logging.
  - **Pattern 3: Trainer** — make_trainer_log emits the per-call record with
    full_transcript + ai_prediction + human_override + final_decision so the
    log is replayable for prompt tuning / fine-tuning.

Adapted from the reference project: vendor_id → provider_match, etc.
"""
from __future__ import annotations

from langgraph.types import interrupt

from .state import WorkflowState


def validator_gate(state: WorkflowState):
    """HITL validator gate.

    In eval mode (batch grading), auto-approves and just builds a
    routing_decision string for the trainer log.

    In interactive mode (live operator workflow), if `needs_human_review` is
    True, fires interrupt() with the classification + risk context so the
    reviewer can approve, override, or reject. Resume semantics are handled
    by LangGraph's Command(resume=...) downstream.
    """
    if state.get("eval_mode"):
        urgency_score = state.get("urgency_score") or 0.0
        classification = state["classification"]
        cat = classification.get("category", "")
        sub = classification.get("subcategory", "")
        if state.get("needs_human_review"):
            routing = f"Flagged for review → {cat}/{sub} (urgency {urgency_score:.2f})"
        else:
            routing = f"Auto-routed → {cat}/{sub} (urgency {urgency_score:.2f})"
        return {"routing_decision": routing}

    # Interactive mode — interrupt for human review
    if state.get("needs_human_review"):
        decision = interrupt({
            "type": "review_required",
            "reason": state.get("urgency_signals", {}).get("reason", "elevated urgency"),
            "urgency_score": state.get("urgency_score"),
            "classification": state["classification"],
            "urgency_signals": state.get("urgency_signals"),
        })

        if decision.get("override_code"):
            return {
                "human_override": decision,
                "routing_decision": f"Human override → {decision.get('override_code')}",
            }
        else:
            return {"routing_decision": f"Human approved → {state['classification'].get('subcategory')}"}

    classification = state["classification"]
    return {
        "routing_decision": f"Auto-routed → {classification.get('category')}/{classification.get('subcategory')} (urgency {(state.get('urgency_score') or 0.0):.2f})",
    }


def override_node(state: WorkflowState):
    """Apply human corrections from validator_gate to the classification."""
    human_override = state.get("human_override")
    if not human_override:
        return {}

    classification = state["classification"].copy()
    if human_override.get("category"):
        classification["category"] = human_override["category"]
    if human_override.get("subcategory"):
        classification["subcategory"] = human_override["subcategory"]
    if human_override.get("urgency_score"):
        classification["urgency_score"] = human_override["urgency_score"]
    classification["human_reviewed"] = True

    return {"classification": classification}


def make_trainer_log(state: WorkflowState):
    """Build the per-call trainer log + ensure call_summary is non-trivial.

    Surfaces the full ai_prediction (including every correction provenance
    field: trap check results, urgency score, clarification reasons, etc.)
    alongside any human_override and the final_decision (AI prediction with
    overrides applied).
    """
    classification = state["classification"]
    category = (classification.get("category") or "").upper()
    subcategory = (classification.get("subcategory") or "").lower()

    address = state.get("address") or classification.get("address")
    city = state.get("city") or classification.get("city")
    provider_matches = state.get("provider_matches") or []
    merged_providers = state.get("merged_providers") or []
    needs_human = state.get("needs_human_review", False)
    needs_clarification = state.get("needs_clarification", False)

    # Call summary with fallback
    call_summary = (classification.get("call_summary") or "").strip()
    if len(call_summary) < 30:
        call_summary = (
            f"Homeowner reported {subcategory.replace('_', ' ')} at "
            f"{address or 'unknown address'}, {city or 'Santa Barbara'}. "
            f"Classified {category}/{subcategory}."
        )

    ai_prediction = {
        "category": category,
        "subcategory": subcategory,
        "address": address,
        "city": city,
        "provider_matches": provider_matches,
        "merged_providers": merged_providers,
        "needs_human_review": needs_human,
        "needs_clarification": needs_clarification,
        "urgency_score": state.get("urgency_score"),
        "urgency_signals": state.get("urgency_signals"),
        "trap_check_result": state.get("trap_check_result"),
        "lead_quality": state.get("lead_quality"),
        "clarification_reasons": list(state.get("clarification_reasons") or ()),
    }

    human_override = None
    if state.get("human_override"):
        human_override = state["human_override"]

    final_decision = ai_prediction.copy()
    if human_override:
        for k, v in human_override.items():
            if k in final_decision and v is not None:
                final_decision[k] = v

    trainer_log = {
        "full_transcript": state["transcript"],
        "ai_prediction": ai_prediction,
        "human_override": human_override,
        "final_decision": final_decision,
    }

    return {
        "trainer_log": trainer_log,
        "call_summary": call_summary,
    }