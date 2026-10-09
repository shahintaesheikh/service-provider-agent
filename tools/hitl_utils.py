"""HITL (Human-In-The-Loop) risk-scoring utilities for the service provider agent.

All functions in this module are **standalone pure functions** — they accept
data as plain dicts / keyword arguments, perform deterministic computation, and
return result dicts.  No LangGraph state, no side effects, no I/O.

The core formula is ported verbatim from the CBRE dispatch agent reference
project (``projects/rich-shahin-finalproject/agent/risk.py``):

    Risk = P(error) × Cost(error)

    P(error) = 0.40 × (1 - llm_confidence)    [LLM's own uncertainty]
             + 0.30 × rag_disagreement         [RAG consensus demotion]
             + 0.30 × historical_mismatch      [P(intake_band != final_band)]

    Cost(error) = 0.50 × severity_weight[risk_level]      [band severity]
                + 0.30 × subcategory_cost_factor[sub]     [historical $ cost]
                + 0.20 × normalized_property_modifier     [home type criticality]

    If Risk >= 0.30 → needs_human_review = True (additive)

Adapted from the commercial-building domain to the home-service domain by
replacing ``BUILDING_TYPE_COST_MODIFIER`` with ``PROPERTY_TYPE_COST_MODIFIER``
(mapping home types instead of commercial building types).
"""

from __future__ import annotations

__all__ = [
    "compute_risk_score",
    "RISK_LEVEL_SEVERITY",
    "PROPERTY_TYPE_COST_MODIFIER",
    "RISK_HITL_THRESHOLD",
]

# ── Severity weights ─────────────────────────────────────────────────────────
# Band-name → severity weight, fed into the Cost(error) term.
# Identical scale to the reference project.
RISK_LEVEL_SEVERITY: dict[str, float] = {
    "LOW": 0.15,
    "MEDIUM": 0.40,
    "HIGH": 0.75,
    "EMERGENCY": 1.0,
}

# ── Property-type cost modifier ──────────────────────────────────────────────
# Adapted from BUILDING_TYPE_COST_MODIFIER in the reference project.
# Maps home property types to a criticality multiplier (raw range [0.8, 1.1]).
# Normalised to [0, 1] inside the Cost(error) formula.
PROPERTY_TYPE_COST_MODIFIER: dict[str, float] = {
    "single_family": 1.0,
    "condo": 0.8,
    "apartment": 1.1,
    "townhouse": 0.9,
}

# Pre-computed normalisation constants for property-type modifier.
_PM_MIN: float = min(PROPERTY_TYPE_COST_MODIFIER.values())
_PM_MAX: float = max(PROPERTY_TYPE_COST_MODIFIER.values())
_PM_RANGE: float = _PM_MAX - _PM_MIN if _PM_MAX > _PM_MIN else 1.0

# ── HITL trigger threshold ───────────────────────────────────────────────────
# Risk = P(error) × Cost(error) at or above this value triggers human review.
# Value kept identical to the reference project.
RISK_HITL_THRESHOLD: float = 0.30

# Weight allocations (matching the reference project exactly)
_W_P_ERROR_CONFIDENCE_UNCERTAINTY: float = 0.40
_W_P_ERROR_RAG_DISAGREEMENT: float = 0.30
_W_P_ERROR_HISTORICAL_MISMATCH: float = 0.30

_W_COST_ERROR_SEVERITY: float = 0.50
_W_COST_ERROR_SUBCATEGORY: float = 0.30
_W_COST_ERROR_PROPERTY: float = 0.20


def compute_risk_score(
    *,
    confidence: float = 0.5,
    risk_level: str = "MEDIUM",
    subcategory: str = "",
    property_type: str = "single_family",
    rag_disagreement: float = 0.0,
    historical_mismatch: float = 0.0,
    subcategory_cost_factor: float = 0.5,
    existing_needs_human_review: bool = False,
) -> dict:
    """Compute the HITL risk score = P(error) × Cost(error).

    This is a **standalone pure function** — it accepts keyword arguments,
    performs deterministic arithmetic, and returns a result dict.  No
    LangGraph state, no side effects, no I/O.

    Parameters
    ----------
    confidence : float (default 0.5)
        Jev Choice confidence (0-1) from category classification.
        Drives the *confidence_uncertainty* = ``1 - confidence`` term in
        P(error).
    risk_level : str (default "MEDIUM")
        One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"EMERGENCY"``.
        Drives the *severity_weight* term in Cost(error) and the hard-rule
        triggers for HIGH/EMERGENCY bands.
    subcategory : str (default "")
        Home-service subcategory identifier (e.g. ``"pipe_leak"``).
        Used for  *historical_mismatch* and *subcategory_cost_factor* lookups
        in the calling code.  Stored in the result for auditability; does *not*
        drive a dict lookup in this function (the caller passes the looked-up
        values directly).
    property_type : str (default "single_family")
        One of ``"single_family"``, ``"condo"``, ``"apartment"``,
        ``"townhouse"``.  Drives the *property_modifier_norm* term in
        Cost(error) via :data:`PROPERTY_TYPE_COST_MODIFIER`.
    rag_disagreement : float (default 0.0)
        RAG-consensus demotion strength (0-1).  Defaults to 0.0 because the
        RAG pipeline is not yet deployed.
    historical_mismatch : float (default 0.0)
        P(intake_band != final_band) for the subcategory.  Defaults to 0.0
        because no historical corpus is available yet.
    subcategory_cost_factor : float (default 0.5)
        Historical cost factor for the subcategory (0-1).  Defaults to 0.5,
        a sensible middle value in the absence of cost data.
    existing_needs_human_review : bool (default False)
        Whether an upstream node already flagged this case for human review.
        The risk trigger is **additive** — it never overrides an existing
        ``False``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``risk_score``
            P(error) × Cost(error), clamped to [0, 1].
        ``risk_details``
            Nested dict with component breakdowns and trigger explanation::

                {
                    "p_error": float,
                    "cost_error": float,
                    "p_error_components": {
                        "confidence_uncertainty": float,
                        "rag_disagreement": float,
                        "historical_mismatch": float,
                    },
                    "cost_error_components": {
                        "severity_weight": float,
                        "sub_cost_factor": float,
                        "property_modifier_norm": float,
                    },
                    "threshold": float,          # RISK_HITL_THRESHOLD
                    "reason": str,               # human-readable trigger
                }

        ``needs_human_review``
            ``True`` when any trigger condition fires:
            risk ≥ threshold, band is HIGH/EMERGENCY, or confidence < 0.6.
            Additive: existing ``True`` inputs remain ``True``.

    Examples
    --------
    >>> # Minimal input — everything at defaults.
    >>> compute_risk_score(confidence=0.8, risk_level="MEDIUM")
    {'risk_score': 0.0..., 'risk_details': {...}, 'needs_human_review': False}

    >>> # High confidence + HIGH band → human review due to band rule.
    >>> result = compute_risk_score(confidence=0.95, risk_level="HIGH")
    >>> result["needs_human_review"]
    True

    >>> # Low confidence → human review.
    >>> result = compute_risk_score(confidence=0.5, risk_level="LOW")
    >>> result["needs_human_review"]
    True
    """

    # ── Clamp & coerce inputs ─────────────────────────────────────────────
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    risk_level = risk_level.upper().strip() if risk_level else "MEDIUM"
    if risk_level not in RISK_LEVEL_SEVERITY:
        risk_level = "MEDIUM"

    property_type = (property_type or "single_family").lower().strip()

    rag_disagreement = max(0.0, min(1.0, float(rag_disagreement)))
    historical_mismatch = max(0.0, min(1.0, float(historical_mismatch)))
    subcategory_cost_factor = max(0.0, min(1.0, float(subcategory_cost_factor)))

    needs_human = bool(existing_needs_human_review)

    # ── P(error) ──────────────────────────────────────────────────────────
    confidence_uncertainty = max(0.0, 1.0 - confidence)

    p_error = (
        _W_P_ERROR_CONFIDENCE_UNCERTAINTY * confidence_uncertainty
        + _W_P_ERROR_RAG_DISAGREEMENT * rag_disagreement
        + _W_P_ERROR_HISTORICAL_MISMATCH * historical_mismatch
    )
    p_error = round(min(1.0, max(0.0, p_error)), 4)

    # ── Cost(error) ───────────────────────────────────────────────────────
    severity_weight = RISK_LEVEL_SEVERITY[risk_level]

    bm_raw = PROPERTY_TYPE_COST_MODIFIER.get(property_type, 1.0)
    property_modifier_norm = (bm_raw - _PM_MIN) / _PM_RANGE  # → [0, 1]

    cost_error = (
        _W_COST_ERROR_SEVERITY * severity_weight
        + _W_COST_ERROR_SUBCATEGORY * subcategory_cost_factor
        + _W_COST_ERROR_PROPERTY * property_modifier_norm
    )
    cost_error = round(min(1.0, max(0.0, cost_error)), 4)

    # ── Risk = P × C ──────────────────────────────────────────────────────
    risk = round(p_error * cost_error, 4)

    # ── Hard rules (keeper from reference) ─────────────────────────────────
    if risk_level in ("HIGH", "EMERGENCY"):
        needs_human = True
    if confidence < 0.6:
        needs_human = True

    # ── Risk-based additive trigger ───────────────────────────────────────
    if risk >= RISK_HITL_THRESHOLD:
        needs_human = True

    # ── Reason string ─────────────────────────────────────────────────────
    reason_parts: list[str] = []
    if risk >= RISK_HITL_THRESHOLD:
        reason_parts.append(
            f"Risk={risk:.2f} (P_err={p_error:.2f} × Cost={cost_error:.2f})"
        )
    if risk_level in ("HIGH", "EMERGENCY"):
        reason_parts.append(f"band={risk_level}")
    if confidence < 0.6:
        reason_parts.append(f"low conf {confidence:.2f}")

    return {
        "risk_score": risk,
        "risk_details": {
            "p_error": p_error,
            "cost_error": cost_error,
            "p_error_components": {
                "confidence_uncertainty": round(confidence_uncertainty, 4),
                "rag_disagreement": round(rag_disagreement, 4),
                "historical_mismatch": round(historical_mismatch, 4),
            },
            "cost_error_components": {
                "severity_weight": severity_weight,
                "sub_cost_factor": subcategory_cost_factor,
                "property_modifier_norm": round(property_modifier_norm, 4),
            },
            "threshold": RISK_HITL_THRESHOLD,
            "reason": " | ".join(reason_parts) if reason_parts else "within normal parameters",
        },
        "needs_human_review": needs_human,
    }