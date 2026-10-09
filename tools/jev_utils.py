"""Jev (TypeSafe System One) API utility functions.

All functions wrap Jev primitives (Choice, Score, Noul) for the home-service-agent
pipeline.  LangGraph nodes import these functions instead of calling Jev directly.

Architecture
------------
- Every function is **synchronous** (LangGraph compatibility).
- API key is read from ``TYPESAFE_API_KEY`` at call time (no global client).
- Confidence thresholds follow the spec and are documented in each function's docstring.

See https://docs.typesafe.ai/ for the full API reference.
"""

from __future__ import annotations

import os
from typing import Any

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30.0


def _get_client() -> TypeSafeClient:
    """Return a synchronous Jev client using the environment API key.

    Raises
    ------
    RuntimeError
        When ``TYPESAFE_API_KEY`` is not set or is empty.
    """
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set. "
            "Set it in your environment or .env file before calling Jev functions."
        )
    return TypeSafeClient(api_key=api_key, timeout=_DEFAULT_TIMEOUT)


# ---------------------------------------------------------------------------
# Category Classification  (Choice)
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, str | None] = {
    "Plumbing": "Pipe leaks, drain clogs, water heater issues, toilet/faucet repair",
    "HVAC": "Heating, ventilation, air conditioning, furnace, heat pump",
    "Electrical": "Wiring, outlets, breakers, lighting, panel issues",
    "Roofing": "Roof leaks, shingle damage, gutter repair, storm damage",
    "Water Damage": "Flooding, water intrusion, mold, basement water",
    "Appliance Repair": "Refrigerator, washer, dryer, dishwasher, oven, range",
    "Pest Control": "Insects, rodents, termites, bed bugs, wildlife",
    "Locksmith": "Locked out, broken lock, rekey, security upgrade",
    "General Handyman": "Minor repairs, drywall, painting, caulking, assembly",
    "Not a home service issue": "Any topic unrelated to home repair or maintenance",
}


def classify_category(problem_description: str) -> dict[str, Any]:
    """Classify a home-service problem into a top-level category.

    Parameters
    ----------
    problem_description : str
        The user's natural-language description of the issue.

    Returns
    -------
    dict
        ``{"category": str, "confidence": float, "probabilities": dict[str, float]}``

    Confidence thresholds
    ---------------------
    - ``>= 0.8``  → auto-accept
    - ``0.4 – 0.8`` → route for human review
    - ``< 0.4`` → escalate (uncertain / out-of-scope)
    """
    client = _get_client()
    result = client.system_one(
        state=problem_description,
        questions={
            "category": Choice(
                instructions="Which home service category best fits this issue?",
                criteria=CATEGORIES,
            ),
        },
    )
    answer = result.answers["category"]
    return {
        "category": answer.choice,
        "confidence": answer.confidence,
        "probabilities": answer.probabilities,
    }


# ---------------------------------------------------------------------------
# Subcategory Classification  (Choice)
# ---------------------------------------------------------------------------

SUBCATEGORIES: dict[str, dict[str, str | None]] = {
    "Plumbing": {
        "pipe_leak": "Leaking pipe under sink, in wall, or exposed",
        "drain_clog": "Slow or blocked sink, shower, or floor drain",
        "water_heater": "No hot water, strange noises, leaking tank",
        "toilet_repair": "Running, clogged, leaking, or broken toilet",
        "faucet_repair": "Dripping, low pressure, or stuck faucet / fixture",
    },
    "HVAC": {
        "ac_not_cooling": "Air conditioner not blowing cold air",
        "heater_not_working": "Furnace or heat pump not producing heat",
        "thermostat_issue": "Thermostat unresponsive, wrong temperature",
        "air_quality": "Filters, humidity, ventilation concerns",
        "noise_or_smell": "Unusual sounds or odours from HVAC system",
    },
    "Electrical": {
        "outage_or_trip": "Breaker tripped, partial or full power loss",
        "outlet_switch": "Dead outlet, sparking switch, loose plate",
        "wiring_repair": "Exposed wire, rodent damage, old wiring",
        "lighting": "Fixture repair, rewiring, flickering lights",
        "panel_upgrade": "Old panel, insufficient capacity, code upgrade",
    },
    "Roofing": {
        "leak_repair": "Water stain on ceiling, active drip after rain",
        "shingle_damage": "Missing, cracked, or curled shingles",
        "gutter_repair": "Clogged, detached, or leaking gutters",
        "storm_damage": "Wind, hail, or debris impact after storm",
        "inspection": "General roof health check for sale or insurance",
    },
    "Water Damage": {
        "flooding": "Active standing water in basement or living area",
        "water_intrusion": "Moisture seepage through walls or floor",
        "mold_remediation": "Visible mold growth or musty odour",
        "drying": "Post-leak drying and dehumidification",
    },
    "Appliance Repair": {
        "refrigerator": "Not cooling, leaking water, ice maker broken",
        "washer_dryer": "Not spinning, not heating, leaking, loud",
        "dishwasher": "Not draining, not cleaning, leaking",
        "oven_range": "Not heating, burner not lighting, temperature off",
    },
    "Pest Control": {
        "insects": "Ants, roaches, spiders, bees, wasps",
        "rodents": "Mice, rats, squirrels in attic or walls",
        "termites": "Wood damage, mud tubes, swarmers",
        "bed_bugs": "Bites, blood spots, live bugs in bedding",
        "wildlife": "Raccoons, possums, bats, birds in structure",
    },
    "Locksmith": {
        "locked_out": "Locked out of home, car, or office",
        "broken_lock": "Key broken in lock, lock mechanism failed",
        "rekey": "Change locks after move-in or lost key",
        "security_upgrade": "Smart lock, deadbolt, high-security install",
    },
    "General Handyman": {
        "drywall_repair": "Holes, cracks, or water-damaged drywall",
        "painting": "Interior or exterior painting, touch-up",
        "caulking_sealing": "Caulk around windows, doors, bathroom fixtures",
        "furniture_assembly": "Flat-pack furniture or shelving assembly",
        "misc_repair": "Squeaky door, loose hinge, shelf install, etc.",
    },
}


def classify_subcategory(problem_description: str, category: str) -> dict[str, Any]:
    """Classify a home-service problem into a specific subcategory.

    Parameters
    ----------
    problem_description : str
        The user's natural-language description of the issue.
    category : str
        The top-level category returned by :func:`classify_category`.

    Returns
    -------
    dict
        ``{"subcategory": str, "confidence": float, "probabilities": dict[str, float]}``

    Confidence thresholds
    ---------------------
    - ``>= 0.8``  → auto-accept
    - ``0.4 – 0.8`` → route for human review
    - ``< 0.4`` → escalate
    """
    options = SUBCATEGORIES.get(category)
    if not options:
        return {
            "subcategory": "general",
            "confidence": 1.0,
            "probabilities": {"general": 1.0},
        }

    client = _get_client()
    result = client.system_one(
        state=problem_description,
        questions={
            "subcategory": Choice(
                instructions=f"Which specific subcategory of {category} best fits this issue?",
                criteria=options,
            ),
        },
    )
    answer = result.answers["subcategory"]
    return {
        "subcategory": answer.choice,
        "confidence": answer.confidence,
        "probabilities": answer.probabilities,
    }


# ---------------------------------------------------------------------------
# API Selection  (Choice)
# ---------------------------------------------------------------------------

API_OPTIONS: dict[str, str | None] = {
    "Google Places": "Query Google Places API for provider names, phone, address, rating, hours",
    "Google Places + CSLB": "Google Places for provider data plus CA Contractors State License Board verification",
    "CSLB only": "CA CSLB license lookup only (no Google Places data)",
    "None": "No provider API query needed (not a serviceable issue or insufficient info)",
}


def select_api(problem_description: str, category: str) -> dict[str, Any]:
    """Decide which provider-data API(s) to query for this issue.

    Parameters
    ----------
    problem_description : str
        The user's natural-language description of the issue.
    category : str
        The top-level category returned by :func:`classify_category`.

    Returns
    -------
    dict
        ``{"api": str, "confidence": float}``

    Confidence thresholds
    ---------------------
    - ``>= 0.7``  → auto-accept
    - ``< 0.7`` → fall back to Google Places (safe default)
    """
    client = _get_client()
    result = client.system_one(
        state=f"Problem: {problem_description}\nCategory: {category}",
        questions={
            "api": Choice(
                instructions="Which provider data API should we query for this issue?",
                criteria=API_OPTIONS,
            ),
        },
    )
    answer = result.answers["api"]
    return {
        "api": answer.choice,
        "confidence": answer.confidence,
    }


# ---------------------------------------------------------------------------
# Urgency Dimension Signals  (Score + Noul)
# ---------------------------------------------------------------------------

# Each factor maps to the appropriate Jev primitive and a human-readable prompt.
_URGENCY_FACTORS: dict[str, dict[str, Any]] = {
    "water_damage_spreading": {
        "primitive": "score",
        "instructions": (
            "Is water or damage actively spreading? "
            "Rate how urgently this is worsening."
        ),
        "criteria": [
            "0 — No water or damage present / issue is stable",
            "1 — Slow seepage or very gradual spread, can wait hours",
            "2 — Steady flow or moderate spread, should be addressed soon",
            "3 — Fast-spreading water or active damage, needs prompt attention",
            "4 — Rapidly worsening (gushing water, structural weakening), emergency",
        ],
    },
    "safety_hazard": {
        "primitive": "noul",
        "instructions": "Is there a safety hazard present?",
        "true_rubric": (
            "Gas leak, exposed electrical, carbon monoxide, "
            "structural instability, fire risk, or biohazard."
        ),
        "false_rubric": (
            "No immediate safety risk — issue is a comfort, "
            "convenience, or property concern only."
        ),
    },
    "user_contained": {
        "primitive": "noul",
        "instructions": "Has the user contained the issue?",
        "true_rubric": (
            "User has shut off water at the main, turned off the breaker, "
            "blocked access, or otherwise stopped the problem from worsening."
        ),
        "false_rubric": (
            "Issue is still active / uncontained — water still flowing, "
            "electrical still live, etc."
        ),
    },
    "same_day_attention": {
        "primitive": "noul",
        "instructions": "Does this need same-day attention?",
        "true_rubric": (
            "Immediate health/safety risk, active damage that worsens by the hour, "
            "or a situation the user cannot safely wait on."
        ),
        "false_rubric": (
            "Can reasonably wait until tomorrow or the next business day "
            "without additional risk or damage."
        ),
    },
}


def assess_urgency_factor(problem_description: str, factor: str) -> dict[str, Any]:
    """Assess a single urgency dimension for a home-service problem.

    Parameters
    ----------
    problem_description : str
        The user's natural-language description of the issue.
    factor : str
        One of the known factor names:

        - ``"water_damage_spreading"``  (Score,  0 – 1  scaled to 0–4)
        - ``"safety_hazard"``           (Noul,   yes / no)
        - ``"user_contained"``          (Noul,   yes / no)  — *inverse signal*
        - ``"same_day_attention"``      (Noul,   yes / no)

    Returns
    -------
    dict
        ``{"dimension": str, "value": float, "confidence": float}``

        For Score primitives *value* is the raw score (0–4); code normalises later.
        For Noul primitives *value* is the probability of "yes" (0–1).
        *confidence* is 0–1 for Score questions, or ``1.0`` for Noul questions
        (the Noul answer does not carry a separate confidence; use ``1 - abs(value - 0.5) * 2``
        as a derived confidence if needed in the composite calculation).

    Notes
    -----
    Jev provides **raw dimension signals only**.  The composite urgency score is
    calculated in ``agent/risk.py`` — see the PRD for the weighted formula.
    """
    factor_def = _URGENCY_FACTORS.get(factor)
    if not factor_def:
        raise ValueError(
            f"Unknown urgency factor: {factor!r}. "
            f"Known factors: {list(_URGENCY_FACTORS)}"
        )

    client = _get_client()

    if factor_def["primitive"] == "score":
        question = Score(
            instructions=factor_def["instructions"],
            criteria=factor_def["criteria"],
        )
    else:  # noul
        question = Noul(
            instructions=factor_def["instructions"],
            criteria={
                "true": factor_def.get("true_rubric"),
                "false": factor_def.get("false_rubric"),
            },
        )

    result = client.system_one(
        state=problem_description,
        questions={"signal": question},
    )

    answer = result.answers["signal"]

    if factor_def["primitive"] == "score":
        return {
            "dimension": factor,
            "value": answer.score,
            "confidence": answer.confidence,
        }
    else:  # noul
        return {
            "dimension": factor,
            "value": answer.noul,
            "confidence": 1.0,
        }


# ---------------------------------------------------------------------------
# Lead Quality  (Noul)
# ---------------------------------------------------------------------------


def assess_lead_quality(conversation_summary: str, provider_name: str) -> dict[str, Any]:
    """Assess whether a lead is likely to result in a successful service engagement.

    Parameters
    ----------
    conversation_summary : str
        A summary of the user's conversation with the agent, including the problem
        description, classified category/subcategory, urgency signals, and any
        user-provided contact details or preferences.
    provider_name : str
        The matched provider's business name.

    Returns
    -------
    dict
        ``{"will_convert": bool, "confidence": float}``

        *will_convert* is ``True`` when the Jev noul probability is ``>= 0.5``,
        ``False`` otherwise.

    Confidence thresholds
    ---------------------
    - ``>= 0.8``  → auto-accept lead
    - ``< 0.8`` → flag for human review
    """
    client = _get_client()
    result = client.system_one(
        state=f"Conversation summary:\n{conversation_summary}\n\nMatched provider: {provider_name}",
        questions={
            "lead_quality": Noul(
                instructions=(
                    "Is this lead likely to result in a successful service engagement? "
                    "Consider whether the user has a genuine need, the provider is a good "
                    "match, and contact is likely to happen."
                ),
                criteria={
                    "true": "Clear need, good provider fit, user likely to reach out",
                    "false": "Vague need, poor match, user unlikely to follow through",
                },
            ),
        },
    )
    answer = result.answers["lead_quality"]
    will_convert = answer.noul >= 0.5
    # Derive confidence from the noul probability:
    #   near 0 or near 1 → high confidence
    #   near 0.5 → low confidence
    confidence = 1.0 - abs(answer.noul - 0.5) * 2.0
    return {
        "will_convert": will_convert,
        "confidence": confidence,
    }