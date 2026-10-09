"""
Static configuration + operational data for the Service Provider Agent.

This module defines the home service taxonomy (10 categories × 3–6
subcategories each), Santa Barbara area constants, provider API key
placeholders, policy thresholds (HITL threshold, confidence thresholds),
and path resolution.

Importing this module also triggers `load_dotenv()` so any module that
constructs an LLM client sees `OPENAI_API_KEY` from the user's .env.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# .env is at the repo root, one level above this package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

# ── Paths ────────────────────────────────────────────────────────────────────

HERE = _REPO_ROOT

# ── Santa Barbara area constants ─────────────────────────────────────────────

SANTA_BARBARA_CITIES = {
    "santa barbara": "Santa Barbara",
    "goleta": "Goleta",
    "carpinteria": "Carpinteria",
    "montecito": "Montecito",
    "summerland": "Summerland",
    "hope ranch": "Hope Ranch",
    "mission canyon": "Mission Canyon",
    "santa barbara county": "Santa Barbara County",
}

DEFAULT_CITY = "Santa Barbara"
PROVIDER_RADIUS_MILES = 50

# ── Home service taxonomy (10 categories × 3–6 subcategories each) ───────────

CATEGORY_SUBCATEGORIES = {
    "PLUMBING": [
        "pipe_leak",
        "drain_clog",
        "water_heater",
        "toilet_repair",
        "faucet_repair",
        "sewer_backup",
    ],
    "HVAC": [
        "ac_repair",
        "heating_repair",
        "furnace_repair",
        "thermostat_issue",
        "duct_cleaning",
        "hvac_maintenance",
    ],
    "ELECTRICAL": [
        "power_outage",
        "outlet_repair",
        "wiring_issue",
        "panel_upgrade",
        "lighting_install",
        "electrical_inspection",
    ],
    "ROOFING": [
        "roof_leak",
        "storm_damage",
        "roof_replacement",
        "gutter_repair",
        "skylight_repair",
    ],
    "WATER_DAMAGE": [
        "flood_cleanup",
        "mold_remediation",
        "leak_detection",
        "water_extraction",
        "drying_service",
    ],
    "APPLIANCE_REPAIR": [
        "refrigerator_repair",
        "washer_dryer_repair",
        "dishwasher_repair",
        "oven_range_repair",
        "appliance_install",
    ],
    "PEST_CONTROL": [
        "rodent_infestation",
        "insect_infestation",
        "termite_damage",
        "bed_bugs",
        "general_pest_control",
    ],
    "LOCKSMITH": [
        "lockout",
        "lock_repair",
        "key_replacement",
        "security_upgrade",
        "rekeying",
    ],
    "HANDYMAN": [
        "general_repair",
        "drywall_repair",
        "painting",
        "furniture_assembly",
        "caulking_sealing",
        "minor_plumbing_electrical",
    ],
    "NOT_APPLICABLE": [
        "general_inquiry",
        "not_a_service_request",
        "wrong_number",
    ],
}

SUBCATEGORY_TO_CATEGORY: dict[str, str] = {
    sub: cat
    for cat, subs in CATEGORY_SUBCATEGORIES.items()
    for sub in subs
}

ALL_SUBCATEGORIES = set(SUBCATEGORY_TO_CATEGORY.keys())

SUBCATEGORY_TO_VENDOR_TYPE = {
    "pipe_leak": "plumber",
    "drain_clog": "plumber",
    "water_heater": "plumber",
    "toilet_repair": "plumber",
    "faucet_repair": "plumber",
    "sewer_backup": "plumber",
    "ac_repair": "hvac",
    "heating_repair": "hvac",
    "furnace_repair": "hvac",
    "thermostat_issue": "hvac",
    "duct_cleaning": "hvac",
    "hvac_maintenance": "hvac",
    "power_outage": "electrician",
    "outlet_repair": "electrician",
    "wiring_issue": "electrician",
    "panel_upgrade": "electrician",
    "lighting_install": "electrician",
    "electrical_inspection": "electrician",
    "roof_leak": "roofer",
    "storm_damage": "roofer",
    "roof_replacement": "roofer",
    "gutter_repair": "roofer",
    "skylight_repair": "roofer",
    "flood_cleanup": "water_damage",
    "mold_remediation": "water_damage",
    "leak_detection": "water_damage",
    "water_extraction": "water_damage",
    "drying_service": "water_damage",
    "refrigerator_repair": "appliance_repair",
    "washer_dryer_repair": "appliance_repair",
    "dishwasher_repair": "appliance_repair",
    "oven_range_repair": "appliance_repair",
    "appliance_install": "appliance_repair",
    "rodent_infestation": "pest_control",
    "insect_infestation": "pest_control",
    "termite_damage": "pest_control",
    "bed_bugs": "pest_control",
    "general_pest_control": "pest_control",
    "lockout": "locksmith",
    "lock_repair": "locksmith",
    "key_replacement": "locksmith",
    "security_upgrade": "locksmith",
    "rekeying": "locksmith",
    "general_repair": "handyman",
    "drywall_repair": "handyman",
    "painting": "handyman",
    "furniture_assembly": "handyman",
    "caulking_sealing": "handyman",
    "minor_plumbing_electrical": "handyman",
    "general_inquiry": "other",
    "not_a_service_request": "other",
    "wrong_number": "other",
}

# ── API key placeholders ─────────────────────────────────────────────────────

# These will be populated from .env or environment variables at runtime.
# Actual API calls are implemented in a separate file (not part of this task).
GOOGLE_PLACES_API_KEY = ""        # TODO: set via GOOGLE_PLACES_API_KEY env var
YELP_FUSION_API_KEY = ""          # TODO: set via YELP_FUSION_API_KEY env var
YELP_BUSINESS_CLIENT_ID = ""      # TODO: set via YELP_FUSION_CLIENT_ID env var

# ── Policy thresholds ────────────────────────────────────────────────────────

# HITL trigger threshold for the composite urgency score.
# When urgency_score >= HITL_THRESHOLD, needs_human_review is set to True.
HITL_THRESHOLD = 0.30

# Confidence thresholds for Jev Choice classification.
# These mirror the reference project's band scheme.
CONFIDENCE_AUTO_ACCEPT = 0.80    # >= 0.80: auto-route
CONFIDENCE_REVIEW = 0.40         # 0.40–0.80: route for human review
                                 # < 0.40: escalate

# Confidence threshold for Jev Noul lead quality check.
LEAD_QUALITY_AUTO_ACCEPT = 0.80  # >= 0.80: lead is good

# Urgency score weights (will be tuned with real data).
URGENCY_WEIGHTS = {
    "water_spread_score": 0.30,
    "safety_hazard_noul": 0.25,
    "contained_inverse": 0.20,
    "keyword_safety_signals": 0.25,
}

# Danger words that trigger over-escalation guard review.
DANGER_WORDS = [
    "fire", "smoke", "gas", "flood", "electrical burning",
    "sparking", "hot panel", "carbon monoxide", "natural gas",
    "propane", "evacuate", "emergency", "collapse", "ceiling fell",
    "structural", "gas leak",
]

# Subcategories that genuinely warrant emergency dispatch.
GENUINE_EMERGENCY_SUBCATEGORIES = {
    "flood_cleanup",  # active flooding
    "sewer_backup",   # raw sewage hazard
}