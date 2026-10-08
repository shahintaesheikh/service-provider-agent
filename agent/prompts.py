"""
All LLM prompt templates for the Service Provider Agent.

  - EXTRACT_SYSTEM_PROMPT + EXTRACT_USER_TEMPLATE: passed to ChatOpenAI
    for entity extraction + quality check (intake).
  - CLASSIFY_SYSTEM_PROMPT + CLASSIFY_USER_TEMPLATE: placeholder structure
    for potential future LLM classification use (Jev handles classification,
    but the prompt structure is kept for fallback or hybrid use).

──────────────── OpenAI prompt-caching layout ────────────────
Each prompt is split into two messages that map directly onto OpenAI's
prompt-cache contract:

  - SYSTEM message: 100% static. Pre-interpolated with the taxonomy at
    module load so the bytes are frozen across every request.
  - USER message: 100% dynamic. The per-call description, gate verdict,
    and transcript all live here so none of them ever interrupt the
    cacheable prefix.

OpenAI auto-caches any messages-array prefix >=1024 tokens; cache hits cost
~50% less and return ~30-80% faster TTFT on the cached portion. No SDK
flag is required — the only lever is prompt structure.
"""
from __future__ import annotations

from .config import CATEGORY_SUBCATEGORIES

# ── Taxonomy block for prompts ───────────────────────────────────────────────

_TAXONOMY_BLOCK = """HOME SERVICE CATEGORIES:

PLUMBING:
  pipe_leak         – water leaking from a pipe, supply line, or fixture
  drain_clog        – clogged sink, shower, or floor drain
  water_heater      – water heater not working, leaking, or making noise
  toilet_repair     – toilet running, clogged, or not flushing
  faucet_repair     – faucet leaking, dripping, or not working
  sewer_backup      – sewer line backup or raw sewage coming up

HVAC:
  ac_repair         – air conditioner not cooling or making strange noises
  heating_repair    – heater / furnace not producing heat
  furnace_repair    – furnace malfunction, pilot light out, unusual odors
  thermostat_issue  – thermostat not responding, incorrect readings
  duct_cleaning     – ducts need cleaning, dust issues, air quality
  hvac_maintenance  – routine HVAC inspection or tune-up

ELECTRICAL:
  power_outage      – partial or total power loss in the home
  outlet_repair     – outlet not working, sparks, loose fit
  wiring_issue      – exposed wires, faulty wiring, breaker trips
  panel_upgrade     – electrical panel needs upgrade or replacement
  lighting_install  – new light fixture installation or repair
  electrical_inspection – general electrical safety inspection

ROOFING:
  roof_leak         – water intrusion from the roof or flashing
  storm_damage      – roof damage from wind, rain, or debris
  roof_replacement  – old roof needs full replacement
  gutter_repair     – gutters leaking, clogged, or detached
  skylight_repair   – skylight leaking or damaged

WATER_DAMAGE:
  flood_cleanup     – standing water, flooding, or water intrusion
  mold_remediation  – visible mold growth or musty odors
  leak_detection    – hidden water leak behind walls or under floors
  water_extraction  – removing standing water from the property
  drying_service    – structural drying after water damage

APPLIANCE_REPAIR:
  refrigerator_repair  – fridge not cooling, ice maker issues
  washer_dryer_repair  – washer/dryer not working, leaking, noisy
  dishwasher_repair    – dishwasher not cleaning, leaking, won't start
  oven_range_repair    – oven/range not heating, burner issues
  appliance_install    – new appliance installation or hookup

PEST_CONTROL:
  rodent_infestation     – rats, mice, or other rodents in the home
  insect_infestation     – ants, roaches, spiders, or other bugs
  termite_damage         – termite infestation or wood damage
  bed_bugs               – bed bug infestation
  general_pest_control   – routine pest prevention or treatment

LOCKSMITH:
  lockout             – locked out of home or room
  lock_repair         – broken or jammed lock
  key_replacement     – lost or broken keys
  security_upgrade    – new deadbolt, smart lock, or security hardware
  rekeying            – rekeying existing locks

HANDYMAN:
  general_repair            – miscellaneous home repairs
  drywall_repair            – holes, cracks, or damaged drywall
  painting                  – interior or exterior painting
  furniture_assembly        – assembling furniture or fixtures
  caulking_sealing          – caulking around windows, doors, fixtures
  minor_plumbing_electrical – small plumbing or electrical fixes

NOT_APPLICABLE:
  general_inquiry         – general questions not requesting a service
  not_a_service_request   – wrong number or misdirected call
  wrong_number            – caller reached the wrong number"""

# ── EXTRACT (ChatOpenAI NLU) ────────────────────────────────────────────────
# System message: 100% static — fully cacheable.
# User message: per-call user description only.

EXTRACT_SYSTEM_PROMPT = (
    "You are a home service intake analyst. A homeowner has described a problem.\n"
    "Extract the key information AND assess quality in a single pass.\n"
    "If a field cannot be determined from the description, leave it as null.\n\n"
    "Severity guidelines:\n"
    "- Critical: Immediate danger to life/safety or active property damage (active gas leak,\n"
    "  confirmed fire, active flooding, sparking electrical panel, structural collapse)\n"
    "- High: Significant disruption or escalation risk (major water leak, broken HVAC in\n"
    "  extreme weather, power outage, sewer backup)\n"
    "- Medium: Needs attention soon but not an emergency (leaky faucet, clogged drain,\n"
    "  appliance not working, minor electrical issue)\n"
    "- Low: Minor/cosmetic (flickering light, dripping faucet, cabinet door off hinge,\n"
    "  painting, furniture assembly)\n\n"
    "IMPORTANT de-escalation rules:\n"
    "- Burnt food smell (popcorn, toast, microwave) without visible fire → Low severity\n"
    "- Water stain on ceiling with NO active dripping → Medium, not Critical\n"
    "- Sump pump running but no standing water → Medium vigilance\n\n"
    "Quality check: set description_specific_enough=true if the problem is a specific symptom.\n"
    "Set needs_clarification=true only if the description is genuinely too vague to classify\n"
    "or a safety-critical detail is ambiguous or location is missing.\n\n"
    "─── CLARIFICATION HEURISTIC (when followup is needed) ───\n"
    "When the description is specific enough to classify but a SAFETY-RELEVANT or\n"
    "DISPATCH-RELEVANT detail is missing, set needs_clarification=true and write ONE\n"
    "targeted followup_question. Pick the dimension whose answer is most likely to change\n"
    "the severity or dispatch decision.\n\n"
    "Dimensions to probe, in priority order:\n"
    "  1. Immediate danger to people — is anyone exposed, trapped, or unable to leave?\n"
    "  2. Sensory confirmation vs. inference — distinguish \"I see flames / I smell gas\"\n"
    "     from \"the alarm is going off\" or \"something smells off\".\n"
    "  3. Scope and spread — is the issue confined to one area or spreading?\n"
    "     (Most relevant for water, fire, and structural issues.)\n"
    "  4. Containment and mitigation — has the caller already shut off a valve, mopped\n"
    "     up, or cleared the area?\n"
    "  5. Location — which neighborhood or area of Santa Barbara?\n\n"
    "For ROUTINE symptoms (painting, furniture assembly, appliance install, light bulb)\n"
    "skip the follow-up."
)

EXTRACT_USER_TEMPLATE = "Homeowner description: {description}"

# ── CLASSIFY (placeholder — Jev handles classification) ─────────────────────
# System message: 100% static. Pre-interpolated with taxonomy at module load.
# Kept as a placeholder for potential future LLM classification fallback.

_CLASSIFY_SYSTEM_TEMPLATE = """You are a home service dispatch AI. Classify the homeowner's request below.

HOME SERVICE TAXONOMY:
{taxonomy}

─── CONFIDENCE GUIDELINES ───
- If the description clearly matches ONE subcategory → high confidence (>=0.8)
- If it could fit 2-3 subcategories → medium confidence (0.4-0.8)
- If it's vague or doesn't match well → low confidence (<0.4)

─── LOCATION ───
All properties are in Santa Barbara, CA or surrounding areas (Goleta, Carpinteria,
Montecito, Summerland, Hope Ranch). If the caller doesn't specify a neighborhood,
default to Santa Barbara proper.

─── HUMAN REVIEW (needs_human_review) ───
TRUE when:
1. Severity is Critical or High
2. The description involves urgent safety concerns
3. Genuinely ambiguous classification
FALSE for routine LOW/MEDIUM requests with clear classification.

Return a JSON object with these fields:
  category, subcategory, confidence (0-1), reasoning,
  needs_human_review, needs_clarification,
  call_summary (2-3 sentence operator-style narrative),
  address, city
"""

CLASSIFY_SYSTEM_PROMPT = _CLASSIFY_SYSTEM_TEMPLATE.format(
    taxonomy=_TAXONOMY_BLOCK,
)

CLASSIFY_USER_TEMPLATE = """─── EXTRACTED ENTITIES ───
{entities}

TRANSCRIPT:
{transcript}

Classify this request. Return a JSON object with these fields:
  category, subcategory, confidence (0-1), reasoning,
  needs_human_review, needs_clarification,
  call_summary (2-3 sentence operator-style narrative),
  address, city"""