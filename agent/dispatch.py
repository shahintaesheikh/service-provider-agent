"""
Dispatch nodes — location resolution and provider merge/ranking.

Maps to the brief's deliverables on:
  - **Location resolution** — resolve_location uses Google Geocoding API
    via tools.provider_search.geocode_location() to canonicalize city/state.
  - **Provider merge & rank** — merge_and_rank calls
    tools.provider_search.merge_and_rank_providers() for dedup + rating sort.
"""
from __future__ import annotations

from .config import CATEGORY_SUBCATEGORIES, SUBCATEGORY_TO_CATEGORY, DEFAULT_CITY
from .state import WorkflowState
from tools.provider_search import geocode_location, merge_and_rank_providers


def resolve_location(state: WorkflowState):
    """Canonicalize the address and city from extracted entities.

    Uses tools.provider_search.geocode_location() to convert the city
    name to lat/lng coordinates for downstream provider search.
    Falls back to Santa Barbara (default city for MVP) when no location
    is extracted.
    """
    entity = state.get("entity") or {}

    address = entity.get("address")
    city = entity.get("city") or DEFAULT_CITY

    # Attempt geocoding — gracefully handles missing API key
    geo = geocode_location(f"{city}, CA")
    latitude = geo.get("latitude")
    longitude = geo.get("longitude")

    return {
        "address": address,
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
    }


def merge_and_rank(state: WorkflowState):
    """Merge, deduplicate, and rank provider results.

    Uses tools.provider_search.merge_and_rank_providers() to
    deduplicate by place ID, sort by rating descending, and return
    the top 5 candidates.
    """
    provider_matches = state.get("provider_matches") or []

    merged = merge_and_rank_providers(provider_matches)

    return {
        "merged_providers": merged,
    }