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
from tools.provider_search import geocode_location, merge_google_and_yelp


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

    Merges Google Places and Yelp results (via phone-number dedup),
    enriches with Yelp review data, and sorts by rating descending.
    Yelp-only entries are included as secondary options.
    """
    provider_matches = state.get("provider_matches") or []
    yelp_results = state.get("yelp_results") or []

    merged = merge_google_and_yelp(provider_matches, yelp_results)

    return {
        "merged_providers": merged,
    }