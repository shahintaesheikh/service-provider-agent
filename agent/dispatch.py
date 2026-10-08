"""
Dispatch nodes — location resolution and provider merge/ranking.

Stubs for:
  - resolve_location: canonicalize address + city from extracted entities
  - merge_and_rank: merge, deduplicate, and rank provider results

Each node returns its expected state fields with None/empty defaults.
No API calling yet — those come in later tasks.
"""
from __future__ import annotations

from .config import CATEGORY_SUBCATEGORIES, SUBCATEGORY_TO_CATEGORY, DEFAULT_CITY
from .state import WorkflowState


def resolve_location(state: WorkflowState):
    """Stub: canonocalize the address and city from extracted entities.

    TODO: fuse entity extraction output, classification hints, and
    any caller profile data into a canonical location record.
    Falls back to Santa Barbara (default city for MVP).
    """
    entity = state.get("entity") or {}

    address = entity.get("address")
    city = entity.get("city") or DEFAULT_CITY

    return {
        "address": address,
        "city": city,
    }


def merge_and_rank(state: WorkflowState):
    """Stub: merge, deduplicate, and rank provider results.

    TODO: implement full merge/rank logic:
      1. Merge Google Places + Yelp results by phone/name
      2. Deduplicate (prefer Google Places as primary)
      3. Filter by: open now, distance < 50mi, rating >= 3.0
      4. Sort by: rating desc, review count desc
      5. Return top 3 candidates
    """
    provider_matches = state.get("provider_matches") or []

    merged = list(provider_matches)

    return {
        "merged_providers": merged,
    }