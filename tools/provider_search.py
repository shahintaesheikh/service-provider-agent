"""
Google Places API (New) + CA CSLB — provider search tool functions.

All functions are synchronous (LangGraph nodes are synchronous).
Uses the ``requests`` library for HTTP calls.

SKU: service-provider-agent
Date: 2025-04-17

Dependencies:
    - requests (install via requirements.txt)
    - GOOGLE_API_KEY environment variable

API reference: https://developers.google.com/maps/documentation/places/web-service/
"""

import os
import re
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")

PLACES_BASE_URL: str = "https://places.googleapis.com/v1"
GEOCODE_BASE_URL: str = "https://geocode.googleapis.com/v1"

TIMEOUT_SECONDS: int = 15

# ---------------------------------------------------------------------------
# Field masks (cost control — request only the fields you need)
# ---------------------------------------------------------------------------

# Pro-tier mask for search results (Text Search / Nearby Search).
# Rating, phone, and businessStatus are Pro-only; avoid Enterprise fields.
SEARCH_FIELD_MASK: str = (
    "places.id,places.displayName,places.formattedAddress,"
    "places.location,places.rating,places.userRatingCount,"
    "places.nationalPhoneNumber,places.primaryType,"
    "places.businessStatus"
)

# Pro-tier mask for Place Details.
DETAILS_FIELD_MASK: str = (
    "places.id,places.displayName,places.formattedAddress,"
    "places.shortFormattedAddress,places.location,"
    "places.googleMapsUri,places.websiteUri,"
    "places.nationalPhoneNumber,places.internationalPhoneNumber,"
    "places.rating,places.userRatingCount,places.businessStatus,"
    "places.primaryType,places.types,places.regularOpeningHours,"
    "places.priceLevel"
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _api_key(key_override: Optional[str] = None) -> str:
    """Return the effective API key or raise ``ValueError``."""
    key = key_override or GOOGLE_API_KEY
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. "
            "Set the GOOGLE_API_KEY environment variable or pass api_key=."
        )
    return key


def _build_headers(api_key: str, field_mask: Optional[str] = None) -> dict[str, str]:
    """Return common request headers."""
    headers: dict[str, str] = {
        "X-Goog-API-Key": api_key,
        "Content-Type": "application/json",
    }
    if field_mask is not None:
        headers["X-Goog-FieldMask"] = field_mask
    return headers


# ---------------------------------------------------------------------------
# 1.  search_providers  —  Google Places Text Search  (Pro SKU)
# ---------------------------------------------------------------------------


def search_providers(
    service_type: str,
    location: str,
    max_results: int = 20,
    api_key: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Search for home-service providers by type and location.

    Calls Google Places **Text Search** (Places API Text Search Pro SKU).

    Args:
        service_type: Category label, e.g. ``"plumber"``, ``"electrician"``.
        location: Human-readable location, e.g. ``"Santa Barbara, CA"``.
        max_results: Maximum number of results returned (clamped to 1‑20).
        api_key: Optional override for the ``GOOGLE_API_KEY`` env var.

    Returns:
        A list of place dicts.  Each dict contains at least the Pro-tier
        fields requested via the field mask.  Returns an empty list on any
        API or network error (never raises).
    """
    try:
        key = _api_key(api_key)
    except ValueError:
        return []

    query = f"{service_type} in {location}"
    payload: dict[str, Any] = {
        "textQuery": query,
        "maxResultPageSize": max(1, min(max_results, 20)),
    }

    headers = _build_headers(key, field_mask=SEARCH_FIELD_MASK)

    try:
        resp = requests.post(
            f"{PLACES_BASE_URL}/places:searchText",
            json=payload,
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        # 403  →  billing not enabled  429  →  rate limited
        if resp.status_code in (403, 429):
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("places", [])
    except (requests.RequestException, ValueError, KeyError):
        return []


# ---------------------------------------------------------------------------
# 2.  search_providers_nearby  —  Google Places Nearby Search  (Pro SKU)
# ---------------------------------------------------------------------------


def search_providers_nearby(
    latitude: float,
    longitude: float,
    radius_meters: float = 5000.0,
    included_types: Optional[list[str]] = None,
    max_results: int = 20,
    api_key: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Search for providers near a geographic point.

    Calls Google Places **Nearby Search** (Places API Nearby Search Pro SKU).

    Args:
        latitude: Centre latitude.
        longitude: Centre longitude.
        radius_meters: Search radius in metres (default 5 km).
        included_types: Google Place Types to filter on, e.g.
            ``["plumber", "electrician", "hvac_contractor"]``.
            If ``None``, defaults to ``["home_services"]``.
        max_results: Maximum number of results (clamped to 1‑20).
        api_key: Optional override for the ``GOOGLE_API_KEY`` env var.

    Returns:
        A list of place dicts (same shape as ``search_providers``).
        Returns an empty list on any API or network error.
    """
    try:
        key = _api_key(api_key)
    except ValueError:
        return []

    if included_types is None:
        included_types = ["home_services"]

    payload: dict[str, Any] = {
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius_meters,
            }
        },
        "includedTypes": included_types,
        "maxResultPageSize": max(1, min(max_results, 20)),
    }

    headers = _build_headers(key, field_mask=SEARCH_FIELD_MASK)

    try:
        resp = requests.post(
            f"{PLACES_BASE_URL}/places:searchNearby",
            json=payload,
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code in (403, 429):
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("places", [])
    except (requests.RequestException, ValueError, KeyError):
        return []


# ---------------------------------------------------------------------------
# 3.  get_place_details  —  Google Place Details  (Pro SKU)
# ---------------------------------------------------------------------------


def get_place_details(
    place_id: str,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Get detailed information for a specific place.

    Calls **Place Details** (Places API Place Details Pro SKU).

    Args:
        place_id: The Google Place ID obtained from search results.
        api_key: Optional override for the ``GOOGLE_API_KEY`` env var.

    Returns:
        A dict with the full Pro-tier place details (name, address, phone,
        hours, website, rating, etc.).  Returns ``{}`` on any API or
        network error.
    """
    if not place_id:
        return {}

    try:
        key = _api_key(api_key)
    except ValueError:
        return {}

    headers = _build_headers(key, field_mask=DETAILS_FIELD_MASK)

    try:
        resp = requests.get(
            f"{PLACES_BASE_URL}/places/{place_id}",
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code in (403, 429):
            return {}
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return {}


# ---------------------------------------------------------------------------
# 4.  geocode_location  —  Google Geocoding API  (Essentials SKU)
# ---------------------------------------------------------------------------


def geocode_location(
    city_state: str,
    api_key: Optional[str] = None,
) -> dict[str, Optional[float]]:
    """Convert a city/state string to latitude/longitude coordinates.

    Calls the **Geocoding** API (Essentials SKU — 10 000 free requests/month).

    Args:
        city_state: Human-readable location, e.g. ``"Santa Barbara, CA"``.
        api_key: Optional override for the ``GOOGLE_API_KEY`` env var.

    Returns:
        ``{"latitude": float, "longitude": float}`` on success, or
        ``{"latitude": None, "longitude": None}`` on any error.
    """
    try:
        key = _api_key(api_key)
    except ValueError:
        return {"latitude": None, "longitude": None}

    headers = _build_headers(key, field_mask=None)  # Geocoding doesn't use field-mask

    try:
        resp = requests.post(
            f"{GEOCODE_BASE_URL}/geocode:search",
            json={"address": city_state},
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code in (403, 429):
            return {"latitude": None, "longitude": None}
        resp.raise_for_status()
        data = resp.json()
        places = data.get("places", [])
        if not places:
            return {"latitude": None, "longitude": None}
        loc = places[0].get("location", {})
        return {
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
        }
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return {"latitude": None, "longitude": None}


# ---------------------------------------------------------------------------
# 5.  verify_contractor_license  —  CA CSLB screen-scrape  (optional)
# ---------------------------------------------------------------------------

# CSLB search URLs
_CSLB_SEARCH_URL = (
    "https://cslb.ca.gov/OnlineServices/CheckLicenseII/CheckLicense.aspx"
)

# Known CSLB classifications that map to home-service trades
_CSLB_CLASSIFICATIONS: dict[str, str] = {
    "A": "General Engineering Contractor",
    "B": "General Building Contractor",
    "C-2": "Insulation and Acoustical Contractor",
    "C-4": "Boiler, Hot Water Heating and Steam Fitting Contractor",
    "C-7": "Building Moving and Demolition Contractor",
    "C-10": "Electrical Contractor",
    "C-16": "Fire Protection Contractor",
    "C-20": "HVAC Contractor",
    "C-21": "Building Moving and Demolition Contractor",
    "C-27": "Landscaping Contractor",
    "C-33": "Painting and Decorating Contractor",
    "C-36": "Plastering Contractor",
    "C-38": "Plumbing Contractor",
    "C-39": "Roofing Contractor",
    "C-43": "Sheet Metal Contractor",
    "C-47": "General Manufactured Housing Contractor",
    "C-48": "Moving Contractor",
    "C-53": "Swimming Pool Contractor",
    "C-54": "Tile (Ceramic and Mosaic) Contractor",
    "C-55": "Water Conditioning Contractor",
    "C-57": "Well Drilling Contractor",
    "C-60": "Washing Machine Contractor",
    "C-61": "Limited Specialty Contractor",
}


def verify_contractor_license(
    business_name: str,
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    """Check a business name against the CA Contractor State License Board.

    **This is a free screen-scrape** of the CSLB public lookup tool.  Because
    it depends on the CSLB web site's ASPX form structure, it may break if
    the site is redesigned.

    Args:
        business_name: The business name to look up (e.g. ``"Lewis Plumbing"``).
        timeout_seconds: HTTP request timeout.

    Returns:
        A dict with keys:
        - ``"found"`` (bool)          — whether a matching license was found.
        - ``"license_number"`` (str)  — CSLB license number, if found.
        - ``"status"`` (str)          — ``"ACTIVE"``, ``"INACTIVE"``, or
          ``"UNKNOWN"``.
        - ``"classification"`` (str)  — trade classification code.
        - ``"classification_name"`` (str) — human-readable trade name.
        - ``"business_name"`` (str)   — name on the license.
        - ``"bond_info"`` (str)       — bond / workers' comp info (raw).
        - ``"error"`` (str)           — error description if the scrape failed.

    Note:
        This is a **best-effort** lookup.  The CSLB site returns at most 50
        results, and matching is done by normalised substring.  If the scrape
        fails structurally, the returned dict will have ``found=False`` and
        an ``error`` key explaining why.
    """
    # Default response — conservative: assume not found.
    default: dict[str, Any] = {
        "found": False,
        "license_number": None,
        "status": "UNKNOWN",
        "classification": None,
        "classification_name": None,
        "business_name": business_name,
        "bond_info": None,
        "error": None,
    }

    try:
        # Step 1 — GET the form to obtain ASP.NET view-state tokens
        session = requests.Session()
        get_resp = session.get(_CSLB_SEARCH_URL, timeout=timeout_seconds)
        get_resp.raise_for_status()

        # Extract ASP.NET hidden fields for the POST
        html = get_resp.text
        view_state = _extract_asp_hidden(html, "__VIEWSTATE")
        view_state_gen = _extract_asp_hidden(html, "__VIEWSTATEGENERATOR")
        event_validation = _extract_asp_hidden(html, "__EVENTVALIDATION")

        # Step 2 — POST the business-name search
        form_data: dict[str, str] = {
            "__VIEWSTATE": view_state,
            "__VIEWSTATEGENERATOR": view_state_gen,
            "__EVENTVALIDATION": event_validation,
            "ctl00$MainContent$txtContractorName": business_name,
            "ctl00$MainContent$txtLicenseNumber": "",
            "ctl00$MainContent$btnSearch": "Search",
        }

        post_resp = session.post(
            _CSLB_SEARCH_URL,
            data=form_data,
            timeout=timeout_seconds,
        )
        post_resp.raise_for_status()

        # Step 3 — parse the results table
        return _parse_cslb_results(post_resp.text, business_name, default)

    except (requests.RequestException, ValueError, KeyError) as exc:
        default["error"] = f"CSLB scrape failed: {exc}"
        return default


# ---------------------------------------------------------------------------
# CSLB helpers
# ---------------------------------------------------------------------------


def _extract_asp_hidden(html: str, field_name: str) -> str:
    """Extract the value of an ASP.NET hidden input field from HTML."""
    # Pattern: <input type="hidden" name="__VIEWSTATE" value="..." />
    pattern = re.escape(field_name) + r'"\s+value="([^"]*)"'
    match = re.search(pattern, html)
    if match:
        return match.group(1)
    # Try alternate attribute order
    pattern2 = r'value="([^"]*)"[^>]*name="' + re.escape(field_name) + r'"'
    match = re.search(pattern2, html)
    if match:
        return match.group(1)
    return ""


def _parse_cslb_results(
    html: str,
    business_name: str,
    default: dict[str, Any],
) -> dict[str, Any]:
    """Parse the CSLB results table, looking for a matching business name."""
    # Look for the results table
    # The CSLB page uses an HTML table with class "table table-striped"
    table_pattern = r'<table[^>]*class="table[^"]*"[^>]*>(.*?)</table>'
    table_match = re.search(table_pattern, html, re.DOTALL | re.IGNORECASE)
    if not table_match:
        # No results table → either no results or the page structure changed
        default["error"] = "No results table found on CSLB page"
        return default

    table_html = table_match.group(1)

    # Parse rows
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)

    if len(rows) <= 1:
        # Header row only → empty results
        default["status"] = "NOT_FOUND"
        default["error"] = "No matching license found"
        return default

    # Normalise the business name for comparison
    norm_query = _normalise_name(business_name)

    # Skip header row (index 0)
    for row in rows[1:]:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if len(cells) < 4:
            continue

        # Typical columns: License #, Business Name, Status, Classification
        license_num = _strip_html(cells[0]).strip()
        row_business = _strip_html(cells[1]).strip()
        status = _strip_html(cells[2]).strip()
        classification = _strip_html(cells[3]).strip()

        # Match by normalised substring
        if norm_query in _normalise_name(row_business) or _normalise_name(
            row_business
        ) in norm_query:
            class_name = _CSLB_CLASSIFICATIONS.get(classification, "")
            result: dict[str, Any] = {
                "found": True,
                "license_number": license_num,
                "status": status.upper() if status else "UNKNOWN",
                "classification": classification,
                "classification_name": class_name,
                "business_name": row_business,
                "bond_info": None,
                "error": None,
            }
            # Attempt to extract bond/workers comp info from remaining columns
            if len(cells) >= 6:
                result["bond_info"] = _strip_html(cells[5]).strip()
            return result

    # No match found
    default["status"] = "NOT_FOUND"
    default["error"] = f"No CSLB license found matching '{business_name}'"
    return default


def _normalise_name(name: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _strip_html(html_fragment: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", html_fragment)


# ---------------------------------------------------------------------------
# 6.  merge_and_rank_providers
# ---------------------------------------------------------------------------


def merge_and_rank_providers(
    google_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate, sort by rating descending, and return top‑5 results.

    Args:
        google_results: Raw results from ``search_providers`` or
            ``search_providers_nearby``.

    Returns:
        At most 5 deduplicated results sorted by ``rating`` descending,
        then by ``userRatingCount`` descending as a tie-breaker.
        List entries that are clearly closed (businessStatus != OPERATIONAL)
        are placed after open ones.
    """
    if not google_results:
        return []

    # Deduplicate by place id (keep first occurrence)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for place in google_results:
        pid = place.get("id") or place.get("place_id", "")
        if pid and pid not in seen:
            seen.add(pid)
            deduped.append(place)
        elif not pid:
            # No ID — include anyway (edge case)
            deduped.append(place)

    # Sort — open now first, then by rating descending, then review count
    def _sort_key(p: dict[str, Any]) -> tuple:
        # Check if the place has opening hours indicating it's open
        hours = p.get("regularOpeningHours") or {}
        open_now = hours.get("openNow", False) if isinstance(hours, dict) else False
        # Also check businessStatus
        is_open = p.get("businessStatus") == "OPERATIONAL"
        # Composite: open_now (1) > is_open (0.5) > closed (0)
        open_score = 1.0 if open_now else (0.5 if is_open else 0.0)
        rating = p.get("rating") or 0.0
        reviews = p.get("userRatingCount") or 0
        # Sort by open_score desc, rating desc, reviews desc
        return (-open_score, -rating, -reviews)

    deduped.sort(key=_sort_key)

    return deduped[:5]


# ---------------------------------------------------------------------------
# 7.  search_yelp_providers  —  Yelp Fusion Business Search
# ---------------------------------------------------------------------------


YELP_API_KEY: str = os.environ.get("YELP_API_KEY", "")

YELP_BASE_URL: str = "https://api.yelp.com/v3"


def search_yelp_providers(
    service_type: str,
    location: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Search Yelp Fusion API for providers by service type and location.

    Free tier: ~500 calls/day limit.  Use sparingly as secondary source.

    Args:
        service_type: Category label, e.g. ``"plumber"``, ``"electrician"``.
        location: Human-readable location, e.g. ``"Santa Barbara, CA"``.
        api_key: Optional override for the ``YELP_API_KEY`` env var.
        max_results: Maximum number of results (clamped to 1\u201350, default 5).

    Returns:
        A list of provider dicts with keys:
        - ``name`` (str)
        - ``phone`` (str) \u2014 national phone number
        - ``address`` (str) \u2014 formatted address
        - ``rating`` (float) \u2014 Yelp rating (1.0\u20135.0)
        - ``review_count`` (int) \u2014 number of Yelp reviews
        - ``yelp_url`` (str) \u2014 URL to the Yelp business page
        - ``categories`` (list[str]) \u2014 Yelp category tags
        - ``source`` (str) \u2014 always ``"yelp"``
        Returns an empty list on any API or network error (never raises).
    """
    key: str = api_key or YELP_API_KEY
    if not key:
        return []

    headers = {
        "Authorization": f"Bearer {key}",
        "accept": "application/json",
    }
    params: dict[str, Any] = {
        "term": service_type,
        "location": location,
        "limit": max(1, min(max_results, 50)),
        "sort_by": "rating",
    }

    try:
        resp = requests.get(
            f"{YELP_BASE_URL}/businesses/search",
            headers=headers,
            params=params,
            timeout=TIMEOUT_SECONDS,
        )
        # 403 \u2192 invalid/revoked key  429 \u2192 rate limited
        if resp.status_code in (403, 429):
            return []
        resp.raise_for_status()
        data = resp.json()
        businesses = data.get("businesses", [])

        results: list[dict[str, Any]] = []
        for biz in businesses:
            location_data = biz.get("location", {}) or {}
            address_parts = location_data.get("display_address", [])
            address = ", ".join(address_parts) if address_parts else ""

            categories = [
                cat.get("title", "")
                for cat in biz.get("categories", [])
                if cat.get("title")
            ]

            results.append(
                {
                    "name": biz.get("name", ""),
                    "phone": biz.get("phone", ""),
                    "address": address,
                    "rating": biz.get("rating"),
                    "review_count": biz.get("review_count", 0),
                    "yelp_url": biz.get("url", ""),
                    "categories": categories,
                    "source": "yelp",
                }
            )

        return results
    except (requests.RequestException, ValueError, KeyError):
        return []


# ---------------------------------------------------------------------------
# 8.  merge_google_and_yelp  \u2014  Merge Google Places + Yelp by phone
# ---------------------------------------------------------------------------


def merge_google_and_yelp(
    google_results: list[dict[str, Any]],
    yelp_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge Google Places and Yelp results, deduplicating by phone number.

    Primary sort: Google rating descending.  Yelp-only results are appended
    as secondary options after all Google-sourced entries.

    Args:
        google_results: Results from ``search_providers`` or
            ``search_providers_nearby`` (Google Places format).
        yelp_results: Results from ``search_yelp_providers`` (Yelp format).

    Returns:
        A merged list sorted by ``rating`` descending.  Google entries are
        enriched with Yelp fields (``yelp_rating``, ``yelp_review_count``,
        ``yelp_url``) when a phone-number match is found.  Providers only
        on Yelp are included as secondary options at the end.
    """
    if not google_results and not yelp_results:
        return []

    # Build a phone \u2192 Yelp entry lookup (normalised to digits)
    yelp_by_phone: dict[str, dict[str, Any]] = {}
    for yelp in yelp_results:
        phone = yelp.get("phone", "")
        if phone:
            norm = re.sub(r"[^\d]", "", phone)
            if norm:
                yelp_by_phone[norm] = yelp

    # Phase 1 \u2014 enrich Google entries with matching Yelp data
    enriched_google: list[dict[str, Any]] = []
    matched_phones: set[str] = set()

    for google in google_results:
        g_phone = google.get("nationalPhoneNumber") or google.get("phone", "")
        g_norm = re.sub(r"[^\d]", "", g_phone) if g_phone else ""

        if g_norm and g_norm in yelp_by_phone:
            yelp_data = yelp_by_phone[g_norm]
            enriched_google.append(
                {
                    **google,
                    "yelp_rating": yelp_data.get("rating"),
                    "yelp_review_count": yelp_data.get("review_count", 0),
                    "yelp_url": yelp_data.get("yelp_url", ""),
                }
            )
            matched_phones.add(g_norm)
        else:
            enriched_google.append(google)

    # Sort Google-sourced entries by rating desc
    def _google_key(p: dict[str, Any]) -> tuple:
        rating = p.get("rating") or 0.0
        reviews = p.get("userRatingCount") or 0
        return (-rating, -reviews)

    enriched_google.sort(key=_google_key)

    # Phase 2 \u2014 append Yelp-only entries as secondary options
    yelp_only: list[dict[str, Any]] = []
    for yelp in yelp_results:
        y_phone = yelp.get("phone", "")
        y_norm = re.sub(r"[^\d]", "", y_phone) if y_phone else ""
        if y_norm and y_norm not in matched_phones:
            yelp_only.append(
                {
                    "name": yelp.get("name", ""),
                    "displayName": yelp.get("name", ""),
                    "formattedAddress": yelp.get("address", ""),
                    "phone": yelp.get("phone", ""),
                    "rating": yelp.get("rating"),
                    "userRatingCount": yelp.get("review_count", 0),
                    "yelp_rating": yelp.get("rating"),
                    "yelp_review_count": yelp.get("review_count", 0),
                    "yelp_url": yelp.get("yelp_url", ""),
                    "source": "yelp",
                }
            )
            matched_phones.add(y_norm)

    def _yelp_only_key(p: dict[str, Any]) -> tuple:
        rating = p.get("rating") or 0.0
        reviews = p.get("userRatingCount") or p.get("yelp_review_count") or 0
        return (-rating, -reviews)

    yelp_only.sort(key=_yelp_only_key)

    return enriched_google + yelp_only