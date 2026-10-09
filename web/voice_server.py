"""Text-only chat server for the Home Service Agent.

Wraps the Service Provider Agent behind a tiny HTTP shim so a browser
frontend can drive the agent with typed text input.

The agent contract matches the home service flow — this server builds the
same `[{"speaker": ..., "text": ...}]` shape and hands it to the same graph
used by the pipeline. The only extra it exposes is `clarification_question`,
which already lives in graph state but may be dropped by the formatter.

Run (from the repo root):
    python web/voice_server.py
Then open http://localhost:8000/ in Chrome or Edge.

Adapted from the reference project (CBRE → Home Service Agent branding).
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# This script lives at web/voice_server.py; the `agent` package lives one
# directory up at the repo root. When invoked as `python web/voice_server.py`,
# Python only puts `web/` on sys.path, so the agent import would fail.
# Prepend the repo root so the rest of the imports resolve regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langchain_openai import ChatOpenAI

from agent import _flatten, get_graph
from agent.intake import make_extract_and_check
from agent.utils import _lookup_profile

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
INDEX_FILE = STATIC_DIR / "index.html"

PORT = 8000

# Voice-mode gating thresholds.
# We deliberately do NOT keep a "minimum caller words" floor — even a 2-word
# input like "water's leaking" should run through the extractor so the LLM can
# generate a contextual followup question instead of falling back to a generic
# prompt. The only hard guard is below: zero caller words → ask "what's happening?".
MAX_FOLLOWUP_TURNS = 4      # after this many follow-ups, give up and escalate to HITL

# Standalone extractor for the voice-mode pre-check. Same LLM the graph uses.
# Lazily initialized so the server starts even without an OPENAI_API_KEY.
_VOICE_EXTRACTOR = None
_VOICE_EXTRACTOR_INIT_DONE = False


def _get_voice_extractor():
    global _VOICE_EXTRACTOR, _VOICE_EXTRACTOR_INIT_DONE
    if _VOICE_EXTRACTOR_INIT_DONE:
        return _VOICE_EXTRACTOR
    _VOICE_EXTRACTOR_INIT_DONE = True
    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=3, request_timeout=20)
        _VOICE_EXTRACTOR = make_extract_and_check(llm)
    except Exception:
        _VOICE_EXTRACTOR = None
    return _VOICE_EXTRACTOR


def _caller_word_count(turns: list[dict]) -> int:
    return sum(len(t.get("text", "").split()) for t in turns if t.get("speaker") == "caller")


def _agent_turn_count(turns: list[dict]) -> int:
    """How many follow-up questions has the agent already asked?"""
    return sum(1 for t in turns if t.get("speaker") == "agent") - 1  # minus the initial greeting


def _build_clarification_response(question: str, profile: dict | None) -> dict:
    """Voice-mode payload: ask the question, don't surface a (partial) classification."""
    return {
        "category": None,
        "subcategory": None,
        "urgency": None,
        "urgency_level": None,
        "confidence": None,
        "needs_human_review": False,
        "needs_clarification": True,
        "clarification_question": question,
        "call_summary": "",
        "address": None,
        "city": None,
        "provider_matches": [],
        "merged_providers": [],
        "lead_quality": None,
        "trainer_log": None,
        "caller_profile": profile,
    }


def _format_provider(provider: dict) -> dict:
    """Normalize a provider dict to the shape the frontend expects.

    Google Places API (New) returns displayName as a ``{text, languageCode}``
    dict, but ``make_call_google_places()`` already unwraps it to a plain
    ``name`` string during normalization.  We still prefer ``name`` first,
    then ``displayName``, then a fallback dash.
    """
    return {
        "name": provider.get("name") or provider.get("displayName", "—"),
        "phone": provider.get("phone") or provider.get("nationalPhoneNumber", "—"),
        "address": provider.get("address") or provider.get("formattedAddress", "—"),
        "rating": provider.get("rating"),
        "open_now": provider.get("open_now"),
        "displayName": provider.get("displayName"),
        "formattedAddress": provider.get("formattedAddress"),
    }


def _to_urgency_label(score: float | None) -> str | None:
    """Map a numeric urgency score to a label for the UI pill."""
    if score is None:
        return None
    if score >= 0.8:
        return "emergency"
    if score >= 0.5:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def classify_voice(turns: list[dict], caller_phone: str | None) -> dict:
    """Voice-side wrapper. Multi-turn aware: returns a clarifying question
    while extraction is incomplete instead of closing the call early.
    Falls back to full classification once enough info accumulates, or
    escalates to HITL after MAX_FOLLOWUP_TURNS questions if still incomplete.
    """
    profile = _lookup_profile(caller_phone)
    transcript = _flatten(turns)

    # ── Voice-mode pre-check: are we ready to classify? ────────────────────
    caller_words = _caller_word_count(turns)
    followups_asked = _agent_turn_count(turns)

    if followups_asked < MAX_FOLLOWUP_TURNS:
        # 1) Zero-caller-words guard: nothing to extract from. Cheap, deterministic.
        if caller_words == 0:
            return _build_clarification_response(
                "I didn't catch that — what's happening?",
                profile,
            )

        # 2) Soft check: run the extractor so the LLM produces a contextual
        # question even for short input.
        pre_state = {
            "transcript": transcript,
            "caller_profile": profile,
            "extraction_complete": False,
            "needs_clarification": False,
        }
        extractor = _get_voice_extractor()
        if extractor is not None:
            try:
                extract_out = extractor(pre_state)
            except Exception:
                extract_out = {}
        else:
            extract_out = {}

        # Fire the extractor's follow-up if it produced one AND either
        # extraction is incomplete or the extractor flagged needs_clarification.
        follow_ups = extract_out.get("follow_ups") or []
        needs_clar = extract_out.get("needs_clarification", False)
        extraction_complete = extract_out.get("extraction_complete", True)
        if follow_ups and (not extraction_complete or needs_clar):
            return _build_clarification_response(follow_ups[0], profile)

        # 3) Location floor: require at least an address or city hint before
        # dispatching. Voice mode needs a real location to find providers.
        ent = extract_out.get("entity") or {}
        has_extracted_location = bool(
            (ent.get("address") or "").strip()
            or (ent.get("city") or "").strip()
            or (ent.get("neighborhood") or "").strip()
        )
        has_profile_location = bool(
            profile and (
                profile.get("address")
                or profile.get("city")
            )
        )
        if not has_extracted_location and not has_profile_location:
            return _build_clarification_response(
                "What's your address or neighborhood? I need a location "
                "to find the right service provider.",
                profile,
            )

    # ── Otherwise: run the full graph ─────────────────────────────────────
    graph = get_graph()

    result = graph.invoke({
        "transcript": transcript,
        "turns_split": turns,
        "caller_phone": caller_phone,
        "caller_profile": profile,
        "eval_mode": True,
        "entity": None,
        "follow_ups": [],
        "extraction_complete": False,
        "needs_clarification": False,
        "gate_clarification_verdict": False,
        "clarification_question": None,
        "clarification_reasons": (),
        "enriched_query": "",
        "retrieved_docs": [],
        "retrieved_context": "",
        "retrieval_attempts": 0,
        "classification": None,
        "consensus_agreement": None,
        "api_selection": None,
        "provider_matches": [],
        "merged_providers": [],
        "trap_check_result": None,
        "risk_score": None,
        "risk_details": None,
        "urgency_score": None,
        "urgency_signals": None,
        "address": None,
        "city": None,
        "needs_human_review": False,
        "lead_quality": None,
        "routing_decision": None,
        "human_override": None,
        "trainer_log": None,
    })

    classification = result.get("classification") or {}
    category = (classification.get("category") or "UNKNOWN").upper()
    subcategory = (classification.get("subcategory") or "").lower()

    urgency_score = result.get("urgency_score") or classification.get("urgency_score")

    # Format provider matches for the frontend
    merged = result.get("merged_providers") or []
    provider_matches = result.get("provider_matches") or []
    formatted_merged = [_format_provider(p) for p in merged]
    formatted_matches = [_format_provider(p) for p in provider_matches]

    lead_quality = result.get("lead_quality")
    # lead_quality may be a dict with a "score" field or a string label
    if isinstance(lead_quality, dict):
        lead_quality_label = lead_quality.get("label") or lead_quality.get("score") or str(lead_quality)
    elif lead_quality is not None:
        lead_quality_label = str(lead_quality)
    else:
        lead_quality_label = None

    # Check if the search radius was expanded (empty results → wider retry)
    search_expanded = result.get("search_expanded", False)

    return {
        "category": category,
        "subcategory": subcategory,
        "urgency": urgency_score,
        "urgency_level": _to_urgency_label(urgency_score),
        "confidence": classification.get("confidence"),
        "needs_human_review": result.get("needs_human_review", False),
        "needs_clarification": result.get("needs_clarification", False),
        "clarification_question": (
            result.get("clarification_question")
            or classification.get("clarification_question")
        ),
        "call_summary": result.get("call_summary", ""),
        "address": result.get("address") or classification.get("address"),
        "city": result.get("city") or classification.get("city"),
        "provider_matches": formatted_matches,
        "merged_providers": formatted_merged,
        "search_expanded": search_expanded,
        "lead_quality": lead_quality_label,
        "trainer_log": result.get("trainer_log"),
        "caller_profile": profile,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        print(f"[{self.address_string()}] {fmt % args}")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):  # CORS preflight (browser-safe even on same origin)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html"):
            if not INDEX_FILE.exists():
                self._send_json(500, {"error": "index.html missing — build the frontend first"})
                return
            self._send_file(INDEX_FILE, "text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = STATIC_DIR / rel
            if not target.is_file() or STATIC_DIR not in target.resolve().parents:
                self.send_error(404)
                return
            ext = target.suffix.lower()
            ctype = {
                ".css": "text/css",
                ".js": "application/javascript",
                ".html": "text/html; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
            }.get(ext, "application/octet-stream")
            self._send_file(target, ctype)
            return

        if path == "/api/health":
            self._send_json(200, {"status": "ok", "agent": "home-service-agent"})
            return

        self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)

        if url.path != "/api/classify":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            self._send_json(400, {"error": "missing body"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"bad json: {e}"})
            return

        turns = payload.get("turns") or []
        caller_phone = payload.get("caller_phone") or None

        if not isinstance(turns, list) or not turns:
            self._send_json(400, {"error": "turns must be a non-empty list"})
            return
        for t in turns:
            if not isinstance(t, dict) or "speaker" not in t or "text" not in t:
                self._send_json(400, {"error": "each turn needs speaker and text"})
                return

        try:
            result = classify_voice(turns, caller_phone)
        except Exception as exc:  # noqa: BLE001 — surface to client
            import traceback
            traceback.print_exc()
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(200, result)


def main() -> None:
    STATIC_DIR.mkdir(exist_ok=True)
    if not INDEX_FILE.exists():
        raise SystemExit(f"missing {INDEX_FILE} — build the frontend first")

    print(f"Warming graph…")
    get_graph()  # eager compile so first request isn't slow

    print(f"Home Service Agent — chat server listening on http://localhost:{PORT}/")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()