"""Voice demo server.

Wraps the Service Provider Agent behind a tiny HTTP shim so a browser
frontend can drive the agent with ElevenLabs STT/TTS or typed input.

The agent contract matches the home service flow — this server builds the
same `[{"speaker": ..., "text": ...}]` shape and hands it to the same graph
used by the pipeline. The only extra it exposes is `clarification_question`,
which already lives in graph state but may be dropped by the formatter.

Run (from the repo root):
    python web/voice_server.py
Then open http://localhost:8000/ in Chrome or Edge.

Adapted from the reference project (CBRE → Home Service Agent branding).
ElevenLabs voice features are optional — text input always works as fallback.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# This script lives at web/voice_server.py; the `agent` package lives one
# directory up at the repo root. When invoked as `python web/voice_server.py`,
# Python only puts `web/` on sys.path, so the agent import would fail.
# Prepend the repo root so the rest of the imports resolve regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from elevenlabs import ElevenLabs
from langchain_openai import ChatOpenAI

from agent import _flatten, get_graph
from agent.config import CATEGORY_SUBCATEGORIES
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

# ── ElevenLabs (Scribe v2 realtime STT + Flash v2.5 streaming TTS) ───────────
# Optional: server runs without it, but /api/scribe-token + /api/tts/stream
# return 503 and the UI falls back to a "voice unavailable" message.
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
_EL: ElevenLabs | None = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# Rachel — warm, conversational. Good for empathetic homeowner conversations.
TTS_VOICE_ID = "21m00Tcm4TzfDVHbD9J"
# Flash v2.5 — ~75ms TTFB, multilingual, the lowest-latency realtime model.
TTS_MODEL_ID = "eleven_flash_v2_5"
# MP3 streams cleanly into a browser <audio> element; cheap to proxy.
TTS_OUTPUT_FORMAT = "mp3_44100_128"


_KEYTERM_MAX_LEN = 20  # ElevenLabs Scribe per-keyterm hard cap


def _shorten_keyterm(name: str) -> str | None:
    """Reduce a term to a ≤20-char anchor that Scribe will accept.

    Drops everything past the longest leading word-prefix that still fits in 20
    chars. "Water Heater Repair" → "Water Heater Repair" (18). If even the
    first word exceeds 20, truncates it. Returns None for empty input.
    """
    name = (name or "").strip()
    if not name:
        return None
    if len(name) <= _KEYTERM_MAX_LEN:
        return name
    words = name.split()
    out = words[0][:_KEYTERM_MAX_LEN]
    for w in words[1:]:
        candidate = f"{out} {w}"
        if len(candidate) > _KEYTERM_MAX_LEN:
            break
        out = candidate
    return out


def _build_keyterms() -> list[str]:
    """Bias Scribe toward domain proper nouns the model would otherwise mishear.

    Source: shortened service category/subcategory phrases + a small set of
    high-value domain words. Each term is ≤20 chars per the Scribe API limit,
    deduped case-insensitively, capped at 100 total.
    """
    raw: list[str] = []
    for subs in CATEGORY_SUBCATEGORIES.values():
        raw.extend(s.replace("_", " ") for s in subs)
    raw.extend([
        "plumbing", "electrical", "HVAC", "leak", "emergency", "gas",
        "water heater", "clogged drain", "broken pipe", "furnace",
        "air conditioner", "roof leak", "pest control", "lockout",
        "Santa Barbara", "Goleta", "home service",
    ])
    seen, out = set(), []
    for t in raw:
        t = t.strip()
        if not t or len(t) > _KEYTERM_MAX_LEN:
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
        if len(out) >= 100:
            break
    return out


_KEYTERMS = _build_keyterms()


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
    """Normalize a provider dict to the shape the frontend expects."""
    return {
        "name": provider.get("name") or provider.get("display_name", "—"),
        "phone": provider.get("phone") or provider.get("formatted_phone_number", "—"),
        "address": provider.get("address") or provider.get("formatted_address", "—"),
        "rating": provider.get("rating"),
        "open_now": provider.get("open_now"),
        "display_name": provider.get("display_name"),
        "formatted_phone_number": provider.get("formatted_phone_number"),
        "formatted_address": provider.get("formatted_address"),
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

    def _stream_tts(self, text: str) -> None:
        """Proxy ElevenLabs Flash v2.5 streaming TTS to the browser as MP3.

        Each yielded chunk is forwarded immediately so playback can begin
        before the full utterance is rendered (~75ms TTFB with Flash).
        """
        try:
            audio_iter = _EL.text_to_speech.stream(
                voice_id=TTS_VOICE_ID,
                text=text,
                model_id=TTS_MODEL_ID,
                output_format=TTS_OUTPUT_FORMAT,
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json(502, {"error": f"tts upstream error: {exc}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            for chunk in audio_iter:
                if chunk:
                    self.wfile.write(chunk)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client navigated away or hit barge-in mid-stream. Expected.
            pass

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

        if path == "/api/keyterms":
            self._send_json(200, {"keyterms": _KEYTERMS})
            return

        if path == "/api/scribe-token":
            if _EL is None:
                self._send_json(503, {"error": "ELEVENLABS_API_KEY not configured"})
                return
            try:
                token = _EL.tokens.single_use.create(token_type="realtime_scribe")
            except Exception as exc:  # noqa: BLE001
                self._send_json(502, {"error": f"token mint failed: {exc}"})
                return
            self._send_json(200, {"token": token.token})
            return

        # GET form for TTS — kept for quick curl testing. POST is preferred
        # for production because long text overflows URL length limits.
        if path == "/api/tts/stream":
            if _EL is None:
                self._send_json(503, {"error": "ELEVENLABS_API_KEY not configured"})
                return
            qs = parse_qs(url.query)
            text = (qs.get("text") or [""])[0].strip()
            if not text:
                self._send_json(400, {"error": "missing text"})
                return
            self._stream_tts(text)
            return

        self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)

        if url.path == "/api/tts/stream":
            if _EL is None:
                self._send_json(503, {"error": "ELEVENLABS_API_KEY not configured"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                self._send_json(400, {"error": "missing or oversized body"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError as e:
                self._send_json(400, {"error": f"bad json: {e}"})
                return
            text = (payload.get("text") or "").strip()
            if not text:
                self._send_json(400, {"error": "text required"})
                return
            self._stream_tts(text)
            return

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

    if _EL is None:
        print("⚠  ELEVENLABS_API_KEY not set — /api/scribe-token + /api/tts/stream "
              "will return 503; UI will fall back to typed-only mode.")
    else:
        print(f"ElevenLabs voice: Scribe v2 realtime STT + {TTS_MODEL_ID} TTS "
              f"(voice {TTS_VOICE_ID}, {len(_KEYTERMS)} keyterms)")

    print(f"Home Service Agent — voice demo listening on http://localhost:{PORT}/")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()