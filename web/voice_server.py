"""Voice demo server.

Wraps the Service Provider Agent behind a tiny HTTP shim so a browser
frontend can drive the agent with Web Speech API STT/TTS.

The agent contract matches the home service flow — this server builds the
same `[{"speaker": ..., "text": ...}]` shape and hands it to the same graph
used by the pipeline. The only extra it exposes is `clarification_question`,
which already lives in graph state but may be dropped by the formatter.

Run (from the repo root):
    python web/voice_server.py
Then open http://localhost:8000/ in Chrome or Edge.

Adapted from the reference project (CBRE → Home Service Agent branding).
All endpoints are stubs — real API integrations come in later tasks.
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

from agent import _flatten, get_graph
from agent.config import CATEGORY_SUBCATEGORIES

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
INDEX_FILE = STATIC_DIR / "index.html"

PORT = 8000

# Voice-mode gating thresholds.
MAX_FOLLOWUP_TURNS = 4      # after this many follow-ups, give up and escalate to HITL


def classify_voice(turns: list[dict], caller_phone: str | None) -> dict:
    """Voice-side wrapper. Stub — runs the full graph and returns prediction.

    TODO: implement multi-turn aware logic that returns a clarifying question
    while extraction is incomplete instead of closing the call early.
    """
    graph = get_graph()

    result = graph.invoke({
        "transcript": _flatten(turns),
        "turns_split": turns,
        "caller_phone": caller_phone,
        "caller_profile": None,
        "eval_mode": True,
        "entity": None,
        "follow_ups": [],
        "extraction_complete": False,
        "needs_clarification": False,
        "gate_clarification_verdict": False,
        "clarification_question": None,
        "clarification_reasons": (),
        "enriched_query": "",
        "classification": None,
        "api_selection": None,
        "provider_matches": [],
        "merged_providers": [],
        "trap_check_result": None,
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

    return {
        "category": category,
        "subcategory": subcategory,
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
        "provider_matches": result.get("provider_matches", []),
        "merged_providers": result.get("merged_providers", []),
        "trainer_log": result.get("trainer_log"),
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

    def do_OPTIONS(self):  # CORS preflight
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

    print(f"Warming graph…")
    get_graph()  # eager compile so first request isn't slow

    print(f"Home Service Agent — voice demo server")
    print(f"ElevenLabs and API integrations not yet connected (stub mode)")
    print(f"Serving static files from {STATIC_DIR}")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Listening on http://localhost:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()