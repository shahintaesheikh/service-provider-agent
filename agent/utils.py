"""
Shared utilities used by multiple nodes.

  - Transcript flattening (LLM input prep)
  - Profile lookup by phone
  - Word-boundary regex compiler (used by trap patterns + danger-word check)
  - Node-latency instrumentation (timed_node + print_timing_report)
  - Cache usage logging
"""
from __future__ import annotations

import logging
import os
import re
import time


# ── Transcript + profile helpers ─────────────────────────────────────────────


def _flatten(turns: list[dict]) -> str:
    """Format the turns list into the [SPEAKER] line per-line text the LLMs see."""
    return "\n".join(f"[{t['speaker'].upper()}] {t['text']}" for t in turns)


def _lookup_profile(phone: str | None) -> dict | None:
    """Stub: look up a caller profile by phone number.

    TODO: implement profile lookup (e.g., from a CRM or past session store).
    """
    return None


# ── Word-boundary keyword regex (used by trap patterns + danger-word check) ──


def _compile_kw_regex(keywords):
    """Word-boundary, case-insensitive, longest-first regex over a keyword list."""
    parts = [re.escape(kw) for kw in sorted(keywords, key=len, reverse=True)]
    return re.compile(r"(?:\b|^)(?:" + "|".join(parts) + r")(?:\b|$)", re.IGNORECASE)


# ── Shared danger-word regex ─────────────────────────────────────────────────

_DANGER_WORDS_RE = _compile_kw_regex([
    "fire", "smoke", "gas", "flood", "electrical burning",
    "sparking", "hot panel", "carbon monoxide", "natural gas",
    "propane", "evacuate", "emergency", "collapse", "ceiling fell",
    "structural", "gas leak",
])


# ── OpenAI prompt-cache telemetry ────────────────────────────────────────────


def _log_cache_usage(node_name: str, response) -> None:
    """Print a one-line cache-hit summary if the response surfaces token usage.

    Silent no-op when the SDK didn't expose token_usage (e.g. on errors).
    """
    try:
        meta = getattr(response, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        prompt = usage.get("prompt_tokens", 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
        if prompt:
            pct = (cached / prompt * 100) if cached else 0
            print(f"[cache] {node_name}: cached={cached}/{prompt} ({pct:.0f}%)")
    except Exception:  # noqa: BLE001 — telemetry never fails the call
        pass


# ── Logging setup ────────────────────────────────────────────────────────────


logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("spa")

# Users can enable debug logging via:
#   export SPA_LOG_LEVEL=DEBUG
# or by setting LOG_LEVEL=DEBUG in .env
if os.environ.get("SPA_LOG_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)


def log_node(node_name: str):
    """Decorator that logs node start/end with elapsed time."""
    def decorator(fn):
        def wrapper(state):
            logger.debug("[%s] started", node_name)
            t0 = time.perf_counter()
            try:
                result = fn(state)
                elapsed = time.perf_counter() - t0
                logger.debug("[%s] done in %.3fs", node_name, elapsed)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.error("[%s] FAILED after %.3fs: %s", node_name, elapsed, e)
                raise
        return wrapper
    return decorator


# ── Per-node latency instrumentation ─────────────────────────────────────────


_node_timings: dict[str, list[float]] = {}


def timed_node(node_name: str, fn):
    """Wrap a node function so the per-call elapsed time is recorded under
    `_node_timings[node_name]`. Used in graph building so the timing report
    covers every node uniformly.
    """
    def wrapper(state):
        start = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - start
        _node_timings.setdefault(node_name, []).append(elapsed)
        return result
    wrapper.__name__ = fn.__name__ if hasattr(fn, "__name__") else node_name
    return wrapper


def print_timing_report():
    """Print a compact per-node + total latency report."""
    if not _node_timings:
        return
    print(f"\n{'='*60}")
    print("  NODE LATENCY REPORT")
    print(f"{'='*60}")
    for name, times in sorted(_node_timings.items()):
        avg = sum(times) / len(times)
        total = sum(times)
        print(f"  {name:<30} avg={avg:.3f}s  total={total:.2f}s  calls={len(times)}")
    total_all = sum(sum(t) for t in _node_timings.values())
    print(f"  {'TOTAL':<30} {total_all:.2f}s")
    print(f"{'='*60}")