"""Tests for agent/intake.py — deterministic entity-field clarification gate.

These tests prove that the ``_missing_entities()`` function (a pure
deterministic check with no LLM involvement) correctly routes vague inputs
back to clarification and lets detailed inputs pass through.
"""
from __future__ import annotations

import pytest

from agent.intake import (
    _missing_entities,
    _sparse_caller,
    decide_clarification,
)


class TestMissingEntities:
    """``_missing_entities()`` — the core deterministic field check."""

    def test_vague_input_triggers_clarification(self):
        """Empty entity → all required fields missing → needs clarification."""
        entity = {}
        missing = _missing_entities(entity)
        assert missing == ["problem_description", "address", "city"]

    def test_detailed_input_passes_clarification(self):
        """All required fields present → no missing entities → clear."""
        entity = {
            "problem_description": "water pipe leaking",
            "address": "123 Anacapa Street",
            "city": "Santa Barbara",
        }
        missing = _missing_entities(entity)
        assert missing == []

    def test_missing_problem_description_triggers_clarification(self):
        """Address + city present but problem_description missing → clarification."""
        entity = {
            "problem_description": None,
            "address": "123 State Street",
            "city": "Santa Barbara",
        }
        missing = _missing_entities(entity)
        assert missing == ["problem_description"]

    def test_missing_address_triggers_clarification(self):
        """Problem present but no address AND no city → clarification."""
        entity = {
            "problem_description": "sink leaking",
            "address": None,
            "city": None,
        }
        missing = _missing_entities(entity)
        assert missing == ["address"]

    def test_address_alone_satisfies_location_check(self):
        """Address present but city missing → location is OK."""
        entity = {
            "problem_description": "water heater broken",
            "address": "500 Chapala Street",
            "city": None,
        }
        missing = _missing_entities(entity)
        # Only problem_description is present → no missing field
        assert missing == []

    def test_city_alone_satisfies_location_check(self):
        """City present but address missing → location is OK."""
        entity = {
            "problem_description": "roof leak",
            "address": None,
            "city": "Santa Barbara",
        }
        missing = _missing_entities(entity)
        # city is set, so "address" is not appended
        assert missing == []


class TestSparseCaller:
    """``_sparse_caller()`` — fires on very short first utterances."""

    def test_sparse_caller_still_fires(self):
        """Single-word utterance triggers sparse_caller even with filled entity."""
        turns = [{"speaker": "caller", "text": "Help"}]
        entity = {
            "problem_description": "some issue",
            "address": "1 Main St",
            "city": "Santa Barbara",
        }

        decision = decide_clarification(turns, entity=entity)

        assert decision.needs_clarification is True
        # The sparse-caller reason should be present
        assert any(r.startswith("sparse_caller:") for r in decision.reasons)

    def test_multi_turn_sparse_still_fires(self):
        """Short turns across multiple exchanges still trigger sparse_caller."""
        turns = [
            {"speaker": "caller", "text": "Help"},
            {"speaker": "bot", "text": "What's the issue?"},
            {"speaker": "caller", "text": "Water leak"},
        ]
        decision = decide_clarification(turns, entity=None)
        assert decision.needs_clarification is True

    def test_detailed_conversation_passes_sparse_check(self):
        """Long first utterance → no sparse_caller trigger."""
        turns = [{
            "speaker": "caller",
            "text": (
                "There is a water pipe leaking in my kitchen "
                "at 123 Anacapa Street in Santa Barbara"
            ),
        }]
        entity = {
            "problem_description": "water pipe leaking",
            "address": "123 Anacapa Street",
            "city": "Santa Barbara",
        }
        decision = decide_clarification(turns, entity=entity)
        assert decision.needs_clarification is False


class TestDecideClarification:
    """Integration of all heuristics in ``decide_clarification()``."""

    def test_full_input_passes_all_checks(self):
        """All entity fields present + non-sparse turns → pass through."""
        turns = [{
            "speaker": "caller",
            "text": (
                "The water heater is broken. It's not heating water. "
                "Located at 500 Chapala Street, Santa Barbara."
            ),
        }]
        entity = {
            "problem_description": "water heater broken",
            "address": "500 Chapala Street",
            "city": "Santa Barbara",
        }
        decision = decide_clarification(turns, entity=entity)
        assert decision.needs_clarification is False
        assert decision.reasons == ()

    def test_only_sparse_triggers(self):
        """Sparse utterance triggers even when all entity fields are present."""
        turns = [{"speaker": "caller", "text": "Help"}]
        entity = {
            "problem_description": "water heater broken",
            "address": "500 Chapala Street",
            "city": "Santa Barbara",
        }
        decision = decide_clarification(turns, entity=entity)
        assert decision.needs_clarification is True
        assert any(r.startswith("sparse_caller:") for r in decision.reasons)
        assert not any(r.startswith("missing_fields:") for r in decision.reasons)

    def test_only_missing_fields_triggers(self):
        """Missing entity fields trigger even when turns are detailed."""
        turns = [{
            "speaker": "caller",
            "text": (
                "I'm at 123 State Street in Santa Barbara and "
                "would like to report a problem"
            ),
        }]
        entity = {
            "problem_description": None,
            "address": "123 State Street",
            "city": "Santa Barbara",
        }
        decision = decide_clarification(turns, entity=entity)
        assert decision.needs_clarification is True
        assert any(r.startswith("missing_fields:") for r in decision.reasons)

    def test_both_heuristics_fire(self):
        """Both sparse caller AND missing fields → both reasons recorded."""
        turns = [{"speaker": "caller", "text": "I have an issue"}]
        decision = decide_clarification(turns, entity={})
        assert decision.needs_clarification is True
        reasons = decision.reasons
        assert any(r.startswith("sparse_caller:") for r in reasons)
        assert any(r.startswith("missing_fields:") for r in reasons)