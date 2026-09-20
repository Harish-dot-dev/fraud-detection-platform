"""Tests for the decision audit record and its sinks."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from serving.decisions import (
    BLOCK,
    TRIGGER_RULE,
    Decision,
    JsonlDecisionSink,
    ListDecisionSink,
)


def _decision(**overrides) -> Decision:
    defaults = {
        "transaction_id": 123,
        "card_token": "a" * 32,
        "decided_at": datetime(2023, 6, 1, 12, 0, tzinfo=UTC),
        "decision": BLOCK,
        "score": 0.91,
        "triggered_by": TRIGGER_RULE,
        "reason": "confirmed compromised",
        "features": {"amount": 100.0},
        "latency_ms": 12.5,
    }
    return Decision(**{**defaults, **overrides})


def test_the_record_carries_everything_needed_to_explain_a_decision() -> None:
    """Requirement: reproduce any decision months later, for a complaints team."""
    payload = _decision().to_dict()

    for field in (
        "transaction_id",
        "decided_at",
        "decision",
        "score",
        "model_version",
        "triggered_by",
        "reason",
        "features",
        "latency_ms",
    ):
        assert field in payload


def test_the_record_holds_a_token_not_a_card() -> None:
    payload = _decision().to_dict()

    assert payload["card_token"] == "a" * 32
    assert "card1" not in payload
    assert "p_emaildomain" not in payload


def test_the_record_serialises_for_the_topic() -> None:
    restored = json.loads(_decision().to_json())

    assert restored["decision"] == BLOCK
    assert restored["decided_at"].startswith("2023-06-01T12:00")


def test_the_list_sink_collects_decisions() -> None:
    sink = ListDecisionSink()

    sink.record(_decision())
    sink.record(_decision(transaction_id=124))

    assert [d.transaction_id for d in sink.decisions] == [123, 124]


def test_the_jsonl_sink_appends_one_line_per_decision(tmp_path) -> None:
    """The fallback when Kafka is not running: a laptop demo still has an audit trail."""
    sink = JsonlDecisionSink(tmp_path / "decisions.jsonl")

    sink.record(_decision())
    sink.record(_decision(transaction_id=124))
    sink.close()

    lines = (tmp_path / "decisions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["transaction_id"] == 124


def test_the_jsonl_sink_creates_its_directory(tmp_path) -> None:
    sink = JsonlDecisionSink(tmp_path / "nested" / "deeper" / "decisions.jsonl")

    sink.record(_decision())

    assert sink.path.exists()
