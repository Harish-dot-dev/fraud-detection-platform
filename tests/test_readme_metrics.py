"""Tests for the README results table.

This table is the project's promise that no published number was invented, so
the rule it enforces is worth testing: a metric measured on the synthetic
fixture, or produced by a scripted generator, is **withheld** rather than
footnoted. A bold caveat next to "100.0%" is still read as 100%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from readme_metrics import (  # noqa: E402
    END_MARKER,
    START_MARKER,
    build_rows,
    render_table,
    update_readme,
)


@pytest.fixture
def reports(tmp_path, monkeypatch):
    """Point the generator at a temporary reports directory."""
    import readme_metrics

    monkeypatch.setattr(readme_metrics, "REPORTS", tmp_path)
    return tmp_path


def _write(reports: Path, name: str, payload: dict) -> None:
    (reports / name).write_text(json.dumps(payload))


def _value(rows, metric: str) -> str:
    return next(value for name, value, _ in rows if name == metric)


def test_missing_reports_produce_no_numbers(reports) -> None:
    rows = build_rows()

    assert rows
    assert all(value.startswith("*not yet measured") for _, value, _ in rows)


def test_fixture_metrics_are_withheld(reports) -> None:
    """The rule that matters most.

    A PR-AUC from synthetic data says nothing about real payments, and a
    results table is exactly where somebody will read it as though it did.
    """
    _write(
        reports,
        "metrics.json",
        {"pr_auc": 0.94, "precision_at_block": 0.93, "dataset": "synthetic fixture"},
    )

    rows = build_rows()

    assert "0.94" not in _value(rows, "PR-AUC (test window)")
    assert "real dataset" in _value(rows, "PR-AUC (test window)")


def test_real_dataset_metrics_are_published(reports) -> None:
    _write(
        reports,
        "metrics.json",
        {
            "pr_auc": 0.8123,
            "precision_at_block": 0.91,
            "recall_at_block": 0.64,
            "recall_including_review": 0.77,
            "false_positive_rate": 0.0012,
            "fraud_value_blocked": 40000.0,
            "fraud_value_reviewed": 5000.0,
            "fraud_value_missed": 9000.0,
            "fraud_value_caught_share": 0.83,
            "dataset": "IEEE-CIS (real)",
        },
    )

    rows = build_rows()

    assert _value(rows, "PR-AUC (test window)") == "0.8123"
    assert "45,000 caught" in _value(rows, "Fraud value caught vs missed (test window)")


def test_thresholds_follow_the_dataset_they_were_tuned_on(reports) -> None:
    """An operating point chosen against synthetic costs is not the real one."""
    _write(reports, "metrics.json", {"pr_auc": 0.9, "dataset": "synthetic fixture"})
    _write(
        reports,
        "thresholds.json",
        {"review_threshold": 0.05, "block_threshold": 0.7, "cost_reduction": 0.9},
    )

    assert "0.05" not in _value(build_rows(), "Chosen thresholds (cost-tuned)")


def test_a_scripted_generator_is_not_an_llm_measurement(reports) -> None:
    _write(
        reports,
        "llm_eval.json",
        {
            "generator": "scripted",
            "embedder": "hashing-stub",
            "factual_accuracy": 1.0,
            "schema_validity": 0.84,
            "retrieval_precision": 0.956,
            "latency_p50_ms": 0.0,
        },
    )

    rows = build_rows()

    assert "100" not in _value(rows, "LLM factual accuracy")
    assert "needs a local LLM" in _value(rows, "LLM factual accuracy")
    assert "needs the real embedder" in _value(rows, "Retrieval quality (same confirmed outcome)")


def test_a_real_generator_is_published(reports) -> None:
    _write(
        reports,
        "llm_eval.json",
        {
            "generator": "llama3.2:3b",
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "factual_accuracy": 0.96,
            "schema_validity": 0.98,
            "retrieval_precision": 0.71,
            "latency_p50_ms": 1840.0,
        },
    )

    rows = build_rows()

    assert _value(rows, "LLM factual accuracy") == "96.0%"
    assert _value(rows, "LLM latency per summary (p50)") == "1840 ms"
    assert _value(rows, "Retrieval quality (same confirmed outcome)") == "71.0%"


def test_latency_carries_the_conditions_it_was_measured_under(reports) -> None:
    """A latency figure without its concurrency is not a measurement."""
    _write(
        reports,
        "latency.json",
        {
            "label": "docker compose",
            "peak_throughput_rps": 71.0,
            "service_latency": {"p50_ms": 12.7, "p95_ms": 20.3, "p99_ms": 21.4, "concurrency": 1},
        },
    )

    value = _value(build_rows(), "/score latency p50 / p95 / p99")

    assert "12.7 / 20.3 / 21.4 ms" in value
    assert "concurrency 1" in value


def test_the_table_renders_as_markdown(reports) -> None:
    table = render_table(build_rows())

    assert table.startswith("| Metric | Value | Produced by |")
    assert table.count("\n") >= 10


def test_the_readme_block_is_replaced_in_place(tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n\n{START_MARKER}\nold table\n{END_MARKER}\n\nafter\n")

    assert update_readme(readme, "| new |") is True

    text = readme.read_text()
    assert "old table" not in text
    assert "| new |" in text
    assert text.startswith("before")
    assert text.endswith("after\n")


def test_a_readme_without_markers_is_an_error(tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers here")

    with pytest.raises(ValueError, match="METRICS:START"):
        update_readme(readme, "| new |")
