"""Tests for the load-test harness.

A measurement tool that is wrong is worse than no measurement, so the parts
that turn timings into a published number are tested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from load_test import LatencyReport, SweepReport, load_payloads  # noqa: E402


def _report(concurrency: int, p50: float, rps: float) -> LatencyReport:
    return LatencyReport(
        label="test",
        api_url="http://localhost:8000",
        requests=100,
        concurrency=concurrency,
        successes=100,
        failures=0,
        p50_ms=p50,
        p95_ms=p50 * 1.3,
        p99_ms=p50 * 1.8,
        mean_ms=p50,
        max_ms=p50 * 2,
        throughput_rps=rps,
        duration_seconds=1.0,
        decision_mix={"allow": 97, "review": 2, "block": 1},
    )


def test_payloads_are_built_from_the_fixture() -> None:
    payloads = load_payloads(5)

    assert len(payloads) == 5
    assert json.loads(payloads[0])["transaction_id"]


def test_payloads_cycle_when_more_are_requested_than_exist() -> None:
    payloads = load_payloads(2500)

    assert len(payloads) == 2500


def test_service_latency_is_the_least_contended_run() -> None:
    """At concurrency 1 you measure the service; above saturation, the queue.

    Quoting a saturated p50 as "our latency" is the most common way a load
    test result misleads, so the report names the unqueued run explicitly.
    """
    sweep = SweepReport(
        label="test",
        api_url="http://localhost:8000",
        note="",
        runs=[_report(8, 105.0, 75), _report(1, 11.4, 84), _report(4, 49.6, 78)],
    )

    assert sweep.service_latency.concurrency == 1
    assert sweep.service_latency.p50_ms == 11.4
    assert sweep.peak_throughput_rps == 84


def test_the_sweep_is_saved_with_every_run(tmp_path) -> None:
    sweep = SweepReport(
        label="test",
        api_url="http://localhost:8000",
        note="synthetic",
        runs=[_report(1, 11.4, 84), _report(8, 105.0, 75)],
    )

    saved = json.loads(sweep.save(tmp_path / "latency.json").read_text())

    assert saved["service_latency"]["p50_ms"] == 11.4
    assert [run["concurrency"] for run in saved["runs"]] == [1, 8]
    assert saved["note"] == "synthetic"


def test_the_summary_reads_as_a_table() -> None:
    sweep = SweepReport(label="demo", api_url="http://x", note="", runs=[_report(1, 11.4, 84)])

    described = sweep.describe()

    assert "concurrency" in described
    assert "service latency (concurrency 1)" in described


def test_an_api_that_is_not_running_fails_loudly() -> None:
    from load_test import run_load_test

    with pytest.raises(RuntimeError, match="is the API running"):
        run_load_test("http://localhost:1", ["{}"], concurrency=1, timeout=0.5)
