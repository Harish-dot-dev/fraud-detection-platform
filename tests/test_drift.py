"""Tests for drift monitoring.

A monitoring tool that reports nonsense is worse than no monitoring, so these
check the numbers agree with each other as well as with the data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quality.drift import build_drift_report, split_by_time


@pytest.fixture
def reference() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "amount": rng.normal(100, 20, 500),
            "card_txn_count_10m": rng.poisson(2, 500).astype(float),
            "hour_of_day": rng.integers(0, 24, 500).astype(float),
        }
    )


def test_identical_windows_show_no_drift(reference: pd.DataFrame) -> None:
    result = build_drift_report(reference, reference.copy())

    assert result.columns_drifted == 0
    assert result.drifted_columns == []
    assert result.dataset_drift is False


def test_a_shifted_population_is_detected(reference: pd.DataFrame) -> None:
    """Payment amounts double: the model is being asked about new territory."""
    current = reference.copy()
    current["amount"] = current["amount"] * 2 + 500

    result = build_drift_report(reference, current)

    assert "amount" in result.drifted_columns
    assert result.columns_drifted >= 1


def test_the_count_and_the_share_describe_the_same_thing(reference: pd.DataFrame) -> None:
    """The regression test for a genuinely embarrassing bug.

    Evidently's DataDriftPreset emits two metrics - a summary and a per-column
    table - and an earlier version read the summary's position for both. The
    report said "0 of 18 columns drifted (66.7%)": internally contradictory,
    and precisely what a monitoring tool must never say.
    """
    current = reference.copy()
    current["amount"] = current["amount"] * 3
    current["card_txn_count_10m"] = current["card_txn_count_10m"] + 10

    result = build_drift_report(reference, current)

    assert result.columns_drifted == len(result.drifted_columns)
    assert result.drift_share == pytest.approx(
        result.columns_drifted / result.columns_checked, abs=1e-6
    )


def test_the_dataset_verdict_uses_the_documented_threshold(reference: pd.DataFrame) -> None:
    """Our threshold, not Evidently's opinion - so it is reviewable."""
    current = reference.copy()
    current["amount"] = current["amount"] * 3

    lenient = build_drift_report(reference, current, drift_share_threshold=0.9)
    strict = build_drift_report(reference, current, drift_share_threshold=0.1)

    assert lenient.dataset_drift is False
    assert strict.dataset_drift is True


def test_an_html_report_is_written_for_a_human(reference: pd.DataFrame, tmp_path) -> None:
    result = build_drift_report(reference, reference.copy(), html_path=tmp_path / "drift.html")

    assert (tmp_path / "drift.html").exists()
    assert result.html_report.endswith("drift.html")


def test_the_summary_is_saved_as_json(reference: pd.DataFrame, tmp_path) -> None:
    result = build_drift_report(reference, reference.copy())

    saved = json.loads(result.save(tmp_path / "drift.json").read_text())

    assert saved["columns_checked"] == 3
    assert "drifted_columns" in saved


def test_windows_are_split_by_time_not_at_random() -> None:
    """A random split would find no drift by construction."""
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2023-01-01", periods=100, freq="h"),
            "amount": range(100),
        }
    )

    reference, current = split_by_time(frame, reference_fraction=0.7)

    assert len(reference) == 70
    assert reference["event_time"].max() < current["event_time"].min()


def test_windows_with_nothing_in_common_are_rejected(reference: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="share no columns"):
        build_drift_report(reference, pd.DataFrame({"unrelated": [1.0, 2.0]}))
