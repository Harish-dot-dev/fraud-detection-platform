"""Tests for the time-based split."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from training.split import time_based_split


def _frame(n: int = 100) -> pd.DataFrame:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        {
            "transaction_id": range(n),
            "event_time": [start + timedelta(hours=i) for i in range(n)],
            "is_fraud": [i % 20 == 0 for i in range(n)],
        }
    )


def test_the_split_is_chronological() -> None:
    """Every training payment happened before every test payment.

    A random split would let the model learn from payments that happened after
    the ones it is scored on. Fraud arrives in campaigns, so that is close to
    showing it the answer.
    """
    train, validation, test, _ = time_based_split(_frame())

    assert train["event_time"].max() < validation["event_time"].min()
    assert validation["event_time"].max() < test["event_time"].min()


def test_the_fractions_are_respected() -> None:
    train, validation, test, boundaries = time_based_split(
        _frame(100), train_fraction=0.7, validation_fraction=0.15
    )

    assert (len(train), len(validation), len(test)) == (70, 15, 15)
    assert boundaries.train_rows == 70


def test_no_payment_is_lost_or_duplicated() -> None:
    frame = _frame(97)

    train, validation, test, _ = time_based_split(frame)

    ids = set(train.transaction_id) | set(validation.transaction_id) | set(test.transaction_id)
    assert len(ids) == len(frame)


def test_unordered_input_is_sorted_first() -> None:
    """The Gold table is not guaranteed to come back in time order."""
    shuffled = _frame(50).sample(frac=1.0, random_state=1)

    train, _, test, _ = time_based_split(shuffled)

    assert train["event_time"].is_monotonic_increasing
    assert train["event_time"].max() < test["event_time"].min()


def test_the_boundaries_are_reported_for_reproducibility() -> None:
    _, _, _, boundaries = time_based_split(_frame())

    assert boundaries.train_end < boundaries.validation_end
    assert "train 70 rows" in boundaries.describe()


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction"),
    [(0.0, 0.15), (1.0, 0.15), (0.9, 0.2)],
)
def test_impossible_fractions_are_rejected(train_fraction, validation_fraction) -> None:
    with pytest.raises(ValueError):
        time_based_split(_frame(), train_fraction, validation_fraction)


def test_a_missing_time_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="event_time"):
        time_based_split(pd.DataFrame({"x": [1, 2, 3]}))
