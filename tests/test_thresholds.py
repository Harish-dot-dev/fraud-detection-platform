"""Tests for cost-based threshold tuning.

The thresholds are a business decision, so these tests are mostly about the
business logic: does the chosen operating point actually respond to what
mistakes cost?
"""

from __future__ import annotations

import numpy as np
import pytest

from training.thresholds import CostModel, expected_cost, tune_thresholds


@pytest.fixture
def scored_window():
    """1000 payments: fraud scores high, legitimate scores low, with overlap."""
    rng = np.random.default_rng(7)
    n = 1000
    is_fraud = (rng.random(n) < 0.04).astype(int)
    # Overlapping distributions - a perfectly separable set would make every
    # threshold look equally good.
    scores = np.clip(rng.normal(np.where(is_fraud, 0.7, 0.2), 0.18), 0.0, 1.0)
    amounts = np.round(np.exp(rng.normal(4.0, 0.8, n)), 2)
    return is_fraud, scores, amounts


def test_the_cost_of_doing_nothing_is_all_the_fraud(scored_window) -> None:
    is_fraud, scores, amounts = scored_window

    # Thresholds above every score: everything is allowed.
    cost = expected_cost(is_fraud, scores, amounts, 1.1, 1.1, CostModel())

    assert cost == pytest.approx(amounts[is_fraud == 1].sum())


def test_blocking_everything_costs_friction_on_every_good_customer(scored_window) -> None:
    is_fraud, scores, amounts = scored_window
    costs = CostModel(false_block=25.0, review=5.0)

    cost = expected_cost(is_fraud, scores, amounts, -1.0, -1.0, costs)

    assert cost == pytest.approx(int((is_fraud == 0).sum()) * costs.false_block)


def test_tuning_beats_both_extremes(scored_window) -> None:
    """The point of tuning: the chosen point is cheaper than either extreme."""
    is_fraud, scores, amounts = scored_window
    costs = CostModel()

    choice = tune_thresholds(is_fraud, scores, amounts, costs)

    allow_everything = expected_cost(is_fraud, scores, amounts, 1.1, 1.1, costs)
    block_everything = expected_cost(is_fraud, scores, amounts, -1.0, -1.0, costs)
    assert choice.expected_cost < allow_everything
    assert choice.expected_cost < block_everything
    assert choice.cost_reduction > 0


def test_the_review_threshold_never_exceeds_the_block_threshold(scored_window) -> None:
    is_fraud, scores, amounts = scored_window

    choice = tune_thresholds(is_fraud, scores, amounts)

    assert choice.review_threshold <= choice.block_threshold


def test_expensive_false_blocks_make_the_model_more_cautious(scored_window) -> None:
    """Raise the cost of annoying a customer and fewer payments get blocked."""
    is_fraud, scores, amounts = scored_window

    cheap = tune_thresholds(is_fraud, scores, amounts, CostModel(false_block=5.0, review=1.0))
    expensive = tune_thresholds(is_fraud, scores, amounts, CostModel(false_block=500.0, review=1.0))

    assert expensive.block_threshold >= cheap.block_threshold
    assert expensive.block_rate <= cheap.block_rate


def test_expensive_reviews_shrink_the_review_queue(scored_window) -> None:
    """If analysts are costly, the band between the thresholds narrows."""
    is_fraud, scores, amounts = scored_window

    cheap = tune_thresholds(is_fraud, scores, amounts, CostModel(false_block=25.0, review=0.5))
    expensive = tune_thresholds(is_fraud, scores, amounts, CostModel(false_block=25.0, review=50.0))

    assert expensive.review_rate <= cheap.review_rate


def test_the_breakdown_adds_up(scored_window) -> None:
    is_fraud, scores, amounts = scored_window

    choice = tune_thresholds(is_fraud, scores, amounts)

    total = choice.missed_fraud_cost + choice.review_cost + choice.false_block_cost
    assert total == pytest.approx(choice.expected_cost)


def test_large_frauds_matter_more_than_small_ones() -> None:
    """Missed fraud costs the transaction amount, so the expensive ones dominate.

    Two identically-scored frauds, one 20 euros and one 5,000: the tuner has to
    reach below the expensive one's score, even though that costs false blocks.
    """
    is_fraud = np.array([1, 1] + [0] * 98)
    scores = np.array([0.8, 0.4] + list(np.linspace(0.0, 0.39, 98)))
    amounts = np.array([20.0, 5000.0] + [50.0] * 98)

    choice = tune_thresholds(is_fraud, scores, amounts, CostModel(false_block=25.0, review=5.0))

    assert choice.review_threshold <= 0.4


def test_the_choice_is_serialisable(tmp_path, scored_window) -> None:
    """Every reported threshold has to come from a file a run produced."""
    import json

    is_fraud, scores, amounts = scored_window

    path = tune_thresholds(is_fraud, scores, amounts).save(tmp_path / "thresholds.json")

    saved = json.loads(path.read_text())
    assert "review_threshold" in saved
    assert "cost_reduction" in saved


def test_an_empty_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        tune_thresholds(np.array([]), np.array([]), np.array([]))
