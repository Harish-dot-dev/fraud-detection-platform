"""Tests for model evaluation.

Written around the failure these metrics exist to expose: a model that looks
good on a headline number while doing something useless.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from training.evaluate import ALLOW, BLOCK, REVIEW, decide, evaluate_model
from training.thresholds import CostModel


def test_three_decisions_not_two() -> None:
    decisions = decide(np.array([0.05, 0.45, 0.95]), review_threshold=0.3, block_threshold=0.8)

    assert list(decisions) == [ALLOW, REVIEW, BLOCK]


def test_a_score_exactly_on_a_threshold_takes_the_stricter_decision() -> None:
    decisions = decide(np.array([0.3, 0.8]), review_threshold=0.3, block_threshold=0.8)

    assert list(decisions) == [REVIEW, BLOCK]


def test_the_do_nothing_model_is_caught_by_pr_auc_not_accuracy() -> None:
    """The reason accuracy is not reported anywhere in this project.

    A model that scores everything 0.01 is right 96% of the time and catches no
    fraud at all. PR-AUC and recall both say so; accuracy would not.
    """
    y_true = np.array([0] * 96 + [1] * 4)
    scores = np.full(100, 0.01)
    amounts = np.full(100, 100.0)

    metrics = evaluate_model(y_true, scores, amounts, 0.3, 0.8)

    assert metrics.recall_at_block == 0.0
    assert metrics.fraud_value_caught_share == 0.0
    assert metrics.pr_auc < 0.1
    # ...and it would have scored 96% on accuracy, which is why we do not use it.


def test_a_perfect_model_catches_every_fraud_and_blocks_nobody_else() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.01, 0.02, 0.97, 0.99])
    amounts = np.array([10.0, 20.0, 500.0, 300.0])

    metrics = evaluate_model(y_true, scores, amounts, 0.3, 0.8)

    assert metrics.precision_at_block == 1.0
    assert metrics.recall_at_block == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.fraud_value_missed == 0.0
    assert metrics.pr_auc == pytest.approx(1.0)


def test_recall_by_count_and_by_value_can_disagree() -> None:
    """The reason fraud value is reported alongside recall.

    This model catches three of four frauds - 75% recall - while missing the
    one that mattered. Over 90% of the fraud value walks out of the door.
    """
    y_true = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.9, 0.9, 0.1, 0.05, 0.05])
    amounts = np.array([10.0, 10.0, 10.0, 5000.0, 50.0, 50.0])

    metrics = evaluate_model(y_true, scores, amounts, 0.3, 0.8)

    assert metrics.recall_at_block == pytest.approx(0.75)
    assert metrics.fraud_value_caught_share < 0.01
    assert metrics.fraud_value_missed == pytest.approx(5000.0)


def test_the_review_queue_counts_as_caught_only_in_its_own_metric() -> None:
    """An analyst sees reviewed payments, so they are not simply misses."""
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.95, 0.5, 0.1, 0.1])
    amounts = np.array([100.0, 200.0, 10.0, 10.0])

    metrics = evaluate_model(y_true, scores, amounts, 0.3, 0.8)

    assert metrics.recall_at_block == pytest.approx(0.5)
    assert metrics.recall_including_review == pytest.approx(1.0)
    assert metrics.fraud_value_reviewed == 200.0
    assert metrics.fraud_value_missed == 0.0


def test_decision_rates_sum_to_one() -> None:
    rng = np.random.default_rng(3)
    scores = rng.random(500)

    metrics = evaluate_model(rng.integers(0, 2, 500), scores, np.full(500, 10.0), 0.3, 0.8)

    assert metrics.allow_rate + metrics.review_rate + metrics.block_rate == pytest.approx(1.0)


def test_expected_cost_uses_the_configured_cost_model() -> None:
    y_true = np.array([1, 0, 0])
    scores = np.array([0.1, 0.9, 0.5])
    amounts = np.array([400.0, 10.0, 10.0])

    metrics = evaluate_model(
        y_true, scores, amounts, 0.3, 0.8, CostModel(false_block=25.0, review=5.0)
    )

    # 400 missed fraud + one false block + one review.
    assert metrics.expected_cost == pytest.approx(400.0 + 25.0 + 5.0)
    assert metrics.baseline_cost == 400.0


def test_metrics_are_saved_as_evidence(tmp_path) -> None:
    """Nothing reaches the README that did not come out of a file like this."""
    metrics = evaluate_model(
        np.array([0, 1]), np.array([0.1, 0.9]), np.array([10.0, 20.0]), 0.3, 0.8
    )

    path = metrics.save(tmp_path / "metrics.json")

    saved = json.loads(path.read_text())
    assert saved["pr_auc"] == pytest.approx(1.0)
    assert "fraud_value_missed" in saved


def test_an_empty_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(np.array([]), np.array([]), np.array([]), 0.3, 0.8)
