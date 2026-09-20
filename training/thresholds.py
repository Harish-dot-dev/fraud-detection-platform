"""Choosing the two thresholds by what mistakes actually cost.

The model outputs a probability. Turning that into *allow / review / block*
needs two cut-offs, and there is no statistically correct place to put them -
only a business one. So the thresholds are chosen by minimising expected cost:

* **missed fraud** costs the transaction amount (the issuer eats the
  chargeback),
* **a wrongly blocked payment** costs a fixed amount of customer friction - a
  support call, a lost sale, some churn risk (``COST_FALSE_BLOCK``),
* **sending a payment to an analyst** costs a fixed handling fee
  (``COST_REVIEW``).

The asymmetry is the whole point. A 20 euro fraud and a 2,000 euro fraud are
not the same miss, so a single "optimal F1" threshold is the wrong tool: it
treats every error as equal when the business does not.

One optimistic assumption is baked in and worth saying out loud: a payment sent
to review is assumed to be resolved correctly. Real analysts are not perfect,
so the review band looks slightly cheaper here than it would be in production.

The search itself is exact over a candidate grid rather than approximate. Sort
the scores once, precompute cumulative sums, and the cost of any threshold pair
is then three array lookups - which makes a 100x100 grid instant even on a
million rows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """What each kind of mistake costs."""

    # A wrongly blocked good customer: support call, lost sale, churn risk.
    false_block: float = 25.0
    # Handling one payment in the analyst queue.
    review: float = 5.0

    def describe(self) -> str:
        return (
            f"missed fraud = transaction amount, "
            f"false block = {self.false_block:.2f}, review = {self.review:.2f}"
        )


@dataclass(frozen=True)
class ThresholdChoice:
    """The chosen operating point and what it is expected to cost."""

    review_threshold: float
    block_threshold: float
    expected_cost: float
    # Cost of doing nothing at all: every fraud is missed.
    baseline_cost: float
    missed_fraud_cost: float
    false_block_cost: float
    review_cost: float
    review_rate: float
    block_rate: float

    @property
    def cost_reduction(self) -> float:
        """Share of the do-nothing cost avoided. Not a profit forecast."""
        if self.baseline_cost <= 0:
            return 0.0
        return 1.0 - self.expected_cost / self.baseline_cost

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "cost_reduction": self.cost_reduction}

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2))
        return destination

    def describe(self) -> str:
        return (
            f"review >= {self.review_threshold:.3f}, block >= {self.block_threshold:.3f} "
            f"(expected cost {self.expected_cost:,.0f} vs {self.baseline_cost:,.0f} "
            f"doing nothing, {self.cost_reduction:.1%} lower; "
            f"{self.review_rate:.2%} reviewed, {self.block_rate:.2%} blocked)"
        )


class _CostSurface:
    """Cumulative sums that make any threshold pair an O(1) lookup."""

    def __init__(self, y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray) -> None:
        order = np.argsort(scores, kind="stable")
        self.scores = scores[order]
        labels = y_true[order].astype(bool)
        amounts = amounts[order]

        # Leading zero so that cumulative[i] means "the first i rows".
        self.cumulative_fraud_amount = np.concatenate([[0.0], np.cumsum(amounts * labels)])
        self.cumulative_legitimate = np.concatenate([[0], np.cumsum(~labels)])
        self.total = len(self.scores)
        self.total_legitimate = int((~labels).sum())
        self.total_fraud_amount = float((amounts * labels).sum())

    def index_at(self, threshold: float) -> int:
        """First position whose score is >= threshold."""
        return int(np.searchsorted(self.scores, threshold, side="left"))

    def cost(self, review: float, block: float, costs: CostModel) -> dict[str, float]:
        review_index = self.index_at(review)
        block_index = max(self.index_at(block), review_index)

        # Everything below the review threshold is allowed: fraud there is lost.
        missed_fraud = float(self.cumulative_fraud_amount[review_index])
        reviewed = block_index - review_index
        blocked_legitimate = self.total_legitimate - int(self.cumulative_legitimate[block_index])

        return {
            "missed_fraud_cost": missed_fraud,
            "review_cost": reviewed * costs.review,
            "false_block_cost": blocked_legitimate * costs.false_block,
            "reviewed": reviewed,
            "blocked": self.total - block_index,
        }


def expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    review_threshold: float,
    block_threshold: float,
    costs: CostModel,
) -> float:
    """Total expected cost of one operating point."""
    surface = _CostSurface(np.asarray(y_true), np.asarray(scores), np.asarray(amounts))
    parts = surface.cost(review_threshold, block_threshold, costs)
    return parts["missed_fraud_cost"] + parts["review_cost"] + parts["false_block_cost"]


def tune_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    costs: CostModel | None = None,
    grid_size: int = 101,
) -> ThresholdChoice:
    """Find the cheapest (review, block) pair.

    Candidates are score quantiles rather than an even split of [0, 1]: model
    scores bunch up near zero, so an even grid would spend most of its
    candidates in a region where nothing changes.
    """
    costs = costs or CostModel()
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    if len(scores) == 0:
        raise ValueError("cannot tune thresholds on an empty set")

    surface = _CostSurface(y_true, scores, amounts)
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, grid_size)))

    best: ThresholdChoice | None = None
    for review in candidates:
        for block in candidates[candidates >= review]:
            parts = surface.cost(float(review), float(block), costs)
            total = parts["missed_fraud_cost"] + parts["review_cost"] + parts["false_block_cost"]
            if best is None or total < best.expected_cost:
                best = ThresholdChoice(
                    review_threshold=float(review),
                    block_threshold=float(block),
                    expected_cost=float(total),
                    baseline_cost=surface.total_fraud_amount,
                    missed_fraud_cost=parts["missed_fraud_cost"],
                    false_block_cost=parts["false_block_cost"],
                    review_cost=parts["review_cost"],
                    review_rate=parts["reviewed"] / surface.total,
                    block_rate=parts["blocked"] / surface.total,
                )

    assert best is not None  # candidates is never empty when scores is not
    return best
