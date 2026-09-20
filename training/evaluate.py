"""Evaluating a fraud model honestly.

Accuracy is not reported anywhere in this project, and the reason is worth
stating plainly: fraud is about 3.5% of payments, so a model that says "not
fraud" to everything is 96.5% accurate and catches nothing.

What is reported instead:

* **PR-AUC** (average precision) - the summary metric for a rare positive
  class. ROC-AUC is logged too, but it flatters imbalanced problems because
  the huge number of true negatives drowns out the false positives.
* **Precision and recall at the chosen operating points**, not at 0.5. The
  thresholds are a business decision (``training/thresholds.py``), so the
  metrics have to be quoted at them.
* **Money**. The fraud value caught versus missed is what the business
  actually cares about, and it is not the same story as the transaction count:
  catching 60% of fraudulent payments while missing the expensive ones is a
  bad model with respectable-looking recall.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from training.thresholds import CostModel

ALLOW = "allow"
REVIEW = "review"
BLOCK = "block"


def decide(scores: np.ndarray, review_threshold: float, block_threshold: float) -> np.ndarray:
    """Map scores onto the three decisions."""
    scores = np.asarray(scores, dtype=float)
    return np.where(
        scores >= block_threshold, BLOCK, np.where(scores >= review_threshold, REVIEW, ALLOW)
    )


@dataclass
class ModelMetrics:
    """Everything worth knowing about a model on one evaluation window."""

    rows: int
    fraud_count: int
    fraud_rate: float
    pr_auc: float
    roc_auc: float
    review_threshold: float
    block_threshold: float
    # Quality at the block threshold - the decision that affects a customer.
    precision_at_block: float
    recall_at_block: float
    false_positive_rate: float
    # Recall counting the review queue as caught, since an analyst sees those.
    recall_including_review: float
    allow_rate: float
    review_rate: float
    block_rate: float
    fraud_value_total: float
    fraud_value_blocked: float
    fraud_value_reviewed: float
    fraud_value_missed: float
    fraud_value_caught_share: float
    legitimate_value_blocked: float
    expected_cost: float
    baseline_cost: float
    cost_model: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2))
        return destination

    def describe(self) -> str:
        return (
            f"PR-AUC {self.pr_auc:.4f} | at block>={self.block_threshold:.3f}: "
            f"precision {self.precision_at_block:.3f}, recall {self.recall_at_block:.3f}, "
            f"FPR {self.false_positive_rate:.4f} | "
            f"fraud value caught {self.fraud_value_caught_share:.1%} "
            f"({self.fraud_value_missed:,.0f} missed) | "
            f"{self.review_rate:.2%} reviewed, {self.block_rate:.2%} blocked"
        )


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_model(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    review_threshold: float,
    block_threshold: float,
    costs: CostModel | None = None,
) -> ModelMetrics:
    """Score a model on one window at a specific operating point."""
    costs = costs or CostModel()
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    if len(y_true) == 0:
        raise ValueError("cannot evaluate an empty window")

    decisions = decide(scores, review_threshold, block_threshold)
    is_fraud = y_true == 1
    blocked = decisions == BLOCK
    reviewed = decisions == REVIEW

    true_positive = int((blocked & is_fraud).sum())
    false_positive = int((blocked & ~is_fraud).sum())
    false_negative = int((~blocked & is_fraud).sum())
    true_negative = int((~blocked & ~is_fraud).sum())

    fraud_value_total = float(amounts[is_fraud].sum())
    fraud_value_blocked = float(amounts[is_fraud & blocked].sum())
    fraud_value_reviewed = float(amounts[is_fraud & reviewed].sum())
    fraud_value_missed = fraud_value_total - fraud_value_blocked - fraud_value_reviewed

    # Both classes must be present for the ranking metrics to mean anything.
    both_classes = 0 < is_fraud.sum() < len(y_true)
    pr_auc = float(average_precision_score(y_true, scores)) if both_classes else 0.0
    roc_auc = float(roc_auc_score(y_true, scores)) if both_classes else 0.0

    missed_fraud_cost = float(amounts[is_fraud & (decisions == ALLOW)].sum())
    expected = (
        missed_fraud_cost + reviewed.sum() * costs.review + false_positive * costs.false_block
    )

    return ModelMetrics(
        rows=len(y_true),
        fraud_count=int(is_fraud.sum()),
        fraud_rate=_safe_divide(is_fraud.sum(), len(y_true)),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        review_threshold=float(review_threshold),
        block_threshold=float(block_threshold),
        precision_at_block=_safe_divide(true_positive, true_positive + false_positive),
        recall_at_block=_safe_divide(true_positive, true_positive + false_negative),
        false_positive_rate=_safe_divide(false_positive, false_positive + true_negative),
        recall_including_review=_safe_divide(
            true_positive + int((reviewed & is_fraud).sum()), int(is_fraud.sum())
        ),
        allow_rate=_safe_divide((decisions == ALLOW).sum(), len(decisions)),
        review_rate=_safe_divide(reviewed.sum(), len(decisions)),
        block_rate=_safe_divide(blocked.sum(), len(decisions)),
        fraud_value_total=fraud_value_total,
        fraud_value_blocked=fraud_value_blocked,
        fraud_value_reviewed=fraud_value_reviewed,
        fraud_value_missed=fraud_value_missed,
        fraud_value_caught_share=_safe_divide(
            fraud_value_blocked + fraud_value_reviewed, fraud_value_total
        ),
        legitimate_value_blocked=float(amounts[~is_fraud & blocked].sum()),
        expected_cost=float(expected),
        baseline_cost=fraud_value_total,
        cost_model={"false_block": costs.false_block, "review": costs.review},
    )
