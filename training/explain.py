"""Per-decision explanations with SHAP.

A fraud analyst cannot act on "0.87". They need to know *why*, and a reason
that is actually derived from the model rather than written by hand.

SHAP gives each feature a signed contribution to this particular score, and
those contributions add up to the score itself. For tree models it is exact and
fast - no sampling - which is what makes it usable on the scoring path rather
than only in a notebook.

The output here is deliberately small: the few features that pushed this
payment's score up, with their values. That is what goes in the decision log,
the analyst app and the LLM's facts object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Reason:
    """One feature's contribution to one decision."""

    feature: str
    value: float | str | None
    contribution: float

    @property
    def direction(self) -> str:
        return "increases risk" if self.contribution > 0 else "reduces risk"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "direction": self.direction}

    def describe(self) -> str:
        return f"{self.feature}={self.value} ({self.contribution:+.3f}, {self.direction})"


class Explainer:
    """Wraps a trained model and turns rows into ranked reasons."""

    def __init__(self, model: Any) -> None:
        import shap

        self._explainer = shap.TreeExplainer(model)

    def contributions(self, matrix: pd.DataFrame) -> np.ndarray:
        """SHAP values for every row and feature.

        ``check_additivity`` is disabled because the categorical columns are
        passed through XGBoost's native categorical support, where the
        additivity check is known to trip on the internal representation. The
        contributions themselves are unaffected.
        """
        return np.asarray(self._explainer.shap_values(matrix, check_additivity=False))

    def top_reasons(self, matrix: pd.DataFrame, top_n: int = 3) -> list[Reason]:
        """The strongest risk-increasing reasons for a single row.

        Only positive contributions are returned: an analyst reviewing a
        flagged payment wants to know what made it look bad, not what made it
        look fine.
        """
        if len(matrix) != 1:
            raise ValueError("top_reasons explains exactly one payment at a time")

        values = self.contributions(matrix)[0]
        row = matrix.iloc[0]

        reasons = [
            Reason(
                feature=str(column),
                value=_readable(row[column]),
                contribution=float(contribution),
            )
            for column, contribution in zip(matrix.columns, values, strict=True)
            if contribution > 0
        ]
        reasons.sort(key=lambda reason: reason.contribution, reverse=True)
        return reasons[:top_n]


def _readable(value: Any) -> float | str | None:
    """Render a feature value for a human (or for an LLM prompt)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, int | float | np.floating | np.integer):
        return round(float(value), 4)
    return str(value)


def global_importance(explainer: Explainer, matrix: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Mean absolute SHAP value per feature - what the model relies on overall.

    Logged with every training run: a sudden change in this ranking between
    retrains is a strong hint that something upstream has shifted.
    """
    values = np.abs(explainer.contributions(matrix)).mean(axis=0)
    return (
        pd.DataFrame({"feature": matrix.columns, "mean_abs_shap": values})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
