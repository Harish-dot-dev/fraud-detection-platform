"""Time-based train / validation / test splits.

Never random. A random split lets the model learn from payments that happened
*after* the ones it is evaluated on, which for a fraud model is close to
cheating: fraud arrives in campaigns, so a card compromised on Tuesday shows up
in both halves of a random split and the model gets credit for recognising a
pattern it was shown the answer to.

Splitting on time also matches how the model will actually be used: trained on
the past, run on the future. The test window is the most recent data, which is
the closest available stand-in for tomorrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

TIME_COLUMN = "event_time"


@dataclass(frozen=True)
class SplitBoundaries:
    """Where the cuts fell - logged with every run so a result is reproducible."""

    train_end: datetime
    validation_end: datetime
    train_rows: int
    validation_rows: int
    test_rows: int

    def describe(self) -> str:
        return (
            f"train {self.train_rows} rows (to {self.train_end:%Y-%m-%d}), "
            f"validation {self.validation_rows} rows (to {self.validation_end:%Y-%m-%d}), "
            f"test {self.test_rows} rows"
        )


def time_based_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    time_column: str = TIME_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitBoundaries]:
    """Split chronologically into train, validation and test.

    Args:
        frame: the training set, one row per payment.
        train_fraction: share of the *time-ordered* rows used for training.
        validation_fraction: share used for validation (early stopping and
            threshold tuning). The remainder is the test window.

    Returns:
        ``(train, validation, test, boundaries)``.

    The cuts are made on row position after sorting by time, not on calendar
    dates, so the three parts have predictable sizes even when payment volume
    varies over the period.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction must leave room for a test window")
    if time_column not in frame.columns:
        raise ValueError(f"{time_column!r} is required for a time-based split")

    ordered = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
    train_cut = int(len(ordered) * train_fraction)
    validation_cut = int(len(ordered) * (train_fraction + validation_fraction))

    train = ordered.iloc[:train_cut]
    validation = ordered.iloc[train_cut:validation_cut]
    test = ordered.iloc[validation_cut:]

    boundaries = SplitBoundaries(
        train_end=train[time_column].max(),
        validation_end=validation[time_column].max(),
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
    )
    return train, validation, test, boundaries
