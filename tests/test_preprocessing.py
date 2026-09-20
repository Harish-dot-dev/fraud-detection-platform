"""Tests for the model input schema.

The matrix built at training time and the matrix built for one payment at
scoring time have to be identical, or the model is being served something it
was never trained on.
"""

from __future__ import annotations

import pandas as pd

from features.definitions import FEATURE_NAMES
from training.preprocessing import (
    CATEGORICAL_COLUMNS,
    MISSING,
    MODEL_COLUMNS,
    OTHER,
    build_matrix,
)


def test_every_behavioural_feature_reaches_the_model() -> None:
    for feature in FEATURE_NAMES:
        assert feature in MODEL_COLUMNS


def test_column_order_is_fixed() -> None:
    """XGBoost matches features by position; a reordering silently corrupts it."""
    first = build_matrix(pd.DataFrame({"amount": [1.0]}))
    second = build_matrix(pd.DataFrame({"product_cd": ["W"], "amount": [1.0]}))

    assert list(first.columns) == MODEL_COLUMNS
    assert list(second.columns) == MODEL_COLUMNS


def test_a_single_payment_produces_the_same_columns_as_a_batch() -> None:
    """The scoring API passes one row; training passes hundreds of thousands."""
    batch = build_matrix(pd.DataFrame({"amount": [1.0, 2.0], "product_cd": ["W", "C"]}))
    single = build_matrix(pd.DataFrame({"amount": [1.0], "product_cd": ["W"]}))

    assert list(batch.columns) == list(single.columns)
    assert batch.dtypes.equals(single.dtypes)


def test_missing_columns_become_missing_values_not_errors() -> None:
    """A payment with no device information must still be scoreable."""
    matrix = build_matrix(pd.DataFrame({"amount": [10.0]}))

    assert matrix["card_txn_count_10m"].isna().all()
    assert matrix["device_type"].iloc[0] == MISSING


def test_unknown_and_absent_categories_are_distinguished() -> None:
    """An unfamiliar email domain and no email domain are different events."""
    matrix = build_matrix(
        pd.DataFrame({"p_emaildomain": ["never-seen.example", None, "gmail.com"]})
    )

    assert list(matrix["p_emaildomain"]) == [OTHER, MISSING, "gmail.com"]


def test_category_matching_ignores_case() -> None:
    """ProductCD is upper case in the source data, card4 is lower."""
    matrix = build_matrix(pd.DataFrame({"product_cd": ["W"], "card4": ["VISA"]}))

    assert matrix["product_cd"].iloc[0] == "W"
    assert matrix["card4"].iloc[0] == "visa"


def test_sparse_map_columns_are_flattened() -> None:
    matrix = build_matrix(
        pd.DataFrame(
            {
                "counts": [{"C1": 3.0, "C14": 1.0}, {}],
                "deltas": [{"D1": 20.0}, {"D15": 5.0}],
            }
        )
    )

    assert matrix["C1"].iloc[0] == 3.0
    assert pd.isna(matrix["C1"].iloc[1])
    assert matrix["D15"].iloc[1] == 5.0


def test_categoricals_are_category_dtype_for_native_xgboost_support() -> None:
    matrix = build_matrix(pd.DataFrame({"product_cd": ["W"]}))

    for column in CATEGORICAL_COLUMNS:
        assert matrix[column].dtype.name == "category"


def test_numeric_columns_are_floats() -> None:
    matrix = build_matrix(pd.DataFrame({"amount": ["12.50"]}))

    assert matrix["amount"].dtype == "float64"
    assert matrix["amount"].iloc[0] == 12.5
