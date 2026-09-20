"""The model's input schema, defined once for training and for serving.

A model is only as reproducible as the columns it was fed. This module fixes
three things so that the matrix built at training time and the matrix built for
a single payment at scoring time are identical:

* **which columns** go in, and in **what order** (XGBoost cares about order),
* **how categoricals are encoded**, and
* **what happens to a value the model has never seen**.

Categorical vocabularies live here, in code, rather than in a fitted encoder
saved alongside the model. That is a deliberate trade:

* an encoder artifact can drift out of step with the model file, and has to be
  loaded and versioned separately at serving time;
* a vocabulary in code is versioned with the code, reviewable in a diff, and
  impossible to forget to ship.

The cost is that a genuinely new email domain is encoded as "other" until
somebody updates the list - a visible, deliberate change rather than a silent
re-encoding on the next retrain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from common.events import COUNT_COLUMNS, DELTA_COLUMNS
from features.definitions import FEATURE_NAMES

# Numeric columns carried straight through from Gold.
RAW_NUMERIC_COLUMNS = ["card3", "card5", "dist1", "dist2"]

# The C and D blocks: counts of addresses/phones/emails linked to the card, and
# "days since" timedeltas. Genuinely predictive in this dataset and available
# at authorisation time, so they earn their place.
MAP_COLUMNS = {"counts": COUNT_COLUMNS, "deltas": DELTA_COLUMNS}

# Categorical vocabularies. Anything outside these becomes OTHER.
OTHER = "other"
MISSING = "missing"

# The domains that make up the overwhelming majority of the dataset. Note
# "anonymous.com" - it is a real value in IEEE-CIS and a meaningful signal.
EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "anonymous.com",
    "aol.com",
    "comcast.net",
    "icloud.com",
    "outlook.com",
    "msn.com",
    "att.net",
    "live.com",
    "verizon.net",
    "sbcglobal.net",
    "ymail.com",
    "bellsouth.net",
    "cox.net",
    "me.com",
    "optonline.net",
    "charter.net",
    "mail.com",
]

CATEGORICAL_VOCABULARIES: dict[str, list[str]] = {
    "product_cd": ["W", "C", "R", "H", "S"],
    "card4": ["visa", "mastercard", "american express", "discover"],
    "card6": ["debit", "credit", "charge card", "debit or credit"],
    "device_type": ["desktop", "mobile"],
    "p_emaildomain": EMAIL_DOMAINS,
    "r_emaildomain": EMAIL_DOMAINS,
}

CATEGORICAL_COLUMNS = list(CATEGORICAL_VOCABULARIES)

# The full model input, in a fixed order.
MODEL_COLUMNS: list[str] = (
    FEATURE_NAMES
    + RAW_NUMERIC_COLUMNS
    + [column for columns in MAP_COLUMNS.values() for column in columns]
    + CATEGORICAL_COLUMNS
)


def _categories_for(column: str) -> list[str]:
    """Vocabulary plus the two catch-alls, in a stable order."""
    return [*CATEGORICAL_VOCABULARIES[column], OTHER, MISSING]


# Built once. Constructing these per request cost around 20 ms on the scoring
# path, which is a fifth of the entire latency budget spent on bookkeeping.
CATEGORICAL_DTYPES: dict[str, pd.CategoricalDtype] = {
    column: pd.CategoricalDtype(categories=_categories_for(column))
    for column in CATEGORICAL_VOCABULARIES
}
_CANONICAL_VALUES: dict[str, dict[str, str]] = {
    column: {value.lower(): value for value in vocabulary}
    for column, vocabulary in CATEGORICAL_VOCABULARIES.items()
}


def flatten_map_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand the Gold ``counts`` / ``deltas`` maps into one column each.

    Gold stores these as maps because the source data is sparse and a map
    survives a schema change; a model needs flat columns.
    """
    flattened = frame.copy()

    for map_column, expected in MAP_COLUMNS.items():
        if map_column not in flattened.columns:
            # Allow a caller (the scoring API) to pass the flat columns already.
            continue
        values = flattened[map_column]
        for column in expected:
            flattened[column] = values.map(
                lambda mapping, key=column: (mapping or {}).get(key, float("nan"))
            )
        flattened = flattened.drop(columns=[map_column])

    return flattened


def _encode_categorical(frame: pd.DataFrame, column: str) -> pd.Categorical:
    """Encode one categorical column against its fixed vocabulary.

    Two catch-alls, and they mean different things: a value the vocabulary does
    not contain becomes "other", while an absent value becomes "missing". A
    payment with an unfamiliar email domain and a payment with no email domain
    at all are not the same event, and the model is allowed to tell them apart.
    """
    categories = _categories_for(column)
    # Matching is case-insensitive (ProductCD is upper case, card4 is lower),
    # but the canonical spelling is kept so SHAP output reads naturally.
    canonical = {value.lower(): value for value in CATEGORICAL_VOCABULARIES[column]}

    if column in frame.columns:
        raw = frame[column].astype("string")
    else:
        raw = pd.Series(pd.NA, index=frame.index, dtype="string")

    encoded = raw.str.lower().map(canonical).fillna(OTHER).mask(raw.isna(), MISSING)
    return pd.Categorical(encoded, categories=categories)


def build_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn Gold-shaped rows into the exact matrix the model expects.

    Missing columns are filled with NaN rather than raising: XGBoost treats NaN
    as "missing" and learns a default direction for it, which is the right
    behaviour for a payment that arrives without, say, device information.

    Returns a frame with :data:`MODEL_COLUMNS` in order, numeric columns as
    float and categoricals as pandas ``category`` dtype with fixed categories.
    """
    flattened = flatten_map_columns(frame)
    matrix = pd.DataFrame(index=flattened.index)

    for column in MODEL_COLUMNS:
        if column in CATEGORICAL_VOCABULARIES:
            matrix[column] = _encode_categorical(flattened, column)
        else:
            if column in flattened.columns:
                matrix[column] = pd.to_numeric(flattened[column], errors="coerce").astype("float64")
            else:
                matrix[column] = pd.Series(float("nan"), index=flattened.index, dtype="float64")

    return matrix[MODEL_COLUMNS]


def _canonical_category(column: str, value: Any) -> str:
    """Encode one categorical value exactly as :func:`build_matrix` would."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value != value:
        return MISSING
    return _CANONICAL_VALUES[column].get(str(value).lower(), OTHER)


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def build_row(row: Mapping[str, Any]) -> pd.DataFrame:
    """The single-payment fast path, for the scoring API.

    ``build_matrix`` is written for a training set: vectorised pandas
    operations over hundreds of thousands of rows. Running it on one row costs
    around 20 ms of pure overhead - constructing 57 Series and six categorical
    dtypes to hold a single value each - which on a 100 ms budget is not
    affordable.

    This builds the same one-row matrix directly. It is a second
    implementation of the same schema, so - exactly like the online/offline
    feature pair - a test asserts the two produce identical output
    (``test_the_fast_path_matches_the_batch_path``).
    """
    flat = dict(row)
    for map_column, expected in MAP_COLUMNS.items():
        mapping = flat.pop(map_column, None) or {}
        for column in expected:
            if column not in flat:
                flat[column] = mapping.get(column, float("nan"))

    data: dict[str, Any] = {}
    for column in MODEL_COLUMNS:
        if column in CATEGORICAL_VOCABULARIES:
            data[column] = pd.Categorical(
                [_canonical_category(column, flat.get(column))],
                dtype=CATEGORICAL_DTYPES[column],
            )
        else:
            data[column] = np.array([_as_float(flat.get(column))], dtype="float64")

    return pd.DataFrame(data, columns=MODEL_COLUMNS)
