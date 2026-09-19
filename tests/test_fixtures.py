"""The committed fixture must stay in sync with the generator.

If someone changes the generator without re-running `make sample`, CI catches
it here rather than in a confusing failure three phases later.
"""

from __future__ import annotations

import pandas as pd

from tests.conftest import IDENTITY_FIXTURE, TRANSACTION_FIXTURE
from tests.synthetic import IDENTITY_COLUMNS, TRANSACTION_COLUMNS, generate_transactions

# Must match the arguments in the `sample` target of the Makefile.
FIXTURE_ROWS = 1000
FIXTURE_SEED = 42


def test_fixture_files_exist() -> None:
    assert TRANSACTION_FIXTURE.exists(), "run `make sample`"
    assert IDENTITY_FIXTURE.exists(), "run `make sample`"


def test_fixture_has_the_full_schema(
    sample_transactions: pd.DataFrame, sample_identity: pd.DataFrame
) -> None:
    assert list(sample_transactions.columns) == TRANSACTION_COLUMNS
    assert list(sample_identity.columns) == IDENTITY_COLUMNS
    assert len(sample_transactions) == FIXTURE_ROWS


def test_fixture_matches_the_generator(sample_transactions: pd.DataFrame) -> None:
    """Regenerating with the documented arguments reproduces the committed file."""
    expected, _ = generate_transactions(n_rows=FIXTURE_ROWS, seed=FIXTURE_SEED)

    pd.testing.assert_series_equal(
        sample_transactions["TransactionAmt"], expected["TransactionAmt"]
    )
    assert sample_transactions["isFraud"].sum() == expected["isFraud"].sum()
