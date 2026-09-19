"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
TRANSACTION_FIXTURE = FIXTURE_DIR / "train_transaction_sample.csv"
IDENTITY_FIXTURE = FIXTURE_DIR / "train_identity_sample.csv"


@pytest.fixture(scope="session")
def sample_transactions() -> pd.DataFrame:
    """The committed synthetic stand-in for train_transaction.csv."""
    return pd.read_csv(TRANSACTION_FIXTURE)


@pytest.fixture(scope="session")
def sample_identity() -> pd.DataFrame:
    """The committed synthetic stand-in for train_identity.csv."""
    return pd.read_csv(IDENTITY_FIXTURE)
