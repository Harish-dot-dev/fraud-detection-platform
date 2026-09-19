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


@pytest.fixture(scope="session")
def spark():
    """One Spark session for the whole test run, with Delta enabled.

    Session-scoped on purpose. Spark resolves ``spark.jars.packages`` when the
    JVM starts and cannot change the classpath afterwards, so a second session
    built with different packages in the same process silently reuses the
    first one's classpath - which shows up later as a baffling
    ClassNotFoundException for the Delta catalog.
    """
    from common.spark import build_spark_session

    session = build_spark_session("tests", master="local[1]")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
