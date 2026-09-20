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


@pytest.fixture
def make_event():
    """Factory for payment events in tests.

    The defaults describe an *ordinary* payment - a Visa debit card, a common
    email domain, the most frequent product code - on a single card at a fixed
    time, so a test only has to state the thing it actually cares about.

    That matters more than it sounds: an event with every attribute missing is
    genuinely unusual to the model, and a test built on one would be asserting
    against an anomaly without meaning to.
    """
    from datetime import UTC, datetime

    from common.events import PaymentEvent

    base_time = datetime(2023, 6, 1, 12, 0, 0, tzinfo=UTC)
    counter = {"n": 0}

    def _make(
        at: datetime | None = None,
        amount: float = 100.0,
        card_key: str = "13926|315|gmail.com",
        device_info: str | None = None,
        r_emaildomain: str | None = None,
        **overrides,
    ) -> PaymentEvent:
        counter["n"] += 1
        moment = at or base_time
        defaults = {
            # The C block (counts of addresses/phones/emails linked to the
            # card) and D block ("days since") are populated on every real
            # payment, and the model was trained with them present. An event
            # without them looks like a data outage, not a normal payment.
            "counts": {f"C{i}": float(i % 3) for i in range(1, 15)},
            "deltas": {"D1": 30.0, "D2": 12.0, "D3": 5.0},
            "product_cd": "W",
            "card3": 150.0,
            "card4": "visa",
            "card5": 226.0,
            "card6": "debit",
            "p_emaildomain": "gmail.com",
        }
        defaults.update(overrides)
        return PaymentEvent(
            transaction_id=3_000_000 + counter["n"],
            event_time=moment,
            transaction_dt=int(moment.timestamp()),
            card_key=card_key,
            amount=amount,
            device_info=device_info,
            r_emaildomain=r_emaildomain,
            **defaults,
        )

    return _make
