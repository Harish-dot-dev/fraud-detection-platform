"""Tests for the Bronze -> Silver transformation.

Silver is where the platform's data-protection promise is kept, so most of
these tests are about what is *not* in the output.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.pii import card_token
from streaming.silver import IDENTIFYING_COLUMNS, bronze_to_silver
from tests.spark_helpers import bronze_dataframe

pytestmark = pytest.mark.needs_spark

SALT = "e" * 64


@pytest.fixture(scope="module")
def silver(spark, sample_transactions: pd.DataFrame, sample_identity: pd.DataFrame):
    bronze = bronze_dataframe(spark, sample_transactions, sample_identity, n=200)
    return bronze_to_silver(bronze, SALT).cache()


def test_every_valid_payment_survives(silver, sample_transactions: pd.DataFrame) -> None:
    expected = (sample_transactions.head(200)["TransactionAmt"] > 0).sum()

    assert silver.count() == expected


def test_raw_card_identifiers_are_gone(silver) -> None:
    """The whole point of tokenising: these columns must not exist downstream."""
    for column in IDENTIFYING_COLUMNS:
        assert column not in silver.columns


def test_the_card_token_matches_the_scoring_api(
    silver, spark, sample_transactions: pd.DataFrame
) -> None:
    """Offline and online must tokenise identically or the join is meaningless."""
    from common.events import payment_event_from_row

    first = sample_transactions.iloc[0]
    event = payment_event_from_row(first.to_dict())
    expected_token = card_token(event.card_key, SALT)

    row = silver.filter(silver.transaction_id == int(first["TransactionID"])).first()
    assert row["card_token"] == expected_token


def test_tokens_are_stable_across_rows_of_the_same_card(silver) -> None:
    """Per-card features depend on this; if it broke, every card would look new."""
    from pyspark.sql import functions as F

    per_card = silver.groupBy("card_token").agg(F.count("*").alias("n"))

    assert per_card.filter(F.col("n") > 1).count() > 0
    # Tokens are 32 hex characters.
    assert all(len(row["card_token"]) == 32 for row in silver.select("card_token").take(20))


def test_duplicate_payments_are_removed(spark, sample_transactions: pd.DataFrame) -> None:
    """A Kafka replay must not double a card's transaction count."""
    bronze = bronze_dataframe(spark, sample_transactions, n=50)
    doubled = bronze.union(bronze)

    result = bronze_to_silver(doubled, SALT)

    assert doubled.count() == 100
    assert result.count() == 50


def test_non_positive_amounts_are_dropped(spark, sample_transactions: pd.DataFrame) -> None:
    """A zero amount is a data problem, and it would poison the card averages."""
    from pyspark.sql import functions as F

    bronze = bronze_dataframe(spark, sample_transactions, n=20)
    with_zero = bronze.withColumn(
        "amount", F.when(F.col("kafka_offset") == 0, F.lit(0.0)).otherwise(F.col("amount"))
    )

    result = bronze_to_silver(with_zero, SALT)

    assert result.count() == 19


def test_useful_payment_attributes_are_kept(silver) -> None:
    """Tokenisation should not cost the model everything it can legitimately use."""
    for column in ("product_cd", "card4", "card6", "p_emaildomain", "counts", "device_type"):
        assert column in silver.columns


def test_event_date_is_derived_for_partitioning(silver) -> None:
    row = silver.first()

    assert row["event_date"] == row["event_time"].date()
