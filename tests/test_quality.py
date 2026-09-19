"""Tests for the Silver data quality suite.

A quality suite that has never been seen to fail is not a quality suite, so
each test here breaks the data in a specific way and checks that the suite
notices.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quality.expectations import validate_silver
from streaming.silver import bronze_to_silver
from tests.spark_helpers import bronze_dataframe

pytestmark = [pytest.mark.needs_spark, pytest.mark.slow]

SALT = "1" * 64


@pytest.fixture(scope="module")
def silver(spark, sample_transactions: pd.DataFrame, sample_identity: pd.DataFrame):
    bronze = bronze_dataframe(spark, sample_transactions, sample_identity, n=150)
    return bronze_to_silver(bronze, SALT).cache()


def test_clean_silver_passes(silver) -> None:
    result = validate_silver(silver)

    assert result.success, result.failures
    assert result.evaluated == result.successful
    assert "silver" in result.describe()


def test_a_leaked_card_identifier_fails_the_suite(silver) -> None:
    """The check that matters most: a data-protection regression, caught in CI.

    If someone adds card1 back into Silver, the column set no longer matches
    and the job refuses to write the table.
    """
    from pyspark.sql import functions as F

    leaked = silver.withColumn("card1", F.lit(13926))

    result = validate_silver(leaked)

    assert not result.success
    assert any("columns_to_match_set" in failure["expectation"] for failure in result.failures)


def test_duplicate_payments_fail_the_suite(silver) -> None:
    """Duplicates would silently corrupt every velocity feature."""
    result = validate_silver(silver.union(silver))

    assert not result.success
    assert any(failure["column"] == "transaction_id" for failure in result.failures)


def test_an_impossible_amount_fails_the_suite(silver) -> None:
    from pyspark.sql import functions as F

    broken = silver.withColumn(
        "amount",
        F.when(F.col("transaction_id") == silver.first()["transaction_id"], F.lit(-1.0)).otherwise(
            F.col("amount")
        ),
    )

    result = validate_silver(broken)

    assert not result.success
    assert any(failure["column"] == "amount" for failure in result.failures)


def test_an_untokenised_card_fails_the_suite(silver) -> None:
    """Proof the tokeniser ran, not just that the column is populated."""
    from pyspark.sql import functions as F

    result = validate_silver(silver.withColumn("card_token", F.lit("13926|315|gmail.com")))

    assert not result.success
    assert any(failure["column"] == "card_token" for failure in result.failures)


def test_the_result_is_saved_as_evidence(silver, tmp_path) -> None:
    """Every quality run leaves a file behind; Airflow publishes it in phase 7."""
    result = validate_silver(silver)

    path = result.save(tmp_path / "quality_silver.json")

    saved = json.loads(path.read_text())
    assert saved["success"] is True
    assert saved["expectations_evaluated"] == result.evaluated
    assert saved["failures"] == []
