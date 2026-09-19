"""Point-in-time correctness: the leakage tests.

Data leakage is the defect that makes a fraud model look excellent offline and
fail in production, and it has two entrances:

1. **Feature leakage** - a feature computed from payments that happened after
   the one being scored.
2. **Label leakage** - training on a label that had not arrived yet.

The second is the subtle one in fraud, because the label genuinely does not
exist for weeks. A model trained "as of today" on labels that only became
known a month later has been shown the future.

Each test here demonstrates the failure it prevents, rather than just asserting
the happy path.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from features.definitions import FEATURE_NAMES
from features.offline import build_gold
from streaming.silver import bronze_to_silver
from tests.spark_helpers import bronze_dataframe
from training.chargebacks import build_chargebacks
from training.dataset import (
    IMMATURE_ASSUME_LEGITIMATE,
    IMMATURE_EXCLUDE,
    build_training_set,
    describe_training_set,
)

pytestmark = pytest.mark.needs_spark


def naive(moment):
    """Spark returns timezone-naive timestamps in the (UTC) session timezone.

    The pandas side is timezone-aware, so comparisons in these tests strip the
    tzinfo rather than pretend the two are interchangeable.
    """
    return moment.replace(tzinfo=None)


SALT = "2" * 64
SAMPLE_ROWS = 300


@pytest.fixture(scope="module")
def gold(spark, sample_transactions: pd.DataFrame, sample_identity: pd.DataFrame):
    bronze = bronze_dataframe(spark, sample_transactions, sample_identity, n=SAMPLE_ROWS)
    return build_gold(bronze_to_silver(bronze, SALT)).cache()


@pytest.fixture(scope="module")
def labels(spark, sample_transactions: pd.DataFrame):
    return spark.createDataFrame(build_chargebacks(sample_transactions.head(SAMPLE_ROWS)))


@pytest.fixture(scope="module")
def last_payment(sample_transactions: pd.DataFrame):
    return build_chargebacks(sample_transactions.head(SAMPLE_ROWS))["event_time"].max()


# --- Label leakage ----------------------------------------------------------


def test_on_the_day_of_the_last_payment_there_is_nothing_to_train_on(
    gold, labels, last_payment
) -> None:
    """Not a bug - the point.

    Every payment in the fixture is less than a day old, and no dispute comes
    back in under a week. A model trained today has nothing from this week to
    learn from, and the builder says so instead of inventing labels.
    """
    training_set = build_training_set(gold, labels, as_of=last_payment)

    assert training_set.count() == 0


def test_only_labels_that_had_arrived_are_included(gold, labels, last_payment) -> None:
    """The core point-in-time invariant, asserted row by row."""
    as_of = last_payment + timedelta(days=30)

    training_set = build_training_set(gold, labels, as_of=as_of)

    rows = training_set.select("event_time", "label_available_at").collect()
    assert rows, "expected some matured labels 30 days later"
    assert all(row["label_available_at"] <= naive(as_of) for row in rows)
    assert all(row["event_time"] <= naive(as_of) for row in rows)


def test_a_chargeback_that_has_not_arrived_yet_is_excluded(gold, labels, last_payment) -> None:
    """The specific row a naive join would leak.

    This fraud is confirmed 50-ish days after the payment. Thirty days in,
    nobody knows about it, so it must not be in a model trained then - even
    though the dataset "knows" the answer.
    """
    from pyspark.sql import functions as F

    as_of = last_payment + timedelta(days=30)
    late = (
        labels.filter((F.col("is_fraud") == 1) & (F.col("label_available_at") > as_of))
        .select("transaction_id")
        .first()
    )
    assert late is not None, "fixture should contain a fraud confirmed later than 30 days"

    training_set = build_training_set(gold, labels, as_of=as_of)

    assert training_set.filter(F.col("transaction_id") == late["transaction_id"]).count() == 0


def test_the_same_payment_appears_once_its_label_has_arrived(gold, labels, last_payment) -> None:
    """...and the row is not lost forever, just deferred."""
    from pyspark.sql import functions as F

    as_of = last_payment + timedelta(days=30)
    late = (
        labels.filter((F.col("is_fraud") == 1) & (F.col("label_available_at") > as_of))
        .select("transaction_id")
        .first()
    )

    later = build_training_set(gold, labels, as_of=last_payment + timedelta(days=61))

    row = later.filter(F.col("transaction_id") == late["transaction_id"]).first()
    assert row is not None
    assert row["is_fraud"] == 1


def test_assuming_legitimate_uses_more_data_and_mislabels_it(gold, labels, last_payment) -> None:
    """The alternative policy, and the bias it buys.

    Including immature payments as non-fraud is what a system that trusts "no
    dispute yet" effectively does. It gets more rows - and labels genuine fraud
    as legitimate. The rows are flagged so the trade is visible.
    """
    from pyspark.sql import functions as F

    as_of = last_payment + timedelta(days=30)

    strict = build_training_set(gold, labels, as_of, immature_policy=IMMATURE_EXCLUDE)
    permissive = build_training_set(gold, labels, as_of, immature_policy=IMMATURE_ASSUME_LEGITIMATE)

    assert permissive.count() > strict.count()
    assumed = permissive.filter(F.col("label_is_assumed"))
    assert assumed.count() > 0
    # Every assumed row is labelled non-fraud, and some of them are not.
    assert assumed.filter(F.col("is_fraud") == 1).count() == 0
    truly_fraudulent = labels.filter(F.col("label_available_at") > as_of).filter(
        F.col("is_fraud") == 1
    )
    assert truly_fraudulent.count() > 0


def test_payments_after_the_as_of_date_are_never_included(gold, labels) -> None:
    """A wrong as_of must not quietly pull in the future."""
    from pyspark.sql import functions as F

    early = gold.agg(F.min("event_time")).first()[0] + timedelta(hours=1)

    training_set = build_training_set(gold, labels, as_of=early + timedelta(days=90))

    assert training_set.count() > 0
    training_set = build_training_set(gold, labels, as_of=early)
    assert training_set.count() == 0


def test_an_unknown_policy_is_rejected(gold, labels, last_payment) -> None:
    with pytest.raises(ValueError, match="immature_policy"):
        build_training_set(gold, labels, as_of=last_payment, immature_policy="whatever")


# --- Feature leakage --------------------------------------------------------


def test_features_do_not_change_when_later_payments_arrive(
    spark, sample_transactions: pd.DataFrame, sample_identity: pd.DataFrame
) -> None:
    """The direct test for feature leakage.

    Build Gold over the first 150 payments, then over the first 300, and
    compare the first 150 rows. If any feature looked forward - a card average
    over the whole file, a count with no upper bound - the numbers would move
    when later payments were added. They must be identical.
    """
    smaller = build_gold(
        bronze_to_silver(bronze_dataframe(spark, sample_transactions, sample_identity, n=150), SALT)
    )
    larger = build_gold(
        bronze_to_silver(bronze_dataframe(spark, sample_transactions, sample_identity, n=300), SALT)
    )

    first = smaller.select("transaction_id", *FEATURE_NAMES).toPandas().set_index("transaction_id")
    second = (
        larger.select("transaction_id", *FEATURE_NAMES)
        .toPandas()
        .set_index("transaction_id")
        .loc[first.index]
    )

    pd.testing.assert_frame_equal(first.sort_index(), second.sort_index())


def test_the_training_set_carries_every_model_feature(gold, labels, last_payment) -> None:
    training_set = build_training_set(gold, labels, as_of=last_payment + timedelta(days=61))

    for feature in FEATURE_NAMES:
        assert feature in training_set.columns
    assert "is_fraud" in training_set.columns


def test_the_summary_describes_what_was_built(gold, labels, last_payment) -> None:
    training_set = build_training_set(gold, labels, as_of=last_payment + timedelta(days=61))

    summary = describe_training_set(training_set)

    assert summary["rows"] == SAMPLE_ROWS
    assert summary["fraud"] > 0
    assert 0 < summary["fraud_rate"] < 0.2
    assert summary["assumed_labels"] == 0
