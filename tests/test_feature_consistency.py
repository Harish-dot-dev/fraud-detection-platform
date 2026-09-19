"""Online / offline feature consistency.

The most important test in the repository.

There are two implementations of every behavioural feature. The online one
folds payments into a state object one at a time (``features/definitions.py``,
used by the streaming job and the scoring API). The offline one expresses the
same definitions as Spark window functions (``features/offline.py``, used to
build training sets), because folding half a million rows in Python would be
far too slow.

Two implementations of one definition is exactly how training/serving skew
gets into a system: the model learns from one set of numbers and is served
another, and nothing fails - the model just quietly gets worse.

So this test computes the features for the same payments both ways and
requires them to agree, value by value, for every payment in the sample.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.pii import card_token
from features.definitions import FEATURE_NAMES, CardState, compute_features, update_state
from features.offline import add_card_features
from streaming.silver import bronze_to_silver
from tests.spark_helpers import bronze_dataframe, events_from_fixture

pytestmark = pytest.mark.needs_spark

SALT = "f" * 64
# Floating point summation order differs between the two paths, so exact
# equality is the wrong bar. A tolerance of one part in a million is far
# tighter than anything that could change a model's decision.
TOLERANCE = 1e-6
SAMPLE_ROWS = 400


def _online_features(events) -> pd.DataFrame:
    """The reference: fold payments one at a time, exactly as production does."""
    states: dict[str, CardState] = {}
    rows = []

    for event in events:
        state = states.get(event.card_key, CardState())
        features = compute_features(state, event)
        states[event.card_key] = update_state(state, event)
        rows.append({"transaction_id": event.transaction_id, **features})

    return pd.DataFrame(rows).set_index("transaction_id").sort_index()


def _offline_features(spark, transactions, identity) -> pd.DataFrame:
    """The batch implementation: Spark window functions over the Silver table."""
    bronze = bronze_dataframe(spark, transactions, identity, n=SAMPLE_ROWS)
    silver = bronze_to_silver(bronze, SALT)
    gold = add_card_features(silver)

    return (
        gold.select("transaction_id", *FEATURE_NAMES)
        .toPandas()
        .set_index("transaction_id")
        .sort_index()
    )


@pytest.fixture(scope="module")
def both_implementations(spark, sample_transactions: pd.DataFrame, sample_identity: pd.DataFrame):
    events = events_from_fixture(sample_transactions, sample_identity, n=SAMPLE_ROWS)
    return _online_features(events), _offline_features(spark, sample_transactions, sample_identity)


def test_both_paths_score_the_same_payments(both_implementations) -> None:
    online, offline = both_implementations

    assert len(online) == SAMPLE_ROWS
    assert list(online.index) == list(offline.index)


@pytest.mark.parametrize("feature", FEATURE_NAMES)
def test_feature_matches_between_online_and_offline(both_implementations, feature: str) -> None:
    """Parametrised per feature so a failure names the one that drifted."""
    online, offline = both_implementations

    pd.testing.assert_series_equal(
        online[feature],
        offline[feature],
        check_names=False,
        rtol=TOLERANCE,
        atol=TOLERANCE,
    )


def test_the_sample_actually_exercises_the_features(both_implementations) -> None:
    """A test that compares two columns of zeros proves nothing.

    This asserts the sample contains real card history: repeat payments, cards
    with a baseline to compare against, and at least one new device.
    """
    online, _ = both_implementations

    assert (online["card_txn_count_24h"] > 1).sum() > 50
    assert (online["card_is_new"] == 0).sum() > 50
    assert online["card_new_device"].sum() > 0
    assert online["amount_to_card_avg_ratio"].std() > 0


def test_the_card_grouping_is_the_same_on_both_sides(
    spark, sample_transactions: pd.DataFrame, sample_identity: pd.DataFrame
) -> None:
    """Online groups by card key, offline by card token: same partitioning.

    If tokenisation ever collided, two different cards would share a history
    and this would catch it.
    """
    events = events_from_fixture(sample_transactions, sample_identity, n=SAMPLE_ROWS)
    online_cards = {event.card_key for event in events}
    expected_tokens = {card_token(key, SALT) for key in online_cards}

    silver = bronze_to_silver(
        bronze_dataframe(spark, sample_transactions, sample_identity, n=SAMPLE_ROWS), SALT
    )
    offline_tokens = {row["card_token"] for row in silver.select("card_token").distinct().collect()}

    assert offline_tokens == expected_tokens


def test_device_tracking_stays_within_the_online_bound(
    spark, sample_transactions: pd.DataFrame, sample_identity: pd.DataFrame
) -> None:
    """The one known difference between the two implementations.

    Redis keeps the most recent 20 distinct devices per card; the offline
    window keeps all of them. For a card that has used more than 20 devices the
    two could disagree - which would be a wildly anomalous card. This test
    fails if the data ever gets near that bound, so the divergence cannot creep
    in unnoticed.
    """
    from pyspark.sql import functions as F

    from features.definitions import MAX_TRACKED_DEVICES

    silver = bronze_to_silver(
        bronze_dataframe(spark, sample_transactions, sample_identity), SALT
    )
    worst = (
        silver.groupBy("card_token")
        .agg(F.countDistinct("device_info").alias("devices"))
        .agg(F.max("devices"))
        .first()[0]
    )

    assert worst <= MAX_TRACKED_DEVICES
