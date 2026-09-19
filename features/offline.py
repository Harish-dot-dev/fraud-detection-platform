"""Offline (batch) feature computation for the Gold layer.

This is the *second* implementation of the features defined in
``features/definitions.py``, and the duplication is on purpose.

The online path folds payments into a state object one at a time, which is the
only thing that makes sense when a payment arrives on its own. Doing that row
by row over half a million historical rows in Python would take minutes and
would not parallelise, so the offline path expresses the same definitions as
Spark window functions instead.

Two implementations of the same thing is a classic source of training/serving
skew, so ``tests/test_feature_consistency.py`` computes a sample both ways and
requires them to agree. That test is the contract between these two files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from features.definitions import (
    FEATURE_NAMES,
    NO_HISTORY,
    WINDOW_1H,
    WINDOW_10M,
    WINDOW_24H,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

# Columns carried through to Gold unchanged: available at authorisation time
# and useful to the model, but needing no history to compute.
PASSTHROUGH_COLUMNS = [
    "product_cd",
    "card3",
    "card4",
    "card5",
    "card6",
    "dist1",
    "dist2",
    "p_emaildomain",
    "r_emaildomain",
    "device_type",
]


def add_card_features(silver: DataFrame) -> DataFrame:
    """Add every feature in :data:`FEATURE_NAMES` to a Silver DataFrame.

    Each window is defined over ``event_seconds`` and ends at the current row,
    so a row can only ever see payments at or before its own timestamp. That is
    what makes the offline features point-in-time correct - the same property
    the online path gets for free by only having past state in hand.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    frame = silver.withColumn("event_seconds", F.col("event_time").cast("long"))

    def trailing(window_seconds: int) -> Window:
        # rangeBetween is inclusive at both ends, matching the half-open
        # (now - window, now] convention in features/definitions.py.
        return (
            Window.partitionBy("card_token")
            .orderBy("event_seconds")
            .rangeBetween(-window_seconds, Window.currentRow)
        )

    # Everything strictly before the current payment. Ordered by time, then by
    # transaction_id so that two payments in the same second are still ordered
    # deterministically - otherwise "lifetime" aggregates would depend on the
    # shuffle.
    history = (
        Window.partitionBy("card_token")
        .orderBy("event_seconds", "transaction_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    previous_event_seconds = F.lag("event_seconds").over(
        Window.partitionBy("card_token").orderBy("event_seconds", "transaction_id")
    )
    lifetime_count = F.count("*").over(history)
    lifetime_avg = F.coalesce(F.avg("amount").over(history), F.lit(0.0))
    prior_devices = F.collect_set("device_info").over(history)
    prior_domains = F.collect_set("r_emaildomain").over(history)

    return (
        frame.withColumn(
            "card_txn_count_10m",
            F.count("*").over(trailing(int(WINDOW_10M.total_seconds()))).cast("double"),
        )
        .withColumn(
            "card_txn_count_1h",
            F.count("*").over(trailing(int(WINDOW_1H.total_seconds()))).cast("double"),
        )
        .withColumn(
            "card_txn_count_24h",
            F.count("*").over(trailing(int(WINDOW_24H.total_seconds()))).cast("double"),
        )
        .withColumn(
            "card_amount_sum_24h",
            F.sum("amount").over(trailing(int(WINDOW_24H.total_seconds()))).cast("double"),
        )
        .withColumn(
            "seconds_since_card_last_txn",
            F.coalesce(
                (F.col("event_seconds") - previous_event_seconds).cast("double"),
                F.lit(NO_HISTORY),
            ),
        )
        .withColumn("card_txn_count_lifetime", lifetime_count.cast("double"))
        .withColumn("card_amount_avg_lifetime", lifetime_avg)
        .withColumn(
            "amount_to_card_avg_ratio",
            F.when(lifetime_avg > 0, F.col("amount") / lifetime_avg).otherwise(F.lit(1.0)),
        )
        .withColumn("card_is_new", (lifetime_count == 0).cast("double"))
        .withColumn(
            "card_new_device",
            (
                F.col("device_info").isNotNull()
                & (lifetime_count > 0)
                & ~F.array_contains(prior_devices, F.col("device_info"))
            ).cast("double"),
        )
        .withColumn(
            "card_new_email_domain",
            (
                F.col("r_emaildomain").isNotNull()
                & (lifetime_count > 0)
                & ~F.array_contains(prior_domains, F.col("r_emaildomain"))
            ).cast("double"),
        )
        .withColumn("card_distinct_devices", F.size(prior_devices).cast("double"))
        # --- Stateless features (must match features.definitions) ---
        .withColumn("amount_log", F.log1p(F.greatest(F.col("amount"), F.lit(0.0))))
        .withColumn("hour_of_day", F.hour("event_time").cast("double"))
        # Spark's dayofweek is 1=Sunday; Python's weekday() is 0=Monday.
        .withColumn("day_of_week", ((F.dayofweek("event_time") + 5) % 7).cast("double"))
        .withColumn("is_night", (F.hour("event_time") < 6).cast("double"))
        .withColumn(
            "has_identity",
            (
                F.col("device_info").isNotNull()
                | (F.size(F.coalesce(F.map_keys("identity_numeric"), F.array())) > 0)
            ).cast("double"),
        )
        .drop("event_seconds")
    )


def build_gold(silver: DataFrame) -> DataFrame:
    """Produce the Gold feature table used to build training sets."""
    from pyspark.sql import functions as F

    features = add_card_features(silver)
    return features.select(
        "transaction_id",
        "event_time",
        "event_date",
        "transaction_dt",
        "card_token",
        *[F.col(name) for name in FEATURE_NAMES],
        *PASSTHROUGH_COLUMNS,
        "counts",
        "deltas",
        "match_flags",
    )
