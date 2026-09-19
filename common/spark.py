"""Spark session construction, in one place.

Every Spark entry point in the project (the streaming Bronze job, the batch
Silver/Gold jobs, the tests) builds its session here, so the Delta extensions,
the UTC session timezone and the local-mode tuning are configured identically
everywhere. A streaming job and a batch job that disagree about the session
timezone will silently produce different windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cost only paid at runtime
    from pyspark.sql import SparkSession

# Pinned to match the pyspark version in pyproject.toml. A mismatch between the
# Kafka connector and Spark itself fails at runtime with an unhelpful error.
SPARK_VERSION = "3.5.3"
SCALA_BINARY_VERSION = "2.12"
KAFKA_PACKAGE = f"org.apache.spark:spark-sql-kafka-0-10_{SCALA_BINARY_VERSION}:{SPARK_VERSION}"


def build_spark_session(
    app_name: str,
    packages: list[str] | None = None,
    master: str | None = "local[*]",
    extra_conf: dict[str, str] | None = None,
    with_delta: bool = True,
) -> SparkSession:
    """Create a Spark session configured for this project.

    Args:
        app_name: shows up in the Spark UI and in logs.
        packages: extra Maven coordinates (e.g. the Kafka connector).
        master: ``None`` leaves it to spark-submit / the environment.
        extra_conf: additional Spark configuration.
        with_delta: register the Delta Lake SQL extensions and catalog.
    """
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)

    # Everything in this platform is UTC. Timestamps are derived from a
    # synthetic epoch (common/timeline.py) and any local-time conversion would
    # silently shift the feature windows.
    builder = builder.config("spark.sql.session.timeZone", "UTC")
    # Laptop-scale data: the default 200 shuffle partitions create hundreds of
    # tiny tasks and tiny files for no benefit.
    builder = builder.config("spark.sql.shuffle.partitions", "4")
    builder = builder.config("spark.sql.adaptive.enabled", "true")

    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    if with_delta:
        # Delta needs both halves: the SQL extension and the catalog override.
        # configure_spark_with_delta_pip then resolves the jars that match the
        # installed delta-spark pin, so the Python and JVM versions cannot drift.
        builder = builder.config(
            "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
        ).config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        builder = configure_spark_with_delta_pip(builder, extra_packages=packages or [])
    elif packages:
        builder = builder.config("spark.jars.packages", ",".join(packages))

    return builder.getOrCreate()
