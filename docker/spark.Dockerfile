# Spark jobs image (streaming Bronze now, batch Silver/Gold later).
#
# Spark 3.5 supports Java 8, 11 and 17 - not 21 - so the JRE is pinned
# explicitly rather than inherited from whatever the base image happens to ship.

FROM python:3.11.10-slim-bookworm

# libgomp1 is the OpenMP runtime XGBoost links against; the training job runs
# in this image too.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless procps curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Without this Spark tries to resolve the container hostname and logs a
    # warning storm before falling back to a loopback address anyway.
    SPARK_LOCAL_IP=127.0.0.1

WORKDIR /app

COPY pyproject.toml README.md docker/stub_packages.py ./
# See docker/stub_packages.py: the package list is read from pyproject rather
# than duplicated here, so it cannot drift out of sync again.
RUN python stub_packages.py \
    && pip install --no-cache-dir ".[spark,quality,ml]" \
    && rm stub_packages.py

# Resolve the Delta and Kafka connector jars at build time so that starting a
# job does not depend on Maven Central being reachable (and does not spend the
# first 90 seconds of every run downloading).
COPY common/spark.py common/spark.py
RUN python -c "\
from delta import configure_spark_with_delta_pip; \
from pyspark.sql import SparkSession; \
b = SparkSession.builder.appName('warm-jars').master('local[1]'); \
s = configure_spark_with_delta_pip(b, extra_packages=['org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3']).getOrCreate(); \
s.stop()"

COPY . .

# Overridden by docker-compose; this is the streaming Bronze ingest.
CMD ["python", "-m", "streaming.bronze"]
