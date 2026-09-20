# Airflow image with the platform's dependencies alongside it.
#
# The DAGs run every task as a subprocess (`python -m ...`), so this image
# needs both Airflow and the platform's own dependencies - including a JVM,
# because the medallion tasks are Spark jobs.
#
# That makes it the largest image in the stack by some way. The alternative -
# having Airflow drive the other containers through DockerOperator - needs the
# Docker socket mounted into a container, which is a meaningful security
# trade-off for a laptop demo.

FROM apache/airflow:2.10.3-python3.11

USER root
# Java 17 for Spark (3.5 does not support 21); libgomp1 for XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless procps libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

USER airflow
WORKDIR /opt/fraud-platform

COPY --chown=airflow:root pyproject.toml README.md ./
RUN mkdir -p features serving producer streaming training genai eval common warehouse quality \
    && touch features/__init__.py serving/__init__.py producer/__init__.py \
       streaming/__init__.py training/__init__.py genai/__init__.py eval/__init__.py \
       common/__init__.py warehouse/__init__.py quality/__init__.py \
    && pip install --no-cache-dir ".[spark,ml,quality,dbt]"

COPY --chown=airflow:root . .
