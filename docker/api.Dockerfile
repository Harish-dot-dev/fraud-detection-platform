# Scoring API image.
#
# Deliberately lean: no torch, no Spark, no sentence-transformers. This container
# sits on the latency-critical path, so it installs the core dependency group
# plus the ML group (XGBoost + SHAP are needed to score and explain), and nothing
# else. Starting it should take seconds, not minutes.

FROM python:3.11.10-slim-bookworm

# libgomp1 is the OpenMP runtime that XGBoost links against at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy only the dependency manifest first so that Docker's layer cache survives
# ordinary source edits.
COPY pyproject.toml README.md ./
# The package directories must exist for setuptools to resolve the project.
RUN mkdir -p features serving producer streaming training genai eval \
    && touch features/__init__.py serving/__init__.py producer/__init__.py \
       streaming/__init__.py training/__init__.py genai/__init__.py eval/__init__.py \
    && pip install --no-cache-dir ".[ml]"

COPY . .

EXPOSE 8000

# Compose overrides this with --reload for development.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
