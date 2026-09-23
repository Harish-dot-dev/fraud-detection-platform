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
COPY pyproject.toml README.md docker/stub_packages.py ./
# setuptools requires every directory in [tool.setuptools] packages to exist,
# even for a dependency-only install. The list is read from pyproject rather
# than repeated here, because repeating it is what broke this build for six
# phases - see docker/stub_packages.py.
RUN python stub_packages.py \
    && pip install --no-cache-dir ".[ml,app]" "psycopg[binary]==3.2.3" "pgvector==0.3.6" \
    && rm stub_packages.py

COPY . .

EXPOSE 8000

# Compose overrides this with --reload for development.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
