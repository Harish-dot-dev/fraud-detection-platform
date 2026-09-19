"""Scoring API.

Phase 1 ships only the health endpoint, so that the container in
docker-compose.yml starts something real rather than a placeholder. The /score
endpoint, the rules engine and the model loader arrive in phase 6.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from common.config import get_settings

API_VERSION = "0.1.0"

app = FastAPI(
    title="Fraud Scoring API",
    version=API_VERSION,
    summary="Rules + XGBoost scoring for payment authorisation decisions.",
)


class HealthResponse(BaseModel):
    """Liveness payload, also used by the container healthcheck."""

    # "model_loaded" collides with pydantic's protected "model_" namespace;
    # the field name is worth keeping, so the protection is switched off here.
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    version: str
    # Surfaced so a demo cannot silently run with the example PII salt.
    pii_salt_configured: bool
    # Populated once a model is registered (phase 5).
    model_loaded: bool


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Report service liveness and basic configuration state."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        pii_salt_configured=not settings.uses_default_pii_salt,
        model_loaded=False,
    )
