"""The scoring API.

    POST /score   a payment in, a decision out
    GET  /health  liveness, plus which model version is serving

The request body is the same ``PaymentEvent`` that travels on the Kafka topic,
so the consumer forwards messages unchanged and there is no second schema to
keep in step.

Everything expensive is built once at startup - the Redis client, the rules,
the model and its SHAP explainer - because building any of them per request
would dominate the latency budget.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, ConfigDict

from common.config import REPO_ROOT, get_settings
from common.events import PaymentEvent
from features.store import FeatureStore, build_redis_client
from serving.decisions import Decision, DecisionSink, JsonlDecisionSink, KafkaDecisionSink
from serving.model import ModelBundle, load_champion
from serving.rules import RulesEngine
from serving.scoring import ScoringService

logger = logging.getLogger("serving.app")

API_VERSION = "0.1.0"

_service: ScoringService | None = None


def build_service(
    bundle: ModelBundle | None = None, sink: DecisionSink | None = None
) -> ScoringService:
    """Assemble the scoring service from configuration."""
    settings = get_settings()

    store = FeatureStore(
        build_redis_client(settings.redis_host, settings.redis_port),
        salt=settings.pii_hash_salt,
        ttl_seconds=settings.redis_feature_ttl_seconds,
    )
    rules = RulesEngine.from_file()
    bundle = bundle or load_champion(
        tracking_uri=settings.mlflow_tracking_uri,
        registered_model=settings.mlflow_registered_model,
        alias=settings.mlflow_champion_alias,
        fallback_review_threshold=settings.threshold_review,
        fallback_block_threshold=settings.threshold_block,
    )
    return ScoringService(store=store, rules=rules, bundle=bundle, sink=sink or _default_sink())


def _default_sink() -> DecisionSink:
    """Kafka when it is reachable, a local JSONL file when it is not.

    A demo on a laptop with no broker should still leave a complete audit
    trail rather than silently discarding decisions.
    """
    settings = get_settings()
    try:
        return KafkaDecisionSink(settings.kafka_bootstrap_servers, settings.kafka_topic_decisions)
    except Exception as error:  # noqa: BLE001
        path = REPO_ROOT / "data" / "decisions" / "decisions.jsonl"
        logger.warning("Kafka unavailable (%s); writing decisions to %s", error, path)
        return JsonlDecisionSink(path)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Build the service once, tear it down cleanly."""
    global _service

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # A service injected before startup (tests, tests/demo_server.py) is left
    # alone: building the real one would reach for Redis and the registry.
    if _service is None:
        _service = build_service()
    logger.info(
        "scoring service ready (model %s from %s)",
        _service.bundle.version,
        _service.bundle.source,
    )
    yield
    _service.sink.close()


app = FastAPI(
    title="Fraud Scoring API",
    version=API_VERSION,
    summary="Rules + XGBoost scoring for payment authorisation decisions.",
    lifespan=lifespan,
)


def get_service() -> ScoringService:
    """FastAPI dependency; overridden in tests."""
    if _service is None:  # pragma: no cover - only before startup completes
        raise RuntimeError("the scoring service is not ready yet")
    return _service


# Annotated rather than a `Depends(...)` default: same behaviour, and it keeps
# the callable out of the function signature's default values.
Service = Annotated[ScoringService, Depends(get_service)]


class HealthResponse(BaseModel):
    """Liveness payload, also used by the container healthcheck."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    version: str
    pii_salt_configured: bool
    model_loaded: bool
    model_version: str
    model_source: str
    review_threshold: float
    block_threshold: float


class ScoreResponse(BaseModel):
    """What the caller gets back. Mirrors the audit record."""

    model_config = ConfigDict(protected_namespaces=())

    transaction_id: int
    decision: str
    score: float | None
    triggered_by: str
    reason: str
    rule_name: str | None
    top_reasons: list[dict[str, Any]]
    model_version: str
    latency_ms: float
    degraded: bool

    @classmethod
    def from_decision(cls, decision: Decision) -> ScoreResponse:
        return cls(
            transaction_id=decision.transaction_id,
            decision=decision.decision,
            score=decision.score,
            triggered_by=decision.triggered_by,
            reason=decision.reason,
            rule_name=decision.rule_name,
            top_reasons=decision.top_reasons,
            model_version=decision.model_version,
            latency_ms=round(decision.latency_ms, 3),
            degraded=decision.degraded,
        )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(service: Service) -> HealthResponse:
    """Report liveness and which model is serving."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        pii_salt_configured=not settings.uses_default_pii_salt,
        model_loaded=service.bundle.is_loaded,
        model_version=service.bundle.version,
        model_source=service.bundle.source,
        review_threshold=service.bundle.review_threshold,
        block_threshold=service.bundle.block_threshold,
    )


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score(event: PaymentEvent, service: Service) -> ScoreResponse:
    """Score one payment: allow, review or block."""
    return ScoreResponse.from_decision(service.score(event))
