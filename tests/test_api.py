"""Tests for the HTTP surface.

The scoring logic is covered in test_scoring.py; these are about the contract a
caller sees - status codes, payload shape, and the health endpoint telling the
truth about which model is serving.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import serving.app as app_module
from serving.decisions import ListDecisionSink


@pytest.fixture(scope="module")
def client():
    """A TestClient wired to an in-process service.

    The service is installed before the app starts, so the lifespan leaves it
    alone instead of reaching for Redis and the MLflow registry.
    """
    from tests.demo_server import build_demo_service

    app_module._service = build_demo_service()
    with TestClient(app_module.app) as test_client:
        yield test_client
    app_module._service = None


@pytest.fixture
def payment(make_event) -> dict:
    return make_event(amount=55.0).model_dump(mode="json")


def test_health_reports_the_serving_model(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "demo"
    assert 0.0 <= body["review_threshold"] <= body["block_threshold"] <= 1.0


def test_scoring_a_payment_returns_a_decision(client, payment) -> None:
    response = client.post("/score", json=payment)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in {"allow", "review", "block"}
    assert body["transaction_id"] == payment["transaction_id"]
    assert body["model_version"] == "demo"
    assert body["latency_ms"] > 0


def test_a_flagged_payment_comes_back_with_reasons(client, make_event) -> None:
    """What the analyst app and the LLM prompt are built from."""
    response = client.post("/score", json=make_event(amount=25_000.0).model_dump(mode="json"))

    body = response.json()
    assert body["decision"] in {"review", "block"}
    assert body["reason"]
    assert body["top_reasons"]


def test_the_response_says_which_rule_or_threshold_decided(client, payment) -> None:
    body = client.post("/score", json=payment).json()

    assert body["triggered_by"] in {"rule", "model", "default"}
    assert body["reason"]


def test_a_malformed_payment_is_rejected(client) -> None:
    response = client.post("/score", json={"amount": "not a number"})

    assert response.status_code == 422


def test_the_kafka_payload_scores_without_translation(client, make_event) -> None:
    """The consumer forwards topic messages unchanged; there is one schema."""
    raw = make_event(amount=88.0).model_dump_json()

    response = client.post("/score", content=raw, headers={"content-type": "application/json"})

    assert response.status_code == 200


def test_every_request_is_recorded_for_audit(client, payment) -> None:
    sink = app_module._service.sink
    assert isinstance(sink, ListDecisionSink)
    before = len(sink.decisions)

    client.post("/score", json=payment)

    assert len(sink.decisions) == before + 1
    assert sink.decisions[-1].features


def test_the_api_documents_itself(client) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/score" in paths
    assert "/health" in paths
