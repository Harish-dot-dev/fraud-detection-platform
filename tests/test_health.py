"""Smoke tests for the scoring API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from serving.app import API_VERSION, app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == API_VERSION
    # No model is registered until phase 5.
    assert body["model_loaded"] is False


def test_openapi_schema_is_served() -> None:
    """The /docs page is the quickest way for a reviewer to try the API."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
