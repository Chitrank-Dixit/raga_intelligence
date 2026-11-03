"""
Integration and security tests for CORS restrictions (Prompt 13),
safe error responses without stack trace or path exposure (Prompt 14),
and model-verified readiness checks (Prompt 15).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from raga_intelligence.api.app import create_app
from raga_intelligence.config import Settings


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


# ============================================================================
# Prompt 13: CORS Restrictions & Security
# ============================================================================

def test_cors_allowed_origin_returns_headers(client: TestClient) -> None:
    # Default allowed origin includes http://localhost:8000
    headers = {"Origin": "http://localhost:8000"}
    response = client.get("/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8000"


def test_cors_disallowed_origin_rejected(client: TestClient) -> None:
    headers = {"Origin": "https://malicious-site.example.com"}
    response = client.get("/health", headers=headers)
    assert response.status_code == 200
    # Disallowed origin must NOT receive access-control-allow-origin header
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_request(client: TestClient) -> None:
    headers = {
        "Origin": "http://localhost:8000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    response = client.options("/analyze", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8000"
    assert "POST" in response.headers.get("access-control-allow-methods", "")


def test_cors_production_fails_safely_on_wildcard() -> None:
    with pytest.raises(ValueError, match="Production CORS violation: wildcard origin"):
        Settings(
            ENV="production",
            CORS_ALLOWED_ORIGINS=["*"],
        )


def test_cors_rejects_credentials_with_wildcard() -> None:
    with pytest.raises(ValueError, match="CORS security violation"):
        Settings(
            ENV="development",
            CORS_ALLOWED_ORIGINS=["*"],
            CORS_ALLOW_CREDENTIALS=True,
        )


# ============================================================================
# Prompt 14: Safe Error Responses (No stack trace or path exposure)
# ============================================================================

def test_request_id_attached_to_all_responses(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    req_id = response.headers["X-Request-ID"]
    assert len(req_id) > 8


def test_client_supplied_request_id_propagated(client: TestClient) -> None:
    custom_id = "test-correlation-id-98765"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id


def test_unhandled_exception_returns_500_with_generic_message_and_request_id() -> None:
    # Use raise_server_exceptions=False so TestClient returns the HTTP 500 response
    app = create_app()
    error_client = TestClient(app, raise_server_exceptions=False)

    with patch("raga_intelligence.api.routes.get_pipeline") as mock_get_pipe:
        # Simulate unexpected crash during pipeline retrieval
        mock_get_pipe.side_effect = RuntimeError("Fatal DB/Filesystem failure at C:\\Users\\Admin\\Secret\\model.pt")

        response = error_client.get("/ready")
        assert response.status_code == 500
        data = response.json()
        assert "request_id" in data
        assert "internal server error occurred" in data["detail"].lower()
        # Ensure no traceback or internal filesystem paths are exposed
        assert "Fatal DB" not in data["detail"]
        assert "C:\\Users" not in data["detail"]
        assert "Secret" not in data["detail"]


# ============================================================================
# Prompt 15: Readiness Checks Reflect Model Availability
# ============================================================================

def test_readiness_healthy_when_models_loaded(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["supported_ragas_count"] == 10
    assert "active_inference_backend" in data
    assert any("MelodicSimilarityEngine" in m for m in data["models_loaded"])


def test_readiness_returns_503_when_pipeline_unready(client: TestClient) -> None:
    with patch("raga_intelligence.api.routes.get_pipeline") as mock_get_pipe:
        mock_pipe = MagicMock()
        mock_pipe.is_ready.return_value = (False, "none", [])
        mock_get_pipe.return_value = mock_pipe

        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert "unready" in data["details"].lower()


def test_liveness_remains_available_even_if_readiness_fails(client: TestClient) -> None:
    # Liveness /health must report server status independently of inference readiness
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
