"""
Endpoint-level tests for POST /auth/generate-code and POST /auth/verify-code.

Confirms generate-code requires the admin key (same protection as the
existing DELETE /ingest/flush-cache) and verify-code does not — NextAuth's
authorize() must be able to reach it before any session exists.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_access_code_store
from src.api.main import app


@pytest.fixture
def mock_store():
    m = AsyncMock()
    m.generate = AsyncMock(return_value=("generated-code-123", "2099-01-01T00:00:00+00:00"))
    m.verify = AsyncMock(return_value=True)
    return m


@pytest.fixture
def client(mock_store):
    app.dependency_overrides[get_access_code_store] = lambda: mock_store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestGenerateCodeEndpoint:
    def test_missing_admin_key_is_rejected(self, client):
        with patch("src.api.admin_auth.ADMIN_KEY", "the-real-key"):
            response = client.post("/auth/generate-code", json={})
        assert response.status_code == 403

    def test_wrong_admin_key_is_rejected(self, client):
        with patch("src.api.admin_auth.ADMIN_KEY", "the-real-key"):
            response = client.post(
                "/auth/generate-code", json={}, headers={"X-Admin-Key": "wrong-key"}
            )
        assert response.status_code == 403

    def test_unconfigured_admin_key_blocks_everything(self, client):
        """Matches /ingest/flush-cache's existing behavior: no ADMIN_KEY set = nothing works."""
        with patch("src.api.admin_auth.ADMIN_KEY", ""):
            response = client.post(
                "/auth/generate-code", json={}, headers={"X-Admin-Key": "anything"}
            )
        assert response.status_code == 503

    def test_correct_admin_key_succeeds(self, client, mock_store):
        with patch("src.api.admin_auth.ADMIN_KEY", "the-real-key"):
            response = client.post(
                "/auth/generate-code",
                json={"ttl_hours": 24},
                headers={"X-Admin-Key": "the-real-key"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "generated-code-123"
        assert data["ttl_hours"] == 24
        mock_store.generate.assert_called_once_with(ttl_hours=24)

    def test_omitted_ttl_uses_server_default(self, client, mock_store):
        from src.api.access_code import ACCESS_CODE_DEFAULT_TTL_HOURS

        with patch("src.api.admin_auth.ADMIN_KEY", "the-real-key"):
            response = client.post(
                "/auth/generate-code", json={}, headers={"X-Admin-Key": "the-real-key"}
            )
        assert response.status_code == 200
        mock_store.generate.assert_called_once_with(ttl_hours=ACCESS_CODE_DEFAULT_TTL_HOURS)


class TestVerifyCodeEndpoint:
    def test_does_not_require_admin_key(self, client, mock_store):
        """authorize() must be able to reach this before any session exists."""
        response = client.post("/auth/verify-code", json={"code": "whatever"})
        assert response.status_code == 200
        assert response.json()["valid"] is True

    def test_returns_false_for_invalid_code(self, client, mock_store):
        mock_store.verify = AsyncMock(return_value=False)
        response = client.post("/auth/verify-code", json={"code": "wrong"})
        assert response.status_code == 200
        assert response.json()["valid"] is False

    def test_never_echoes_stored_code(self, client):
        """Response shape should only ever contain `valid`, never the actual code."""
        response = client.post("/auth/verify-code", json={"code": "whatever"})
        assert set(response.json().keys()) == {"valid"}
