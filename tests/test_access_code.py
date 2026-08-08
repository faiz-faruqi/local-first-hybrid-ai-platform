"""
Unit tests for AccessCodeStore.

All Redis operations are mocked — no live Redis required.
"""

from unittest.mock import AsyncMock

import pytest

from src.api.access_code import AccessCodeStore


class FakeCache:
    """Minimal stand-in exposing ._client, matching what AccessCodeStore reads off ResponseCache."""

    def __init__(self, client):
        self._client = client


class TestAccessCodeStore:
    @pytest.fixture
    def mock_client(self):
        m = AsyncMock()
        m.set = AsyncMock()
        m.get = AsyncMock(return_value=None)
        return m

    @pytest.fixture
    def store(self, mock_client):
        return AccessCodeStore(FakeCache(mock_client))

    async def test_generate_writes_code_with_ttl(self, store, mock_client):
        code, expires_at = await store.generate(ttl_hours=1)

        assert code  # non-empty
        assert expires_at  # ISO timestamp string
        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        assert call_args[0][0] == "auth:access_code"
        assert call_args[0][1] == code
        assert call_args[1]["ex"] == 3600

    async def test_generate_produces_different_codes(self, store):
        code1, _ = await store.generate(ttl_hours=1)
        code2, _ = await store.generate(ttl_hours=1)
        assert code1 != code2

    async def test_verify_true_for_current_code(self, store, mock_client):
        mock_client.get.return_value = "the-current-code"
        assert await store.verify("the-current-code") is True

    async def test_verify_false_for_wrong_code(self, store, mock_client):
        mock_client.get.return_value = "the-current-code"
        assert await store.verify("some-other-guess") is False

    async def test_verify_false_when_no_code_set(self, store, mock_client):
        """Covers both 'never generated' and 'expired' — Redis returns None either way."""
        mock_client.get.return_value = None
        assert await store.verify("anything") is False

    async def test_verify_false_for_empty_code(self, store):
        assert await store.verify("") is False

    async def test_verify_fails_closed_on_redis_error(self, store, mock_client):
        """Unlike RateLimiter/BudgetTracker, an auth check must deny on error, not allow."""
        mock_client.get.side_effect = ConnectionError("Redis down")
        assert await store.verify("anything") is False

    async def test_new_code_retires_previous_one(self, store, mock_client):
        """Generating a second code should invalidate the first (single active code)."""
        first_code, _ = await store.generate(ttl_hours=1)
        second_code, _ = await store.generate(ttl_hours=1)

        # Simulate Redis now holding only the second code (the SET call overwrote it).
        mock_client.get.return_value = second_code

        assert await store.verify(second_code) is True
        assert await store.verify(first_code) is False
