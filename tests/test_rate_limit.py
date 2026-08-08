"""
Unit tests for RateLimiter and get_client_ip.

All Redis operations are mocked — no live Redis required.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.api.rate_limit import RateLimiter, get_client_ip


class FakeCache:
    """Minimal stand-in exposing ._client, matching what RateLimiter reads off ResponseCache."""

    def __init__(self, client):
        self._client = client


def make_incr_counter():
    """
    A fake Redis INCR that returns an incrementing count per distinct key,
    mirroring real Redis semantics (atomic per-key counter) closely enough
    for these tests.
    """
    counts: dict[str, int] = {}

    async def incr(key: str) -> int:
        counts[key] = counts.get(key, 0) + 1
        return counts[key]

    return incr, counts


class TestRateLimiter:
    @pytest.fixture
    def limiter(self):
        incr, _counts = make_incr_counter()
        mock_client = AsyncMock()
        mock_client.incr = AsyncMock(side_effect=incr)
        mock_client.expire = AsyncMock()
        return RateLimiter(FakeCache(mock_client), per_minute=5, per_day=30), mock_client

    async def test_requests_under_minute_cap_all_pass(self, limiter):
        instance, _ = limiter
        for _ in range(5):
            await instance.check("1.2.3.4")  # should not raise

    async def test_request_over_minute_cap_raises_429(self, limiter):
        instance, _ = limiter
        for _ in range(5):
            await instance.check("1.2.3.4")
        with pytest.raises(HTTPException) as exc_info:
            await instance.check("1.2.3.4")
        assert exc_info.value.status_code == 429
        assert "minute" in exc_info.value.detail

    async def test_different_ips_have_independent_limits(self, limiter):
        instance, _ = limiter
        for _ in range(5):
            await instance.check("1.2.3.4")
        await instance.check("5.6.7.8")  # different IP, should not raise

    async def test_expire_set_only_on_first_increment(self, limiter):
        instance, mock_client = limiter
        await instance.check("1.2.3.4")
        await instance.check("1.2.3.4")
        # Two checks -> two keys touched (minute + day) per check, but
        # expire should only fire once per key (on the count==1 increment).
        assert mock_client.expire.call_count == 2  # minute key + day key, first check only

    async def test_daily_cap_enforced_independently_of_minute_cap(self):
        incr, counts = make_incr_counter()
        mock_client = AsyncMock()
        mock_client.incr = AsyncMock(side_effect=incr)
        mock_client.expire = AsyncMock()
        # Generous per-minute cap so only the daily cap can trip.
        instance = RateLimiter(FakeCache(mock_client), per_minute=1000, per_day=2)

        await instance.check("9.9.9.9")
        await instance.check("9.9.9.9")
        with pytest.raises(HTTPException) as exc_info:
            await instance.check("9.9.9.9")
        assert exc_info.value.status_code == 429
        assert "day" in exc_info.value.detail.lower()

    async def test_redis_failure_fails_open(self):
        mock_client = AsyncMock()
        mock_client.incr = AsyncMock(side_effect=ConnectionError("Redis down"))
        instance = RateLimiter(FakeCache(mock_client), per_minute=5, per_day=30)
        await instance.check("1.2.3.4")  # should not raise


class TestGetClientIp:
    def _make_request(self, forwarded_for: str | None, client_host: str | None):
        request = MagicMock()
        request.headers = {"X-Forwarded-For": forwarded_for} if forwarded_for else {}
        if client_host:
            request.client.host = client_host
        else:
            request.client = None
        return request

    def test_prefers_x_forwarded_for_first_entry(self):
        request = self._make_request(forwarded_for="203.0.113.5, 10.0.0.1", client_host="10.0.0.1")
        assert get_client_ip(request) == "203.0.113.5"

    def test_falls_back_to_client_host_when_header_absent(self):
        request = self._make_request(forwarded_for=None, client_host="127.0.0.1")
        assert get_client_ip(request) == "127.0.0.1"

    def test_returns_unknown_when_neither_available(self):
        request = self._make_request(forwarded_for=None, client_host=None)
        assert get_client_ip(request) == "unknown"
