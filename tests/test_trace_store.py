"""
Unit tests for TraceStore.

All Redis operations are mocked — no live Redis required.
"""

import json
from unittest.mock import patch

import pytest

from src.cache.redis_cache import ResponseCache
from src.observability.trace_store import TraceStore


def make_trace(trace_id="abc123", started_at=1000.0, **overrides):
    trace = {
        "trace_id": trace_id,
        "route": "/query/",
        "query_preview": "test query",
        "started_at": started_at,
        "total_latency_ms": 120.0,
        "spans": [],
    }
    trace.update(overrides)
    return trace


class TestTraceStore:
    @pytest.fixture
    def store(self, mock_redis_client):
        """TraceStore instance with an injected mock Redis client."""
        with patch("src.cache.redis_cache.aioredis.Redis", return_value=mock_redis_client):
            cache = ResponseCache(redis_url="", host="localhost", port=6379)
            cache._client = mock_redis_client
            instance = TraceStore(cache)
            return instance, mock_redis_client

    async def test_write_trace_calls_set_and_zadd(self, store):
        instance, mock_client = store
        await instance.write_trace(make_trace())

        mock_client.set.assert_called_once()
        set_args = mock_client.set.call_args
        assert set_args[0][0] == "trace:abc123"
        assert json.loads(set_args[0][1])["trace_id"] == "abc123"
        assert set_args[1]["ex"] == instance._ttl

        mock_client.zadd.assert_called_once_with("traces:index", {"abc123": 1000.0})
        mock_client.zremrangebyrank.assert_called_once()

    async def test_write_trace_failure_does_not_raise(self, store):
        instance, mock_client = store
        mock_client.set.side_effect = ConnectionError("Redis down")
        await instance.write_trace(make_trace())  # should not raise

    async def test_get_trace_returns_none_when_missing(self, store):
        instance, mock_client = store
        mock_client.get.return_value = None
        assert await instance.get_trace("missing") is None

    async def test_get_trace_returns_parsed_record(self, store):
        instance, mock_client = store
        trace = make_trace()
        mock_client.get.return_value = json.dumps(trace)
        assert await instance.get_trace("abc123") == trace

    async def test_get_trace_failure_returns_none(self, store):
        instance, mock_client = store
        mock_client.get.side_effect = ConnectionError("Redis down")
        assert await instance.get_trace("abc123") is None

    async def test_get_recent_returns_empty_when_no_ids(self, store):
        instance, mock_client = store
        mock_client.zrevrange.return_value = []
        assert await instance.get_recent(50) == []
        mock_client.mget.assert_not_called()

    async def test_get_recent_fetches_and_parses_blobs(self, store):
        instance, mock_client = store
        trace1 = make_trace(trace_id="t1")
        trace2 = make_trace(trace_id="t2")
        mock_client.zrevrange.return_value = ["t1", "t2"]
        mock_client.mget.return_value = [json.dumps(trace1), json.dumps(trace2)]

        result = await instance.get_recent(2)

        assert result == [trace1, trace2]
        mock_client.mget.assert_called_once_with(["trace:t1", "trace:t2"])

    async def test_get_recent_skips_expired_blobs(self, store):
        """A trace ID still in the index but whose blob already expired is skipped, not an error."""
        instance, mock_client = store
        trace1 = make_trace(trace_id="t1")
        mock_client.zrevrange.return_value = ["t1", "t2"]
        mock_client.mget.return_value = [json.dumps(trace1), None]

        assert await instance.get_recent(2) == [trace1]

    async def test_get_recent_failure_returns_empty(self, store):
        instance, mock_client = store
        mock_client.zrevrange.side_effect = ConnectionError("Redis down")
        assert await instance.get_recent(50) == []

    async def test_get_stats_empty_window(self, store):
        instance, mock_client = store
        mock_client.zrangebyscore.return_value = []

        stats = await instance.get_stats(3600)

        assert stats["total_requests"] == 0
        assert stats["cache_hit_rate"] == 0.0
        assert stats["latency_p50_ms"] is None
        assert stats["provider_distribution"] == {}

    async def test_get_stats_computes_percentiles_and_distributions(self, store):
        instance, mock_client = store
        traces = [
            make_trace(trace_id="t1", total_latency_ms=100.0, provider="local", cached=False, rag_needed=True),
            make_trace(
                trace_id="t2", total_latency_ms=200.0, provider="cloud", cached=True,
                rag_needed=False, model_alias="gpt-4o",
            ),
            make_trace(
                trace_id="t3", total_latency_ms=300.0, provider="cloud", cached=False,
                rag_needed=False, error="boom", model_alias="gpt-4o",
            ),
        ]
        mock_client.zrangebyscore.return_value = ["t1", "t2", "t3"]
        mock_client.mget.return_value = [json.dumps(t) for t in traces]

        stats = await instance.get_stats(3600)

        assert stats["total_requests"] == 3
        assert stats["error_count"] == 1
        assert stats["cache_hit_rate"] == round(1 / 3, 4)
        assert stats["rag_ratio"] == round(1 / 3, 4)
        assert stats["provider_distribution"] == {"local": 1, "cloud": 2}
        assert stats["model_distribution"] == {"gpt-4o": 2}
        assert stats["latency_p50_ms"] is not None
        assert stats["latency_p95_ms"] is not None

    async def test_get_stats_avg_ttft_only_over_streaming_traces(self, store):
        instance, mock_client = store
        traces = [
            make_trace(trace_id="t1", ttft_ms=50.0),
            make_trace(trace_id="t2", ttft_ms=None),
            make_trace(trace_id="t3", ttft_ms=150.0),
        ]
        mock_client.zrangebyscore.return_value = ["t1", "t2", "t3"]
        mock_client.mget.return_value = [json.dumps(t) for t in traces]

        stats = await instance.get_stats(3600)

        assert stats["avg_ttft_ms"] == 100.0

    async def test_get_stats_failure_returns_empty_stats(self, store):
        instance, mock_client = store
        mock_client.zrangebyscore.side_effect = ConnectionError("Redis down")

        stats = await instance.get_stats(3600)

        assert stats["total_requests"] == 0
        assert stats["provider_distribution"] == {}
