"""
Redis-backed trace store for lightweight request observability.

Reuses the existing Redis connection (same instance as the response cache)
under a separate key namespace — the same reuse pattern as BudgetTracker
(src/routing/budget.py).

Storage shape: a sorted set indexing trace IDs by start time, plus one
JSON blob per trace. This (rather than a single capped list) is what
makes time-windowed reads for the /telemetry/stats endpoint a
ZRANGEBYSCORE + MGET instead of a full scan, and gives each trace its own
TTL instead of relying purely on list length to bound memory.
"""

import json
import logging
import time

from src.cache.redis_cache import ResponseCache

logger = logging.getLogger(__name__)

TRACE_INDEX_KEY = "traces:index"
TRACE_KEY_PREFIX = "trace:"
TRACE_TTL_SECONDS = 86400
MAX_TRACES = 5000


class TraceStore:
    """
    Records and reads structured request traces.

    Every method degrades gracefully on a Redis failure (logs a warning,
    returns None/empty) rather than raising — matching ResponseCache's
    and BudgetTracker's existing convention, so an observability outage
    never breaks the request path it's observing.
    """

    def __init__(
        self,
        cache: ResponseCache,
        ttl: int = TRACE_TTL_SECONDS,
        max_traces: int = MAX_TRACES,
    ) -> None:
        self._client = cache._client  # reuse the underlying Redis connection
        self._ttl = ttl
        self._max_traces = max_traces

    async def write_trace(self, trace: dict) -> None:
        """Persist a completed trace record."""
        trace_id = trace["trace_id"]
        started_at = trace["started_at"]
        try:
            await self._client.set(
                f"{TRACE_KEY_PREFIX}{trace_id}", json.dumps(trace), ex=self._ttl
            )
            await self._client.zadd(TRACE_INDEX_KEY, {trace_id: started_at})
            # Bound the index independent of TTL timing, so a traffic burst
            # followed by silence doesn't keep serving stale entries forever.
            await self._client.zremrangebyrank(
                TRACE_INDEX_KEY, 0, -(self._max_traces + 1)
            )
        except Exception as exc:
            logger.warning("Trace write failed: %s", exc)

    async def get_trace(self, trace_id: str) -> dict | None:
        """Return the full record for a single trace, or None if missing/expired."""
        try:
            raw = await self._client.get(f"{TRACE_KEY_PREFIX}{trace_id}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("Trace read failed: %s", exc)
            return None

    async def get_recent(self, limit: int = 50) -> list[dict]:
        """Return the most recent traces, newest first."""
        try:
            trace_ids = await self._client.zrevrange(TRACE_INDEX_KEY, 0, limit - 1)
            return await self._fetch_blobs(trace_ids)
        except Exception as exc:
            logger.warning("Trace recent-fetch failed: %s", exc)
            return []

    async def get_window(self, window_seconds: int) -> list[dict]:
        """Return every trace started within the last `window_seconds`."""
        try:
            since = time.time() - window_seconds
            trace_ids = await self._client.zrangebyscore(TRACE_INDEX_KEY, since, "+inf")
            return await self._fetch_blobs(trace_ids)
        except Exception as exc:
            logger.warning("Trace window-fetch failed: %s", exc)
            return []

    async def _fetch_blobs(self, trace_ids: list[str]) -> list[dict]:
        if not trace_ids:
            return []
        keys = [f"{TRACE_KEY_PREFIX}{tid}" for tid in trace_ids]
        raw_values = await self._client.mget(keys)
        return [json.loads(v) for v in raw_values if v]

    async def get_stats(self, window_seconds: int = 3600) -> dict:
        """
        Aggregate stats over a recent time window.

        Computed on read from the raw trace blobs rather than maintained as
        incremental counters — avoids a second, driftable source of truth,
        and is cheap enough at the trace volumes this store is sized for
        (thousands, not millions).
        """
        traces = await self.get_window(window_seconds)
        total = len(traces)
        empty_stats = {
            "window_seconds": window_seconds,
            "total_requests": 0,
            "error_count": 0,
            "cache_hit_rate": 0.0,
            "rag_ratio": 0.0,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "latency_p99_ms": None,
            "avg_ttft_ms": None,
            "provider_distribution": {},
            "model_distribution": {},
            "department_distribution": {},
        }
        if total == 0:
            return empty_stats

        latencies = sorted(
            t["total_latency_ms"] for t in traces if t.get("total_latency_ms") is not None
        )
        ttfts = [t["ttft_ms"] for t in traces if t.get("ttft_ms") is not None]
        cached_count = sum(1 for t in traces if t.get("cached"))
        rag_count = sum(1 for t in traces if t.get("rag_needed"))
        error_count = sum(1 for t in traces if t.get("error"))

        provider_distribution: dict[str, int] = {}
        model_distribution: dict[str, int] = {}
        department_distribution: dict[str, int] = {}
        for t in traces:
            provider = t.get("provider")
            if provider:
                provider_distribution[provider] = provider_distribution.get(provider, 0) + 1
            model_alias = t.get("model_alias")
            if model_alias:
                model_distribution[model_alias] = model_distribution.get(model_alias, 0) + 1
            department = t.get("department")
            if department:
                department_distribution[department] = department_distribution.get(department, 0) + 1

        return {
            "window_seconds": window_seconds,
            "total_requests": total,
            "error_count": error_count,
            "cache_hit_rate": round(cached_count / total, 4),
            "rag_ratio": round(rag_count / total, 4),
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "latency_p99_ms": _percentile(latencies, 0.99),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "provider_distribution": provider_distribution,
            "model_distribution": model_distribution,
            "department_distribution": department_distribution,
        }


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    idx = min(int(len(sorted_values) * pct), len(sorted_values) - 1)
    return round(sorted_values[idx], 2)
