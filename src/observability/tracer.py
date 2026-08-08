"""
Lightweight per-request tracer.

Records named spans (start offset + duration) as a request moves through
classify -> retrieve -> cache lookup -> decision engine -> inference ->
cache write, plus top-level outcome fields (provider, cached, routing
decision, errors), then writes the assembled record to a TraceStore.

Created fresh per request in the route handler — unlike TraceStore, this
is never a singleton.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from src.observability.trace_store import TraceStore

logger = logging.getLogger(__name__)


class Tracer:
    def __init__(self, store: TraceStore, route: str, query: str) -> None:
        self._store = store
        self.trace_id = uuid.uuid4().hex
        self._route = route
        self._query_preview = query[:300]
        self._started_at = time.time()
        self._start_perf = time.perf_counter()
        self._spans: list[dict] = []
        self._fields: dict[str, Any] = {}
        self._ttft_recorded = False
        self._finished = False

    @asynccontextmanager
    async def span(self, name: str) -> AsyncIterator[None]:
        """
        Time a named stage. Records duration even if the wrapped block
        raises (e.g. the RAG-miss 404), so failures still show up in the
        trace instead of leaving a gap.
        """
        start = time.perf_counter()
        start_offset_ms = (start - self._start_perf) * 1000
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._spans.append({
                "name": name,
                "start_ms": round(start_offset_ms, 2),
                "duration_ms": round(duration_ms, 2),
            })

    def mark_ttft(self) -> None:
        """Record time-to-first-token, once. Only meaningful for streaming requests."""
        if self._ttft_recorded:
            return
        self._ttft_recorded = True
        self._fields["ttft_ms"] = round((time.perf_counter() - self._start_perf) * 1000, 2)

    def set(self, **fields: Any) -> None:
        """Merge top-level trace fields (provider, cached, model_alias, error, ...)."""
        self._fields.update(fields)

    async def finish(self) -> None:
        """Assemble the final trace record and persist it. Idempotent."""
        if self._finished:
            return
        self._finished = True
        total_latency_ms = round((time.perf_counter() - self._start_perf) * 1000, 2)
        record = {
            "trace_id": self.trace_id,
            "route": self._route,
            "query_preview": self._query_preview,
            "started_at": self._started_at,
            "total_latency_ms": total_latency_ms,
            "spans": self._spans,
            **self._fields,
        }
        await self._store.write_trace(record)
