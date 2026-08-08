"""
Telemetry endpoints.

Read-only access to the trace data recorded by the Tracer for every
/query/ and /query/stream request — powers the observability dashboard.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_trace_store
from src.models.telemetry import TelemetryStats, TraceRecord, TraceSummary
from src.observability.trace_store import TraceStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats", response_model=TelemetryStats)
async def stats(
    window_seconds: int = 3600,
    trace_store: TraceStore = Depends(get_trace_store),
) -> TelemetryStats:
    """Aggregated latency/cache/routing stats over a recent time window."""
    data = await trace_store.get_stats(window_seconds)
    return TelemetryStats(**data)


@router.get("/traces", response_model=list[TraceSummary])
async def recent_traces(
    limit: int = 50,
    trace_store: TraceStore = Depends(get_trace_store),
) -> list[TraceSummary]:
    """The most recent traces, newest first."""
    traces = await trace_store.get_recent(limit)
    return [TraceSummary(**t) for t in traces]


@router.get("/traces/{trace_id}", response_model=TraceRecord)
async def trace_detail(
    trace_id: str,
    trace_store: TraceStore = Depends(get_trace_store),
) -> TraceRecord:
    """Full span-level detail for a single trace."""
    trace = await trace_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return TraceRecord(**trace)
