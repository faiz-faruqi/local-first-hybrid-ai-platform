"""
Telemetry schemas.

A `TraceRecord` captures one request's journey through the pipeline
(classify -> retrieve -> cache lookup -> decision engine -> inference ->
cache write) as a list of named spans plus top-level outcome fields.
Written by the Tracer (src/observability/tracer.py) and read back by the
telemetry router to power the dashboard.
"""

from pydantic import BaseModel, Field

from src.models.classification import QueryProfile
from src.models.routing import RoutingDecision


class TraceSpan(BaseModel):
    """A single named stage within a traced request."""

    name: str
    start_ms: float = Field(description="Offset from request start, in milliseconds.")
    duration_ms: float


class TraceRecord(BaseModel):
    """Full span-level record for one request."""

    model_config = {"protected_namespaces": ()}

    trace_id: str
    route: str = Field(description="Which endpoint handled the request, e.g. '/query/' or '/query/stream'.")
    query_preview: str
    started_at: float = Field(description="Unix epoch seconds when the request started.")
    total_latency_ms: float
    ttft_ms: float | None = Field(
        default=None,
        description="Time to first token, in milliseconds. Only set for streaming requests.",
    )
    provider: str | None = None
    cached: bool = False
    model_alias: str | None = None
    rag_needed: bool | None = None
    department: str | None = None
    routing_mode: str | None = Field(
        default=None,
        description="Which of the three routing paths fired: explicit_model, force_cloud, or automatic.",
    )
    classification: QueryProfile | None = None
    routing_decision: RoutingDecision | None = None
    error: str | None = None
    spans: list[TraceSpan] = Field(default_factory=list)


class TraceSummary(BaseModel):
    """Compact per-trace row for the recent-traces list."""

    model_config = {"protected_namespaces": ()}

    trace_id: str
    route: str
    query_preview: str
    started_at: float
    total_latency_ms: float
    ttft_ms: float | None = None
    provider: str | None = None
    cached: bool = False
    model_alias: str | None = None
    rag_needed: bool | None = None
    department: str | None = None
    error: str | None = None


class TelemetryStats(BaseModel):
    """Aggregated stats over a recent time window, computed on read."""

    model_config = {"protected_namespaces": ()}

    window_seconds: int
    total_requests: int
    error_count: int
    cache_hit_rate: float
    rag_ratio: float
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    avg_ttft_ms: float | None = Field(
        default=None,
        description="Average time-to-first-token across streaming requests only.",
    )
    provider_distribution: dict[str, int] = Field(default_factory=dict)
    model_distribution: dict[str, int] = Field(default_factory=dict)
    department_distribution: dict[str, int] = Field(default_factory=dict)
