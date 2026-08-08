"""
Request and response schemas.

Using Pydantic v2 for runtime validation and auto-generated OpenAPI docs.
"""

from enum import Enum

from pydantic import BaseModel, Field

from src.models.classification import QueryProfile
from src.models.routing import RoutingDecision


class InferenceProvider(str, Enum):
    """Which inference provider handled the request."""
    local = "local"
    cloud = "cloud"
    cache = "cache"


class QueryRequest(BaseModel):
    """Incoming user query payload."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question over enterprise documents.",
        examples=["What are the termination conditions in the vendor contract?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve from the vector store.",
    )
    force_cloud: bool = Field(
        default=False,
        description="Bypass local inference and use cloud provider directly.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional model alias to force a specific provider "
            "(e.g. 'gpt-4o', 'claude-sonnet', 'local-gemma'). "
            "If omitted, the gateway selects the model automatically (Phase 3)."
        ),
    )
    department: str | None = Field(
        default=None,
        description=(
            "Self-reported org department (e.g. 'IT', 'Sales', 'Marketing', 'Finance') "
            "for usage-attribution demo purposes. Not validated against a fixed enum — "
            "this is a labeling attribute for the telemetry dashboard, not an access-control field."
        ),
    )


class SourceDocument(BaseModel):
    """A retrieved document chunk included as context."""

    chunk_id: str
    document_name: str
    content_preview: str = Field(..., max_length=300)
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Response returned to the user."""

    model_config = {"protected_namespaces": ()}

    answer: str
    provider: InferenceProvider
    cached: bool
    sources: list[SourceDocument]
    latency_ms: float
    model_alias: str | None = Field(
        default=None,
        description="The model alias that actually handled the request (e.g. 'gpt-4o').",
    )
    classification: QueryProfile | None = Field(
        default=None,
        description="Multi-dimensional query classification (Phase 2).",
    )
    routing_decision: RoutingDecision | None = Field(
        default=None,
        description="The Decision Engine's model selection verdict (Phase 3).",
    )


class IngestRequest(BaseModel):
    """Request to ingest a document into the vector store."""

    document_name: str
    content: str
    collection: str = Field(default="enterprise-docs")


class IngestResponse(BaseModel):
    chunks_indexed: int
    collection: str
    document_name: str


class ModelInfo(BaseModel):
    """Public description of a routable model (returned by GET /models)."""

    model_config = {"protected_namespaces": ()}

    alias: str
    vendor: str
    model_id: str
    tier: str
    context_window: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    latency_tier: str
    is_local: bool
    description: str
