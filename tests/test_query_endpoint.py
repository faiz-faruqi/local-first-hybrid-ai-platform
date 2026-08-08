"""
Integration-style tests for the POST /query/ endpoint.

Uses FastAPI TestClient with all external dependencies mocked via
dependency_overrides — no real Redis, Qdrant, or LLM calls are made.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fastapi import HTTPException

from src.api.main import app
from src.api.dependencies import (
    get_cache,
    get_classifier,
    get_decision_engine,
    get_embedder,
    get_inference_router,
    get_rate_limiter,
    get_trace_store,
    get_vector_store_dep,
)
from src.inference.router import ProviderResult
from src.routing.classifier import QueryClassifier
from src.routing.decision_engine import DecisionEngine


def make_mock_hit(doc_name: str, content: str, score: float = 0.88):
    """Build a mock vector search result."""
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "chunk_id": "test-chunk-001",
        "document_name": doc_name,
        "content": content,
        "chunk_index": 0,
    }
    return hit


@pytest.fixture
def mock_embedder():
    m = MagicMock()
    m.embed_single.return_value = [0.1] * 384
    return m


@pytest.fixture
def mock_store():
    m = AsyncMock()
    m.search = AsyncMock(
        return_value=[
            make_mock_hit(
                "sample-vendor-contract.md",
                "Either party may terminate this agreement with 90 days written notice.",
                0.91,
            )
        ]
    )
    return m


@pytest.fixture
def mock_cache():
    m = AsyncMock()
    m.get = AsyncMock(return_value=None)
    m.set = AsyncMock()
    return m


@pytest.fixture
def mock_trace_store():
    """
    Mocked TraceStore — without this override, the endpoint constructs a
    real TraceStore(ResponseCache()) that tries to reach an actual Redis
    instance and eats the 2s socket-connect timeout on every request.
    """
    m = AsyncMock()
    m.write_trace = AsyncMock()
    return m


@pytest.fixture
def mock_rate_limiter():
    """
    Mocked RateLimiter — without this override, the endpoint constructs a
    real RateLimiter(ResponseCache()) that tries to reach an actual Redis
    instance and eats the 2s socket-connect timeout on every request.
    """
    m = AsyncMock()
    m.check = AsyncMock(return_value=None)
    return m


@pytest.fixture
def mock_registry():
    """A real registry with mocked underlying clients."""
    with patch(
        "src.inference.providers.ollama_provider.OllamaClient"
    ), patch(
        "src.inference.providers.openrouter_provider.OpenRouterClient"
    ):
        from src.inference.provider_registry import ProviderRegistry
        registry = ProviderRegistry()
        # Mock all providers' complete() to return a fixed answer.
        for provider in registry.all():
            provider.complete = AsyncMock(
                return_value=f"Mocked answer from {provider.alias}."
            )
    return registry


@pytest.fixture
def mock_router(mock_ollama, mock_openrouter, mock_registry):
    from src.inference.router import InferenceRouter
    return InferenceRouter(
        ollama=mock_ollama, openrouter=mock_openrouter, registry=mock_registry
    )


@pytest.fixture
def mock_classifier():
    """A real classifier — it's pure and fast, no mocking needed."""
    return QueryClassifier()


@pytest.fixture
def mock_decision_engine(mock_registry):
    """A Decision Engine wired to the mock registry."""
    return DecisionEngine(registry=mock_registry, budget_tracker=None)


@pytest.fixture
def client(
    mock_embedder, mock_store, mock_cache, mock_router,
    mock_classifier, mock_decision_engine, mock_trace_store, mock_rate_limiter,
):
    """TestClient with all dependencies overridden."""
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_vector_store_dep] = lambda: mock_store
    app.dependency_overrides[get_cache] = lambda: mock_cache
    app.dependency_overrides[get_inference_router] = lambda: mock_router
    app.dependency_overrides[get_classifier] = lambda: mock_classifier
    app.dependency_overrides[get_decision_engine] = lambda: mock_decision_engine
    app.dependency_overrides[get_trace_store] = lambda: mock_trace_store
    app.dependency_overrides[get_rate_limiter] = lambda: mock_rate_limiter
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestQueryEndpoint:
    def test_query_returns_200_with_local_provider(self, client, mock_cache, mock_ollama, mock_trace_store):
        import os
        mock_cache.get.return_value = None  # cache miss
        response = client.post(
            "/query/",
            json={"query": "What are the termination conditions?", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        # In DEMO_MODE the router routes to cloud; otherwise local.
        demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
        assert data["provider"] == ("cloud" if demo_mode else "local")
        assert data["cached"] is False
        assert len(data["sources"]) == 1
        assert "latency_ms" in data
        assert isinstance(data["answer"], str)
        # Cache-miss path should produce exactly one trace with routing metadata.
        mock_trace_store.write_trace.assert_called_once()
        record = mock_trace_store.write_trace.call_args[0][0]
        assert record["cached"] is False
        assert record["routing_mode"] == "automatic"
        assert record["classification"] is not None
        assert record["routing_decision"] is not None

    def test_query_returns_cached_response(self, client, mock_cache, mock_trace_store):
        mock_cache.get.return_value = "This is a cached answer."
        response = client.post(
            "/query/",
            json={"query": "What are the termination conditions?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "cache"
        assert data["cached"] is True
        assert data["answer"] == "This is a cached answer."
        # Cache-hit path is still traced, just without a routing decision.
        mock_trace_store.write_trace.assert_called_once()
        record = mock_trace_store.write_trace.call_args[0][0]
        assert record["cached"] is True
        assert record["provider"] == "cache"

    def test_query_force_cloud_routes_to_cloud(self, client, mock_cache, mock_ollama, mock_openrouter):
        mock_cache.get.return_value = None
        response = client.post(
            "/query/",
            json={"query": "Complex multi-hop reasoning query", "force_cloud": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "cloud"
        mock_ollama.complete.assert_not_called()

    def test_query_returns_404_when_no_docs(
        self, mock_embedder, mock_cache, mock_router,
        mock_classifier, mock_decision_engine, mock_trace_store, mock_rate_limiter,
    ):
        """When RAG is needed and vector search returns empty results, endpoint returns 404."""
        empty_store = AsyncMock()
        empty_store.search = AsyncMock(return_value=[])
        app.dependency_overrides[get_embedder] = lambda: mock_embedder
        app.dependency_overrides[get_vector_store_dep] = lambda: empty_store
        app.dependency_overrides[get_cache] = lambda: mock_cache
        app.dependency_overrides[get_inference_router] = lambda: mock_router
        app.dependency_overrides[get_classifier] = lambda: mock_classifier
        app.dependency_overrides[get_decision_engine] = lambda: mock_decision_engine
        app.dependency_overrides[get_trace_store] = lambda: mock_trace_store
        app.dependency_overrides[get_rate_limiter] = lambda: mock_rate_limiter
        with TestClient(app) as c:
            # Use a query that triggers rag_needed=True (contains 'contracts')
            response = c.post("/query/", json={"query": "Which contracts have termination clauses?"})
        app.dependency_overrides.clear()
        assert response.status_code == 404
        # A 404 is still a traced outcome, not a gap in the trace record.
        mock_trace_store.write_trace.assert_called_once()
        record = mock_trace_store.write_trace.call_args[0][0]
        assert record["error"] is not None

    def test_query_bypasses_rag_for_general_knowledge(
        self, mock_embedder, mock_cache, mock_router,
        mock_classifier, mock_decision_engine, mock_store, mock_trace_store, mock_rate_limiter,
    ):
        """When rag_needed=False, the endpoint should not call the vector store."""
        mock_cache.get.return_value = None
        app.dependency_overrides[get_embedder] = lambda: mock_embedder
        app.dependency_overrides[get_vector_store_dep] = lambda: mock_store
        app.dependency_overrides[get_cache] = lambda: mock_cache
        app.dependency_overrides[get_inference_router] = lambda: mock_router
        app.dependency_overrides[get_classifier] = lambda: mock_classifier
        app.dependency_overrides[get_decision_engine] = lambda: mock_decision_engine
        app.dependency_overrides[get_trace_store] = lambda: mock_trace_store
        app.dependency_overrides[get_rate_limiter] = lambda: mock_rate_limiter
        with TestClient(app) as c:
            # 'What is machine learning' has no RAG-trigger keywords
            response = c.post("/query/", json={"query": "What is machine learning?"})
        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        # Sources should be empty — no retrieval happened
        assert data["sources"] == []
        # rag_needed flag should be False in the classification
        assert data["classification"]["rag_needed"] is False

    def test_query_validates_min_length(self, client):
        """Query shorter than 3 chars should be rejected by Pydantic."""
        response = client.post("/query/", json={"query": "ab"})
        assert response.status_code == 422

    def test_query_validates_max_top_k(self, client):
        """top_k > 20 should be rejected."""
        response = client.post(
            "/query/",
            json={"query": "Valid question text", "top_k": 99},
        )
        assert response.status_code == 422

    def test_health_endpoint_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "service" in response.json()

    def test_query_includes_classification(self, client, mock_cache):
        """Phase 2: response should include classification metadata."""
        mock_cache.get.return_value = None
        response = client.post(
            "/query/",
            json={"query": "What are the termination conditions?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "classification" in data
        assert data["classification"] is not None
        assert "complexity" in data["classification"]
        assert "domain" in data["classification"]

    def test_query_includes_routing_decision(self, client, mock_cache):
        """Phase 3: response should include the routing decision."""
        mock_cache.get.return_value = None
        response = client.post(
            "/query/",
            json={"query": "What are the termination conditions?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "routing_decision" in data
        assert data["routing_decision"] is not None
        assert "selected_model" in data["routing_decision"]
        assert "reason" in data["routing_decision"]

    def test_query_with_explicit_model(self, client, mock_cache):
        """Phase 1: explicit model selection should work."""
        mock_cache.get.return_value = None
        response = client.post(
            "/query/",
            json={"query": "What are the termination conditions?", "model": "gpt-4o"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_alias"] == "gpt-4o"

    def test_query_returns_429_when_rate_limited(self, client, mock_rate_limiter):
        """
        Proves the rate-limit dependency is actually wired onto the route —
        RateLimiter's own unit tests (test_rate_limit.py) only prove it's
        correct in isolation, not that the endpoint calls it.
        """
        mock_rate_limiter.check = AsyncMock(
            side_effect=HTTPException(status_code=429, detail="Rate limit exceeded: max 5 requests/minute.")
        )
        response = client.post("/query/", json={"query": "What are the termination conditions?"})
        assert response.status_code == 429
        assert "detail" in response.json()

    def test_models_endpoint(self, client):
        """Phase 1: GET /models should return the model catalog."""
        response = client.get("/models/")
        assert response.status_code == 200
        models = response.json()
        assert len(models) > 0
        assert "alias" in models[0]
        assert "tier" in models[0]
