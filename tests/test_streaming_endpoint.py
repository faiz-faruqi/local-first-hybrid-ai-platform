"""
Tests for POST /query/stream.

Note: Starlette's TestClient buffers the full SSE body synchronously
rather than doing a true async incremental read — fine for asserting
final shape and content, not true incrementality.

`stream_complete_with_model` is faked with a plain async-generator
function rather than AsyncMock, since AsyncMock can't produce one.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

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


class FakeStreamingRouter:
    """
    Hand-rolled fake standing in for InferenceRouter — the endpoint only
    needs `stream_complete_with_model` / `stream_complete` to be async
    generators, which AsyncMock cannot produce.
    """

    def __init__(self, chunks=None, raise_exc=None, provider=ProviderResult.CLOUD):
        self._chunks = chunks or []
        self._raise_exc = raise_exc
        self._provider = provider

    async def stream_complete_with_model(self, prompt, model_alias, provider_result=None):
        if self._raise_exc is not None:
            raise self._raise_exc
        if provider_result is not None:
            provider_result["provider"] = self._provider
        for chunk in self._chunks:
            yield chunk

    async def stream_complete(self, prompt, force_cloud=False):
        if self._raise_exc is not None:
            raise self._raise_exc
        for chunk in self._chunks:
            yield chunk


def parse_sse_events(body: str) -> list[dict]:
    events = []
    for frame in body.strip().split("\n\n"):
        frame = frame.strip()
        if frame.startswith("data:"):
            events.append(json.loads(frame[len("data:"):].strip()))
    return events


@pytest.fixture
def mock_embedder():
    m = MagicMock()
    m.embed_single.return_value = [0.1] * 384
    return m


@pytest.fixture
def mock_store():
    m = AsyncMock()
    m.search = AsyncMock(return_value=[])
    return m


@pytest.fixture
def mock_cache():
    m = AsyncMock()
    m.get = AsyncMock(return_value=None)
    m.set = AsyncMock()
    return m


@pytest.fixture
def mock_trace_store():
    m = AsyncMock()
    m.write_trace = AsyncMock()
    return m


@pytest.fixture
def mock_rate_limiter():
    m = AsyncMock()
    m.check = AsyncMock(return_value=None)
    return m


@pytest.fixture
def mock_classifier():
    return QueryClassifier()


@pytest.fixture
def mock_registry():
    with patch(
        "src.inference.providers.ollama_provider.OllamaClient"
    ), patch(
        "src.inference.providers.openrouter_provider.OpenRouterClient"
    ):
        from src.inference.provider_registry import ProviderRegistry
        return ProviderRegistry()


@pytest.fixture
def mock_decision_engine(mock_registry):
    return DecisionEngine(registry=mock_registry, budget_tracker=None)


def override_deps(
    router, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine,
    mock_trace_store, mock_rate_limiter,
):
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_vector_store_dep] = lambda: mock_store
    app.dependency_overrides[get_cache] = lambda: mock_cache
    app.dependency_overrides[get_inference_router] = lambda: router
    app.dependency_overrides[get_classifier] = lambda: mock_classifier
    app.dependency_overrides[get_decision_engine] = lambda: mock_decision_engine
    app.dependency_overrides[get_trace_store] = lambda: mock_trace_store
    app.dependency_overrides[get_rate_limiter] = lambda: mock_rate_limiter


class TestQueryStreamEndpoint:
    def test_returns_sse_content_type(
        self, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine, mock_trace_store,
        mock_rate_limiter,
    ):
        router = FakeStreamingRouter(chunks=["Hello", " world"])
        override_deps(
            router, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine,
            mock_trace_store, mock_rate_limiter,
        )
        with TestClient(app) as c:
            response = c.post("/query/stream", json={"query": "What is machine learning?"})
        app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_delivers_delta_chunks_then_done_event(
        self, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine, mock_trace_store,
        mock_rate_limiter,
    ):
        router = FakeStreamingRouter(chunks=["Hello", " world"], provider=ProviderResult.CLOUD)
        override_deps(
            router, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine,
            mock_trace_store, mock_rate_limiter,
        )
        with TestClient(app) as c:
            response = c.post("/query/stream", json={"query": "What is machine learning?"})
        app.dependency_overrides.clear()

        events = parse_sse_events(response.text)
        deltas = [e["delta"] for e in events if "delta" in e]
        done_events = [e for e in events if e.get("event") == "done"]

        assert "".join(deltas) == "Hello world"
        assert len(done_events) == 1
        assert done_events[0]["provider"] == "cloud"
        assert done_events[0]["cached"] is False
        assert "classification" in done_events[0]

    def test_cache_hit_streams_as_single_chunk(
        self, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine, mock_trace_store,
        mock_rate_limiter,
    ):
        mock_cache.get.return_value = "Cached answer."
        router = FakeStreamingRouter(chunks=[])  # never invoked on a cache hit
        override_deps(
            router, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine,
            mock_trace_store, mock_rate_limiter,
        )
        with TestClient(app) as c:
            response = c.post("/query/stream", json={"query": "What is machine learning?"})
        app.dependency_overrides.clear()

        events = parse_sse_events(response.text)
        assert events[0]["delta"] == "Cached answer."
        assert events[-1]["event"] == "done"
        assert events[-1]["cached"] is True
        assert events[-1]["provider"] == "cache"

    def test_error_before_first_chunk_yields_error_event(
        self, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine, mock_trace_store,
        mock_rate_limiter,
    ):
        router = FakeStreamingRouter(raise_exc=RuntimeError("boom"))
        override_deps(
            router, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine,
            mock_trace_store, mock_rate_limiter,
        )
        with TestClient(app) as c:
            response = c.post("/query/stream", json={"query": "What is machine learning?"})
        app.dependency_overrides.clear()

        # Headers are already committed by the time a streaming failure
        # happens, so the HTTP status stays 200 — the failure surfaces as
        # a terminal SSE event instead.
        assert response.status_code == 200
        events = parse_sse_events(response.text)
        assert events[-1]["event"] == "error"
        assert "boom" in events[-1]["message"]

    def test_writes_trace_with_ttft(
        self, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine, mock_trace_store,
        mock_rate_limiter,
    ):
        router = FakeStreamingRouter(chunks=["Hello"])
        override_deps(
            router, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine,
            mock_trace_store, mock_rate_limiter,
        )
        with TestClient(app) as c:
            c.post("/query/stream", json={"query": "What is machine learning?"})
        app.dependency_overrides.clear()

        mock_trace_store.write_trace.assert_called_once()
        record = mock_trace_store.write_trace.call_args[0][0]
        assert record["route"] == "/query/stream"
        assert record["ttft_ms"] is not None

    def test_error_trace_has_no_ttft(
        self, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine, mock_trace_store,
        mock_rate_limiter,
    ):
        """A failure before any chunk is yielded should leave ttft_ms unset."""
        router = FakeStreamingRouter(raise_exc=RuntimeError("boom"))
        override_deps(
            router, mock_embedder, mock_store, mock_cache, mock_classifier, mock_decision_engine,
            mock_trace_store, mock_rate_limiter,
        )
        with TestClient(app) as c:
            c.post("/query/stream", json={"query": "What is machine learning?"})
        app.dependency_overrides.clear()

        record = mock_trace_store.write_trace.call_args[0][0]
        assert record.get("ttft_ms") is None
        assert record["error"] is not None
