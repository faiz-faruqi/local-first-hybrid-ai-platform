"""
Shared pytest fixtures for the test suite.

All fixtures mock external dependencies (Redis, Qdrant, OpenRouter, Ollama)
so tests run without any live services — safe in CI and on developer machines.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_ollama():
    """Async mock for OllamaClient."""
    m = AsyncMock()
    m.complete = AsyncMock(return_value="Local model answer from Ollama.")
    m.health_check = AsyncMock(return_value=True)
    return m


@pytest.fixture
def mock_openrouter():
    """Async mock for OpenRouterClient."""
    m = AsyncMock()
    m.complete = AsyncMock(return_value="Cloud model answer from OpenRouter.")
    m.health_check = AsyncMock(return_value=True)
    return m


@pytest.fixture
def mock_redis_client():
    """Async mock for aioredis client."""
    m = AsyncMock()
    m.get = AsyncMock(return_value=None)
    m.set = AsyncMock()
    m.setex = AsyncMock()
    m.keys = AsyncMock(return_value=[])
    m.delete = AsyncMock(return_value=0)
    m.ping = AsyncMock(return_value=True)
    m.mget = AsyncMock(return_value=[])
    m.zadd = AsyncMock()
    m.zremrangebyrank = AsyncMock()
    m.zrevrange = AsyncMock(return_value=[])
    m.zrangebyscore = AsyncMock(return_value=[])
    return m


@pytest.fixture
def fixed_embedding():
    """A fixed 384-dimensional unit vector for deterministic embedding tests."""
    import math
    raw = [float(i % 10) for i in range(384)]
    magnitude = math.sqrt(sum(x * x for x in raw))
    return [x / magnitude for x in raw]


@pytest.fixture
def sample_qdrant_hit():
    """A mock ScoredPoint-like object as returned by Qdrant search."""
    hit = MagicMock()
    hit.score = 0.91
    hit.payload = {
        "chunk_id": "chunk-001",
        "document_name": "sample-vendor-contract.md",
        "content": "Either party may terminate this agreement with 90 days written notice.",
        "chunk_index": 3,
    }
    return hit
