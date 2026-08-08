"""
Unit tests for OllamaClient.stream_complete() and OpenRouterClient.stream_complete().

Uses httpx.MockTransport to simulate NDJSON/SSE chunk framing without any
real network calls. httpx.AsyncClient is patched at construction time to
inject the mock transport, since neither client accepts one directly.
"""

from unittest.mock import patch

import httpx
import pytest

from src.inference.ollama_client import OllamaClient
from src.inference.openrouter_client import OpenRouterClient

# Captured before any patching — `httpx.AsyncClient` gets patched in-place
# below (patch() mutates the real, shared httpx module), so factories must
# build clients from this original class rather than the module attribute,
# or they'd recurse into their own patch.
_RealAsyncClient = httpx.AsyncClient


def mock_transport_client(body: str):
    """Return a patch target that builds a real httpx.AsyncClient wired to a MockTransport."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    transport = httpx.MockTransport(handler)

    def factory(**kwargs):
        return _RealAsyncClient(transport=transport)

    return factory


class TestOllamaStreamComplete:
    async def test_yields_response_fragments_in_order(self):
        body = (
            '{"response": "Hel", "done": false}\n'
            '{"response": "lo", "done": false}\n'
            '{"response": "", "done": true, "total_duration": 123}\n'
        )
        client = OllamaClient(base_url="http://ollama.test", model="test-model")

        with patch("src.inference.ollama_client.httpx.AsyncClient", side_effect=mock_transport_client(body)):
            chunks = [c async for c in client.stream_complete("hello")]

        assert chunks == ["Hel", "lo"]

    async def test_stops_at_done_even_with_trailing_lines(self):
        body = (
            '{"response": "only", "done": true}\n'
            '{"response": "should not appear", "done": false}\n'
        )
        client = OllamaClient(base_url="http://ollama.test", model="test-model")

        with patch("src.inference.ollama_client.httpx.AsyncClient", side_effect=mock_transport_client(body)):
            chunks = [c async for c in client.stream_complete("hello")]

        assert chunks == ["only"]

    async def test_raises_on_http_error(self):
        client = OllamaClient(base_url="http://ollama.test", model="test-model")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, content="server error")

        transport = httpx.MockTransport(handler)

        def factory(**kwargs):
            return _RealAsyncClient(transport=transport)

        with patch("src.inference.ollama_client.httpx.AsyncClient", side_effect=factory):
            with pytest.raises(httpx.HTTPStatusError):
                async for _ in client.stream_complete("hello"):
                    pass


class TestOpenRouterStreamComplete:
    async def test_yields_delta_fragments_and_stops_at_done(self):
        body = (
            'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n'
            'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        client = OpenRouterClient(api_key="test-key", model="test-model")

        with patch("src.inference.openrouter_client.httpx.AsyncClient", side_effect=mock_transport_client(body)):
            chunks = [c async for c in client.stream_complete("hello")]

        assert chunks == ["Hel", "lo"]

    async def test_skips_deltas_with_no_content(self):
        body = (
            'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
            'data: {"choices": [{"delta": {"content": "answer"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        client = OpenRouterClient(api_key="test-key", model="test-model")

        with patch("src.inference.openrouter_client.httpx.AsyncClient", side_effect=mock_transport_client(body)):
            chunks = [c async for c in client.stream_complete("hello")]

        assert chunks == ["answer"]

    async def test_raises_without_api_key(self):
        client = OpenRouterClient(api_key="", model="test-model")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            async for _ in client.stream_complete("hello"):
                pass
