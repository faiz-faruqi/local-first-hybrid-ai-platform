"""
Unit tests for Tracer.

The TraceStore dependency is mocked — these tests only exercise the
span-timing and record-assembly logic.
"""

from unittest.mock import AsyncMock

import pytest

from src.observability.tracer import Tracer


class TestTracer:
    @pytest.fixture
    def mock_store(self):
        m = AsyncMock()
        m.write_trace = AsyncMock()
        return m

    @pytest.fixture
    def tracer(self, mock_store):
        return Tracer(mock_store, route="/query/", query="What are the termination conditions?")

    async def test_span_records_duration(self, tracer):
        async with tracer.span("classify"):
            pass
        await tracer.finish()

        record = tracer._store.write_trace.call_args[0][0]
        assert len(record["spans"]) == 1
        assert record["spans"][0]["name"] == "classify"
        assert record["spans"][0]["duration_ms"] >= 0

    async def test_span_records_duration_even_when_block_raises(self, tracer):
        """A 404 raised mid-span (e.g. no RAG hits) should still leave a span behind."""
        with pytest.raises(ValueError):
            async with tracer.span("retrieve"):
                raise ValueError("no hits")
        await tracer.finish()

        record = tracer._store.write_trace.call_args[0][0]
        assert [s["name"] for s in record["spans"]] == ["retrieve"]

    async def test_multiple_spans_are_recorded_in_order(self, tracer):
        async with tracer.span("classify"):
            pass
        async with tracer.span("cache_lookup"):
            pass
        await tracer.finish()

        record = tracer._store.write_trace.call_args[0][0]
        assert [s["name"] for s in record["spans"]] == ["classify", "cache_lookup"]

    async def test_mark_ttft_is_idempotent(self, tracer):
        tracer.mark_ttft()
        first_value = tracer._fields["ttft_ms"]
        tracer.mark_ttft()
        assert tracer._fields["ttft_ms"] == first_value

    async def test_set_merges_fields_into_record(self, tracer):
        tracer.set(provider="local", cached=False)
        await tracer.finish()

        record = tracer._store.write_trace.call_args[0][0]
        assert record["provider"] == "local"
        assert record["cached"] is False

    async def test_finish_writes_exactly_once(self, tracer):
        await tracer.finish()
        await tracer.finish()  # idempotent — should not write twice
        assert tracer._store.write_trace.call_count == 1

    async def test_finish_includes_core_fields(self, tracer):
        await tracer.finish()

        record = tracer._store.write_trace.call_args[0][0]
        assert record["route"] == "/query/"
        assert record["query_preview"] == "What are the termination conditions?"
        assert "trace_id" in record
        assert "started_at" in record
        assert "total_latency_ms" in record

    async def test_query_preview_is_truncated(self, mock_store):
        long_query = "x" * 500
        tracer = Tracer(mock_store, route="/query/", query=long_query)
        await tracer.finish()

        record = tracer._store.write_trace.call_args[0][0]
        assert len(record["query_preview"]) == 300
