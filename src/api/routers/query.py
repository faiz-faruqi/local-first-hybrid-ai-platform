"""
Query endpoint.

This is the core of the platform — the endpoint that:
  1. Classifies the user query to produce a QueryProfile
  2. Conditionally embeds and retrieves documents (RAG) if rag_needed=True
  3. Assembles a grounded (RAG) or direct prompt
  4. Checks the Redis cache
  5. Routes to the optimal inference model via the Decision Engine
  6. Writes the response to cache
  7. Returns the answer with source attribution and routing metadata

RAG is bypassed entirely when the classifier determines the query is a
general knowledge question (rag_needed=False), saving embedding compute
and vector-store round-trips for those requests.

Each step maps to a distinct architectural component, keeping the
orchestration logic readable and each component independently testable.

Steps 1-4 (classify -> conditional RAG -> prompt assembly -> cache lookup)
are shared with the streaming endpoint via `_prepare()`, so both entry
points produce identically-shaped traces and identical routing behaviour.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

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
from src.api.rate_limit import RateLimiter, get_client_ip
from src.cache.redis_cache import ResponseCache
from src.inference.router import InferenceRouter, ProviderResult
from src.models.classification import QueryProfile
from src.models.schemas import (
    InferenceProvider,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from src.observability.tracer import Tracer
from src.observability.trace_store import TraceStore
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store_factory import VectorStoreType
from src.routing.classifier import QueryClassifier
from src.routing.decision_engine import DecisionEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Used when RAG retrieval has run — grounds the answer in document context.
GROUNDED_PROMPT_TEMPLATE = """You are an enterprise document assistant. Answer the user's question \
using ONLY the information in the provided context. If the answer is not present in the \
context, say so clearly. Do not speculate or use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""

# Used when RAG is skipped — the model answers from its own knowledge.
DIRECT_PROMPT_TEMPLATE = """You are a helpful enterprise AI assistant. \
Answer the following question clearly and concisely.

Question: {question}

Answer:"""


def _sse_event(data: dict) -> str:
    """Format a single Server-Sent Events frame."""
    return f"data: {json.dumps(data)}\n\n"


async def _enforce_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """
    Per-IP cost guard for the public demo. Applied only to the two query
    routes below — /health, /models, /telemetry/* stay unaffected.
    """
    await rate_limiter.check(get_client_ip(request))


async def _prepare(
    request: QueryRequest,
    classifier: QueryClassifier,
    embedder: Embedder,
    store: VectorStoreType,
    cache: ResponseCache,
    tracer: Tracer,
) -> tuple[str, QueryProfile, list[SourceDocument], str | None]:
    """
    Shared pipeline steps for both POST / and POST /stream.

    Returns (prompt, classification_profile, sources, cached_answer).
    `cached_answer` is None on a cache miss.
    """
    async with tracer.span("classify"):
        profile = classifier.classify(request.query)
    tracer.set(
        classification=profile.model_dump(mode="json"),
        rag_needed=profile.rag_needed,
        department=request.department,
    )

    sources: list[SourceDocument] = []

    if profile.rag_needed:
        async with tracer.span("retrieve"):
            query_vector = embedder.embed_single(request.query)
            hits = await store.search(query_vector, top_k=request.top_k)
            if not hits:
                raise HTTPException(
                    status_code=404,
                    detail="No relevant documents found. Ensure documents have been ingested.",
                )

            sources = [
                SourceDocument(
                    chunk_id=hit.payload["chunk_id"],
                    document_name=hit.payload["document_name"],
                    content_preview=hit.payload["content"][:300],
                    relevance_score=round(hit.score, 4),
                )
                for hit in hits
            ]

            context_blocks = "\n\n".join(
                f"[{hit.payload['document_name']}]\n{hit.payload['content']}"
                for hit in hits
            )
        prompt = GROUNDED_PROMPT_TEMPLATE.format(
            context=context_blocks,
            question=request.query,
        )
        logger.info("RAG pipeline active: %d chunks retrieved.", len(hits))
    else:
        prompt = DIRECT_PROMPT_TEMPLATE.format(question=request.query)
        logger.info("RAG bypassed: classifier flagged query as general knowledge.")

    async with tracer.span("cache_lookup"):
        cached_answer = await cache.get(prompt)

    return prompt, profile, sources, cached_answer


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    embedder: Embedder = Depends(get_embedder),
    store: VectorStoreType = Depends(get_vector_store_dep),
    cache: ResponseCache = Depends(get_cache),
    inference_router: InferenceRouter = Depends(get_inference_router),
    classifier: QueryClassifier = Depends(get_classifier),
    decision_engine: DecisionEngine = Depends(get_decision_engine),
    trace_store: TraceStore = Depends(get_trace_store),
    _rate_limit: None = Depends(_enforce_rate_limit),
) -> QueryResponse:
    """
    Process a natural language query over indexed enterprise documents.

    Steps:
      1. Classify the query (Phase 2) — produces rag_needed flag
      2. If rag_needed: embed → retrieve → build grounded prompt
         If not rag_needed: build a direct prompt, skip vector store
      3. Check Redis cache
      4. Run inference via Decision Engine (Phase 3)
      5. Cache and return response with source attribution

    Every request is traced (see src/observability/) regardless of outcome —
    cache hits, automatic/explicit/force-cloud routing, and errors all
    produce a trace record for the /telemetry endpoints.
    """
    start = time.perf_counter()
    tracer = Tracer(trace_store, route="/query/", query=request.query)

    try:
        prompt, profile, sources, cached_answer = await _prepare(
            request, classifier, embedder, store, cache, tracer
        )

        if cached_answer:
            latency = (time.perf_counter() - start) * 1000
            tracer.set(provider=InferenceProvider.cache.value, cached=True)
            return QueryResponse(
                answer=cached_answer,
                provider=InferenceProvider.cache,
                cached=True,
                sources=sources,
                latency_ms=round(latency, 2),
                classification=profile,
            )

        # ── Inference ────────────────────────────────────────────
        # Phase 3: automatic model selection via the Decision Engine.
        #   - If the caller specified an explicit model (request.model),
        #     use it directly (Phase 1 explicit selection).
        #   - If force_cloud is set, use the legacy cloud path.
        #   - Otherwise, let the Decision Engine select the optimal model
        #     based on the QueryProfile and cost budget.
        routing_decision = None

        async with tracer.span("routing"):
            if request.model:
                tracer.set(routing_mode="explicit_model")
                answer, used_provider, model_alias = await inference_router.complete_with_model(
                    prompt=prompt,
                    model_alias=request.model,
                )
            elif request.force_cloud:
                tracer.set(routing_mode="force_cloud")
                answer, used_provider = await inference_router.complete(
                    prompt=prompt,
                    force_cloud=True,
                )
                model_alias = None
            else:
                tracer.set(routing_mode="automatic")
                routing_decision = await decision_engine.decide(profile)
                answer, used_provider, model_alias = await inference_router.complete_with_model(
                    prompt=prompt,
                    model_alias=routing_decision.selected_model,
                )

        provider_enum = (
            InferenceProvider.local
            if used_provider == ProviderResult.LOCAL
            else InferenceProvider.cloud
        )

        async with tracer.span("cache_write"):
            await cache.set(prompt, answer)

        latency = (time.perf_counter() - start) * 1000
        tracer.set(
            provider=provider_enum.value,
            cached=False,
            model_alias=model_alias,
            routing_decision=routing_decision.model_dump(mode="json") if routing_decision else None,
        )
        return QueryResponse(
            answer=answer,
            provider=provider_enum,
            cached=False,
            sources=sources,
            latency_ms=round(latency, 2),
            model_alias=model_alias,
            classification=profile,
            routing_decision=routing_decision,
        )
    except Exception as exc:
        tracer.set(error=str(exc))
        raise
    finally:
        await tracer.finish()


@router.post("/stream")
async def query_stream(
    request: QueryRequest,
    embedder: Embedder = Depends(get_embedder),
    store: VectorStoreType = Depends(get_vector_store_dep),
    cache: ResponseCache = Depends(get_cache),
    inference_router: InferenceRouter = Depends(get_inference_router),
    classifier: QueryClassifier = Depends(get_classifier),
    decision_engine: DecisionEngine = Depends(get_decision_engine),
    trace_store: TraceStore = Depends(get_trace_store),
    _rate_limit: None = Depends(_enforce_rate_limit),
) -> StreamingResponse:
    """
    Server-Sent Events variant of POST / — streams the answer token by
    token as it's generated instead of waiting for the full completion.

    Shares Steps 1-4 (classify, conditional RAG, prompt assembly, cache
    lookup) with POST / via `_prepare()`. A streamed response has no
    separate JSON body, so the final `event: done` frame carries the same
    metadata (`provider`, `cached`, `latency_ms`, `sources`, `classification`,
    `routing_decision`) that POST / returns directly.

    Cache hits are still streamed as a single-chunk SSE response (rather
    than returned as plain JSON) so the frontend's SSE parser never has to
    special-case response shape by cache status.
    """
    start = time.perf_counter()
    tracer = Tracer(trace_store, route="/query/stream", query=request.query)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            prompt, profile, sources, cached_answer = await _prepare(
                request, classifier, embedder, store, cache, tracer
            )
            source_payload = [s.model_dump(mode="json") for s in sources]
            classification_payload = profile.model_dump(mode="json")

            if cached_answer:
                tracer.set(provider=InferenceProvider.cache.value, cached=True)
                yield _sse_event({"delta": cached_answer})
                latency = (time.perf_counter() - start) * 1000
                yield _sse_event({
                    "event": "done",
                    "provider": InferenceProvider.cache.value,
                    "cached": True,
                    "latency_ms": round(latency, 2),
                    "model_alias": None,
                    "sources": source_payload,
                    "classification": classification_payload,
                    "routing_decision": None,
                })
                return

            routing_decision = None
            provider_result: dict = {}

            async with tracer.span("routing"):
                if request.model:
                    tracer.set(routing_mode="explicit_model")
                    model_alias = request.model
                    gen = inference_router.stream_complete_with_model(
                        prompt, model_alias, provider_result=provider_result
                    )
                elif request.force_cloud:
                    tracer.set(routing_mode="force_cloud")
                    model_alias = None
                    provider_result["provider"] = ProviderResult.CLOUD
                    gen = inference_router.stream_complete(prompt, force_cloud=True)
                else:
                    tracer.set(routing_mode="automatic")
                    routing_decision = await decision_engine.decide(profile)
                    model_alias = routing_decision.selected_model
                    gen = inference_router.stream_complete_with_model(
                        prompt, model_alias, provider_result=provider_result
                    )

                full_answer = ""
                first_chunk = True
                async for chunk in gen:
                    if first_chunk:
                        tracer.mark_ttft()
                        first_chunk = False
                    full_answer += chunk
                    yield _sse_event({"delta": chunk})

            provider_enum = (
                InferenceProvider.local
                if provider_result.get("provider") == ProviderResult.LOCAL
                else InferenceProvider.cloud
            )

            async with tracer.span("cache_write"):
                await cache.set(prompt, full_answer)

            latency = (time.perf_counter() - start) * 1000
            tracer.set(
                provider=provider_enum.value,
                cached=False,
                model_alias=model_alias,
                routing_decision=routing_decision.model_dump(mode="json") if routing_decision else None,
            )
            yield _sse_event({
                "event": "done",
                "provider": provider_enum.value,
                "cached": False,
                "latency_ms": round(latency, 2),
                "model_alias": model_alias,
                "sources": source_payload,
                "classification": classification_payload,
                "routing_decision": routing_decision.model_dump(mode="json") if routing_decision else None,
            })
        except Exception as exc:
            # Headers/status are already committed to the client by the time a
            # mid-stream failure can happen, so an HTTP error status is not an
            # option here — signal failure as a terminal SSE event instead.
            tracer.set(error=str(exc))
            yield _sse_event({"event": "error", "message": str(exc)})
        finally:
            await tracer.finish()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
