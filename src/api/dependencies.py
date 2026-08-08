"""
FastAPI dependency injection — shared component singletons.

All heavy objects (embedding model, vector store, Redis client,
inference router) are instantiated once at startup and reused
across requests via FastAPI's Depends() mechanism.

Usage in a router:
    from src.api.dependencies import get_embedder, get_vector_store_dep

    @router.post("/")
    async def query(
        request: QueryRequest,
        embedder: Embedder = Depends(get_embedder),
        store: VectorStoreType = Depends(get_vector_store_dep),
        cache: ResponseCache = Depends(get_cache),
        router: InferenceRouter = Depends(get_inference_router),
    ) -> QueryResponse:
        ...
"""

import logging
from functools import lru_cache

from src.api.access_code import AccessCodeStore
from src.api.rate_limit import RateLimiter
from src.cache.redis_cache import ResponseCache
from src.inference.ollama_client import OllamaClient
from src.inference.openrouter_client import OpenRouterClient
from src.inference.provider_registry import ProviderRegistry
from src.inference.router import InferenceRouter
from src.observability.trace_store import TraceStore
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store_factory import VectorStoreType, get_vector_store
from src.routing.budget import BudgetTracker
from src.routing.classifier import QueryClassifier
from src.routing.decision_engine import DecisionEngine

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Singleton embedding model. Loaded once; reused across all requests."""
    logger.info("Initialising Embedder singleton.")
    return Embedder()


@lru_cache(maxsize=1)
def get_cache() -> ResponseCache:
    """Singleton Redis response cache."""
    logger.info("Initialising ResponseCache singleton.")
    return ResponseCache()


@lru_cache(maxsize=1)
def get_inference_router() -> InferenceRouter:
    """
    Singleton inference router with local (Ollama) and cloud (OpenRouter) clients.

    In DEMO_MODE, the 'local' client actually routes through OpenRouter using
    mistral-7b-instruct — documented in the UI and CLAUDE.md.

    The router is wired with the ProviderRegistry so it can resolve explicit
    model aliases (Phase 1) while preserving the legacy local→cloud fallback.
    """
    logger.info("Initialising InferenceRouter singleton.")
    return InferenceRouter(
        ollama=OllamaClient(),
        openrouter=OpenRouterClient(),
        registry=get_provider_registry(),
    )


@lru_cache(maxsize=1)
def get_provider_registry() -> ProviderRegistry:
    """
    Singleton provider registry — the gateway's live model catalogue.

    Builds a Provider instance for every entry in src/config/models.py.
    Used by the query endpoint (explicit model selection) and, from Phase 3,
    by the Decision Engine for automatic model selection.
    """
    logger.info("Initialising ProviderRegistry singleton.")
    return ProviderRegistry()


@lru_cache(maxsize=1)
def get_classifier() -> QueryClassifier:
    """
    Singleton query classifier (Phase 2).

    Stateless and fast (<1ms per query) — rule-based, no LLM call.
    Produces a QueryProfile consumed by the Decision Engine (Phase 3).
    """
    logger.info("Initialising QueryClassifier singleton.")
    return QueryClassifier()


@lru_cache(maxsize=1)
def get_budget_tracker() -> BudgetTracker:
    """
    Singleton budget tracker (Phase 3).

    Reuses the existing Redis connection from the response cache.
    Budget caps are configured via BUDGET_DAILY_USD / BUDGET_MONTHLY_USD.
    """
    logger.info("Initialising BudgetTracker singleton.")
    return BudgetTracker(get_cache())


@lru_cache(maxsize=1)
def get_trace_store() -> TraceStore:
    """
    Singleton trace store (observability).

    Reuses the existing Redis connection from the response cache — same
    reuse pattern as BudgetTracker — under a separate key namespace.
    """
    logger.info("Initialising TraceStore singleton.")
    return TraceStore(get_cache())


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    """
    Singleton per-IP rate limiter (public demo cost guard).

    Reuses the existing Redis connection from the response cache — same
    reuse pattern as BudgetTracker/TraceStore — under a separate key
    namespace.
    """
    logger.info("Initialising RateLimiter singleton.")
    return RateLimiter(get_cache())


@lru_cache(maxsize=1)
def get_access_code_store() -> AccessCodeStore:
    """
    Singleton demo access-code store.

    Reuses the existing Redis connection from the response cache — same
    reuse pattern as BudgetTracker/RateLimiter/TraceStore — under a
    separate key namespace.
    """
    logger.info("Initialising AccessCodeStore singleton.")
    return AccessCodeStore(get_cache())


@lru_cache(maxsize=1)
def get_decision_engine() -> DecisionEngine:
    """
    Singleton decision engine (Phase 3).

    The core routing brain — consumes a QueryProfile and budget state
    to select the optimal model. Wired with the provider registry and
    budget tracker.
    """
    logger.info("Initialising DecisionEngine singleton.")
    return DecisionEngine(
        registry=get_provider_registry(),
        budget_tracker=get_budget_tracker(),
    )


def get_vector_store_dep() -> VectorStoreType:
    """
    Vector store dependency.

    Not cached with lru_cache because the pgvector client manages its own
    internal connection pool. A new instance per request is cheap since
    the pool is lazily initialised and reused internally.

    For Qdrant (local dev), the instance is also lightweight.
    """
    return get_vector_store()
