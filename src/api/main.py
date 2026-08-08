"""
FastAPI application entry point.

Registers routers, configures middleware, and exposes a /health
endpoint for container orchestration health checks.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import auth, ingest, models, query, telemetry

logging.basicConfig(level=os.getenv("LOG_LEVEL", "info").upper())
logger = logging.getLogger(__name__)

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

app = FastAPI(
    title="Enterprise AI Gateway",
    description=(
        "Intelligent Multi-LLM Routing Platform — a policy-driven gateway that "
        "classifies queries by complexity, domain, sensitivity, and cost, then "
        "routes each request to the optimal model (GPT-4o, Claude, Gemini, local Llama). "
        "Includes RAG orchestration, semantic caching, budget enforcement, and "
        "intelligent fallback.\n\n"
        + ("**Demo mode active** — local inference tier uses mistral-7b-instruct via OpenRouter." if DEMO_MODE else "")
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(telemetry.router, prefix="/telemetry", tags=["Telemetry"])
app.include_router(auth.router, prefix="/auth", tags=["Auth"])


@app.get("/health", tags=["System"])
async def health() -> dict:
    """
    Liveness + readiness probe.

    Returns component status for Redis, vector store, and (if configured) Ollama.
    Railway and Docker healthchecks use this endpoint.
    """
    from src.api.dependencies import get_cache, get_vector_store_dep
    from src.inference.ollama_client import OllamaClient, OLLAMA_BASE_URL
    from src.retrieval.vector_store_factory import VECTOR_STORE_BACKEND

    status: dict = {
        "status": "ok",
        "version": "0.3.0",
        "demo_mode": DEMO_MODE,
        "vector_store": VECTOR_STORE_BACKEND,
        "components": {},
    }

    # Redis health
    try:
        cache = get_cache()
        redis_ok = await cache.health_check()
        status["components"]["redis"] = "ok" if redis_ok else "unreachable"
    except Exception as exc:
        status["components"]["redis"] = f"error: {exc}"

    # Vector store health
    try:
        store = get_vector_store_dep()
        if hasattr(store, "health_check"):
            vs_ok = await store.health_check()
            status["components"]["vector_store"] = "ok" if vs_ok else "unreachable"
        else:
            status["components"]["vector_store"] = "unknown"
    except Exception as exc:
        status["components"]["vector_store"] = f"error: {exc}"

    # Ollama health (optional — only if OLLAMA_BASE_URL is set and not demo mode)
    if not DEMO_MODE and OLLAMA_BASE_URL:
        try:
            ollama = OllamaClient()
            ollama_ok = await ollama.health_check()
            status["components"]["ollama"] = "ok" if ollama_ok else "unreachable"
        except Exception as exc:
            status["components"]["ollama"] = f"error: {exc}"

    return status


@app.get("/", tags=["System"])
async def root() -> dict:
    return {
        "service": "enterprise-ai-gateway",
        "version": "0.3.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
        "demo_mode": DEMO_MODE,
    }
