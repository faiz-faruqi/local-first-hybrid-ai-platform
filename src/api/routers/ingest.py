"""
Document ingestion endpoints.

POST /ingest/text   — ingest a single document by text content
POST /ingest/batch  — ingest multiple documents in one request
DELETE /ingest/flush-cache — flush all Redis cache entries (admin)

These endpoints allow the seed_demo.py script to populate the vector
store against any deployed backend (local or Railway) via HTTP, without
needing direct database access.

The flush-cache endpoint is protected by an X-Admin-Key header to
prevent accidental or unauthorised cache invalidation in production.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.admin_auth import verify_admin_key
from src.api.dependencies import get_cache, get_embedder, get_vector_store_dep
from src.cache.redis_cache import ResponseCache
from src.models.schemas import IngestRequest, IngestResponse
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store_factory import VectorStoreType

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


@router.post("/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(
    request: IngestRequest,
    embedder: Embedder = Depends(get_embedder),
    store: VectorStoreType = Depends(get_vector_store_dep),
) -> IngestResponse:
    """
    Ingest a single document by text content.

    The document is chunked, embedded, and indexed into the vector store.
    Existing chunks for the same document name are NOT automatically removed —
    re-ingesting the same document will create duplicate entries.
    Use the /flush-cache endpoint and re-index to refresh a document.
    """
    chunks = _chunk_text(request.content)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document content produced no usable chunks after splitting.",
        )

    chunk_dicts = [
        {
            "document_name": request.document_name,
            "content": chunk,
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]

    vectors = embedder.embed([c["content"] for c in chunk_dicts])
    count = await store.upsert_chunks(chunk_dicts, vectors)

    logger.info(
        "Ingested document '%s' → %d chunks into collection '%s'.",
        request.document_name,
        count,
        request.collection,
    )

    return IngestResponse(
        chunks_indexed=count,
        collection=request.collection,
        document_name=request.document_name,
    )


@router.post("/batch", status_code=status.HTTP_201_CREATED)
async def ingest_batch(
    requests: list[IngestRequest],
    embedder: Embedder = Depends(get_embedder),
    store: VectorStoreType = Depends(get_vector_store_dep),
) -> dict:
    """
    Ingest multiple documents in a single request.

    Returns a summary of chunks indexed per document.
    """
    results = []
    total = 0
    for req in requests:
        chunks = _chunk_text(req.content)
        if not chunks:
            logger.warning("Document '%s' produced no chunks — skipping.", req.document_name)
            continue
        chunk_dicts = [
            {"document_name": req.document_name, "content": c, "chunk_index": i}
            for i, c in enumerate(chunks)
        ]
        vectors = embedder.embed([c["content"] for c in chunk_dicts])
        count = await store.upsert_chunks(chunk_dicts, vectors)
        total += count
        results.append({"document_name": req.document_name, "chunks_indexed": count})
        logger.info("Batch: ingested '%s' → %d chunks.", req.document_name, count)

    return {"total_chunks_indexed": total, "documents": results}


@router.delete("/flush-cache", dependencies=[Depends(verify_admin_key)])
async def flush_cache(
    cache: ResponseCache = Depends(get_cache),
) -> dict:
    """
    Flush all LLM response cache entries.

    Protected by X-Admin-Key header. Use after re-indexing documents
    to prevent stale cached responses from being served.
    """
    deleted = await cache.flush()
    logger.info("Cache flushed — %d keys deleted.", deleted)
    return {"keys_deleted": deleted, "status": "cache flushed"}
