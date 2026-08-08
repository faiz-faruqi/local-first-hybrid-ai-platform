"""
Production-scale traffic simulation (dry run).

Sends a mix of streaming/non-streaming, RAG/general-knowledge, and
automatic/explicit/force-cloud queries from many concurrent virtual users
against a running gateway instance. Intended to run against the stub LLM
server (docker-compose.loadtest.yml) so it costs nothing and needs no real
Ollama/OpenRouter credentials — its purpose is to populate the /telemetry
trace store with realistic volume for the dashboard to render, and to
exercise the classifier/decision-engine/cache/routing pipeline under
concurrency.

Usage:
    docker compose -f docker-compose.yml -f docker-compose.loadtest.yml up -d --build
    python scripts/ingest_documents.py --input-dir ./sample-docs
    python scripts/loadtest/run_loadtest.py --concurrency 50 --duration 180
"""

import argparse
import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field

import httpx

RAG_QUERIES = [
    "What are the termination conditions in the vendor contracts?",
    "Which contracts have data protection obligations?",
    "What is the total contract value across the portfolio?",
    "What SLA commitments are specified in the agreements?",
    "What liability caps are defined in the indemnity clauses?",
    "Summarize the data processing agreement's confidentiality terms.",
]

GENERAL_QUERIES = [
    "What is the capital of France?",
    "Explain how neural networks learn.",
    "What's the difference between TCP and UDP?",
    "Give me a short overview of supply and demand.",
    "What is machine learning?",
    "How does photosynthesis work?",
]

EXPLICIT_MODELS = ["gpt-4o-mini", "claude-haiku", "gemini-flash", "local-gemma"]

DEPARTMENTS = ["IT", "Sales", "Marketing", "Finance"]

# Routing-mode weights: mostly automatic (exercises the Decision Engine),
# occasional explicit/force-cloud for distribution variety.
ROUTING_MODES = ["automatic", "automatic", "automatic", "explicit", "force_cloud"]


@dataclass
class Metrics:
    latencies_ms: list[float] = field(default_factory=list)
    statuses: dict[int, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    streamed: int = 0
    non_streamed: int = 0
    total: int = 0

    def record(self, status: int, latency_ms: float, streaming: bool, error: str | None = None):
        self.total += 1
        self.statuses[status] = self.statuses.get(status, 0) + 1
        self.latencies_ms.append(latency_ms)
        if streaming:
            self.streamed += 1
        else:
            self.non_streamed += 1
        if error:
            self.errors.append(error)


def build_request(rng: random.Random, department: str) -> dict:
    is_rag = rng.random() < 0.5
    query = rng.choice(RAG_QUERIES if is_rag else GENERAL_QUERIES)
    body: dict = {"query": query, "top_k": 5, "department": department}

    mode = rng.choice(ROUTING_MODES)
    if mode == "explicit":
        body["model"] = rng.choice(EXPLICIT_MODELS)
    elif mode == "force_cloud":
        body["force_cloud"] = True

    return body


async def drain_stream(response: httpx.Response) -> None:
    """Fully consume a streamed response body, like a real client would."""
    async for _ in response.aiter_lines():
        pass


async def send_one(
    client: httpx.AsyncClient, rng: random.Random, streaming_ratio: float, metrics: Metrics, department: str
) -> None:
    body = build_request(rng, department)
    streaming = rng.random() < streaming_ratio
    path = "/query/stream" if streaming else "/query/"
    start = time.perf_counter()
    status = 0
    error = None
    try:
        if streaming:
            async with client.stream("POST", path, json=body) as response:
                status = response.status_code
                await drain_stream(response)
        else:
            response = await client.post(path, json=body)
            status = response.status_code
    except httpx.HTTPError as exc:
        error = str(exc)
    latency_ms = (time.perf_counter() - start) * 1000
    metrics.record(status, latency_ms, streaming, error)


async def virtual_user(
    user_id: int,
    client: httpx.AsyncClient,
    duration_s: float,
    streaming_ratio: float,
    metrics: Metrics,
) -> None:
    rng = random.Random(user_id * 7919 + int(time.time()))
    # A stable simulated identity for this virtual user's whole run — models
    # "someone from Finance" using the demo, not a department picked per query.
    department = rng.choice(DEPARTMENTS)
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        await send_one(client, rng, streaming_ratio, metrics, department)
        await asyncio.sleep(rng.uniform(0.5, 3.0))


def print_summary(metrics: Metrics, duration_s: float) -> None:
    print()
    print("=" * 60)
    print("Load simulation summary")
    print("=" * 60)
    print(f"Total requests:     {metrics.total}")
    print(f"Duration:           {duration_s:.0f}s")
    print(f"Throughput:         {metrics.total / duration_s:.2f} req/s")
    print(f"Streaming:          {metrics.streamed}  Non-streaming: {metrics.non_streamed}")
    print(f"Status codes:       {dict(sorted(metrics.statuses.items()))}")
    if metrics.errors:
        print(f"Transport errors:   {len(metrics.errors)} (first: {metrics.errors[0]})")
    if metrics.latencies_ms:
        sorted_lat = sorted(metrics.latencies_ms)
        p50 = sorted_lat[int(len(sorted_lat) * 0.50)]
        p95 = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)]
        p99 = sorted_lat[min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)]
        print(f"Latency (client):   mean={statistics.mean(sorted_lat):.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms")
    print("=" * 60)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--duration", type=float, default=180.0, help="seconds")
    parser.add_argument("--streaming-ratio", type=float, default=0.4)
    args = parser.parse_args()

    metrics = Metrics()
    print(
        f"Starting dry-run load simulation: {args.concurrency} virtual users, "
        f"{args.duration:.0f}s, streaming_ratio={args.streaming_ratio} -> {args.api_url}"
    )

    async with httpx.AsyncClient(base_url=args.api_url, timeout=30.0) as client:
        users = [
            virtual_user(i, client, args.duration, args.streaming_ratio, metrics)
            for i in range(args.concurrency)
        ]
        await asyncio.gather(*users)

    print_summary(metrics, args.duration)


if __name__ == "__main__":
    asyncio.run(main())
