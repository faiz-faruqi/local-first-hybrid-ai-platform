"""
Ollama local inference client.

Wraps the Ollama REST API for on-premises LLM inference.
Configured via OLLAMA_BASE_URL and OLLAMA_MODEL environment variables.
"""

import json
import logging
import os
from collections.abc import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")


class OllamaClient:
    """
    Thin async client for the Ollama /api/generate endpoint.

    Raises httpx.HTTPError on connection failures, allowing the
    InferenceRouter to detect unavailability and trigger fallback.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._timeout = timeout

    async def complete(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the generated text."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(  # fix: was missing `await`
                f"{self._base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]

    async def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream a prompt completion from Ollama as it's generated.

        Ollama's /api/generate returns newline-delimited JSON objects when
        stream=True, each carrying an incremental `response` fragment and
        a final object with `done: true`.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", f"{self._base_url}/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("response"):
                        yield data["response"]
                    if data.get("done"):
                        break

    async def health_check(self) -> bool:
        """Return True if the Ollama node is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
