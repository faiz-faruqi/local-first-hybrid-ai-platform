"""
OpenRouter cloud inference client.

OpenRouter provides a unified API over multiple LLM providers
(OpenAI, Anthropic, Mistral, etc.), acting as an abstraction layer.
This satisfies the vendor-neutrality design principle — swapping
the underlying cloud model requires only an environment variable change.
"""

import json
import logging
import os
from collections.abc import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


class OpenRouterClient:
    """
    Async client for OpenRouter's OpenAI-compatible completions API.

    OpenRouter is used as the cloud fallback path (see ADR 0001).
    Model selection is configurable — default targets a cost-efficient
    model appropriate for document Q&A.
    """

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model: str = OPENROUTER_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def complete(self, prompt: str) -> str:
        """Send a prompt to OpenRouter and return the generated text."""
        if not self._api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Cloud fallback is unavailable."
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/faiz-faruqi/local-first-hybrid-ai-platform",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream a prompt completion from OpenRouter as it's generated.

        OpenRouter's chat/completions endpoint is OpenAI-compatible: with
        stream=True it returns Server-Sent Events, each `data: {...}` line
        carrying an incremental `delta.content` fragment, terminated by a
        literal `data: [DONE]` line.
        """
        if not self._api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Cloud fallback is unavailable."
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/faiz-faruqi/local-first-hybrid-ai-platform",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    if not raw_line.startswith("data:"):
                        continue
                    data_str = raw_line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
