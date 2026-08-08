"""
OpenRouter-backed Provider.

Wraps the existing `OpenRouterClient` behind the unified `Provider` interface.
Because OpenRouter is a unified gateway over OpenAI, Anthropic, Google, Meta,
etc., a single OpenRouterClient class can serve any cloud model — we just
instantiate it with the model_id from the catalog entry.

This is the key reuse insight: we do NOT need separate SDK integrations per
vendor. One OpenRouterClient per model_id gives us GPT-4o, Claude, Gemini,
and Llama through a single, consistent API surface.
"""

from collections.abc import AsyncGenerator

from src.config.models import ModelDefinition
from src.inference.base_provider import Provider
from src.inference.openrouter_client import OpenRouterClient


class OpenRouterProvider(Provider):
    """
    Cloud inference provider backed by OpenRouter.

    Each instance targets exactly one model (its catalog model_id).
    The Decision Engine selects among many such instances.
    """

    def __init__(self, model_def: ModelDefinition) -> None:
        self._info = model_def
        self._client = OpenRouterClient(model=model_def.model_id)

    @property
    def info(self) -> ModelDefinition:
        return self._info

    async def complete(self, prompt: str) -> str:
        return await self._client.complete(prompt)

    async def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        async for chunk in self._client.stream_complete(prompt):
            yield chunk

    async def health_check(self) -> bool:
        """
        For cloud providers, a health check is a lightweight reachability probe.

        We treat the presence of an API key as sufficient readiness for the
        registry — actual failures are handled by the fallback chain (Phase 4).
        """
        from src.inference.openrouter_client import OPENROUTER_API_KEY
        return bool(OPENROUTER_API_KEY)
