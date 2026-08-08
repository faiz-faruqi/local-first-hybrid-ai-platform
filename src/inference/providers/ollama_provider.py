"""
Ollama-backed Provider.

Wraps the existing `OllamaClient` behind the unified `Provider` interface.
The OllamaClient already handles the REST call to the local inference node;
this adapter just attaches catalog metadata and exposes the `Provider` contract.
"""

from collections.abc import AsyncGenerator

from src.config.models import ModelDefinition
from src.inference.base_provider import Provider
from src.inference.ollama_client import OllamaClient


class OllamaProvider(Provider):
    """
    Local inference provider backed by Ollama.

    Reuses the existing OllamaClient — no new HTTP logic is introduced.
    The model_id from the catalog is passed to OllamaClient so a single
    Ollama node can serve multiple local models (gemma2, llama3.1, etc.).
    """

    def __init__(self, model_def: ModelDefinition) -> None:
        self._info = model_def
        self._client = OllamaClient(model=model_def.model_id)

    @property
    def info(self) -> ModelDefinition:
        return self._info

    async def complete(self, prompt: str) -> str:
        return await self._client.complete(prompt)

    async def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        async for chunk in self._client.stream_complete(prompt):
            yield chunk

    async def health_check(self) -> bool:
        return await self._client.health_check()
