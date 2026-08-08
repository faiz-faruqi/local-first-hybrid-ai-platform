"""
Abstract Provider interface for the Enterprise AI Gateway.

Every routable model is wrapped behind a unified `Provider` so the
Decision Engine (Phase 3) and Fallback Chain (Phase 4) can treat
local Ollama models and cloud OpenRouter models polymorphically.

A `Provider` carries:
  - `info`: static metadata (cost, tier, context window) from the catalog
  - `complete(prompt)`: async generation
  - `health_check()`: liveness probe used by the fallback chain
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from src.config.models import ModelDefinition


class Provider(ABC):
    """Unified interface for all inference backends."""

    @property
    @abstractmethod
    def info(self) -> ModelDefinition:
        """Static metadata describing this provider's model."""

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""

    @abstractmethod
    def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream a completion for the given prompt, chunk by chunk."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and ready."""

    @property
    def alias(self) -> str:
        """Convenience accessor for the model alias."""
        return self.info.alias

    @property
    def is_local(self) -> bool:
        """Whether this provider runs on-premises (no external API)."""
        return self.info.is_local
