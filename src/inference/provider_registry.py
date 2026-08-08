"""
Provider Registry — the gateway's model catalogue as live objects.

Builds `Provider` instances from the static `MODEL_CATALOG` and exposes
them by alias. The registry is the single place the Decision Engine
(Phase 3) and Fallback Chain (Phase 4) look up available models.

Design:
  - Local models (is_local=True) → OllamaProvider
  - Cloud models (is_local=False) → OpenRouterProvider
  - The registry is a singleton (instantiated once via DI).

Adding a model requires only a new entry in src/config/models.py — the
registry picks it up automatically.
"""

import logging
import os

from src.config.models import MODEL_CATALOG, ModelDefinition, list_models
from src.inference.base_provider import Provider
from src.inference.providers.ollama_provider import OllamaProvider
from src.inference.providers.openrouter_provider import OpenRouterProvider
from src.routing.policies import tier_index

logger = logging.getLogger(__name__)

# Caps which model tiers are visible/routable at all — not just
# de-prioritized. Default "premium" is unrestricted (every tier is <=
# premium in TIER_ORDER). Set to e.g. "standard" on a public demo to make
# premium models (gpt-5, claude-opus) invisible everywhere: GET /models,
# automatic tier selection, and fallback chains all key off this same
# registry, so filtering here is the single choke point for all of them.
MAX_MODEL_TIER = os.getenv("MAX_MODEL_TIER", "premium")


class ProviderRegistry:
    """
    Holds live `Provider` instances keyed by model alias.

    Usage:
        registry = ProviderRegistry()
        provider = registry.get("gpt-4o")
        answer = await provider.complete(prompt)

        # Iterate all providers of a given tier:
        for p in registry.by_tier("premium"):
            ...
    """

    def __init__(self, max_tier: str = MAX_MODEL_TIER) -> None:
        self._max_tier = max_tier
        self._providers: dict[str, Provider] = {}
        self._build()

    def _build(self) -> None:
        """Instantiate a Provider for every catalog entry at or below max_tier."""
        max_tier_index = tier_index(self._max_tier)
        for model_def in list_models():
            if tier_index(model_def.tier) > max_tier_index:
                logger.info(
                    "Skipping provider: alias=%s tier=%s (above MAX_MODEL_TIER=%s)",
                    model_def.alias, model_def.tier, self._max_tier,
                )
                continue
            provider = self._make_provider(model_def)
            self._providers[model_def.alias] = provider
            logger.info(
                "Registered provider: alias=%s vendor=%s tier=%s local=%s",
                model_def.alias, model_def.vendor, model_def.tier, model_def.is_local,
            )

    @staticmethod
    def _make_provider(model_def: ModelDefinition) -> Provider:
        if model_def.is_local:
            return OllamaProvider(model_def)
        return OpenRouterProvider(model_def)

    def get(self, alias: str) -> Provider | None:
        """Return the Provider for a model alias, or None if unknown."""
        return self._providers.get(alias)

    def all(self) -> list[Provider]:
        """Return every registered provider."""
        return list(self._providers.values())

    def by_tier(self, tier: str) -> list[Provider]:
        """Return all providers matching a tier (local/cheap/standard/premium)."""
        return [p for p in self._providers.values() if p.info.tier == tier]

    def local_providers(self) -> list[Provider]:
        """Return only on-premises (local) providers."""
        return [p for p in self._providers.values() if p.info.is_local]

    def cloud_providers(self) -> list[Provider]:
        """Return only cloud-hosted providers."""
        return [p for p in self._providers.values() if not p.info.is_local]

    def aliases(self) -> list[str]:
        """Return all registered model aliases."""
        return list(self._providers.keys())

    def describe(self) -> list[dict]:
        """Return a JSON-serialisable summary of every model (for /models endpoint)."""
        return [
            {
                "alias": p.info.alias,
                "vendor": p.info.vendor,
                "model_id": p.info.model_id,
                "tier": p.info.tier,
                "context_window": p.info.context_window,
                "cost_per_1k_input": p.info.cost_per_1k_input,
                "cost_per_1k_output": p.info.cost_per_1k_output,
                "latency_tier": p.info.latency_tier,
                "is_local": p.info.is_local,
                "description": p.info.description,
            }
            for p in self._providers.values()
        ]
