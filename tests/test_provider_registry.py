"""
Tests for the Phase 1 provider abstraction and registry.

Verifies:
  - The model catalog is non-empty and well-formed.
  - The registry builds a Provider for every catalog entry.
  - Provider type selection (local vs cloud) matches is_local.
  - by_tier filtering works.
  - Explicit model selection routes to the right provider.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.models import MODEL_CATALOG, ModelDefinition, get_model, list_models
from src.inference.base_provider import Provider
from src.inference.provider_registry import ProviderRegistry
from src.inference.providers.ollama_provider import OllamaProvider
from src.inference.providers.openrouter_provider import OpenRouterProvider
from src.inference.router import InferenceRouter, ProviderResult


# ── Model catalog ────────────────────────────────────────────────────────────


class TestModelCatalog:
    def test_catalog_has_entries(self):
        assert len(MODEL_CATALOG) >= 6, "Catalog should have multiple models."

    def test_every_entry_has_required_fields(self):
        for alias, model in MODEL_CATALOG.items():
            assert model.alias == alias
            assert model.vendor
            assert model.model_id
            assert model.tier in {"local", "cheap", "standard", "premium"}
            assert model.context_window > 0
            assert model.cost_per_1k_input >= 0.0
            assert model.cost_per_1k_output >= 0.0
            assert model.latency_tier in {"fast", "balanced", "slow"}
            assert isinstance(model.is_local, bool)

    def test_local_models_have_zero_cost(self):
        for model in MODEL_CATALOG.values():
            if model.is_local:
                assert model.cost_per_1k_input == 0.0
                assert model.cost_per_1k_output == 0.0

    def test_get_model_returns_definition(self):
        model = get_model("gpt-4o")
        assert model is not None
        assert model.vendor == "openai"

    def test_get_unknown_model_returns_none(self):
        assert get_model("nonexistent-model") is None

    def test_list_models_returns_all(self):
        models = list_models()
        assert len(models) == len(MODEL_CATALOG)

    def test_all_tiers_represented(self):
        tiers = {m.tier for m in MODEL_CATALOG.values()}
        assert "local" in tiers
        assert "cheap" in tiers
        assert "standard" in tiers
        assert "premium" in tiers


# ── Provider Registry ────────────────────────────────────────────────────────


class TestProviderRegistry:
    @pytest.fixture
    def registry(self):
        # Patch the underlying clients so no real network calls happen on build.
        with patch(
            "src.inference.providers.ollama_provider.OllamaClient"
        ), patch(
            "src.inference.providers.openrouter_provider.OpenRouterClient"
        ):
            return ProviderRegistry()

    def test_builds_provider_for_every_catalog_entry(self, registry):
        assert len(registry.all()) == len(MODEL_CATALOG)

    def test_all_entries_are_providers(self, registry):
        for provider in registry.all():
            assert isinstance(provider, Provider)

    def test_local_models_become_ollama_providers(self, registry):
        for provider in registry.local_providers():
            assert isinstance(provider, OllamaProvider)
            assert provider.is_local is True

    def test_cloud_models_become_openrouter_providers(self, registry):
        for provider in registry.cloud_providers():
            assert isinstance(provider, OpenRouterProvider)
            assert provider.is_local is False

    def test_get_returns_provider(self, registry):
        provider = registry.get("gpt-4o")
        assert provider is not None
        assert provider.alias == "gpt-4o"

    def test_get_unknown_returns_none(self, registry):
        assert registry.get("nope") is None

    def test_by_tier(self, registry):
        premium = registry.by_tier("premium")
        assert len(premium) >= 1
        for p in premium:
            assert p.info.tier == "premium"

    def test_aliases(self, registry):
        aliases = registry.aliases()
        assert "gpt-4o" in aliases
        assert "local-gemma" in aliases

    def test_describe_is_json_serialisable(self, registry):
        import json
        description = registry.describe()
        # Must not raise.
        json.dumps(description)
        assert len(description) == len(MODEL_CATALOG)
        first = description[0]
        assert "alias" in first
        assert "cost_per_1k_input" in first


class TestMaxModelTierCap:
    """
    MAX_MODEL_TIER should make disallowed tiers invisible everywhere the
    registry is consulted — all(), by_tier(), get(), aliases(), describe()
    — since callers (decision engine tier selection, fallback chains,
    explicit model selection, GET /models) each go through different
    accessors and none of them should leak a capped-out model.
    """

    @staticmethod
    def _build(max_tier: str) -> ProviderRegistry:
        with patch(
            "src.inference.providers.ollama_provider.OllamaClient"
        ), patch(
            "src.inference.providers.openrouter_provider.OpenRouterClient"
        ):
            return ProviderRegistry(max_tier=max_tier)

    def test_default_is_unrestricted(self):
        registry = self._build("premium")
        assert len(registry.all()) == len(MODEL_CATALOG)

    def test_capping_at_standard_excludes_premium_from_by_tier(self):
        registry = self._build("standard")
        assert registry.by_tier("premium") == []

    def test_capping_at_standard_excludes_premium_from_get(self):
        registry = self._build("standard")
        assert registry.get("gpt-5") is None
        assert registry.get("claude-opus") is None

    def test_capping_at_standard_excludes_premium_from_aliases(self):
        registry = self._build("standard")
        aliases = registry.aliases()
        assert "gpt-5" not in aliases
        assert "claude-opus" not in aliases

    def test_capping_at_standard_excludes_premium_from_describe(self):
        registry = self._build("standard")
        assert all(m["tier"] != "premium" for m in registry.describe())

    def test_capping_at_standard_keeps_lower_tiers(self):
        registry = self._build("standard")
        remaining_tiers = {p.info.tier for p in registry.all()}
        assert remaining_tiers == {"local", "cheap", "standard"}
        assert len(registry.all()) < len(MODEL_CATALOG)


# ── Explicit model selection via router ─────────────────────────────────────


class TestExplicitModelSelection:
    @pytest.fixture
    def registry_with_mocks(self):
        """Build a registry where every provider's complete() is mocked."""
        with patch(
            "src.inference.providers.ollama_provider.OllamaClient"
        ), patch(
            "src.inference.providers.openrouter_provider.OpenRouterClient"
        ):
            reg = ProviderRegistry()
            # Replace complete() on every provider with an AsyncMock.
            for provider in reg.all():
                provider.complete = AsyncMock(
                    return_value=f"Mocked answer from {provider.alias}."
                )
            return reg

    @pytest.fixture
    def router(self, registry_with_mocks, mock_ollama, mock_openrouter):
        return InferenceRouter(
            ollama=mock_ollama,
            openrouter=mock_openrouter,
            registry=registry_with_mocks,
        )

    async def test_cloud_model_called_directly(self, router, registry_with_mocks):
        answer, provider, alias = await router.complete_with_model(
            "prompt", "gpt-4o"
        )
        assert alias == "gpt-4o"
        assert provider == ProviderResult.CLOUD
        assert "gpt-4o" in answer
        gpt_provider = registry_with_mocks.get("gpt-4o")
        gpt_provider.complete.assert_called_once_with("prompt")

    async def test_unknown_model_raises(self, router):
        with pytest.raises(ValueError, match="Unknown model alias"):
            await router.complete_with_model("prompt", "does-not-exist")

    async def test_router_without_registry_raises(self, mock_ollama, mock_openrouter):
        bare_router = InferenceRouter(
            ollama=mock_ollama, openrouter=mock_openrouter, registry=None
        )
        with pytest.raises(RuntimeError, match="ProviderRegistry not configured"):
            await bare_router.complete_with_model("prompt", "gpt-4o")

    async def test_legacy_complete_still_works(self, router, mock_ollama):
        """The original complete() path must remain unaffected."""
        answer, provider = await router.complete("prompt")
        assert provider == ProviderResult.LOCAL
        assert answer == "Local model answer from Ollama."


# ── Streaming: explicit model selection via router ──────────────────────────


def make_stream_complete(chunks=None, exc=None):
    """
    Build a `stream_complete(prompt)` async-generator function for a fixture
    provider/client. AsyncMock can't produce an async generator, so tests
    that need one construct it by hand instead.
    """
    async def _stream_complete(prompt):
        if exc is not None:
            raise exc
        for chunk in chunks or []:
            yield chunk
    return _stream_complete


class TestStreamingModelSelection:
    @pytest.fixture
    def registry_with_stream_mocks(self):
        """Build a registry where every provider's stream_complete() is mocked."""
        with patch(
            "src.inference.providers.ollama_provider.OllamaClient"
        ), patch(
            "src.inference.providers.openrouter_provider.OpenRouterClient"
        ):
            reg = ProviderRegistry()
            for provider in reg.all():
                provider.stream_complete = make_stream_complete([f"chunk-from-{provider.alias}"])
            return reg

    @pytest.fixture
    def router(self, registry_with_stream_mocks, mock_ollama, mock_openrouter):
        return InferenceRouter(
            ollama=mock_ollama,
            openrouter=mock_openrouter,
            registry=registry_with_stream_mocks,
        )

    async def test_cloud_model_streams_directly(self, router):
        provider_result: dict = {}
        chunks = [
            c async for c in router.stream_complete_with_model("prompt", "gpt-4o", provider_result)
        ]
        assert chunks == ["chunk-from-gpt-4o"]
        assert provider_result["provider"] == ProviderResult.CLOUD

    async def test_local_model_streams_directly_when_healthy(self, router):
        provider_result: dict = {}
        chunks = [
            c async for c in router.stream_complete_with_model("prompt", "local-gemma", provider_result)
        ]
        assert chunks == ["chunk-from-local-gemma"]
        assert provider_result["provider"] == ProviderResult.LOCAL

    async def test_local_model_falls_back_to_cloud_before_first_chunk(
        self, registry_with_stream_mocks, mock_ollama, mock_openrouter,
    ):
        """A connection failure on the very first chunk should fall back to cloud transparently."""
        import httpx

        local_provider = registry_with_stream_mocks.get("local-gemma")
        local_provider.stream_complete = make_stream_complete(exc=httpx.ConnectError("unreachable"))
        mock_openrouter.stream_complete = make_stream_complete(["fallback-chunk"])
        router = InferenceRouter(
            ollama=mock_ollama, openrouter=mock_openrouter, registry=registry_with_stream_mocks,
        )

        provider_result: dict = {}
        chunks = [
            c async for c in router.stream_complete_with_model("prompt", "local-gemma", provider_result)
        ]

        assert chunks == ["fallback-chunk"]
        assert provider_result["provider"] == ProviderResult.CLOUD

    async def test_unknown_model_raises(self, router):
        with pytest.raises(ValueError, match="Unknown model alias"):
            async for _ in router.stream_complete_with_model("prompt", "does-not-exist"):
                pass

    async def test_router_without_registry_raises(self, mock_ollama, mock_openrouter):
        bare_router = InferenceRouter(
            ollama=mock_ollama, openrouter=mock_openrouter, registry=None
        )
        with pytest.raises(RuntimeError, match="ProviderRegistry not configured"):
            async for _ in bare_router.stream_complete_with_model("prompt", "gpt-4o"):
                pass

    async def test_legacy_stream_complete_local_first(self, router, mock_ollama):
        """Streaming counterpart to the legacy complete() path."""
        mock_ollama.stream_complete = make_stream_complete(["local-legacy-chunk"])
        chunks = [c async for c in router.stream_complete("prompt")]
        assert chunks == ["local-legacy-chunk"]

    async def test_legacy_stream_complete_force_cloud(self, router, mock_ollama, mock_openrouter):
        mock_openrouter.stream_complete = make_stream_complete(["cloud-legacy-chunk"])
        chunks = [c async for c in router.stream_complete("prompt", force_cloud=True)]
        assert chunks == ["cloud-legacy-chunk"]
        mock_ollama.stream_complete.assert_not_called()
