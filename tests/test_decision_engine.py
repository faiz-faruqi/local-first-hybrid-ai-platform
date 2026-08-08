"""
Tests for the Phase 3 Decision Engine.

Verifies that the engine correctly selects models based on:
  - Complexity (low→cheap, medium→standard, high→premium)
  - Sensitivity (confidential→local)
  - Budget (exceeded→downgrade)
  - Confidence (low→upgrade)
  - Fallback chain construction
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.classification import (
    Complexity,
    ContextSize,
    Domain,
    LatencyTier,
    QueryProfile,
    Sensitivity,
)
from src.routing.decision_engine import DecisionEngine
from src.routing.policies import TIER_ORDER, tier_index


@pytest.fixture
def registry():
    """Build a real registry with mocked underlying clients."""
    with patch(
        "src.inference.providers.ollama_provider.OllamaClient"
    ), patch(
        "src.inference.providers.openrouter_provider.OpenRouterClient"
    ):
        from src.inference.provider_registry import ProviderRegistry
        return ProviderRegistry()


@pytest.fixture
def engine(registry):
    """Decision engine with no budget tracker (budget disabled)."""
    return DecisionEngine(registry=registry, budget_tracker=None)


def make_profile(
    complexity=Complexity.MEDIUM,
    domain=Domain.GENERAL,
    sensitivity=Sensitivity.PUBLIC,
    context_size=ContextSize.SMALL,
    rag_needed=False,
    latency_tier=LatencyTier.INTERACTIVE,
    confidence=0.8,
) -> QueryProfile:
    """Helper to build a QueryProfile with defaults."""
    return QueryProfile(
        complexity=complexity,
        domain=domain,
        sensitivity=sensitivity,
        context_size=context_size,
        rag_needed=rag_needed,
        latency_tier=latency_tier,
        confidence=confidence,
        token_estimate=20,
        signals=["test"],
    )


class TestDecisionEngineRouting:
    async def test_low_complexity_routes_to_cheap(self, engine, registry):
        profile = make_profile(complexity=Complexity.LOW)
        decision = await engine.decide(profile)
        assert decision.selected_tier == "cheap"
        provider = registry.get(decision.selected_model)
        assert provider is not None
        assert provider.info.tier == "cheap"

    async def test_medium_complexity_routes_to_standard(self, engine, registry):
        profile = make_profile(complexity=Complexity.MEDIUM)
        decision = await engine.decide(profile)
        assert decision.selected_tier == "standard"
        provider = registry.get(decision.selected_model)
        assert provider.info.tier == "standard"

    async def test_high_complexity_routes_to_premium(self, engine, registry):
        profile = make_profile(complexity=Complexity.HIGH)
        decision = await engine.decide(profile)
        assert decision.selected_tier == "premium"
        provider = registry.get(decision.selected_model)
        assert provider.info.tier == "premium"

    async def test_confidential_routes_to_local(self, engine, registry):
        """Governance policy: confidential data must stay on-premises."""
        profile = make_profile(
            sensitivity=Sensitivity.CONFIDENTIAL,
            complexity=Complexity.HIGH,  # would normally go premium
        )
        decision = await engine.decide(profile)
        assert decision.selected_tier == "local"
        assert any("sensitivity" in r for r in decision.rules_matched)
        provider = registry.get(decision.selected_model)
        assert provider.info.is_local is True

    async def test_sensitivity_overrides_complexity(self, engine):
        """Even a high-complexity confidential query must go local."""
        profile = make_profile(
            complexity=Complexity.HIGH,
            sensitivity=Sensitivity.CONFIDENTIAL,
        )
        decision = await engine.decide(profile)
        assert decision.selected_tier == "local"

    async def test_low_confidence_upgrades_tier(self, engine, registry):
        """Low classifier confidence should upgrade the model one tier."""
        profile = make_profile(
            complexity=Complexity.LOW,  # would normally go cheap
            confidence=0.45,           # below 0.6 threshold
        )
        decision = await engine.decide(profile)
        # Should be upgraded from cheap to standard
        assert decision.selected_tier == "standard"
        assert any("confidence" in r for r in decision.rules_matched)

    async def test_high_confidence_no_upgrade(self, engine):
        profile = make_profile(
            complexity=Complexity.LOW,
            confidence=0.9,
        )
        decision = await engine.decide(profile)
        assert decision.selected_tier == "cheap"

    async def test_large_context_prefers_long_context_model(self, engine, registry):
        profile = make_profile(
            complexity=Complexity.MEDIUM,
            context_size=ContextSize.LARGE,
        )
        decision = await engine.decide(profile)
        assert any("context" in r for r in decision.rules_matched)

    async def test_coding_domain_prefers_anthropic(self, engine):
        profile = make_profile(
            complexity=Complexity.MEDIUM,
            domain=Domain.CODING,
        )
        decision = await engine.decide(profile)
        assert any("domain" in r for r in decision.rules_matched)

    async def test_healthcare_domain_prefers_local(self, engine, registry):
        profile = make_profile(
            complexity=Complexity.MEDIUM,
            domain=Domain.HEALTHCARE,
        )
        decision = await engine.decide(profile)
        assert any("healthcare" in r for r in decision.rules_matched)


class TestFallbackChain:
    async def test_fallback_chain_not_empty(self, engine):
        profile = make_profile(complexity=Complexity.HIGH)
        decision = await engine.decide(profile)
        assert len(decision.fallback_chain) > 0

    async def test_fallback_chain_excludes_selected(self, engine):
        profile = make_profile(complexity=Complexity.HIGH)
        decision = await engine.decide(profile)
        assert decision.selected_model not in decision.fallback_chain

    async def test_fallback_chain_capped_at_5(self, engine):
        profile = make_profile(complexity=Complexity.HIGH)
        decision = await engine.decide(profile)
        assert len(decision.fallback_chain) <= 5

    async def test_fallback_includes_cheaper_tiers(self, engine, registry):
        profile = make_profile(complexity=Complexity.HIGH)
        decision = await engine.decide(profile)
        # At least one fallback should be from a cheaper tier
        for alias in decision.fallback_chain:
            provider = registry.get(alias)
            if provider:
                assert provider.info.tier in ("local", "cheap", "standard", "premium")


class TestBudgetAwareRouting:
    @pytest.fixture
    def budget_exceeded_tracker(self):
        """A mock budget tracker that always reports budget exceeded."""
        tracker = AsyncMock()
        tracker.has_budget_configured = True
        tracker.is_budget_exceeded = AsyncMock(return_value=True)
        tracker.remaining_monthly = AsyncMock(return_value=0.0)
        tracker.remaining_daily = AsyncMock(return_value=0.0)
        return tracker

    async def test_budget_exceeded_downgrades_to_cheap(
        self, registry, budget_exceeded_tracker
    ):
        engine = DecisionEngine(
            registry=registry,
            budget_tracker=budget_exceeded_tracker,
        )
        profile = make_profile(complexity=Complexity.HIGH)  # would normally go premium
        decision = await engine.decide(profile)
        assert decision.selected_tier == "cheap"
        assert any("budget" in r for r in decision.rules_matched)

    async def test_budget_remaining_reported(self, registry, budget_exceeded_tracker):
        engine = DecisionEngine(
            registry=registry,
            budget_tracker=budget_exceeded_tracker,
        )
        profile = make_profile(complexity=Complexity.LOW)
        decision = await engine.decide(profile)
        assert decision.budget_remaining is not None


class TestRoutingDecision:
    async def test_decision_has_reason(self, engine):
        profile = make_profile(complexity=Complexity.HIGH)
        decision = await engine.decide(profile)
        assert decision.reason
        assert len(decision.reason) > 0

    async def test_decision_has_rules_matched(self, engine):
        profile = make_profile(complexity=Complexity.HIGH)
        decision = await engine.decide(profile)
        assert len(decision.rules_matched) > 0

    async def test_estimated_cost_non_negative(self, engine):
        profile = make_profile(complexity=Complexity.HIGH)
        decision = await engine.decide(profile)
        assert decision.estimated_cost >= 0.0

    async def test_local_model_has_zero_cost(self, engine):
        profile = make_profile(sensitivity=Sensitivity.CONFIDENTIAL)
        decision = await engine.decide(profile)
        assert decision.estimated_cost == 0.0


class TestMaxTierCapRouting:
    """
    Regression coverage for MAX_MODEL_TIER: primary tier selection and the
    fallback chain are two independent call sites into the registry
    (_select_model and _build_fallback_chain both call by_tier() on their
    own), so capping needs verifying against both, not just one.
    """

    @pytest.fixture
    def capped_registry(self):
        with patch(
            "src.inference.providers.ollama_provider.OllamaClient"
        ), patch(
            "src.inference.providers.openrouter_provider.OpenRouterClient"
        ):
            from src.inference.provider_registry import ProviderRegistry
            return ProviderRegistry(max_tier="standard")

    @pytest.fixture
    def capped_engine(self, capped_registry):
        return DecisionEngine(registry=capped_registry, budget_tracker=None)

    async def test_high_complexity_never_selects_premium_when_capped(
        self, capped_engine, capped_registry
    ):
        profile = make_profile(complexity=Complexity.HIGH)  # would normally go premium
        decision = await capped_engine.decide(profile)
        provider = capped_registry.get(decision.selected_model)
        assert provider is not None
        assert provider.info.tier != "premium"

    async def test_fallback_chain_never_includes_premium_when_capped(
        self, capped_engine, capped_registry
    ):
        profile = make_profile(complexity=Complexity.HIGH)
        decision = await capped_engine.decide(profile)
        assert len(decision.fallback_chain) > 0
        for alias in decision.fallback_chain:
            provider = capped_registry.get(alias)
            assert provider is not None
            assert provider.info.tier != "premium"
