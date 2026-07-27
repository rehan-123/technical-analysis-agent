from __future__ import annotations

import pytest

from config.settings import Settings
from data.providers.finnhub_provider import FinnhubProvider
from data.providers.news_exceptions import NewsConfigurationError
from data.providers.news_provider import NewsProvider
from data.providers.provider_factory import (
    _PROVIDER_REGISTRY,
    available_news_providers,
    create_news_provider,
)


class TestProviderSelection:
    def test_returns_the_configured_provider(self):
        provider = create_news_provider(
            Settings(news_sources=["finnhub"], news_finnhub_api_key="k")
        )
        assert isinstance(provider, FinnhubProvider)
        assert isinstance(provider, NewsProvider)  # callers bind to the abstraction

    def test_provider_names_are_case_and_whitespace_insensitive(self):
        provider = create_news_provider(
            Settings(news_sources=["  FinnHub  "], news_finnhub_api_key="k")
        )
        assert isinstance(provider, FinnhubProvider)

    def test_unknown_source_is_skipped_in_favour_of_a_known_one(self):
        """A typo or not-yet-implemented source must degrade, not crash."""
        provider = create_news_provider(
            Settings(news_sources=["polygon", "finnhub"], news_finnhub_api_key="k")
        )
        assert isinstance(provider, FinnhubProvider)

    def test_injected_client_is_passed_through(self):
        import httpx

        client = httpx.AsyncClient()
        provider = create_news_provider(
            Settings(news_sources=["finnhub"], news_finnhub_api_key="k"), client=client
        )
        assert provider._client is client  # noqa: SLF001 — verifying DI wiring


class TestConfigurationErrors:
    def test_empty_source_list_raises(self):
        with pytest.raises(NewsConfigurationError, match="empty"):
            create_news_provider(Settings(news_sources=[], news_finnhub_api_key="k"))

    def test_all_sources_unknown_raises(self):
        with pytest.raises(NewsConfigurationError, match="registered"):
            create_news_provider(
                Settings(news_sources=["polygon", "newsapi"], news_finnhub_api_key="k")
            )

    def test_error_message_lists_available_providers(self):
        with pytest.raises(NewsConfigurationError, match="finnhub"):
            create_news_provider(Settings(news_sources=["nope"], news_finnhub_api_key="k"))

    def test_provider_level_misconfiguration_propagates(self):
        """A missing API key surfaces from the provider's own constructor."""
        with pytest.raises(NewsConfigurationError):
            create_news_provider(Settings(news_sources=["finnhub"], news_finnhub_api_key=""))


class TestRegistry:
    def test_available_providers_is_sorted_and_includes_finnhub(self):
        names = available_news_providers()
        assert "finnhub" in names
        assert list(names) == sorted(names)

    def test_registry_is_immutable_at_runtime(self):
        """A fixed registry keeps the usable-provider set statically auditable."""
        with pytest.raises(TypeError):
            _PROVIDER_REGISTRY["polygon"] = lambda settings, client: None  # type: ignore[index]

    def test_registry_holds_builders_not_instances(self):
        """Deferred construction is why importing this module does not explode
        for providers nobody configured."""
        assert all(callable(builder) for builder in _PROVIDER_REGISTRY.values())
