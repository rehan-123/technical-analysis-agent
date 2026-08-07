from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Final, Mapping

import httpx

from config.settings import Settings, get_settings
from data.providers.finnhub_provider import FinnhubProvider
from data.providers.news_exceptions import NewsConfigurationError
from data.providers.news_provider import NewsProvider
from utils.logger import get_logger

logger = get_logger(__name__)

#: Signature every provider builder must satisfy. Taking the injected
#: ``Settings`` and optional shared ``httpx.AsyncClient`` means the factory
#: passes dependencies *through* to providers rather than letting them reach
#: for globals, preserving Dependency Injection end to end. Each builder
#: decides for itself which of these its provider actually needs.
NewsProviderBuilder = Callable[[Settings, httpx.AsyncClient | None], NewsProvider]


def _build_finnhub(settings: Settings, client: httpx.AsyncClient | None) -> NewsProvider:
    return FinnhubProvider(settings=settings, client=client)


#: Fixed registry of known providers, keyed by the name used in
#: ``Settings.news_sources``. A registry rather than an if/elif chain so the
#: factory is open for extension and closed for modification: adding
#: NewsAPI/AlphaVantage/Polygon means adding a builder and one entry here,
#: never editing ``create_news_provider`` and never touching a single caller.
#:
#: Wrapped in ``MappingProxyType`` so it is immutable at *runtime*, not merely
#: by convention — ``Final`` alone is only a static-checker hint. The set of
#: providers a deployment can use is therefore fixed at import time and fully
#: auditable by reading this file, with no possibility of order-dependent
#: registration or state leaking between tests.
#:
#: Values are *builders*, not instances — construction is deferred until a
#: provider is actually selected. This matters because provider constructors
#: validate their configuration (``FinnhubProvider`` raises
#: ``NewsConfigurationError`` when no API key is set); eager instantiation
#: would make importing this module fail for providers nobody enabled.
_PROVIDER_REGISTRY: Final[Mapping[str, NewsProviderBuilder]] = MappingProxyType(
    {
        FinnhubProvider.SOURCE_NAME: _build_finnhub,
    }
)


def available_news_providers() -> tuple[str, ...]:
    """Return the names of every registered provider, sorted.

    Useful for diagnostics, configuration validation, and error messages that
    tell an operator what they *could* have configured.
    """
    return tuple(sorted(_PROVIDER_REGISTRY))


def create_news_provider(
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> NewsProvider:
    """Construct the configured news provider.

    Reads ``Settings.news_sources`` (an ordered list, mirroring the
    market-data layer's ``data_sources``) and returns an instance of the first
    recognised entry. Unknown names are logged and skipped rather than being
    fatal, so a typo or a not-yet-implemented source degrades to the next
    configured option instead of taking the application down.

    Only a *single* provider is returned. ``news_sources`` is a list purely to
    keep the configuration shape forward-compatible with a future
    ``FallbackNewsProvider`` chain; that chain does not exist yet, so any
    entries beyond the first are explicitly reported as unused rather than
    silently ignored — silence there would look like a working failover that
    isn't.

    This function only *constructs*. It performs no retrieval, caching,
    deduplication, sorting, filtering, or retrying: those belong to the
    provider (transport) and ``NewsService`` (deterministic processing).

    Args:
        settings: Injected configuration. Falls back to the cached global
            settings when omitted.
        client: Optional shared HTTP client passed through to providers that
            accept one, so callers (and tests) control transport and
            connection-pool lifetime.

    Returns:
        A ready-to-use ``NewsProvider``.

    Raises:
        NewsConfigurationError: If no usable provider is configured — either
            ``news_sources`` is empty or none of its entries are registered.
            Also propagated from a provider's own constructor when that
            provider is misconfigured (e.g. a missing API key).
    """
    settings = settings or get_settings()

    configured = [name.strip().lower() for name in settings.news_sources if name and name.strip()]
    if not configured:
        raise NewsConfigurationError(
            "No news provider configured: 'news_sources' is empty. "
            f"Available providers: {', '.join(available_news_providers())}"
        )

    for position, name in enumerate(configured):
        builder = _PROVIDER_REGISTRY.get(name)
        if builder is None:
            logger.warning(
                "Unknown news provider '%s' in configuration — skipping. Available: %s",
                name, ", ".join(available_news_providers()),
            )
            continue

        provider = builder(settings, client)

        unused = configured[position + 1:]
        if unused:
            logger.warning(
                "Multiple news sources configured %s but only '%s' is used; "
                "provider failover is not implemented yet, so %s will be ignored",
                configured, name, unused,
            )
        logger.info("News provider '%s' selected (%s)", name, type(provider).__name__)
        return provider

    raise NewsConfigurationError(
        f"None of the configured news sources {configured} are registered. "
        f"Available providers: {', '.join(available_news_providers())}"
    )
