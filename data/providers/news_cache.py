from __future__ import annotations

from abc import ABC, abstractmethod

from models.news import NewsAnalysisResult, NewsRequest


class NewsCache(ABC):
    """Abstract cache for News Agent results.

    INTERFACE ONLY — there is intentionally no concrete implementation yet.
    This defines the seam so caching (in-memory TTL, Redis, etc.) can be
    added later behind Dependency Injection without changing the service or
    agent. Because nothing implements or is wired to this interface, the
    current system performs no caching; a provider is always consulted.

    Design choices:
      * **Keyed by ``NewsRequest``, not a raw string.** The cache owns its own
        key derivation from the request's identity (ticker, lookback, limit,
        language), keeping call sites type-safe and preventing two callers
        from computing inconsistent keys. Implementations MUST derive keys
        deterministically so identical requests map to identical entries.
      * **Caches the assembled ``NewsAnalysisResult``**, i.e. the final
        deterministic output — so a cache hit skips both retrieval and the
        dedup/sort/filter pipeline.
      * **Async-first**, matching every other I/O boundary in the platform, so
        a networked backend (e.g. Redis) drops in without signature changes.

    A future implementation will be injected into ``NewsService`` (checked
    before calling the provider, populated after a miss); the service depends
    only on this abstraction, never on a concrete cache.
    """

    @abstractmethod
    async def get(self, request: NewsRequest) -> NewsAnalysisResult | None:
        """Return the cached result for ``request``, or ``None`` on a miss
        (including entries that have expired)."""
        raise NotImplementedError

    @abstractmethod
    async def set(
        self,
        request: NewsRequest,
        result: NewsAnalysisResult,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store ``result`` under ``request``'s derived key.

        ``ttl_seconds`` optionally bounds entry lifetime; ``None`` defers to
        the implementation's configured default.
        """
        raise NotImplementedError

    @abstractmethod
    async def invalidate(self, request: NewsRequest) -> None:
        """Remove the cached entry for ``request`` if present (a no-op on a
        miss)."""
        raise NotImplementedError
