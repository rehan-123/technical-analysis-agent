from __future__ import annotations

from abc import ABC, abstractmethod

from models.news import NewsArticle, NewsRequest


class NewsProvider(ABC):
    """Abstract news source.

    Concrete providers (Finnhub, and any future source such as a second
    vendor or a broker feed) all implement this one contract, so the service
    and agent layers never depend on where the news actually comes from.
    Adding a source later means writing one new class here — nothing else in
    the codebase changes. This is the direct analog of ``MarketDataProvider``
    in the market-data layer.

    Responsibility boundary (Single Responsibility): a provider is a pure
    *adapter*. It knows how to talk to exactly one upstream source and how to
    translate that source's payload into the common ``NewsArticle`` schema.
    It does **not** deduplicate, sort, apply business filters (recency window,
    language, limit), or assemble the final result — those are source-agnostic
    concerns owned entirely by ``NewsService``.

    Contract of ``get_news``:
      * Retrieve raw articles for ``request.ticker`` within the requested
        lookback window from the upstream source.
      * Adapt each raw item into a ``NewsArticle``. Items that fail model
        validation (missing title, invalid URL, non-timezone-aware or
        implausibly future-dated timestamp) are silently skipped — this is
        how the 'filter invalid articles' responsibility is met at the source
        boundary. A single malformed item never fails the whole call.
      * Return the list of valid articles. The list may be empty (a valid
        outcome), and is returned unsorted and un-deduplicated — the service
        owns ordering and dedup so those rules stay identical across sources.
      * Raise ``NewsProviderError`` (or a subclass) if the *retrieval itself*
        fails: transport error, non-success HTTP status, or a response so
        malformed that no articles can be extracted.
    """

    @abstractmethod
    async def get_news(self, request: NewsRequest) -> list[NewsArticle]:
        """Return normalized, validation-passed articles for the request.

        Async-first: implementations must not block the event loop (network
        I/O should use an async client, or be offloaded to a worker thread).
        """
        raise NotImplementedError
