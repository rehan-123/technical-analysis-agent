"""Deterministic test doubles for the News Agent.

Mirrors the role ``SyntheticDataProvider`` plays for market data: it lets the
entire News pipeline — service, agent, and API — be exercised with **zero
network access**, so the suite is fast, hermetic, and never flakes on a
vendor outage or rate limit.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from data.providers.news_provider import NewsProvider
from models.news import NewsArticle, NewsRequest

#: Fixed reference instant. Article timestamps are expressed relative to
#: "now" at call time so recency filtering behaves realistically, but every
#: offset is deterministic, so tests never depend on wall-clock drift.
_BASE_OFFSETS_HOURS = (1, 2, 3, 4, 5)


def make_article(
    *,
    title: str = "Example headline",
    url: str = "https://example.com/story",
    hours_ago: float = 1.0,
    publisher: str = "Example Wire",
    ticker: str = "AAPL",
    language: str = "en",
    summary: str = "",
    source: str = "fake",
) -> NewsArticle:
    """Build a valid ``NewsArticle`` with sensible defaults.

    Keyword-only so every call site reads explicitly at the point of use.
    """
    return NewsArticle(
        title=title,
        source=source,
        publisher=publisher,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        url=url,
        summary=summary,
        ticker=ticker,
        language=language,
    )


class FakeNewsProvider(NewsProvider):
    """Returns a canned, caller-supplied list of articles.

    Records the requests it receives so tests can assert the agent and service
    pass parameters through correctly.
    """

    def __init__(self, articles: list[NewsArticle] | None = None) -> None:
        self.articles = articles if articles is not None else []
        self.calls: list[NewsRequest] = []

    async def get_news(self, request: NewsRequest) -> list[NewsArticle]:
        self.calls.append(request)
        return list(self.articles)


class FailingNewsProvider(NewsProvider):
    """Raises a supplied exception, for verifying error propagation."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def get_news(self, request: NewsRequest) -> list[NewsArticle]:
        raise self.error


def default_article_set(ticker: str = "AAPL") -> list[NewsArticle]:
    """A mixed, deliberately messy set exercising the whole pipeline.

    Contains, by construction:
      * articles supplied **out of chronological order** (sorting must fix it),
      * an exact URL duplicate differing only by tracking parameters,
      * a syndicated reprint: same headline, different outlet and URL, minutes
        apart (the title-plus-window rule must catch it),
      * a same-headline article far outside the dedup window (must be KEPT —
        it is a genuinely different story, not a reprint),
      * a non-English article (language filtering),
      * an article well outside a short lookback window (recency filtering).
    """
    return [
        make_article(title="Beta news", url="https://example.com/beta", hours_ago=3, ticker=ticker),
        make_article(title="Alpha news", url="https://example.com/alpha", hours_ago=1, ticker=ticker),
        # Same document as "Alpha news", only tracking params differ.
        make_article(
            title="Alpha news",
            url="https://www.example.com/alpha?utm_source=newsletter",
            hours_ago=1,
            ticker=ticker,
        ),
        # Syndicated reprint: different outlet + URL, minutes apart.
        make_article(
            title="Alpha news!",
            url="https://other-outlet.com/alpha-story",
            hours_ago=1.2,
            publisher="Other Outlet",
            ticker=ticker,
        ),
        # Same headline, but days later — a distinct story, must survive.
        make_article(
            title="Alpha news",
            url="https://example.com/alpha-followup",
            hours_ago=72,
            ticker=ticker,
        ),
        make_article(
            title="Nachrichten auf Deutsch",
            url="https://example.de/nachricht",
            hours_ago=2,
            ticker=ticker,
            language="de",
        ),
        # Far outside a short lookback window.
        make_article(title="Ancient news", url="https://example.com/old", hours_ago=24 * 30, ticker=ticker),
    ]
