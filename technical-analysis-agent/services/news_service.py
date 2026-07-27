from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlparse

from config.settings import Settings, get_settings
from data.providers.news_provider import NewsProvider
from models.news import NewsAnalysisResult, NewsArticle, NewsRequest
from utils.logger import get_logger

logger = get_logger(__name__)

#: Query parameters that identify a *referral*, not a document. Two URLs that
#: differ only by these point at the same article, so they are stripped before
#: comparison. Anything not listed here is preserved, because many sites carry
#: meaningful state in the query string (``?id=123``, ``?page=2``) and blindly
#: discarding it would merge genuinely different articles.
_TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "fbclid", "gclid", "mc_cid", "mc_eid", "igshid",
        "ref", "ref_src", "source", "cmpid", "partner",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)


class NewsService:
    """Deterministic processing pipeline for retrieved news.

    Depends **only** on the ``NewsProvider`` abstraction, never on a concrete
    source, so every rule implemented here applies identically no matter where
    the articles came from. That is the whole point of the split: providers
    adapt one upstream API each; this class owns the source-agnostic
    behaviour, so adding a provider cannot change how news is filtered,
    deduplicated, or ordered.

    The pipeline is:

    1. **Recency filter** — drop anything outside the requested lookback
       window. Providers query an approximate date range; this enforces the
       exact boundary.
    2. **Language filter** — applied only when the request specifies one.
    3. **Sort** — newest first, with a deterministic tiebreaker.
    4. **Deduplicate** — remove repeats of the same story.
    5. **Limit** — take the top *N*.

    Ordering of these steps is deliberate. Sorting *before* deduplication
    makes the survivor of a duplicate group well-defined (the newest copy is
    encountered first and kept) instead of depending on provider response
    order. Limiting *last* guarantees the caller receives *N* distinct
    articles rather than *N* raw ones that might collapse to fewer.

    Every step is pure and deterministic: identical input always yields
    identical output. No LLM, no sentiment scoring, no advice — those are out
    of scope by design.

    Caching is intentionally absent. ``NewsCache`` defines the seam, but no
    implementation exists yet and none is wired here, so every call consults
    the provider. Adding it later is a purely additive constructor parameter.
    """

    def __init__(self, provider: NewsProvider, settings: Settings | None = None) -> None:
        """Args:
        provider: The injected news source. Only the abstract type is
            referenced, so any implementation — including a test fake — works
            without changes here.
        settings: Injected configuration. Falls back to the cached global
            settings when omitted.
        """
        self._provider = provider
        self._settings = settings or get_settings()

    # --- Normalization helpers (pure, static, individually testable) ----------

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Reduce a URL to a canonical identity for duplicate detection.

        Drops the scheme (so ``http`` and ``https`` copies match), lowercases
        the host, strips a leading ``www.`` and any trailing slash, removes
        tracking parameters, and sorts the parameters that remain so their
        order cannot affect the result.
        """
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path.rstrip("/") or "/"

        retained = sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in _TRACKING_PARAMS
        )
        query = urlencode(retained)
        return f"{host}{path}?{query}" if query else f"{host}{path}"

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Reduce a headline to a canonical form: lowercased, punctuation
        removed, whitespace collapsed.

        This is what lets the same story syndicated across outlets with
        cosmetically different punctuation be recognised as one story.
        """
        text = _PUNCTUATION_RE.sub(" ", title.lower())
        return _WHITESPACE_RE.sub(" ", text).strip()

    # --- Pipeline stages -------------------------------------------------------

    @staticmethod
    def _filter_by_recency(articles: list[NewsArticle], request: NewsRequest) -> list[NewsArticle]:
        """Keep only articles published within the requested lookback window.

        Necessary even though providers are given a date range: upstream date
        filters are day-granular and inclusive, so they routinely return items
        slightly outside the intended window.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=request.lookback_days)
        return [article for article in articles if article.published_at >= cutoff]

    @staticmethod
    def _filter_by_language(articles: list[NewsArticle], request: NewsRequest) -> list[NewsArticle]:
        """Keep only articles in the requested language.

        A ``None`` language means "no filtering" — the deliberate default, so
        that a source which does not report language never has its articles
        silently discarded by an assumed value.
        """
        if not request.language:
            return list(articles)
        return [article for article in articles if article.language == request.language]

    @staticmethod
    def _sort(articles: list[NewsArticle]) -> list[NewsArticle]:
        """Order newest first, breaking ties by canonical URL.

        Implemented as two stable sorts rather than one composite key: the
        secondary key is ascending while the primary is descending, and
        Python's sort stability composes them exactly. (Negating a timestamp
        to invert one key inside a tuple would work but risks float precision
        loss at microsecond granularity.)

        The tiebreaker matters for determinism — without it, two articles
        sharing a timestamp would retain provider response order, making the
        output depend on the source rather than the data.
        """
        by_url = sorted(articles, key=lambda article: str(article.url))
        return sorted(by_url, key=lambda article: article.published_at, reverse=True)

    def _deduplicate(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Remove repeated stories, keeping the first occurrence.

        Because the input is already sorted newest-first, "first occurrence"
        deterministically means the most recent copy.

        Two independent identity rules are applied:

        * **Canonical URL** — an exact republication of the same document.
          Always applied; this is the unambiguous case.
        * **Normalized title within a time window** — the same story
          syndicated across outlets, which has different URLs and so is
          invisible to the first rule. Bounded by
          ``news_dedup_time_window_minutes`` because an identical headline
          months apart is far more likely to be a genuinely new article than a
          reprint.

        The window is checked pairwise against previously kept publication
        times rather than by bucketing timestamps: bucketing would fail to
        match two articles minutes apart that happen to straddle a bucket
        boundary. Article counts here are bounded by the request limit, so the
        cost is negligible.
        """
        if not self._settings.news_deduplicate:
            return list(articles)

        window = timedelta(minutes=self._settings.news_dedup_time_window_minutes)
        seen_urls: set[str] = set()
        seen_titles: dict[str, list[datetime]] = {}
        kept: list[NewsArticle] = []

        for article in articles:
            url_key = self._normalize_url(str(article.url))
            if url_key in seen_urls:
                logger.debug("Dropping duplicate URL for %s: %s", article.ticker, url_key)
                continue

            title_key = self._normalize_title(article.title)
            # A title consisting only of punctuation normalizes to an empty
            # string; treating that as an identity would collapse unrelated
            # articles, so title-matching is skipped for it.
            if title_key:
                previous = seen_titles.get(title_key)
                if previous and any(abs(article.published_at - seen) <= window for seen in previous):
                    logger.debug(
                        "Dropping duplicate headline for %s within %s: %r",
                        article.ticker, window, article.title,
                    )
                    continue

            kept.append(article)
            seen_urls.add(url_key)
            if title_key:
                seen_titles.setdefault(title_key, []).append(article.published_at)

        return kept

    @staticmethod
    def _limit(articles: list[NewsArticle], request: NewsRequest) -> list[NewsArticle]:
        """Truncate to the requested maximum, applied last so the caller
        receives that many *distinct* articles."""
        return articles[: request.limit]

    # --- Public API ---------------------------------------------------------------

    async def get_news(self, request: NewsRequest) -> NewsAnalysisResult:
        """Retrieve and deterministically process news for ``request``.

        Provider failures (``NewsProviderError``, ``NewsRateLimitError``, …)
        are deliberately allowed to propagate: this layer has no basis on
        which to decide whether a retrieval failure should be retried, failed
        over, or surfaced, so it does not silently convert an error into an
        empty result — which would be indistinguishable from "no news exists"
        and is exactly the kind of silence that hides outages.

        Returns:
            A ``NewsAnalysisResult`` whose ``articles`` are unique, in the
            requested language, within the lookback window, newest first, and
            no longer than the requested limit. An empty list is a valid
            result meaning no qualifying news was found.
        """
        retrieved = await self._provider.get_news(request)

        after_recency = self._filter_by_recency(retrieved, request)
        after_language = self._filter_by_language(after_recency, request)
        ordered = self._sort(after_language)
        deduplicated = self._deduplicate(ordered)
        final = self._limit(deduplicated, request)

        logger.info(
            "News pipeline for %s: retrieved=%d recency=%d language=%d deduplicated=%d returned=%d",
            request.ticker,
            len(retrieved),
            len(after_recency),
            len(after_language),
            len(deduplicated),
            len(final),
        )

        return NewsAnalysisResult(ticker=request.ticker, articles=final)
