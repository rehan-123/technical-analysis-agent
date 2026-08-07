from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from pydantic import ValidationError

from config.settings import Settings, get_settings
from data.providers.news_exceptions import (
    NewsConfigurationError,
    NewsProviderError,
    NewsRateLimitError,
    NewsValidationError,
)
from data.providers.news_provider import NewsProvider
from models.news import NewsArticle, NewsRequest
from utils.logger import get_logger

logger = get_logger(__name__)

# HTTP statuses worth retrying: transient server-side faults only. Client
# errors (4xx) are deterministic — retrying an unauthorized or malformed
# request just wastes quota — and 429 is handled separately.
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


class FinnhubProvider(NewsProvider):
    """Retrieves company news from the Finnhub API.

    A pure adapter, per the ``NewsProvider`` contract: it fetches from exactly
    one upstream source and translates that source's payload into the common
    ``NewsArticle`` schema. It performs **no** deduplication, sorting,
    recency/language filtering, limiting, or caching — every one of those is a
    source-agnostic concern owned by ``NewsService``. Keeping them out of here
    is what allows a second provider to be added later without any of those
    behaviours changing or being reimplemented.

    Dependency Injection: an ``httpx.AsyncClient`` may be supplied by the
    caller (tests inject one backed by ``httpx.MockTransport``, so the whole
    provider is exercisable with zero network). If none is supplied, the
    provider lazily creates and exclusively owns one.
    """

    #: Value written to ``NewsArticle.source`` — provenance of *how* the
    #: article was retrieved, distinct from the publishing outlet.
    SOURCE_NAME = "finnhub"

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()

        api_key = (self._settings.news_finnhub_api_key or "").strip()
        if not api_key:
            # Fail fast at wiring time rather than surfacing an opaque 401 on
            # the first user request.
            raise NewsConfigurationError(
                "Finnhub news provider is enabled but no API key is configured "
                "(set TA_NEWS_FINNHUB_API_KEY)"
            )
        self._api_key = api_key
        self._base_url = self._settings.news_finnhub_base_url.rstrip("/")
        self._timeout = self._settings.news_request_timeout
        self._max_retries = max(1, self._settings.news_max_retries)
        self._retry_backoff = self._settings.news_retry_backoff

        # Client lifecycle: we only ever close a client we created ourselves.
        # Closing a caller-supplied client would be a side effect on an object
        # we do not own and could break other users of a shared connection pool.
        self._client = client
        self._owns_client = client is None
        self._client_lock = asyncio.Lock()

    # --- Async resource management -------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the HTTP client, creating one lazily if we own it.

        Lazy creation keeps ``__init__`` free of I/O and of any event-loop
        dependency, so the provider can be constructed at import/wiring time.
        The lock makes concurrent first-use safe — without it, simultaneous
        requests could each build a client and leak all but one.
        """
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:  # re-check inside the lock
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
                self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Release the HTTP client if (and only if) this provider owns it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "FinnhubProvider":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # --- Retrieval ------------------------------------------------------------

    def _build_params(self, request: NewsRequest) -> dict[str, str]:
        """Finnhub's company-news endpoint takes an inclusive ``YYYY-MM-DD``
        date range. ``lookback_days`` is a *retrieval* parameter (how much to
        ask the API for), not a business filter — precise recency filtering
        remains the service's job."""
        to_date: date = datetime.now(timezone.utc).date()
        from_date: date = to_date - timedelta(days=request.lookback_days)
        return {
            "symbol": request.ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        }

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        """Read the ``Retry-After`` hint when the server sends one.

        Only the delta-seconds form is interpreted; the alternate HTTP-date
        form is ignored (returns ``None``) rather than guessed at, so callers
        never receive a fabricated delay.
        """
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return float(raw.strip())
        except (TypeError, ValueError):
            return None

    async def _fetch(self, request: NewsRequest) -> Any:
        """Perform the HTTP call with retries and return the decoded payload."""
        client = await self._get_client()
        url = f"{self._base_url}/company-news"
        params = self._build_params(request)
        # The API key travels in a header rather than the query string so it
        # cannot leak into access logs, proxy logs, or exception messages that
        # echo the URL.
        headers = {
            "X-Finnhub-Token": self._api_key,
            "Accept": "application/json",
            "User-Agent": "news-agent/1.0",
        }

        last_error: NewsProviderError | None = None

        for attempt in range(self._max_retries):
            try:
                response = await client.get(url, params=params, headers=headers)
            except httpx.TimeoutException as exc:
                last_error = NewsProviderError(
                    f"request timed out after {self._timeout}s: {exc}",
                    provider=self.SOURCE_NAME, ticker=request.ticker, url=url,
                )
            except httpx.TransportError as exc:
                # Connection resets, DNS failures, TLS errors — transient
                # enough to be worth another attempt.
                last_error = NewsProviderError(
                    f"transport error: {exc}",
                    provider=self.SOURCE_NAME, ticker=request.ticker, url=url,
                )
            else:
                status = response.status_code

                if status == 429:
                    # Deliberately NOT retried here. A rate limit is a quota
                    # signal, not a transient fault; retrying within our short
                    # backoff would almost certainly fail again and consume
                    # more quota. Raise immediately with the server's own
                    # Retry-After hint so the caller can back off correctly or
                    # fail over to another source.
                    raise NewsRateLimitError(
                        "Finnhub rate limit exceeded",
                        retry_after=self._parse_retry_after(response),
                        provider=self.SOURCE_NAME, ticker=request.ticker, url=url,
                    )

                if status in _RETRYABLE_STATUS:
                    last_error = NewsProviderError(
                        f"upstream server error (HTTP {status})",
                        provider=self.SOURCE_NAME, ticker=request.ticker,
                        status_code=status, url=url,
                    )
                elif status >= 400:
                    # Deterministic client error (401 bad key, 403, 404…).
                    # Retrying cannot change the outcome, so fail now.
                    raise NewsProviderError(
                        f"request rejected with HTTP {status}",
                        provider=self.SOURCE_NAME, ticker=request.ticker,
                        status_code=status, url=url,
                    )
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise NewsProviderError(
                            f"response was not valid JSON: {exc}",
                            provider=self.SOURCE_NAME, ticker=request.ticker,
                            status_code=status, url=url,
                        ) from exc

            if attempt < self._max_retries - 1:
                delay = self._retry_backoff * (2**attempt)
                logger.warning(
                    "Finnhub fetch for %s failed (attempt %d/%d): %s — retrying in %.2fs",
                    request.ticker, attempt + 1, self._max_retries, last_error, delay,
                )
                await asyncio.sleep(delay)

        assert last_error is not None  # unreachable: loop runs >= 1 time
        raise last_error

    # --- Normalization ---------------------------------------------------------

    def _to_article(self, raw: dict[str, Any], ticker: str) -> NewsArticle:
        """Map one raw Finnhub item onto ``NewsArticle``.

        Raises on anything unmappable; the caller turns that into a skip.
        ``NewsArticle``'s own validators do the heavy lifting (empty title,
        non-http URL, future-dated timestamp), so this method only has to
        handle Finnhub-specific shape concerns.
        """
        timestamp = raw.get("datetime")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or timestamp <= 0:
            raise ValueError(f"missing or invalid 'datetime': {timestamp!r}")
        published_at = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)

        return NewsArticle(
            title=str(raw.get("headline") or ""),
            source=self.SOURCE_NAME,
            publisher=str(raw.get("source") or ""),
            published_at=published_at,
            url=str(raw.get("url") or ""),
            summary=str(raw.get("summary") or ""),
            ticker=ticker,
            # Finnhub's company-news payload carries no language field, so the
            # model default ("en") applies. Documented as an assumption rather
            # than inferred, since guessing a language would be fabrication.
        )

    def _normalize(self, payload: Any, request: NewsRequest) -> list[NewsArticle]:
        """Convert a decoded payload into validated articles.

        Individual malformed items are skipped, never fatal: one bad record
        must not cost the caller an otherwise good result set. Whole-response
        corruption is a different matter and raises.
        """
        if isinstance(payload, dict):
            # Finnhub reports some failures as a JSON object rather than an
            # array, e.g. {"error": "..."}.
            message = payload.get("error") or payload.get("message")
            if message:
                raise NewsProviderError(
                    f"Finnhub returned an error: {message}",
                    provider=self.SOURCE_NAME, ticker=request.ticker,
                )

        if not isinstance(payload, list):
            raise NewsValidationError(
                f"expected a JSON array of articles from Finnhub, got {type(payload).__name__}"
            )

        articles: list[NewsArticle] = []
        skipped = 0

        for position, raw in enumerate(payload):
            if not isinstance(raw, dict):
                skipped += 1
                logger.debug(
                    "Skipping non-object item at index %d for %s (got %s)",
                    position, request.ticker, type(raw).__name__,
                )
                continue
            try:
                articles.append(self._to_article(raw, request.ticker))
            except (ValidationError, ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
                skipped += 1
                logger.debug(
                    "Skipping invalid article at index %d for %s (id=%s url=%s): %s",
                    position, request.ticker, raw.get("id"), raw.get("url"), exc,
                )

        if skipped:
            # Summary at INFO so a systematically broken feed is visible
            # without enabling debug logging; per-item detail stays at DEBUG.
            logger.info(
                "Finnhub: skipped %d of %d invalid article(s) for %s",
                skipped, len(payload), request.ticker,
            )

        return articles

    # --- NewsProvider contract --------------------------------------------------

    async def get_news(self, request: NewsRequest) -> list[NewsArticle]:
        """Fetch and normalize Finnhub company news for ``request``.

        Returns articles unsorted and un-deduplicated, exactly as the
        ``NewsProvider`` contract specifies — ordering and dedup belong to the
        service so they stay identical across every source. An empty list is a
        valid outcome (no recent news); retrieval failures raise.
        """
        payload = await self._fetch(request)
        articles = self._normalize(payload, request)
        logger.info("Finnhub returned %d article(s) for %s", len(articles), request.ticker)
        return articles
