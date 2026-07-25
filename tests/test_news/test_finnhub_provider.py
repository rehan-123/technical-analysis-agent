from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from config.settings import Settings
from data.providers.finnhub_provider import FinnhubProvider
from data.providers.news_exceptions import (
    NewsConfigurationError,
    NewsProviderError,
    NewsRateLimitError,
    NewsValidationError,
)
from models.news import NewsRequest


def _settings(**overrides) -> Settings:
    base = dict(news_finnhub_api_key="test-key", news_max_retries=2, news_retry_backoff=0.0)
    base.update(overrides)
    return Settings(**base)


def _client(handler) -> httpx.AsyncClient:
    """An AsyncClient wired to an in-process transport — never touches a socket."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _valid_item(**overrides) -> dict:
    item = {
        "category": "company news",
        "datetime": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()),
        "headline": "Apple beats estimates",
        "id": 12345,
        "related": "AAPL",
        "source": "Reuters",
        "summary": "Apple reported quarterly results.",
        "url": "https://example.com/apple-beats",
    }
    item.update(overrides)
    return item


REQUEST = NewsRequest(ticker="AAPL", lookback_days=7, limit=50)


class TestConfiguration:
    def test_missing_api_key_fails_fast_at_construction(self):
        """Surfacing this at wiring time beats an opaque 401 on first request."""
        with pytest.raises(NewsConfigurationError):
            FinnhubProvider(settings=Settings(news_finnhub_api_key=""))

    def test_constructing_with_a_key_succeeds(self):
        assert FinnhubProvider(settings=_settings()) is not None


class TestSuccessfulRetrieval:
    @pytest.mark.asyncio
    async def test_maps_finnhub_payload_onto_news_article(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_valid_item()])

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            articles = await provider.get_news(REQUEST)

        assert len(articles) == 1
        article = articles[0]
        assert article.title == "Apple beats estimates"
        assert article.publisher == "Reuters"       # Finnhub's "source" is the outlet
        assert article.source == "finnhub"          # ours is provenance
        assert article.ticker == "AAPL"
        assert article.published_at.tzinfo == timezone.utc
        assert article.domain == "example.com"

    @pytest.mark.asyncio
    async def test_api_key_is_sent_as_a_header_not_a_query_param(self):
        """Keeps the secret out of access logs, proxy logs, and error strings."""
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["header"] = request.headers.get("X-Finnhub-Token")
            seen["url"] = str(request.url)
            return httpx.Response(200, json=[])

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            await provider.get_news(REQUEST)

        assert seen["header"] == "test-key"
        assert "test-key" not in seen["url"]

    @pytest.mark.asyncio
    async def test_sends_symbol_and_date_range(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(dict(request.url.params))
            return httpx.Response(200, json=[])

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            await provider.get_news(REQUEST)

        assert seen["symbol"] == "AAPL"
        assert "from" in seen and "to" in seen

    @pytest.mark.asyncio
    async def test_empty_array_is_a_valid_empty_result_not_an_error(self):
        async with FinnhubProvider(settings=_settings(), client=_client(lambda r: httpx.Response(200, json=[]))) as p:
            assert await p.get_news(REQUEST) == []


class TestMalformedArticleHandling:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_item",
        [
            {"headline": "", "url": "https://e.com/x", "datetime": 1700000000},   # empty title
            {"headline": "X", "url": "not-a-url", "datetime": 1700000000},        # bad URL
            {"headline": "X", "url": "https://e.com/x"},                          # missing datetime
            {"headline": "X", "url": "https://e.com/x", "datetime": 0},           # zero timestamp
            {"headline": "X", "url": "https://e.com/x", "datetime": "yesterday"}, # wrong type
            {"headline": "X", "url": "https://e.com/x", "datetime": 99999999999999},  # absurd future
        ],
    )
    async def test_one_bad_article_is_skipped_without_failing_the_request(self, bad_item):
        """A single malformed record must never cost the caller a good result."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[bad_item, _valid_item()])

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            articles = await provider.get_news(REQUEST)

        assert len(articles) == 1
        assert articles[0].title == "Apple beats estimates"

    @pytest.mark.asyncio
    async def test_non_object_entries_are_skipped(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=["a string", 42, None, _valid_item()])

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            assert len(await provider.get_news(REQUEST)) == 1

    @pytest.mark.asyncio
    async def test_all_articles_invalid_yields_empty_list_not_an_exception(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[{"headline": "", "url": "bad"}])

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            assert await provider.get_news(REQUEST) == []


class TestWholeResponseCorruption:
    @pytest.mark.asyncio
    async def test_non_array_payload_raises_validation_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            with pytest.raises(NewsValidationError):
                await provider.get_news(REQUEST)

    @pytest.mark.asyncio
    async def test_error_object_payload_raises_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": "Invalid API key"})

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            with pytest.raises(NewsProviderError, match="Invalid API key"):
                await provider.get_news(REQUEST)

    @pytest.mark.asyncio
    async def test_invalid_json_raises_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>not json</html>")

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            with pytest.raises(NewsProviderError):
                await provider.get_news(REQUEST)


class TestHttpErrorHandling:
    @pytest.mark.asyncio
    async def test_rate_limit_maps_to_news_rate_limit_error_with_retry_after(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "30"}, json={})

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            with pytest.raises(NewsRateLimitError) as exc_info:
                await provider.get_news(REQUEST)

        assert exc_info.value.retry_after == 30.0
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider == "finnhub"
        assert exc_info.value.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_rate_limit_is_not_retried(self):
        """Retrying inside a sub-second backoff would fail again and burn quota."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(429, json={})

        async with FinnhubProvider(settings=_settings(news_max_retries=3), client=_client(handler)) as provider:
            with pytest.raises(NewsRateLimitError):
                await provider.get_news(REQUEST)

        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_retry_after_absent_leaves_hint_none(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={})

        async with FinnhubProvider(settings=_settings(), client=_client(handler)) as provider:
            with pytest.raises(NewsRateLimitError) as exc_info:
                await provider.get_news(REQUEST)
        assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_client_error_is_not_retried(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401, json={})

        async with FinnhubProvider(settings=_settings(news_max_retries=3), client=_client(handler)) as provider:
            with pytest.raises(NewsProviderError) as exc_info:
                await provider.get_news(REQUEST)

        assert calls["n"] == 1
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_server_error_is_retried_then_raises(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, json={})

        async with FinnhubProvider(settings=_settings(news_max_retries=3), client=_client(handler)) as provider:
            with pytest.raises(NewsProviderError):
                await provider.get_news(REQUEST)

        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_transient_server_error_then_success_recovers(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, json={})
            return httpx.Response(200, json=[_valid_item()])

        async with FinnhubProvider(settings=_settings(news_max_retries=3), client=_client(handler)) as provider:
            articles = await provider.get_news(REQUEST)

        assert calls["n"] == 2
        assert len(articles) == 1

    @pytest.mark.asyncio
    async def test_transport_error_is_retried_then_raises_with_context(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection reset", request=request)

        async with FinnhubProvider(settings=_settings(news_max_retries=2), client=_client(handler)) as provider:
            with pytest.raises(NewsProviderError) as exc_info:
                await provider.get_news(REQUEST)

        assert exc_info.value.provider == "finnhub"
        assert exc_info.value.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_timeout_is_reported_as_a_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        async with FinnhubProvider(settings=_settings(news_max_retries=1), client=_client(handler)) as provider:
            with pytest.raises(NewsProviderError, match="timed out"):
                await provider.get_news(REQUEST)


class TestErrorContextRendering:
    def test_provider_context_is_rendered_into_the_message(self):
        error = NewsProviderError("boom", provider="finnhub", ticker="AAPL", status_code=502, url="https://x/y")
        text = str(error)
        assert "finnhub" in text and "AAPL" in text and "502" in text

    def test_rate_limit_error_is_a_provider_error(self):
        """Existing handlers catching the parent must keep working."""
        assert issubclass(NewsRateLimitError, NewsProviderError)

    def test_retry_after_appears_in_the_message(self):
        assert "retry_after=12.0s" in str(NewsRateLimitError("limited", retry_after=12.0))


class TestResourceManagement:
    @pytest.mark.asyncio
    async def test_injected_client_is_not_closed_by_the_provider(self):
        """Closing a caller-owned client would sabotage a shared pool."""
        client = _client(lambda r: httpx.Response(200, json=[]))
        provider = FinnhubProvider(settings=_settings(), client=client)
        await provider.get_news(REQUEST)
        await provider.aclose()
        assert not client.is_closed
        await client.aclose()

    @pytest.mark.asyncio
    async def test_context_manager_closes_only_owned_clients(self):
        client = _client(lambda r: httpx.Response(200, json=[]))
        async with FinnhubProvider(settings=_settings(), client=client) as provider:
            await provider.get_news(REQUEST)
        assert not client.is_closed
        await client.aclose()

    def test_construction_performs_no_io_and_needs_no_event_loop(self):
        """Constructible at import/wiring time, outside any running loop."""
        assert FinnhubProvider(settings=_settings()) is not None
