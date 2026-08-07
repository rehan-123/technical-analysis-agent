from __future__ import annotations

import pytest

from config.settings import Settings
from models.news import NewsRequest
from services.news_service import NewsService
from tests.test_news.fakes import FailingNewsProvider, FakeNewsProvider, make_article
from data.providers.news_exceptions import NewsProviderError

def _service(articles, **setting_overrides) -> NewsService:
    settings = Settings(
        news_finnhub_api_key="test-key",
        news_deduplicate=setting_overrides.pop("news_deduplicate", True),
        news_dedup_time_window_minutes=setting_overrides.pop("news_dedup_time_window_minutes", 60),
        **setting_overrides,
    )
    return NewsService(provider=FakeNewsProvider(articles), settings=settings)

class TestNormalizationHelpers:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("https://example.com/x", "http://example.com/x"),            # scheme differs
            ("https://www.example.com/x", "https://example.com/x"),        # www differs
            ("https://example.com/x/", "https://example.com/x"),           # trailing slash
            ("https://example.com/x?utm_source=a", "https://example.com/x?utm_source=b"),
            ("https://example.com/x?a=1&b=2", "https://example.com/x?b=2&a=1"),  # param order
        ],
    )
    def test_urls_that_denote_the_same_document_normalize_equally(self, a, b):
        assert NewsService._normalize_url(a) == NewsService._normalize_url(b)

    def test_meaningful_query_params_are_preserved(self):
        """Blanket query-stripping would wrongly merge distinct articles."""
        assert NewsService._normalize_url("https://e.com/n?id=1") != NewsService._normalize_url("https://e.com/n?id=2")

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Apple beats estimates", "apple beats estimates"),
            ("Apple beats estimates!", "Apple beats estimates"),
            ("Apple   beats  estimates", "Apple beats estimates"),
        ],
    )
    def test_cosmetically_different_headlines_normalize_equally(self, a, b):
        assert NewsService._normalize_title(a) == NewsService._normalize_title(b)

    def test_distinct_headlines_do_not_collide(self):
        assert NewsService._normalize_title("Apple up") != NewsService._normalize_title("Apple down")

class TestPipeline:
    @pytest.mark.asyncio
    async def test_articles_are_sorted_newest_first(self):
        articles = [
            make_article(title="old", url="https://e.com/1", hours_ago=5),
            make_article(title="new", url="https://e.com/2", hours_ago=1),
            make_article(title="mid", url="https://e.com/3", hours_ago=3),
        ]
        result = await _service(articles).get_news(NewsRequest(ticker="AAPL"))
        assert [a.title for a in result.articles] == ["new", "mid", "old"]

    @pytest.mark.asyncio
    async def test_exact_duplicate_urls_are_removed(self):
        articles = [
            make_article(title="A", url="https://example.com/a", hours_ago=1),
            make_article(title="A", url="https://www.example.com/a?utm_source=x", hours_ago=1),
        ]
        result = await _service(articles).get_news(NewsRequest(ticker="AAPL"))
        assert result.article_count == 1

    @pytest.mark.asyncio
    async def test_syndicated_reprint_within_window_is_removed(self):
        """Same headline, different outlet and URL, minutes apart."""
        articles = [
            make_article(title="Apple beats estimates", url="https://a.com/x", hours_ago=1.0),
            make_article(title="Apple beats estimates!", url="https://b.com/y", hours_ago=1.3),
        ]
        result = await _service(articles).get_news(NewsRequest(ticker="AAPL"))
        assert result.article_count == 1

    @pytest.mark.asyncio
    async def test_same_headline_outside_window_is_kept(self):
        """An identical headline days later is a new story, not a reprint."""
        articles = [
            make_article(title="Apple beats estimates", url="https://a.com/x", hours_ago=1),
            make_article(title="Apple beats estimates", url="https://a.com/y", hours_ago=48),
        ]
        result = await _service(articles).get_news(NewsRequest(ticker="AAPL"))
        assert result.article_count == 2

    @pytest.mark.asyncio
    async def test_deduplication_keeps_the_newest_copy(self):
        """Sorting before dedup makes the survivor well-defined."""
        articles = [
            make_article(title="Story", url="https://a.com/older", hours_ago=1.5, publisher="Older"),
            make_article(title="Story", url="https://b.com/newer", hours_ago=1.0, publisher="Newer"),
        ]
        result = await _service(articles).get_news(NewsRequest(ticker="AAPL"))
        assert result.article_count == 1
        assert result.articles[0].publisher == "Newer"

    @pytest.mark.asyncio
    async def test_deduplication_can_be_disabled(self):
        articles = [
            make_article(title="A", url="https://example.com/a", hours_ago=1),
            make_article(title="A", url="https://www.example.com/a?utm_source=x", hours_ago=1),
        ]
        service = _service(articles, news_deduplicate=False)
        result = await service.get_news(NewsRequest(ticker="AAPL"))
        assert result.article_count == 2

    @pytest.mark.asyncio
    async def test_articles_outside_lookback_window_are_filtered(self):
        articles = [
            make_article(title="recent", url="https://e.com/1", hours_ago=2),
            make_article(title="ancient", url="https://e.com/2", hours_ago=24 * 30),
        ]
        result = await _service(articles).get_news(NewsRequest(ticker="AAPL", lookback_days=7))
        assert [a.title for a in result.articles] == ["recent"]

    @pytest.mark.asyncio
    async def test_language_filter_applies_only_when_requested(self):
        articles = [
            make_article(title="english", url="https://e.com/1", hours_ago=1, language="en"),
            make_article(title="german", url="https://e.com/2", hours_ago=2, language="de"),
        ]
        unfiltered = await _service(articles).get_news(NewsRequest(ticker="AAPL"))
        assert unfiltered.article_count == 2

        filtered = await _service(articles).get_news(NewsRequest(ticker="AAPL", language="en"))
        assert [a.title for a in filtered.articles] == ["english"]

    @pytest.mark.asyncio
    async def test_limit_is_applied_after_deduplication(self):
        """The caller must receive N *distinct* articles, not N raw ones."""
        articles = [
            make_article(title="A", url="https://e.com/a", hours_ago=1),
            make_article(title="A", url="https://www.e.com/a?utm_source=z", hours_ago=1),  # dup of above
            make_article(title="B", url="https://e.com/b", hours_ago=2),
            make_article(title="C", url="https://e.com/c", hours_ago=3),
        ]
        result = await _service(articles).get_news(NewsRequest(ticker="AAPL", limit=2))
        assert result.article_count == 2
        assert [a.title for a in result.articles] == ["A", "B"]

    @pytest.mark.asyncio
    async def test_empty_provider_result_is_a_valid_empty_result(self):
        result = await _service([]).get_news(NewsRequest(ticker="AAPL"))
        assert result.article_count == 0
        assert result.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_pipeline_is_deterministic_across_input_orderings(self):
        """Identical content in a different order must yield identical output —
        the URL tiebreaker exists precisely for this."""
        articles = [
            make_article(title="X", url="https://e.com/1", hours_ago=1),
            make_article(title="Y", url="https://e.com/2", hours_ago=1),  # identical timestamp
            make_article(title="Z", url="https://e.com/3", hours_ago=1),
        ]
        first = await _service(articles).get_news(NewsRequest(ticker="AAPL"))
        second = await _service(list(reversed(articles))).get_news(NewsRequest(ticker="AAPL"))
        assert [str(a.url) for a in first.articles] == [str(a.url) for a in second.articles]

    @pytest.mark.asyncio
    async def test_provider_errors_propagate_and_are_not_swallowed(self):
        """Converting a failure into an empty result would make an outage
        indistinguishable from 'no news exists'."""
        service = NewsService(
            provider=FailingNewsProvider(NewsProviderError("upstream down", provider="fake")),
            settings=Settings(news_finnhub_api_key="k"),
        )
        with pytest.raises(NewsProviderError):
            await service.get_news(NewsRequest(ticker="AAPL"))

    @pytest.mark.asyncio
    async def test_request_is_passed_through_to_the_provider(self):
        provider = FakeNewsProvider([])
        service = NewsService(provider=provider, settings=Settings(news_finnhub_api_key="k"))
        request = NewsRequest(ticker="MSFT", lookback_days=3, limit=5)
        await service.get_news(request)
        assert provider.calls == [request]
