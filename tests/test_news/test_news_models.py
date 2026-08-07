from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from models.news import NewsAnalysisResult, NewsArticle, NewsRequest
from tests.test_news.fakes import make_article


class TestNewsRequest:
    def test_ticker_is_normalized_to_uppercase(self):
        assert NewsRequest(ticker="  aapl  ").ticker == "AAPL"

    def test_language_is_normalized_to_lowercase(self):
        assert NewsRequest(ticker="AAPL", language="EN").language == "en"

    def test_empty_ticker_is_rejected(self):
        with pytest.raises(ValidationError):
            NewsRequest(ticker="")

    @pytest.mark.parametrize("lookback", [0, -1, 366])
    def test_lookback_days_out_of_range_is_rejected(self, lookback):
        with pytest.raises(ValidationError):
            NewsRequest(ticker="AAPL", lookback_days=lookback)

    @pytest.mark.parametrize("limit", [0, -5, 251])
    def test_limit_out_of_range_is_rejected(self, limit):
        with pytest.raises(ValidationError):
            NewsRequest(ticker="AAPL", limit=limit)

    def test_is_immutable(self):
        """Immutability underpins deterministic cache-key derivation."""
        request = NewsRequest(ticker="AAPL")
        with pytest.raises(ValidationError):
            request.ticker = "MSFT"

    def test_is_hashable_and_equal_requests_hash_equally(self):
        a = NewsRequest(ticker="aapl", lookback_days=7, limit=10, language="EN")
        b = NewsRequest(ticker="AAPL", lookback_days=7, limit=10, language="en")
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1  # usable directly as a dict/set key

    def test_model_copy_produces_a_variant_without_mutating(self):
        original = NewsRequest(ticker="AAPL", limit=10)
        variant = original.model_copy(update={"limit": 20})
        assert original.limit == 10
        assert variant.limit == 20


class TestNewsArticle:
    def test_valid_article_is_accepted(self):
        article = make_article(title="Headline", url="https://example.com/x")
        assert article.title == "Headline"
        assert article.ticker == "AAPL"

    def test_empty_title_is_rejected(self):
        with pytest.raises(ValidationError):
            make_article(title="")

    @pytest.mark.parametrize("url", ["not-a-url", "ftp://example.com/x", "", "example.com/x"])
    def test_non_http_url_is_rejected(self, url):
        with pytest.raises(ValidationError):
            make_article(url=url)

    def test_naive_datetime_is_rejected(self):
        """A naive timestamp is an ambiguous instant; sorting and dedup would
        be undefined, so it must not enter the pipeline."""
        with pytest.raises(ValidationError):
            NewsArticle(
                title="X", source="fake", published_at=datetime(2024, 1, 1),
                url="https://example.com/x", ticker="AAPL",
            )

    def test_non_utc_timestamp_is_normalized_to_utc(self):
        tz = timezone(timedelta(hours=5, minutes=30))
        moment = datetime.now(tz) - timedelta(hours=1)
        article = NewsArticle(
            title="X", source="fake", published_at=moment,
            url="https://example.com/x", ticker="AAPL",
        )
        assert article.published_at.tzinfo == timezone.utc
        assert article.published_at == moment  # same instant, different representation

    def test_implausibly_future_dated_article_is_rejected(self):
        with pytest.raises(ValidationError):
            make_article(hours_ago=-24)  # a full day in the future

    def test_small_clock_skew_into_the_future_is_tolerated(self):
        """Publisher clocks drift; a couple of minutes ahead must not discard
        otherwise-valid news."""
        article = make_article(hours_ago=-(2 / 60))  # ~2 minutes ahead
        assert article.published_at > datetime.now(timezone.utc)

    def test_ticker_and_language_are_normalized(self):
        article = make_article(ticker="aapl", language="EN")
        assert article.ticker == "AAPL"
        assert article.language == "en"

    def test_is_immutable(self):
        article = make_article()
        with pytest.raises(ValidationError):
            article.title = "changed"

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.reuters.com/article/x", "reuters.com"),
            ("https://reuters.com/article/x", "reuters.com"),
            ("http://SUB.Example.CO.UK/path", "sub.example.co.uk"),
        ],
    )
    def test_domain_is_derived_deterministically(self, url, expected):
        assert make_article(url=url).domain == expected


class TestNewsAnalysisResult:
    def test_article_count_is_derived_from_articles(self):
        result = NewsAnalysisResult(ticker="AAPL", articles=[make_article(), make_article(url="https://e.com/2")])
        assert result.article_count == 2

    def test_article_count_cannot_drift_from_articles(self):
        """It is computed, so no caller can set an inconsistent value."""
        result = NewsAnalysisResult(ticker="AAPL", articles=[make_article()])
        assert result.article_count == len(result.articles)
        assert "article_count" in result.model_dump()

    def test_empty_result_is_valid(self):
        result = NewsAnalysisResult(ticker="AAPL")
        assert result.article_count == 0
        assert result.articles == []

    def test_agent_name_and_retrieved_at_defaults(self):
        result = NewsAnalysisResult(ticker="AAPL")
        assert result.agent == "news_agent"
        assert result.retrieved_at.tzinfo == timezone.utc

    def test_serializes_to_json(self):
        result = NewsAnalysisResult(ticker="AAPL", articles=[make_article()])
        payload = result.model_dump_json()
        assert '"article_count":1' in payload.replace(" ", "")
