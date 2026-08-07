from __future__ import annotations

from datetime import datetime, timezone

from config.settings import Settings
from models.news import NewsAnalysisResult
from scanner.news_scoring import score_news
from tests.test_news.fakes import make_article


def test_none_news_returns_neutral():
    score, notes = score_news(None, settings=Settings())
    assert score == 50
    assert notes


def test_empty_articles_returns_neutral():
    result = NewsAnalysisResult(ticker="AAPL", articles=[])
    score, _ = score_news(result, settings=Settings())
    assert score == 50


def test_many_recent_articles_saturate_near_100():
    settings = Settings()
    articles = [make_article(hours_ago=0.1, ticker="AAPL") for _ in range(settings.scanner_news_saturation_count * 2)]
    result = NewsAnalysisResult(ticker="AAPL", articles=articles)
    score, _ = score_news(result, settings=settings)
    assert score == 100


def test_old_articles_score_lower_than_fresh():
    settings = Settings()
    fresh = NewsAnalysisResult(ticker="AAPL", articles=[make_article(hours_ago=1, ticker="AAPL") for _ in range(3)])
    old = NewsAnalysisResult(
        ticker="AAPL", articles=[make_article(hours_ago=24 * 30, ticker="AAPL") for _ in range(3)]
    )
    fresh_score, _ = score_news(fresh, settings=settings)
    old_score, _ = score_news(old, settings=settings)
    assert fresh_score > old_score


def test_score_bounded_0_100():
    settings = Settings()
    articles = [make_article(hours_ago=h, ticker="AAPL") for h in range(0, 200, 5)]
    result = NewsAnalysisResult(ticker="AAPL", articles=articles)
    score, _ = score_news(result, settings=settings)
    assert 0 <= score <= 100


def test_deterministic_given_fixed_now():
    settings = Settings()
    now = datetime.now(timezone.utc)
    result = NewsAnalysisResult(ticker="AAPL", articles=[make_article(hours_ago=2, ticker="AAPL")])
    first, _ = score_news(result, settings=settings, now=now)
    second, _ = score_news(result, settings=settings, now=now)
    assert first == second
