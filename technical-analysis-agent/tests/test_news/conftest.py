from __future__ import annotations

import pytest

from agent.news_agent import NewsAgent
from config.settings import Settings
from models.news import NewsRequest
from services.news_service import NewsService
from tests.test_news.fakes import FakeNewsProvider, default_article_set


@pytest.fixture
def news_settings() -> Settings:
    """Settings with a dummy Finnhub key present.

    The key is required only so ``FinnhubProvider`` can be constructed in
    tests that exercise it; no test ever performs a real network call.
    """
    return Settings(
        news_finnhub_api_key="test-key-not-a-real-secret",
        news_deduplicate=True,
        news_dedup_time_window_minutes=60,
    )


@pytest.fixture
def fake_provider() -> FakeNewsProvider:
    """A provider returning the deliberately messy default article set."""
    return FakeNewsProvider(default_article_set())


@pytest.fixture
def empty_provider() -> FakeNewsProvider:
    """A provider returning no articles — a valid, non-error outcome."""
    return FakeNewsProvider([])


@pytest.fixture
def news_service(fake_provider: FakeNewsProvider, news_settings: Settings) -> NewsService:
    return NewsService(provider=fake_provider, settings=news_settings)


@pytest.fixture
def news_agent(news_service: NewsService, news_settings: Settings) -> NewsAgent:
    return NewsAgent(service=news_service, settings=news_settings)


@pytest.fixture
def basic_request() -> NewsRequest:
    return NewsRequest(ticker="AAPL", lookback_days=7, limit=50)
