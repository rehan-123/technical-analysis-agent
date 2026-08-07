from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.news_routes as news_routes
from agent.news_agent import NewsAgent
from config.settings import Settings
from data.providers.news_exceptions import (
    NewsConfigurationError,
    NewsProviderError,
    NewsRateLimitError,
)
from main import app
from services.news_service import NewsService
from tests.test_news.fakes import FailingNewsProvider, FakeNewsProvider, default_article_set


@pytest.fixture(autouse=True)
def _reset_agent_singleton():
    """Each test controls the module-level agent singleton in isolation.

    ``get_news_agent`` caches its result, so without resetting, one test's
    agent (or a deliberately un-built ``None``) would leak into the next.
    """
    original = news_routes._agent_singleton
    news_routes._agent_singleton = None
    yield
    news_routes._agent_singleton = original


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _install_agent(provider) -> None:
    settings = Settings(news_finnhub_api_key="k")
    news_routes._agent_singleton = NewsAgent(
        service=NewsService(provider=provider, settings=settings), settings=settings
    )


class TestConfigurationErrorMapping:
    """The reported bug: a missing API key made agent construction raise
    NewsConfigurationError during FastAPI dependency resolution, escaping the
    handler and surfacing as a bare 500. It must be a structured 503."""

    def test_post_returns_503_when_provider_is_unconfigured(self, client, monkeypatch):
        def _boom() -> NewsAgent:
            raise NewsConfigurationError(
                "Finnhub news provider is enabled but no API key is configured"
            )

        monkeypatch.setattr(news_routes, "get_news_agent", _boom)
        response = client.post("/news", json={"ticker": "AAPL"})
        assert response.status_code == 503
        assert "API key" in response.json()["detail"]

    def test_get_returns_503_when_provider_is_unconfigured(self, client, monkeypatch):
        def _boom() -> NewsAgent:
            raise NewsConfigurationError("no API key")

        monkeypatch.setattr(news_routes, "get_news_agent", _boom)
        response = client.get("/news/AAPL")
        assert response.status_code == 503
        assert response.json()["detail"] == "no API key"

    def test_health_still_reports_configured_false_without_raising(self, client, monkeypatch):
        def _boom() -> NewsAgent:
            raise NewsConfigurationError("no API key")

        monkeypatch.setattr(news_routes, "get_news_agent", _boom)
        response = client.get("/news/health")
        assert response.status_code == 200
        assert response.json()["configured"] is False


class TestSuccessAndErrorMapping:
    def test_post_returns_articles_on_success(self, client):
        _install_agent(FakeNewsProvider(default_article_set()))
        response = client.post("/news", json={"ticker": "AAPL"})
        assert response.status_code == 200
        body = response.json()
        assert body["ticker"] == "AAPL"
        assert body["agent"] == "news_agent"
        assert "article_count" in body

    def test_get_returns_articles_on_success(self, client):
        _install_agent(FakeNewsProvider(default_article_set()))
        response = client.get("/news/AAPL?limit=3")
        assert response.status_code == 200
        assert response.json()["ticker"] == "AAPL"

    def test_rate_limit_maps_to_429_with_retry_after_header(self, client):
        _install_agent(
            FailingNewsProvider(NewsRateLimitError("limited", retry_after=30.0, provider="finnhub"))
        )
        response = client.post("/news", json={"ticker": "AAPL"})
        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "30"

    def test_provider_error_maps_to_502(self, client):
        _install_agent(FailingNewsProvider(NewsProviderError("upstream down", provider="finnhub")))
        response = client.post("/news", json={"ticker": "AAPL"})
        assert response.status_code == 502

    def test_invalid_request_body_is_422(self, client):
        _install_agent(FakeNewsProvider([]))
        response = client.post("/news", json={"ticker": ""})  # fails min_length
        assert response.status_code == 422
