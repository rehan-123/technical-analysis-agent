from __future__ import annotations

import inspect

import pytest

from agent.base import BaseAgent
from agent.news_agent import NewsAgent
from config.settings import Settings
from data.providers.news_exceptions import NewsProviderError
from models.news import NewsAnalysisResult
from services.news_service import NewsService
from tests.test_news.fakes import FailingNewsProvider, FakeNewsProvider, make_article


class TestBaseAgentConformance:
    def test_implements_base_agent(self):
        assert issubclass(NewsAgent, BaseAgent)

    def test_exposes_a_stable_agent_name(self):
        assert NewsAgent.name == "news_agent"

    def test_run_is_async_and_accepts_ticker_plus_kwargs(self):
        """The orchestration contract a Chief Decision Agent relies on."""
        assert inspect.iscoroutinefunction(NewsAgent.run)
        params = inspect.signature(NewsAgent.run).parameters
        assert "ticker" in params
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    @pytest.mark.asyncio
    async def test_run_and_get_news_are_equivalent(self, news_agent):
        via_run = await news_agent.run("AAPL")
        via_direct = await news_agent.get_news("AAPL")
        assert [str(a.url) for a in via_run.articles] == [str(a.url) for a in via_direct.articles]


class TestOrchestration:
    @pytest.mark.asyncio
    async def test_returns_a_news_analysis_result(self, news_agent):
        result = await news_agent.run("AAPL")
        assert isinstance(result, NewsAnalysisResult)
        assert result.agent == "news_agent"
        assert result.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_ticker_is_normalized_before_reaching_the_provider(self, news_settings):
        provider = FakeNewsProvider([])
        agent = NewsAgent(service=NewsService(provider, news_settings), settings=news_settings)
        await agent.run("  aapl  ")
        assert provider.calls[0].ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_settings_defaults_are_applied_when_kwargs_omitted(self):
        settings = Settings(
            news_finnhub_api_key="k",
            news_default_lookback_days=3,
            news_default_limit=11,
            news_default_language="en",
        )
        provider = FakeNewsProvider([])
        agent = NewsAgent(service=NewsService(provider, settings), settings=settings)
        await agent.run("AAPL")
        request = provider.calls[0]
        assert (request.lookback_days, request.limit, request.language) == (3, 11, "en")

    @pytest.mark.asyncio
    async def test_explicit_kwargs_override_settings_defaults(self):
        settings = Settings(
            news_finnhub_api_key="k",
            news_default_lookback_days=3,
            news_default_limit=11,
        )
        provider = FakeNewsProvider([])
        agent = NewsAgent(service=NewsService(provider, settings), settings=settings)
        await agent.run("AAPL", lookback_days=30, limit=7, language="de")
        request = provider.calls[0]
        assert (request.lookback_days, request.limit, request.language) == (30, 7, "de")

    @pytest.mark.asyncio
    async def test_provider_errors_propagate_through_the_agent(self, news_settings):
        service = NewsService(
            provider=FailingNewsProvider(NewsProviderError("boom", provider="fake")),
            settings=news_settings,
        )
        agent = NewsAgent(service=service, settings=news_settings)
        with pytest.raises(NewsProviderError):
            await agent.run("AAPL")

    @pytest.mark.asyncio
    async def test_agent_returns_the_services_result_unmodified(self, news_settings):
        articles = [make_article(title="Only", url="https://e.com/only", hours_ago=1)]
        service = NewsService(FakeNewsProvider(articles), news_settings)
        agent = NewsAgent(service=service, settings=news_settings)

        from_service = await service.get_news(
            __import__("models.news", fromlist=["NewsRequest"]).NewsRequest(
                ticker="AAPL",
                lookback_days=news_settings.news_default_lookback_days,
                limit=news_settings.news_default_limit,
                language=news_settings.news_default_language,
            )
        )
        from_agent = await agent.run("AAPL")
        assert [str(a.url) for a in from_agent.articles] == [str(a.url) for a in from_service.articles]

    @pytest.mark.asyncio
    async def test_agent_requires_an_injected_service(self):
        """Provider selection is deliberately not the agent's job, so there is
        no default service to fall back on."""
        with pytest.raises(TypeError):
            NewsAgent()  # type: ignore[call-arg]
