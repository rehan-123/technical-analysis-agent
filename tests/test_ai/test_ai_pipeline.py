from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.ai_routes as ai_routes
from agent.ai_pipeline_agent import AIPipelineAgent
from agent.base import BaseAgent
from data.providers.news_exceptions import NewsProviderError
from main import app
from models.ai_analysis import AIAnalysisResult, Recommendation
from utils.exceptions import DataFetchError, InsufficientDataError, TechnicalAgentError


# --- fakes --------------------------------------------------------------------


class FakeTechnicalAgent:
    name = "technical_analysis_agent"

    def __init__(self, *, result="TECH", error=None):
        self._result, self._error = result, error
        self.calls = []

    async def analyze(self, ticker, *, period="1y", interval="1d"):
        self.calls.append({"ticker": ticker, "period": period, "interval": interval})
        if self._error:
            raise self._error
        return self._result


class FakeNewsAgent:
    name = "news_agent"

    def __init__(self, *, result="NEWS", error=None):
        self._result, self._error = result, error
        self.calls = []

    async def run(self, ticker, **kwargs):
        self.calls.append(ticker)
        if self._error:
            raise self._error
        return self._result


class FakeAIAgent:
    name = "ai_analysis_agent"

    def __init__(self):
        self.received = None

    async def analyze(self, ticker, *, technical=None, news=None, additional_inputs=None, model=None):
        self.received = {"ticker": ticker, "technical": technical, "news": news, "model": model}
        return AIAnalysisResult(
            ticker=ticker, recommendation=Recommendation.BUY, confidence=70,
            investment_thesis="Pipeline thesis.",
        )


def _pipeline(*, tech=None, news=None, ai=None):
    ai = ai or FakeAIAgent()
    return AIPipelineAgent(
        technical_agent=tech or FakeTechnicalAgent(),  # type: ignore[arg-type]
        news_agent=news if news is not False else None,  # type: ignore[arg-type]
        ai_agent=ai,  # type: ignore[arg-type]
    ), ai


# --- composite agent ----------------------------------------------------------


class TestPipelineAgent:
    def test_is_a_base_agent(self):
        p, _ = _pipeline(news=FakeNewsAgent())
        assert isinstance(p, BaseAgent)
        assert p.name == "ai_pipeline_agent"

    @pytest.mark.asyncio
    async def test_gathers_both_inputs_and_forwards_them(self):
        p, ai = _pipeline(news=FakeNewsAgent())
        result = await p.analyze("AAPL")
        assert result.recommendation is Recommendation.BUY
        assert ai.received["technical"] == "TECH"
        assert ai.received["news"] == "NEWS"

    @pytest.mark.asyncio
    async def test_period_and_interval_are_forwarded_to_technical(self):
        tech = FakeTechnicalAgent()
        p, _ = _pipeline(tech=tech, news=FakeNewsAgent())
        await p.analyze("AAPL", period="6mo", interval="1h")
        assert tech.calls[0] == {"ticker": "AAPL", "period": "6mo", "interval": "1h"}

    @pytest.mark.asyncio
    async def test_model_override_forwarded_to_ai_agent(self):
        p, ai = _pipeline(news=FakeNewsAgent())
        await p.analyze("AAPL", model="qwen2.5:14b")
        assert ai.received["model"] == "qwen2.5:14b"

    @pytest.mark.asyncio
    async def test_runs_without_a_news_agent(self):
        p, ai = _pipeline(news=False)
        await p.analyze("AAPL")
        assert ai.received["news"] is None
        assert ai.received["technical"] == "TECH"

    @pytest.mark.asyncio
    async def test_news_failure_is_tolerated(self):
        """News is enrichment — a news outage must not fail the request."""
        p, ai = _pipeline(news=FakeNewsAgent(error=NewsProviderError("down", provider="finnhub")))
        result = await p.analyze("AAPL")
        assert result.confidence == 70
        assert ai.received["news"] is None
        assert ai.received["technical"] == "TECH"

    @pytest.mark.asyncio
    async def test_technical_failure_propagates_unchanged(self):
        """Technical is the primary input — the caller must see the real cause."""
        p, _ = _pipeline(tech=FakeTechnicalAgent(error=DataFetchError("yahoo down")),
                         news=FakeNewsAgent())
        with pytest.raises(DataFetchError):
            await p.analyze("AAPL")

    @pytest.mark.asyncio
    async def test_non_news_error_from_news_agent_is_not_swallowed(self):
        """A programming bug must surface, not be treated as 'news unavailable'."""
        p, _ = _pipeline(news=FakeNewsAgent(error=TypeError("bug")))
        with pytest.raises(TypeError):
            await p.analyze("AAPL")

    @pytest.mark.asyncio
    async def test_run_delegates_to_analyze(self):
        p, ai = _pipeline(news=FakeNewsAgent())
        result = await p.run("AAPL", period="2y")
        assert result.ticker == "AAPL"
        assert ai.received["technical"] == "TECH"


# --- new by-ticker endpoint ---------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    ai_routes._agent_singleton = None
    ai_routes._pipeline_singleton = None
    yield
    ai_routes._agent_singleton = None
    ai_routes._pipeline_singleton = None


client = TestClient(app, raise_server_exceptions=False)


class FakePipeline:
    def __init__(self, *, error=None):
        self._error = error
        self.calls = []

    async def analyze(self, ticker, *, period="1y", interval="1d", model=None):
        self.calls.append({"ticker": ticker, "period": period, "interval": interval, "model": model})
        if self._error:
            raise self._error
        return AIAnalysisResult(
            ticker=ticker, recommendation=Recommendation.HOLD, confidence=55,
            investment_thesis="From pipeline.",
        )


def _install(monkeypatch, pipeline):
    monkeypatch.setattr(ai_routes, "get_ai_pipeline_agent", lambda: pipeline)


class TestByTickerEndpoint:
    def test_returns_200_with_ticker_only(self, monkeypatch):
        _install(monkeypatch, FakePipeline())
        resp = client.get("/ai/analyze/AAPL")
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "AAPL"
        assert resp.json()["recommendation"] == "HOLD"

    def test_query_params_are_forwarded(self, monkeypatch):
        pipe = FakePipeline()
        _install(monkeypatch, pipe)
        client.get("/ai/analyze/MSFT?period=6mo&interval=1h&model=custom")
        assert pipe.calls[0] == {
            "ticker": "MSFT", "period": "6mo", "interval": "1h", "model": "custom",
        }

    def test_defaults_applied_when_params_omitted(self, monkeypatch):
        pipe = FakePipeline()
        _install(monkeypatch, pipe)
        client.get("/ai/analyze/AAPL")
        assert pipe.calls[0]["period"] == "1y"
        assert pipe.calls[0]["interval"] == "1d"

    @pytest.mark.parametrize(
        "error, expected",
        [
            (InsufficientDataError("too few bars"), 422),
            (DataFetchError("provider down"), 502),
            (NewsProviderError("news down", provider="finnhub"), 502),
            (TechnicalAgentError("internal"), 500),
        ],
    )
    def test_upstream_failures_map_to_correct_status(self, monkeypatch, error, expected):
        _install(monkeypatch, FakePipeline(error=error))
        resp = client.get("/ai/analyze/AAPL")
        assert resp.status_code == expected
        assert "Traceback" not in resp.text


class TestBackwardCompatibility:
    def test_post_analyze_contract_is_unchanged(self):
        """The original POST endpoint must still accept the full request body."""
        spec = client.get("/openapi.json").json()
        post_body = spec["paths"]["/ai/analyze"]["post"]["requestBody"]
        ref = post_body["content"]["application/json"]["schema"]["$ref"]
        assert ref.endswith("AIAnalysisRequest")

    def test_new_endpoint_has_no_request_body(self):
        """Ticker-only entry point takes path + query params, no body."""
        spec = client.get("/openapi.json").json()
        get_op = spec["paths"]["/ai/analyze/{ticker}"]["get"]
        assert "requestBody" not in get_op
        names = {p["name"] for p in get_op["parameters"]}
        assert {"ticker", "period", "interval"} <= names

    def test_both_endpoints_return_the_same_response_model(self):
        spec = client.get("/openapi.json").json()
        post_ref = spec["paths"]["/ai/analyze"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        get_ref = spec["paths"]["/ai/analyze/{ticker}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        assert post_ref == get_ref

    def test_existing_routers_still_registered(self):
        spec = client.get("/openapi.json").json()
        assert "/analyze/{ticker}" in spec["paths"]
        assert any(p.startswith("/news") for p in spec["paths"])
