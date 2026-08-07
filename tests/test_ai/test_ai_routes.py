from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.ai_routes as ai_routes
from llm.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
)
from main import app
from models.ai_analysis import AIAnalysisResult, Recommendation
from services.ai_exceptions import InvalidAIResponse, ResponseParseError
from services.prompt_sections.exceptions import PromptBuildError


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Isolate each test: clear the cached agent before and after."""
    ai_routes._agent_singleton = None
    yield
    ai_routes._agent_singleton = None


client = TestClient(app, raise_server_exceptions=False)


class FakeAgent:
    """Returns a canned result or raises a scripted error. No LLM/network."""

    name = "ai_analysis_agent"

    def __init__(self, *, result=None, error=None):
        self._result = result
        self._error = error

    async def run(self, ticker, **kwargs):
        if self._error is not None:
            raise self._error
        return self._result


def _install_agent(monkeypatch, *, result=None, error=None):
    monkeypatch.setattr(ai_routes, "get_ai_agent", lambda: FakeAgent(result=result, error=error))


def _ok_result(ticker="AAPL"):
    return AIAnalysisResult(
        ticker=ticker, recommendation=Recommendation.BUY, confidence=80,
        investment_thesis="Thesis.",
    )


def _body():
    return {"ticker": "AAPL"}


class TestSuccess:
    def test_analyze_returns_200_and_result(self, monkeypatch):
        _install_agent(monkeypatch, result=_ok_result())
        resp = client.post("/ai/analyze", json=_body())
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "AAPL"
        assert data["recommendation"] == "BUY"
        assert data["confidence"] == 80

    def test_disclaimer_present_in_response(self, monkeypatch):
        _install_agent(monkeypatch, result=_ok_result())
        resp = client.post("/ai/analyze", json=_body())
        assert resp.json()["disclaimer"]


class TestValidation:
    def test_missing_ticker_returns_422(self, monkeypatch):
        _install_agent(monkeypatch, result=_ok_result())
        resp = client.post("/ai/analyze", json={})
        assert resp.status_code == 422

    def test_malformed_body_returns_422(self, monkeypatch):
        _install_agent(monkeypatch, result=_ok_result())
        resp = client.post("/ai/analyze", json={"ticker": ""})
        assert resp.status_code == 422


class TestErrorMapping:
    @pytest.mark.parametrize(
        "error, expected",
        [
            (LLMRateLimitError("slow down", provider="ollama"), 429),
            (LLMConfigurationError("no backend"), 503),
            (LLMProviderError("upstream 500", provider="ollama"), 502),
            (ResponseParseError("not json"), 502),
            (InvalidAIResponse("bad fields"), 502),
            (LLMResponseError("weird envelope"), 502),
            (PromptBuildError("nothing to build"), 500),
            (RuntimeError("boom"), 500),
        ],
    )
    def test_exception_maps_to_status(self, monkeypatch, error, expected):
        _install_agent(monkeypatch, error=error)
        resp = client.post("/ai/analyze", json=_body())
        assert resp.status_code == expected
        # structured JSON, never a stack trace
        assert "detail" in resp.json()
        assert "Traceback" not in resp.text

    def test_unexpected_error_message_is_generic(self, monkeypatch):
        _install_agent(monkeypatch, error=RuntimeError("secret internals"))
        resp = client.post("/ai/analyze", json=_body())
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"
        assert "secret internals" not in resp.text

    def test_rate_limit_message_is_surfaced(self, monkeypatch):
        _install_agent(monkeypatch, error=LLMRateLimitError("try later", provider="ollama"))
        resp = client.post("/ai/analyze", json=_body())
        assert resp.status_code == 429


class TestExistingEndpointsUnaffected:
    def test_openapi_lists_ai_routes(self):
        spec = client.get("/openapi.json").json()
        assert "/ai/analyze" in spec["paths"]
        assert "/ai/health" in spec["paths"]

    def test_existing_news_and_technical_paths_still_present(self):
        spec = client.get("/openapi.json").json()
        # news router still mounted
        assert any(p.startswith("/news") for p in spec["paths"])
