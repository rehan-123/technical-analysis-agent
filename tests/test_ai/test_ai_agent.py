from __future__ import annotations

import pytest

from agent.ai_analysis_agent import AIAnalysisAgent
from agent.base import BaseAgent
from models.ai_analysis import AIAnalysisRequest, AIAnalysisResult, Recommendation


class FakeService:
    """Stand-in for AIAnalysisService: records the request and returns a
    canned result. No prompt building, no LLM, no network."""

    def __init__(self) -> None:
        self.received: AIAnalysisRequest | None = None

    async def analyze(self, request: AIAnalysisRequest) -> AIAnalysisResult:
        self.received = request
        return AIAnalysisResult(
            ticker=request.ticker,
            recommendation=Recommendation.BUY,
            confidence=71,
            investment_thesis="Fake thesis.",
        )


def _agent() -> tuple[AIAnalysisAgent, FakeService]:
    svc = FakeService()
    return AIAnalysisAgent(service=svc), svc  # type: ignore[arg-type]


class TestConformance:
    def test_is_a_base_agent(self):
        agent, _ = _agent()
        assert isinstance(agent, BaseAgent)

    def test_has_expected_name(self):
        agent, _ = _agent()
        assert agent.name == "ai_analysis_agent"

    def test_requires_a_service_dependency(self):
        with pytest.raises(TypeError):
            AIAnalysisAgent()  # type: ignore[call-arg]


class TestDelegation:
    @pytest.mark.asyncio
    async def test_analyze_returns_service_result(self):
        agent, _ = _agent()
        result = await agent.analyze("AAPL")
        assert isinstance(result, AIAnalysisResult)
        assert result.recommendation is Recommendation.BUY
        assert result.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_analyze_builds_request_with_ticker(self):
        agent, svc = _agent()
        await agent.analyze("msft")
        assert svc.received is not None
        # ticker normalization is the model's job; agent just forwards it
        assert svc.received.ticker in ("MSFT", "msft")

    @pytest.mark.asyncio
    async def test_inputs_are_forwarded_into_the_request(self):
        agent, svc = _agent()
        await agent.analyze("AAPL", model="qwen2.5:14b")
        assert svc.received.model == "qwen2.5:14b"

    @pytest.mark.asyncio
    async def test_run_delegates_to_analyze(self):
        agent, svc = _agent()
        result = await agent.run("AAPL", model="custom")
        assert isinstance(result, AIAnalysisResult)
        assert svc.received.model == "custom"

    @pytest.mark.asyncio
    async def test_run_and_analyze_are_equivalent(self):
        agent_a, svc_a = _agent()
        agent_b, svc_b = _agent()
        r1 = await agent_a.run("AAPL")
        r2 = await agent_b.analyze("AAPL")
        assert r1.ticker == r2.ticker == "AAPL"
        assert svc_a.received.ticker == svc_b.received.ticker

    @pytest.mark.asyncio
    async def test_agent_performs_no_business_logic(self):
        """The agent must not alter the service's result."""
        agent, _ = _agent()
        result = await agent.run("AAPL")
        assert result.confidence == 71  # exactly what the service returned
