from __future__ import annotations

import pytest

from agent.technical_analysis_agent import TechnicalAnalysisAgent
from config.settings import Settings
from models.analysis_result import TechnicalAnalysisResult
from utils.exceptions import InsufficientDataError


@pytest.mark.asyncio
async def test_analyze_returns_valid_result(synthetic_provider):
    agent = TechnicalAnalysisAgent(settings=Settings(), data_provider=synthetic_provider)
    result = await agent.analyze("AAPL", period="1y", interval="1d")

    assert isinstance(result, TechnicalAnalysisResult)
    assert result.ticker == "AAPL"
    assert result.trend in ("Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish")
    assert 0 <= result.strength <= 100
    assert 0 <= result.confidence <= 100
    assert result.entry_zone[0] < result.entry_zone[1]
    assert len(result.signals) > 0
    assert result.indicators.rsi >= 0


@pytest.mark.asyncio
async def test_bullish_synthetic_data_trends_bullish(synthetic_provider):
    agent = TechnicalAnalysisAgent(settings=Settings(), data_provider=synthetic_provider)
    result = await agent.analyze("AAPL")

    assert result.trend in ("Bullish", "Strong Bullish")
    assert result.stop_loss < result.indicators.close
    assert all(t > result.indicators.close for t in result.targets)


@pytest.mark.asyncio
async def test_bearish_synthetic_data_trends_bearish(bearish_provider):
    agent = TechnicalAnalysisAgent(settings=Settings(), data_provider=bearish_provider)
    result = await agent.analyze("AAPL")

    assert result.trend in ("Bearish", "Strong Bearish")
    assert result.stop_loss > result.indicators.close


@pytest.mark.asyncio
async def test_insufficient_data_raises(synthetic_provider):
    agent = TechnicalAnalysisAgent(settings=Settings(), data_provider=synthetic_provider)
    with pytest.raises(InsufficientDataError):
        await agent.analyze("AAPL", period="1mo")


@pytest.mark.asyncio
async def test_run_matches_base_agent_interface(synthetic_provider):
    """The orchestration entry point (`run`) must behave like `analyze`."""
    agent = TechnicalAnalysisAgent(settings=Settings(), data_provider=synthetic_provider)
    result = await agent.run("AAPL", period="1y", interval="1d")
    assert isinstance(result, TechnicalAnalysisResult)
