from __future__ import annotations

import pytest

from agent.technical_analysis_agent import TechnicalAnalysisAgent
from config.settings import Settings
from data.synthetic_provider import SyntheticDataProvider
from models.analysis_result import TechnicalAnalysisResult


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
async def bullish_technical(synthetic_provider, settings) -> TechnicalAnalysisResult:
    """A real, fully-populated TechnicalAnalysisResult for a clearly
    trending-up synthetic ticker — reuses the existing top-level
    ``synthetic_provider`` fixture so strategy tests exercise the actual
    Technical Engine output shape rather than a hand-built stub."""
    agent = TechnicalAnalysisAgent(settings=settings, data_provider=synthetic_provider)
    return await agent.analyze("AAPL", period="1y", interval="1d")


@pytest.fixture
async def bearish_technical(bearish_provider, settings) -> TechnicalAnalysisResult:
    agent = TechnicalAnalysisAgent(settings=settings, data_provider=bearish_provider)
    return await agent.analyze("AAPL", period="1y", interval="1d")


@pytest.fixture
async def choppy_technical(settings) -> TechnicalAnalysisResult:
    """A low-drift, higher-volatility series that tends to land closer to
    Neutral / ranging structure — useful for exercising the "not applicable"
    branches every strategy must handle without raising."""
    provider = SyntheticDataProvider(seed=99, start_price=80.0, drift=0.00005, volatility=0.03)
    agent = TechnicalAnalysisAgent(settings=settings, data_provider=provider)
    return await agent.analyze("CHOP", period="1y", interval="1d")
