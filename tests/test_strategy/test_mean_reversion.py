from __future__ import annotations

import pytest

from config.settings import Settings
from models.strategy import StrategyDirection, StrategyName
from strategy.mean_reversion import MeanReversionStrategy
from tests.test_strategy.factories import make_indicators, make_technical


@pytest.fixture
def strategy() -> MeanReversionStrategy:
    return MeanReversionStrategy(Settings())


def test_long_when_oversold_at_lower_band(strategy):
    settings = Settings()
    technical = make_technical(
        confidence=60,
        indicators=make_indicators(rsi=settings.rsi_oversold - 2.0, bb_percent_b=0.02),
        market_structure={"structure": "uptrend"},
    )
    signal = strategy.evaluate(technical)

    assert signal.strategy is StrategyName.MEAN_REVERSION
    assert signal.applicable is True
    assert signal.direction is StrategyDirection.LONG


def test_short_when_overbought_at_upper_band(strategy):
    settings = Settings()
    technical = make_technical(
        confidence=60,
        indicators=make_indicators(rsi=settings.rsi_overbought + 2.0, bb_percent_b=0.98),
        market_structure={"structure": "downtrend"},
    )
    signal = strategy.evaluate(technical)

    assert signal.applicable is True
    assert signal.direction is StrategyDirection.SHORT


def test_not_applicable_when_rsi_extreme_but_band_disagrees(strategy):
    settings = Settings()
    technical = make_technical(indicators=make_indicators(rsi=settings.rsi_oversold - 2.0, bb_percent_b=0.5))
    signal = strategy.evaluate(technical)
    assert signal.applicable is False
    assert signal.direction is StrategyDirection.NONE


def test_not_applicable_in_neutral_zone(strategy):
    technical = make_technical(indicators=make_indicators(rsi=50.0, bb_percent_b=0.5))
    signal = strategy.evaluate(technical)
    assert signal.applicable is False


def test_counter_trend_call_scores_lower_than_agreeing_call(strategy):
    settings = Settings()
    agreeing = strategy.evaluate(
        make_technical(
            indicators=make_indicators(rsi=settings.rsi_oversold - 2.0, bb_percent_b=0.02),
            market_structure={"structure": "uptrend"},
        )
    )
    counter = strategy.evaluate(
        make_technical(
            indicators=make_indicators(rsi=settings.rsi_oversold - 2.0, bb_percent_b=0.02),
            market_structure={"structure": "downtrend"},
        )
    )
    assert counter.applicable is True and agreeing.applicable is True
    assert counter.score <= agreeing.score
    assert counter.confidence <= agreeing.confidence
    assert any("fades the primary trend" in r for r in counter.reasoning)


def test_deterministic(strategy):
    settings = Settings()
    technical = make_technical(indicators=make_indicators(rsi=settings.rsi_oversold - 2.0, bb_percent_b=0.02))
    first = strategy.evaluate(technical)
    second = strategy.evaluate(technical)
    assert first.model_dump(exclude={"generated_at"}) == second.model_dump(exclude={"generated_at"})


@pytest.mark.asyncio
async def test_smoke_against_real_engine_output(strategy, bullish_technical, bearish_technical, choppy_technical):
    for technical in (bullish_technical, bearish_technical, choppy_technical):
        signal = strategy.evaluate(technical)
        assert signal.ticker == technical.ticker
        assert 0 <= signal.score <= 100
        assert 0 <= signal.confidence <= 100
