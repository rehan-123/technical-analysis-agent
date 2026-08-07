from __future__ import annotations

import pytest

from config.settings import Settings
from models.strategy import StrategyDirection, StrategyName
from strategy.breakout import BreakoutStrategy
from tests.test_strategy.factories import make_indicators, make_technical


@pytest.fixture
def strategy() -> BreakoutStrategy:
    return BreakoutStrategy(Settings())


def test_long_on_breakout_above_resistance(strategy):
    technical = make_technical(
        confidence=65,
        indicators=make_indicators(close=110.0, relative_volume=2.0),
        resistance=[105.0, 106.0],
        breakout=True,
        volatility={"bollinger_squeeze": True, "breakout_probability": 70},
    )
    signal = strategy.evaluate(technical)

    assert signal.strategy is StrategyName.BREAKOUT
    assert signal.applicable is True
    assert signal.direction is StrategyDirection.LONG
    assert any("volume" in s.lower() for s in signal.signals)


def test_short_on_breakdown_below_support(strategy):
    technical = make_technical(
        confidence=65,
        indicators=make_indicators(close=90.0, relative_volume=1.9),
        support=[95.0, 96.0],
        breakout=True,
    )
    signal = strategy.evaluate(technical)

    assert signal.applicable is True
    assert signal.direction is StrategyDirection.SHORT


def test_direction_falls_back_to_trend_when_no_levels(strategy):
    technical = make_technical(trend="Bullish", breakout=True, support=[], resistance=[])
    signal = strategy.evaluate(technical)
    assert signal.applicable is True
    assert signal.direction is StrategyDirection.LONG


def test_not_applicable_when_no_breakout_flag(strategy):
    technical = make_technical(breakout=False, resistance=[105.0], indicators=make_indicators(close=110.0))
    signal = strategy.evaluate(technical)
    assert signal.applicable is False
    assert signal.direction is StrategyDirection.NONE


def test_not_applicable_when_direction_indeterminate(strategy):
    technical = make_technical(trend="Neutral", breakout=True, support=[], resistance=[])
    signal = strategy.evaluate(technical)
    assert signal.applicable is False


def test_volume_and_squeeze_increase_score(strategy):
    confirmed = strategy.evaluate(
        make_technical(
            indicators=make_indicators(close=110.0, relative_volume=2.0),
            resistance=[105.0],
            breakout=True,
            volatility={"bollinger_squeeze": True},
        )
    )
    weak = strategy.evaluate(
        make_technical(
            indicators=make_indicators(close=110.0, relative_volume=0.8),
            resistance=[105.0],
            breakout=True,
            volatility={},
        )
    )
    assert confirmed.score >= weak.score
    assert confirmed.confidence >= weak.confidence


def test_deterministic(strategy):
    technical = make_technical(
        indicators=make_indicators(close=110.0, relative_volume=2.0), resistance=[105.0], breakout=True
    )
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
