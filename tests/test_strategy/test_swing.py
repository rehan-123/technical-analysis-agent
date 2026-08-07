from __future__ import annotations

import pytest

from config.settings import Settings
from models.strategy import StrategyDirection, StrategyName
from strategy.swing import SwingStrategy
from tests.test_strategy.factories import make_indicators, make_technical


@pytest.fixture
def strategy() -> SwingStrategy:
    return SwingStrategy(Settings())


def test_long_on_pullback_within_uptrend(strategy):
    technical = make_technical(
        confidence=60,
        indicators=make_indicators(close=100.0, rsi=48.0),
        support=[99.5],
        pullback=True,
        market_structure={"structure": "uptrend"},
    )
    signal = strategy.evaluate(technical)

    assert signal.strategy is StrategyName.SWING
    assert signal.applicable is True
    assert signal.direction is StrategyDirection.LONG
    assert any("support" in s.lower() for s in signal.signals)


def test_short_on_bounce_within_downtrend(strategy):
    technical = make_technical(
        confidence=60,
        indicators=make_indicators(close=100.0, rsi=52.0),
        resistance=[100.3],
        pullback=True,
        market_structure={"structure": "downtrend"},
    )
    signal = strategy.evaluate(technical)

    assert signal.applicable is True
    assert signal.direction is StrategyDirection.SHORT


def test_not_applicable_without_pullback_flag(strategy):
    technical = make_technical(pullback=False, market_structure={"structure": "uptrend"})
    signal = strategy.evaluate(technical)
    assert signal.applicable is False
    assert signal.direction is StrategyDirection.NONE


def test_not_applicable_when_ranging(strategy):
    technical = make_technical(pullback=True, market_structure={"structure": "ranging"})
    signal = strategy.evaluate(technical)
    assert signal.applicable is False


def test_near_level_increases_score_and_confidence(strategy):
    near = strategy.evaluate(
        make_technical(
            indicators=make_indicators(close=100.0, rsi=48.0),
            support=[99.8],
            pullback=True,
            market_structure={"structure": "uptrend"},
        )
    )
    far = strategy.evaluate(
        make_technical(
            indicators=make_indicators(close=100.0, rsi=48.0),
            support=[50.0],
            pullback=True,
            market_structure={"structure": "uptrend"},
        )
    )
    assert near.score >= far.score
    assert near.confidence >= far.confidence


def test_deterministic(strategy):
    technical = make_technical(
        indicators=make_indicators(close=100.0, rsi=48.0),
        support=[99.5],
        pullback=True,
        market_structure={"structure": "uptrend"},
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
