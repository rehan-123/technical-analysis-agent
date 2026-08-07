from __future__ import annotations

import pytest

from config.settings import Settings
from models.strategy import StrategyDirection, StrategyName
from strategy.momentum import MomentumStrategy
from tests.test_strategy.factories import make_indicators, make_technical


@pytest.fixture
def strategy() -> MomentumStrategy:
    return MomentumStrategy(Settings())


def test_long_when_macd_and_rsi_agree_bullish(strategy):
    technical = make_technical(
        confidence=70,
        indicators=make_indicators(rsi=62.0, macd_line=1.2, macd_signal=0.8, macd_histogram=0.4, relative_volume=1.5),
    )
    signal = strategy.evaluate(technical)

    assert signal.strategy is StrategyName.MOMENTUM
    assert signal.applicable is True
    assert signal.direction is StrategyDirection.LONG
    assert any("volume" in s.lower() for s in signal.signals)


def test_short_when_macd_and_rsi_agree_bearish(strategy):
    technical = make_technical(
        confidence=70,
        indicators=make_indicators(rsi=38.0, macd_line=-1.2, macd_signal=-0.8, macd_histogram=-0.4, relative_volume=1.2),
    )
    signal = strategy.evaluate(technical)

    assert signal.applicable is True
    assert signal.direction is StrategyDirection.SHORT


def test_not_applicable_when_macd_and_rsi_disagree(strategy):
    technical = make_technical(
        indicators=make_indicators(rsi=62.0, macd_line=-1.0, macd_signal=-0.5, macd_histogram=-0.5),
    )
    signal = strategy.evaluate(technical)
    assert signal.applicable is False
    assert signal.direction is StrategyDirection.NONE


def test_not_applicable_when_rsi_already_overbought(strategy):
    """Momentum zone deliberately excludes the extremes — that is Mean
    Reversion's territory."""
    settings = Settings()
    technical = make_technical(
        indicators=make_indicators(
            rsi=settings.rsi_overbought + 5.0, macd_line=1.2, macd_signal=0.8, macd_histogram=0.4
        ),
    )
    signal = strategy.evaluate(technical)
    assert signal.applicable is False


def test_low_volume_reduces_score_but_stays_applicable(strategy):
    high_vol = strategy.evaluate(
        make_technical(indicators=make_indicators(rsi=62.0, macd_line=1.2, macd_signal=0.8, macd_histogram=0.4, relative_volume=1.8))
    )
    low_vol = strategy.evaluate(
        make_technical(indicators=make_indicators(rsi=62.0, macd_line=1.2, macd_signal=0.8, macd_histogram=0.4, relative_volume=0.4))
    )
    assert high_vol.applicable is True and low_vol.applicable is True
    assert low_vol.score < high_vol.score


def test_deterministic(strategy):
    technical = make_technical(indicators=make_indicators(rsi=62.0, macd_line=1.2, macd_signal=0.8, macd_histogram=0.4))
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
