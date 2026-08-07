from __future__ import annotations

import pytest

from config.settings import Settings
from models.strategy import StrategyDirection, StrategyName
from strategy.trend_following import TrendFollowingStrategy
from tests.test_strategy.factories import make_indicators, make_technical


@pytest.fixture
def strategy() -> TrendFollowingStrategy:
    return TrendFollowingStrategy(Settings())


def test_long_when_uptrend_and_ema_stack_agree(strategy):
    technical = make_technical(
        trend="Bullish",
        strength=75,
        confidence=80,
        indicators=make_indicators(close=110.0, ema_20=105.0, ema_50=100.0, ema_200=90.0),
        market_structure={"structure": "uptrend", "last_label": "HH", "break_of_structure": True},
        confluence={"net_bias": "bullish", "bullish_score": 70.0, "bearish_score": 10.0},
    )
    signal = strategy.evaluate(technical)

    assert signal.strategy is StrategyName.TREND_FOLLOWING
    assert signal.applicable is True
    assert signal.direction is StrategyDirection.LONG
    assert signal.is_actionable is True
    assert 0 <= signal.score <= 100
    assert 0 <= signal.confidence <= 100
    assert any("EMA" in s for s in signal.signals)


def test_short_when_downtrend_and_ema_stack_agree(strategy):
    technical = make_technical(
        trend="Bearish",
        strength=20,
        confidence=70,
        indicators=make_indicators(close=90.0, ema_20=95.0, ema_50=100.0, ema_200=110.0),
        market_structure={"structure": "downtrend", "last_label": "LL"},
        confluence={"net_bias": "bearish", "bullish_score": 5.0, "bearish_score": 65.0},
    )
    signal = strategy.evaluate(technical)

    assert signal.applicable is True
    assert signal.direction is StrategyDirection.SHORT


def test_not_applicable_when_structure_disagrees_with_ema_stack(strategy):
    """Real-world case observed against the platform's own synthetic data
    generator: EMA stacking reads bullish while the swing-structure engine's
    most recent read is a downtrend. The strategy must stay conservative
    rather than firing on a single agreeing signal."""
    technical = make_technical(
        trend="Strong Bullish",
        strength=80,
        confidence=88,
        indicators=make_indicators(close=193.0, ema_20=188.0, ema_50=184.0, ema_200=173.0),
        market_structure={"structure": "downtrend", "last_label": "LL", "change_of_character": True},
        confluence={"net_bias": "bullish", "bullish_score": 61.0, "bearish_score": 0.0},
    )
    signal = strategy.evaluate(technical)

    assert signal.applicable is False
    assert signal.direction is StrategyDirection.NONE
    assert signal.is_actionable is False
    assert signal.score < 50


def test_not_applicable_when_no_market_structure_available(strategy):
    technical = make_technical(indicators=make_indicators(), market_structure={}, confluence={})
    signal = strategy.evaluate(technical)
    assert signal.applicable is False
    assert signal.direction is StrategyDirection.NONE


def test_change_of_character_lowers_confidence(strategy):
    base = dict(
        trend="Bullish",
        strength=75,
        confidence=80,
        indicators=make_indicators(close=110.0, ema_20=105.0, ema_50=100.0, ema_200=90.0),
        confluence={"net_bias": "bullish", "bullish_score": 70.0, "bearish_score": 10.0},
    )
    stable = strategy.evaluate(make_technical(**base, market_structure={"structure": "uptrend"}))
    choppy = strategy.evaluate(
        make_technical(**base, market_structure={"structure": "uptrend", "change_of_character": True})
    )
    assert choppy.confidence <= stable.confidence


def test_deterministic(strategy):
    technical = make_technical(
        trend="Bullish",
        strength=75,
        confidence=80,
        indicators=make_indicators(close=110.0, ema_20=105.0, ema_50=100.0, ema_200=90.0),
        market_structure={"structure": "uptrend"},
        confluence={"net_bias": "bullish", "bullish_score": 70.0, "bearish_score": 10.0},
    )
    first = strategy.evaluate(technical)
    second = strategy.evaluate(technical)
    assert first.model_dump(exclude={"generated_at"}) == second.model_dump(exclude={"generated_at"})


@pytest.mark.asyncio
async def test_smoke_against_real_engine_output(strategy, bullish_technical, bearish_technical, choppy_technical):
    """Never raises, and always returns a well-formed signal, regardless of
    how the real Technical Engine's structure/confluence/EMA reads happen to
    line up for a given synthetic series."""
    for technical in (bullish_technical, bearish_technical, choppy_technical):
        signal = strategy.evaluate(technical)
        assert signal.ticker == technical.ticker
        assert 0 <= signal.score <= 100
        assert 0 <= signal.confidence <= 100
