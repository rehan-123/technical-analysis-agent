from __future__ import annotations

import pytest

from config.settings import Settings
from models.strategy import StrategyDirection, StrategyName
from strategy.engine import StrategyEngine, UnknownStrategyError
from tests.test_strategy.factories import make_indicators, make_technical


@pytest.fixture
def engine() -> StrategyEngine:
    return StrategyEngine(settings=Settings())


def test_evaluate_all_returns_one_signal_per_registered_strategy(engine):
    technical = make_technical()
    signals = engine.evaluate_all(technical)
    assert {s.strategy for s in signals} == set(StrategyName)
    assert len(signals) == 5


def test_evaluate_all_is_deterministically_ordered(engine):
    technical = make_technical()
    first = [s.strategy for s in engine.evaluate_all(technical)]
    second = [s.strategy for s in engine.evaluate_all(technical)]
    assert first == second == sorted(first, key=lambda n: n.value)


def test_evaluate_one_matches_evaluate_all(engine):
    technical = make_technical(
        indicators=make_indicators(close=110.0, ema_20=105.0, ema_50=100.0, ema_200=90.0),
        market_structure={"structure": "uptrend"},
        confluence={"net_bias": "bullish", "bullish_score": 70.0, "bearish_score": 5.0},
    )
    one = engine.evaluate_one(technical, StrategyName.TREND_FOLLOWING)
    all_signals = {s.strategy: s for s in engine.evaluate_all(technical)}
    assert one.model_dump(exclude={"generated_at"}) == all_signals[StrategyName.TREND_FOLLOWING].model_dump(
        exclude={"generated_at"}
    )


def test_evaluate_one_unknown_strategy_raises(engine):
    with pytest.raises(UnknownStrategyError):
        engine.evaluate_one(make_technical(), "not_real")  # type: ignore[arg-type]


def test_best_picks_the_highest_score_among_actionable_signals(engine):
    technical = make_technical(
        confidence=80,
        indicators=make_indicators(
            close=110.0, ema_20=105.0, ema_50=100.0, ema_200=90.0, rsi=62.0,
            macd_line=1.2, macd_signal=0.8, macd_histogram=0.4, relative_volume=1.8,
        ),
        market_structure={"structure": "uptrend", "break_of_structure": True},
        confluence={"net_bias": "bullish", "bullish_score": 80.0, "bearish_score": 0.0},
    )
    best = engine.best(technical)
    assert best is not None
    assert best.is_actionable is True
    all_signals = engine.evaluate_all(technical)
    actionable = [s for s in all_signals if s.is_actionable]
    assert best.score == max(s.score for s in actionable)


def test_best_returns_none_when_nothing_is_actionable(engine):
    """A perfectly neutral setup: no strategy should claim a directional
    edge, so ``best`` must return None rather than force a pick."""
    technical = make_technical(
        strength=50,
        confidence=50,
        indicators=make_indicators(rsi=50.0, macd_line=0.0, macd_signal=0.0, macd_histogram=0.0, bb_percent_b=0.5),
        market_structure={},
        confluence={},
    )
    assert engine.best(technical) is None
