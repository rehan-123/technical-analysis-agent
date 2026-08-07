from __future__ import annotations

from config.settings import Settings
from scanner.ranking_engine import RankingEngine
from tests.test_scanner.factories import make_opportunity


def test_combine_equal_weights_averages():
    settings = Settings(
        scanner_weight_technical=1.0,
        scanner_weight_news=1.0,
        scanner_weight_portfolio=1.0,
        scanner_weight_opportunity=1.0,
    )
    engine = RankingEngine(settings)
    combined = engine.combine(technical_score=80, news_score=60, portfolio_score=40, opportunity_score=20)
    assert combined == 50


def test_combine_bounded_at_extremes():
    engine = RankingEngine(Settings())
    assert engine.combine(technical_score=100, news_score=100, portfolio_score=100, opportunity_score=100) == 100
    assert engine.combine(technical_score=0, news_score=0, portfolio_score=0, opportunity_score=0) == 0


def test_rank_orders_descending_by_combined_score():
    engine = RankingEngine(Settings())
    opps = [
        make_opportunity(ticker="A", combined_score=50),
        make_opportunity(ticker="B", combined_score=90),
        make_opportunity(ticker="C", combined_score=70),
    ]
    ranked = engine.rank(opps)
    assert [o.ticker for o in ranked] == ["B", "C", "A"]
    assert [o.ranking for o in ranked] == [1, 2, 3]


def test_rank_ties_broken_by_confidence_then_ticker():
    engine = RankingEngine(Settings())
    opps = [
        make_opportunity(ticker="Z", combined_score=50, confidence=60),
        make_opportunity(ticker="A", combined_score=50, confidence=60),
        make_opportunity(ticker="M", combined_score=50, confidence=80),
    ]
    ranked = engine.rank(opps)
    assert [o.ticker for o in ranked] == ["M", "A", "Z"]


def test_rank_empty_list_returns_empty():
    engine = RankingEngine(Settings())
    assert engine.rank([]) == []


def test_rank_is_deterministic():
    engine = RankingEngine(Settings())
    opps = [make_opportunity(ticker=t, combined_score=s) for t, s in [("A", 50), ("B", 90), ("C", 70)]]
    first = [o.ticker for o in engine.rank(opps)]
    second = [o.ticker for o in engine.rank(opps)]
    assert first == second
