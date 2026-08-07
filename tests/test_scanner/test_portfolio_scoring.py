from __future__ import annotations

from portfolio.portfolio_models import PortfolioAction
from scanner.portfolio_scoring import score_portfolio_fit
from tests.test_scanner.factories import make_portfolio_context


def test_none_context_returns_neutral():
    score, notes = score_portfolio_fit(None)
    assert score == 50
    assert notes


def test_add_new_with_capital_scores_high():
    score, _ = score_portfolio_fit(
        make_portfolio_context(suggested_action=PortfolioAction.ADD_NEW, suggested_capital=1000.0)
    )
    assert score >= 70


def test_add_new_without_capital_scores_low():
    score, _ = score_portfolio_fit(
        make_portfolio_context(suggested_action=PortfolioAction.ADD_NEW, suggested_capital=0.0)
    )
    assert score <= 25


def test_exit_scores_lowest():
    score, _ = score_portfolio_fit(make_portfolio_context(suggested_action=PortfolioAction.EXIT, suggested_capital=0.0))
    assert score <= 10


def test_breached_limits_reduce_score():
    clean, _ = score_portfolio_fit(make_portfolio_context(breached_limits=()))
    breached, _ = score_portfolio_fit(make_portfolio_context(breached_limits=("max_position_pct",)))
    assert breached < clean


def test_sector_at_limit_reduces_score():
    under, _ = score_portfolio_fit(make_portfolio_context(candidate_sector_pct=10.0, max_sector_pct=35.0))
    at_limit, _ = score_portfolio_fit(make_portfolio_context(candidate_sector_pct=35.0, max_sector_pct=35.0))
    assert at_limit < under


def test_score_always_bounded_0_100():
    score, _ = score_portfolio_fit(
        make_portfolio_context(
            suggested_action=PortfolioAction.EXIT,
            breached_limits=("max_position_pct", "max_sector_pct", "min_cash_pct"),
            candidate_sector_pct=90.0,
            max_sector_pct=35.0,
        )
    )
    assert 0 <= score <= 100
