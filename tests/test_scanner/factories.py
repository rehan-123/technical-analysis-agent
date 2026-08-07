from __future__ import annotations

from models.opportunity import Opportunity
from models.strategy import StrategyDirection, StrategyName
from portfolio.portfolio_models import (
    Allocation,
    PortfolioAction,
    PortfolioPerformance,
    PortfolioRecommendationContext,
    PortfolioRisk,
    PortfolioStatistics,
    RiskLevel,
)


def make_opportunity(
    *,
    ticker: str = "TEST",
    strategy: StrategyName = StrategyName.TREND_FOLLOWING,
    direction: StrategyDirection = StrategyDirection.LONG,
    opportunity_score: int = 60,
    technical_score: int = 60,
    news_score: int = 50,
    portfolio_score: int = 50,
    combined_score: int = 60,
    confidence: int = 60,
    risk: str = "Medium",
    trend: str = "Bullish",
) -> Opportunity:
    return Opportunity(
        ticker=ticker,
        strategy=strategy,
        direction=direction,
        opportunity_score=opportunity_score,
        technical_score=technical_score,
        news_score=news_score,
        portfolio_score=portfolio_score,
        combined_score=combined_score,
        confidence=confidence,
        risk=risk,  # type: ignore[arg-type]
        trend=trend,  # type: ignore[arg-type]
        entry_zone=(99.0, 100.0),
        stop_loss=95.0,
        targets=[105.0, 110.0],
    )


def make_portfolio_context(
    *,
    suggested_action: PortfolioAction = PortfolioAction.ADD_NEW,
    suggested_capital: float = 1000.0,
    breached_limits: tuple[str, ...] = (),
    candidate_sector_pct: float = 0.0,
    max_sector_pct: float = 35.0,
) -> PortfolioRecommendationContext:
    stats = PortfolioStatistics(
        total_value=10_000.0,
        cash_available=5_000.0,
        holdings_value=5_000.0,
        position_count=2,
        invested_pct=50.0,
        cash_pct=50.0,
        largest_position=Allocation(symbol="MSFT", market_value=5_000.0, weight_pct=50.0),
    )
    performance = PortfolioPerformance()
    risk = PortfolioRisk(risk_score=20, risk_level=RiskLevel.LOW, breached_limits=breached_limits)
    return PortfolioRecommendationContext(
        candidate_symbol="AAPL",
        statistics=stats,
        performance=performance,
        risk=risk,
        suggested_action=suggested_action,
        suggested_capital=suggested_capital,
        candidate_sector_pct=candidate_sector_pct,
        max_sector_pct=max_sector_pct,
    )
