from __future__ import annotations

from portfolio.allocation_engine import AllocationEngine
from portfolio.performance_tracker import PerformanceTracker
from portfolio.portfolio_models import (
    Portfolio,
    PortfolioAction,
    PortfolioPerformance,
    PortfolioRecommendationContext,
    PortfolioRisk,
    PortfolioStatistics,
)
from portfolio.portfolio_validation import normalize_symbol
from portfolio.position_sizer import PositionSizer
from portfolio.risk_limits import RiskEngine, RiskLimits


class PortfolioService:
    """Read-side facade over the portfolio engines.

    Composes the allocation, risk, sizing, and performance components into the
    views the API and AI layers consume, and owns none of their logic. State
    transitions belong to ``PortfolioManager``; this class only derives.

    Its most important output is :meth:`build_context`, the projection handed to
    the AI. That projection is what turns "is NVDA a good buy?" into "should
    NVDA be added to *this* portfolio?".
    """

    def __init__(
        self,
        *,
        limits: RiskLimits | None = None,
        allocation_engine: AllocationEngine | None = None,
        risk_engine: RiskEngine | None = None,
        position_sizer: PositionSizer | None = None,
        performance_tracker: PerformanceTracker | None = None,
    ) -> None:
        self._limits = limits or RiskLimits()
        self._allocations = allocation_engine or AllocationEngine(self._limits)
        self._risk = risk_engine or RiskEngine(self._limits)
        self._sizer = position_sizer or PositionSizer(self._limits)
        self._performance = performance_tracker or PerformanceTracker()

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def statistics(self, portfolio: Portfolio) -> PortfolioStatistics:
        """Structural snapshot of the portfolio."""
        allocations = self._allocations.allocations(portfolio)
        return PortfolioStatistics(
            total_value=portfolio.total_value,
            cash_available=portfolio.cash.available,
            holdings_value=portfolio.holdings_value,
            position_count=len(portfolio.holdings),
            closed_position_count=len(portfolio.closed_positions),
            trade_count=len(portfolio.trades),
            invested_pct=portfolio.invested_pct,
            cash_pct=portfolio.cash_pct,
            largest_position=allocations[0] if allocations else None,
            sector_count=len(self._allocations.sector_exposure(portfolio)),
        )

    def performance(self, portfolio: Portfolio) -> PortfolioPerformance:
        return self._performance.track(portfolio)

    def risk(self, portfolio: Portfolio) -> PortfolioRisk:
        return self._risk.assess(
            portfolio,
            allocations=self._allocations.allocations(portfolio),
            sector_exposure=self._allocations.sector_exposure(portfolio),
        )

    def summary(self, portfolio: Portfolio) -> dict[str, object]:
        """Combined view for the API. Plain models, serialized by FastAPI."""
        return {
            "name": portfolio.name,
            "statistics": self.statistics(portfolio),
            "performance": self.performance(portfolio),
            "risk": self.risk(portfolio),
            "allocations": self._allocations.allocations(portfolio),
            "sector_exposure": self._allocations.sector_exposure(portfolio),
        }

    def build_context(
        self,
        portfolio: Portfolio,
        *,
        symbol: str,
        sector: str | None = None,
        price: float | None = None,
    ) -> PortfolioRecommendationContext:
        """Project the portfolio into the context the AI reasons over.

        A projection, not the raw portfolio: it carries the decision-relevant
        facts for one candidate — existing exposure, headroom, breached limits,
        and a suggested action — and omits incidental history. This is the model
        passed through ``additional_inputs["portfolio"]``.
        """
        target = normalize_symbol(symbol)
        existing = portfolio.holding_for(target)
        candidate_sector = sector or (existing.sector if existing else None)
        capacity = self._allocations.capacity_for(portfolio, symbol=target, sector=candidate_sector)
        risk = self.risk(portfolio)

        notes = list(self._allocations.diversification_notes(portfolio))
        if capacity <= 0:
            notes.append(
                f"No headroom to add {target} without breaching a position, sector, or cash limit."
            )

        return PortfolioRecommendationContext(
            candidate_symbol=target,
            statistics=self.statistics(portfolio),
            performance=self.performance(portfolio),
            risk=risk,
            allocations=self._allocations.allocations(portfolio),
            sector_exposure=self._allocations.sector_exposure(portfolio),
            existing_holding=existing,
            candidate_sector=candidate_sector,
            candidate_sector_pct=self._allocations.sector_weight(portfolio, candidate_sector),
            max_position_pct=self._limits.max_position_pct,
            max_sector_pct=self._limits.max_sector_pct,
            min_cash_pct=self._limits.min_cash_pct,
            suggested_action=self._suggest_action(portfolio, existing, capacity, risk),
            suggested_capital=capacity,
            constraint_notes=tuple(notes),
        )

    @staticmethod
    def _suggest_action(portfolio, existing, capacity: float, risk: PortfolioRisk) -> PortfolioAction:
        """A deterministic starting point derived purely from portfolio state.

        Intentionally conservative and signal-free — it knows nothing about the
        technical or news view. It frames the portfolio question; the AI weighs
        it against the other sections and may disagree.
        """
        if existing is not None and "max_position_pct" in risk.breached_limits:
            return PortfolioAction.REDUCE
        if capacity <= 0:
            return PortfolioAction.HOLD if existing is not None else PortfolioAction.STAY_IN_CASH
        return PortfolioAction.INCREASE if existing is not None else PortfolioAction.ADD_NEW

    def size_position(self, portfolio: Portfolio, **kwargs):
        """Delegate to the injected sizer so callers need one entry point."""
        return self._sizer.size(portfolio, **kwargs)
