from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from portfolio.portfolio_models import Portfolio
from portfolio.portfolio_validation import PortfolioValidationError
from portfolio.risk_limits import RiskLimits


class SizingMethod(str, Enum):
    FIXED_CAPITAL = "FIXED_CAPITAL"          # deploy a stated cash amount
    FIXED_RISK = "FIXED_RISK"                # risk a fixed % of equity to the stop
    PERCENT_ALLOCATION = "PERCENT_ALLOCATION"  # target a % of portfolio value


class PositionSizeResult(BaseModel):
    """The outcome of a sizing request, with the reasoning preserved.

    ``capped_by`` names the binding constraint, so a small size is explainable
    rather than mysterious — the same property the risk score aims for.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    method: SizingMethod
    shares: float = Field(..., ge=0.0)
    capital: float = Field(..., ge=0.0)
    portfolio_pct: float = Field(..., ge=0.0, le=100.0)
    risk_amount: float = Field(default=0.0, ge=0.0)
    risk_reward_ratio: float | None = None
    capped_by: str | None = None
    rejected_reason: str | None = None

    @property
    def is_actionable(self) -> bool:
        return self.shares > 0 and self.rejected_reason is None


class PositionSizer:
    """Computes position sizes under portfolio and risk constraints.

    Three interchangeable methods share one constraint pipeline: whatever the
    method proposes is then capped by available cash, the per-position limit,
    and the cash reserve. Keeping the caps in one place means a new method
    cannot accidentally bypass risk policy.

    Broker-neutral: no lot sizes, margin, or commission schedules are assumed.
    Fractional shares are returned; callers that need whole shares round down.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self._limits = limits or RiskLimits()

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def size(
        self,
        portfolio: Portfolio,
        *,
        symbol: str,
        price: float,
        method: SizingMethod = SizingMethod.PERCENT_ALLOCATION,
        capital: float | None = None,
        allocation_pct: float | None = None,
        stop_price: float | None = None,
        target_price: float | None = None,
    ) -> PositionSizeResult:
        """Size a position in ``symbol``.

        Raises:
            PortfolioValidationError: for a non-positive price, or when the
                arguments a method requires are missing.
        """
        if price <= 0:
            raise PortfolioValidationError("price must be positive")

        proposed = self._proposed_capital(
            portfolio, method=method, price=price,
            capital=capital, allocation_pct=allocation_pct, stop_price=stop_price,
        )
        final, capped_by = self._apply_constraints(portfolio, proposed)
        shares = round(final / price, 6) if final > 0 else 0.0
        total = portfolio.total_value

        risk_amount = 0.0
        if stop_price is not None and stop_price < price:
            risk_amount = round(shares * (price - stop_price), 2)

        return PositionSizeResult(
            symbol=symbol.upper(),
            method=method,
            shares=shares,
            capital=round(final, 2),
            portfolio_pct=round((final / total) * 100.0, 2) if total else 0.0,
            risk_amount=risk_amount,
            risk_reward_ratio=self.risk_reward(price, stop_price, target_price),
            capped_by=capped_by,
            rejected_reason=None if shares > 0 else "no capital available after constraints",
        )

    def _proposed_capital(
        self, portfolio: Portfolio, *, method: SizingMethod, price: float,
        capital: float | None, allocation_pct: float | None, stop_price: float | None,
    ) -> float:
        """The capital a method asks for, before constraints."""
        if method is SizingMethod.FIXED_CAPITAL:
            if capital is None or capital < 0:
                raise PortfolioValidationError("FIXED_CAPITAL requires a non-negative capital amount")
            return capital

        if method is SizingMethod.PERCENT_ALLOCATION:
            if allocation_pct is None:
                raise PortfolioValidationError("PERCENT_ALLOCATION requires allocation_pct")
            if not 0.0 <= allocation_pct <= 100.0:
                raise PortfolioValidationError("allocation_pct must be between 0 and 100")
            return portfolio.total_value * (allocation_pct / 100.0)

        # FIXED_RISK: risk a fixed slice of equity across the distance to the
        # stop, so a wider stop buys fewer shares for the same money at risk.
        if stop_price is None:
            raise PortfolioValidationError("FIXED_RISK requires a stop_price")
        if stop_price >= price:
            raise PortfolioValidationError("stop_price must be below the entry price")
        risk_budget = portfolio.total_value * (self._limits.max_risk_per_trade_pct / 100.0)
        return (risk_budget / (price - stop_price)) * price

    def _apply_constraints(self, portfolio: Portfolio, proposed: float) -> tuple[float, str | None]:
        """Clamp ``proposed`` to the binding constraint and name it."""
        total = portfolio.total_value
        candidates: list[tuple[float, str]] = [(proposed, "")]

        max_position = total * (self._limits.max_position_pct / 100.0)
        candidates.append((max_position, "max_position_pct"))

        reserve = total * (self._limits.min_cash_pct / 100.0)
        deployable = max(0.0, portfolio.cash.available - reserve)
        candidates.append((deployable, "min_cash_pct"))

        exposure_cap = total * (self._limits.exposure_ceiling_pct / 100.0)
        headroom = max(0.0, exposure_cap - portfolio.holdings_value)
        candidates.append((headroom, "max_portfolio_exposure_pct"))

        final, label = min(candidates, key=lambda pair: pair[0])
        return max(0.0, final), (label or None)

    @staticmethod
    def risk_reward(price: float, stop_price: float | None, target_price: float | None) -> float | None:
        """Reward-to-risk ratio, or ``None`` when either leg is unusable."""
        if stop_price is None or target_price is None:
            return None
        risk = price - stop_price
        reward = target_price - price
        if risk <= 0 or reward <= 0:
            return None
        return round(reward / risk, 2)

    def meets_risk_reward(self, ratio: float | None, *, minimum: float = 2.0) -> bool:
        """Whether a trade clears a minimum reward-to-risk bar."""
        return ratio is not None and ratio >= minimum
