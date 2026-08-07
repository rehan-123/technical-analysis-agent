from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from portfolio.portfolio_models import (
    Allocation,
    Portfolio,
    PortfolioRisk,
    RiskLevel,
    SectorExposure,
)

#: Score weights for each breached rule. Explicit and additive so the resulting
#: score is explainable — every point traces to a named condition.
_CONCENTRATION_WEIGHT = 35
_SECTOR_WEIGHT = 25
_CASH_WEIGHT = 20
_DIVERSIFICATION_WEIGHT = 20

#: Below this many positions a portfolio is treated as under-diversified.
_MIN_DIVERSIFIED_POSITIONS = 5


class RiskLimits(BaseModel):
    """Portfolio risk policy. Pure configuration — injected, never global."""

    model_config = ConfigDict(frozen=True)

    max_position_pct: float = Field(default=20.0, gt=0.0, le=100.0)
    max_sector_pct: float = Field(default=35.0, gt=0.0, le=100.0)
    min_cash_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    max_risk_per_trade_pct: float = Field(default=1.0, gt=0.0, le=100.0)
    #: Ceiling on invested exposure. Left unset it is *derived* from the cash
    #: floor rather than defaulted to a fixed number — see ``_derive_exposure``.
    max_portfolio_exposure_pct: float | None = Field(default=None, gt=0.0, le=100.0)

    @model_validator(mode="before")
    @classmethod
    def _derive_exposure(cls, data: object) -> object:
        """Derive the exposure ceiling from the cash floor when it is not given.

        Invested and cash always sum to 100% of the portfolio, so these two
        limits are not independent: the exposure ceiling *is* ``100 -
        min_cash_pct``. A fixed default silently assumed one particular cash
        floor and rejected any caller who raised it — e.g. ``min_cash_pct=10``
        against a 95% default summed to 105% and failed, despite being a
        perfectly coherent policy.

        Deriving it keeps the default behaviour identical (5% cash -> 95%
        exposure) while letting either knob be set alone. An explicit value is
        still honoured, and still checked below.
        """
        if not isinstance(data, dict):
            return data
        if data.get("max_portfolio_exposure_pct") is None:
            min_cash = data.get("min_cash_pct")
            if min_cash is None:
                min_cash = cls.model_fields["min_cash_pct"].default
            try:
                derived = 100.0 - float(min_cash)
            except (TypeError, ValueError):
                return data  # let field validation report the bad input
            if derived > 0:
                data = {**data, "max_portfolio_exposure_pct": derived}
        return data

    @model_validator(mode="after")
    def _coherent(self) -> "RiskLimits":
        """Reject policies that cannot all hold at once."""
        if self.max_position_pct > self.max_sector_pct:
            raise ValueError("max_position_pct cannot exceed max_sector_pct")
        if self.min_cash_pct + self.exposure_ceiling_pct > 100.0:
            raise ValueError(
                "min_cash_pct and max_portfolio_exposure_pct cannot exceed 100 combined; "
                f"got {self.min_cash_pct} + {self.exposure_ceiling_pct}"
            )
        return self

    @property
    def exposure_ceiling_pct(self) -> float:
        """The effective exposure ceiling, always a concrete number.

        Every consumer reads this rather than the raw optional field, so no
        call site has to repeat the ``None`` fallback.
        """
        if self.max_portfolio_exposure_pct is not None:
            return self.max_portfolio_exposure_pct
        return max(0.0, 100.0 - self.min_cash_pct)


class RiskEngine:
    """Scores portfolio-level risk against a set of limits.

    Deterministic and side-effect free: the same portfolio and limits always
    produce the same score, warnings, and breach list. Scoring is additive over
    named rules rather than an opaque formula, so a score can be explained —
    which matters when the output feeds an LLM prompt.

    Broker-specific margin is deliberately out of scope.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self._limits = limits or RiskLimits()

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def assess(
        self,
        portfolio: Portfolio,
        *,
        allocations: tuple[Allocation, ...],
        sector_exposure: tuple[SectorExposure, ...],
    ) -> PortfolioRisk:
        """Evaluate ``portfolio`` and return a scored risk report."""
        warnings: list[str] = []
        breached: list[str] = []
        score = 0

        largest = max((a.weight_pct for a in allocations), default=0.0)
        if largest > self._limits.max_position_pct:
            score += _CONCENTRATION_WEIGHT
            breached.append("max_position_pct")
            warnings.append(
                f"Largest position is {largest:.1f}% of the portfolio, above the "
                f"{self._limits.max_position_pct:.1f}% limit."
            )

        largest_sector = max((s.weight_pct for s in sector_exposure), default=0.0)
        if largest_sector > self._limits.max_sector_pct:
            score += _SECTOR_WEIGHT
            breached.append("max_sector_pct")
            warnings.append(
                f"Largest sector is {largest_sector:.1f}% of the portfolio, above the "
                f"{self._limits.max_sector_pct:.1f}% limit."
            )

        if portfolio.cash_pct < self._limits.min_cash_pct:
            score += _CASH_WEIGHT
            breached.append("min_cash_pct")
            warnings.append(
                f"Cash is {portfolio.cash_pct:.1f}% of the portfolio, below the "
                f"{self._limits.min_cash_pct:.1f}% reserve."
            )

        if portfolio.invested_pct > self._limits.exposure_ceiling_pct:
            breached.append("max_portfolio_exposure_pct")
            warnings.append(
                f"Invested exposure is {portfolio.invested_pct:.1f}%, above the "
                f"{self._limits.exposure_ceiling_pct:.1f}% limit."
            )

        if portfolio.holdings and len(portfolio.holdings) < _MIN_DIVERSIFIED_POSITIONS:
            score += _DIVERSIFICATION_WEIGHT
            warnings.append(
                f"Only {len(portfolio.holdings)} position(s) held; concentration risk is elevated."
            )

        score = min(100, score)
        return PortfolioRisk(
            risk_score=score,
            risk_level=self.classify(score),
            concentration_pct=round(largest, 2),
            largest_sector_pct=round(largest_sector, 2),
            cash_pct=portfolio.cash_pct,
            position_count=len(portfolio.holdings),
            warnings=tuple(warnings),
            breached_limits=tuple(breached),
        )

    @staticmethod
    def classify(score: int) -> RiskLevel:
        """Map a 0-100 score onto a coarse level."""
        if score >= 70:
            return RiskLevel.CRITICAL
        if score >= 45:
            return RiskLevel.HIGH
        if score >= 20:
            return RiskLevel.MODERATE
        return RiskLevel.LOW
