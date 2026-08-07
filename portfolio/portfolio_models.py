from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from portfolio.portfolio_validation import normalize_symbol, require_unique_symbols

UNCLASSIFIED_SECTOR = "Unclassified"


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    CRYPTO = "CRYPTO"
    FUND = "FUND"
    OTHER = "OTHER"


class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PortfolioAction(str, Enum):
    """What the portfolio layer believes should happen to a candidate — the
    portfolio-aware counterpart to a bare BUY/SELL signal."""

    ADD_NEW = "ADD_NEW"
    INCREASE = "INCREASE"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    STAY_IN_CASH = "STAY_IN_CASH"


class CashBalance(BaseModel):
    """Settled cash and the portion already committed to pending orders."""

    model_config = ConfigDict(frozen=True)

    currency: str = Field(default="USD", min_length=3, max_length=3)
    amount: float = Field(..., ge=0.0, description="Settled cash; never negative")
    reserved: float = Field(default=0.0, ge=0.0, description="Committed to open orders")

    @model_validator(mode="after")
    def _reserved_within_balance(self) -> "CashBalance":
        if self.reserved > self.amount:
            raise ValueError("reserved cash cannot exceed the cash balance")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def available(self) -> float:
        """Cash free to deploy. Buying power without leverage — no margin is
        modelled, since that is broker-specific."""
        return round(self.amount - self.reserved, 2)


class Holding(BaseModel):
    """One open position."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    quantity: float = Field(..., gt=0.0)
    average_cost: float = Field(..., gt=0.0)
    current_price: float = Field(..., gt=0.0)
    sector: str = Field(default=UNCLASSIFIED_SECTOR, min_length=1)
    asset_class: AssetClass = AssetClass.EQUITY

    @field_validator("symbol")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return normalize_symbol(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def market_value(self) -> float:
        return round(self.quantity * self.current_price, 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_basis(self) -> float:
        return round(self.quantity * self.average_cost, 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unrealized_pnl(self) -> float:
        return round(self.market_value - self.cost_basis, 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unrealized_pnl_pct(self) -> float:
        return round((self.unrealized_pnl / self.cost_basis) * 100.0, 2) if self.cost_basis else 0.0


class Trade(BaseModel):
    """An executed fill. Immutable history — corrections are new records."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: TradeSide
    quantity: float = Field(..., gt=0.0)
    price: float = Field(..., gt=0.0)
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fees: float = Field(default=0.0, ge=0.0)

    @field_validator("symbol")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return normalize_symbol(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gross_value(self) -> float:
        return round(self.quantity * self.price, 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_value(self) -> float:
        """Cash effect: a buy costs gross + fees, a sell returns gross - fees."""
        sign = 1.0 if self.side is TradeSide.BUY else -1.0
        return round(sign * self.gross_value + self.fees, 2)


class ClosedPosition(BaseModel):
    """A fully exited position with its realized result."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    quantity: float = Field(..., gt=0.0)
    entry_price: float = Field(..., gt=0.0)
    exit_price: float = Field(..., gt=0.0)
    opened_at: datetime | None = None
    closed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fees: float = Field(default=0.0, ge=0.0)

    @field_validator("symbol")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return normalize_symbol(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def realized_pnl(self) -> float:
        return round((self.exit_price - self.entry_price) * self.quantity - self.fees, 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def realized_pnl_pct(self) -> float:
        basis = self.entry_price * self.quantity
        return round((self.realized_pnl / basis) * 100.0, 2) if basis else 0.0


class Portfolio(BaseModel):
    """The complete portfolio: cash, open holdings, closed positions, history.

    Frozen. Every mutation returns a new instance (see ``PortfolioManager``),
    which keeps state changes explicit and makes the model safe to share.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(default="default", min_length=1)
    cash: CashBalance
    holdings: tuple[Holding, ...] = ()
    closed_positions: tuple[ClosedPosition, ...] = ()
    trades: tuple[Trade, ...] = ()

    @model_validator(mode="after")
    def _no_duplicate_holdings(self) -> "Portfolio":
        require_unique_symbols([h.symbol for h in self.holdings])
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def holdings_value(self) -> float:
        return round(sum(h.market_value for h in self.holdings), 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_value(self) -> float:
        """Cash plus marked-to-market holdings."""
        return round(self.cash.amount + self.holdings_value, 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def invested_pct(self) -> float:
        return round((self.holdings_value / self.total_value) * 100.0, 2) if self.total_value else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cash_pct(self) -> float:
        return round(100.0 - self.invested_pct, 2) if self.total_value else 100.0

    def holding_for(self, symbol: str) -> Holding | None:
        """Return the holding for ``symbol``, or ``None``. O(n) over a list that
        is small by construction — one entry per symbol."""
        target = normalize_symbol(symbol)
        return next((h for h in self.holdings if h.symbol == target), None)


class Allocation(BaseModel):
    """One symbol's share of the portfolio."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    market_value: float = Field(..., ge=0.0)
    weight_pct: float = Field(..., ge=0.0, le=100.0)


class SectorExposure(BaseModel):
    """One sector's share of the portfolio."""

    model_config = ConfigDict(frozen=True)

    sector: str = Field(..., min_length=1)
    market_value: float = Field(..., ge=0.0)
    weight_pct: float = Field(..., ge=0.0, le=100.0)
    symbols: tuple[str, ...] = ()


class PortfolioStatistics(BaseModel):
    """Structural snapshot — counts, values, and concentration."""

    model_config = ConfigDict(frozen=True)

    total_value: float = Field(..., ge=0.0)
    cash_available: float = Field(..., ge=0.0)
    holdings_value: float = Field(..., ge=0.0)
    position_count: int = Field(..., ge=0)
    closed_position_count: int = Field(default=0, ge=0)
    trade_count: int = Field(default=0, ge=0)
    invested_pct: float = Field(..., ge=0.0, le=100.0)
    cash_pct: float = Field(..., ge=0.0, le=100.0)
    largest_position: Allocation | None = None
    sector_count: int = Field(default=0, ge=0)


class PortfolioPerformance(BaseModel):
    """Realized and unrealized results."""

    model_config = ConfigDict(frozen=True)

    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    cost_basis: float = Field(default=0.0, ge=0.0)
    return_pct: float = 0.0
    win_count: int = Field(default=0, ge=0)
    loss_count: int = Field(default=0, ge=0)
    win_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    best_trade_pnl: float | None = None
    worst_trade_pnl: float | None = None


class PortfolioRisk(BaseModel):
    """Portfolio-level risk assessment.

    ``beta`` is intentionally unpopulated: it needs a benchmark price history
    the platform does not yet ingest. Declaring it as an explicit optional is
    honest about the gap rather than emitting a fabricated number.
    """

    model_config = ConfigDict(frozen=True)

    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    concentration_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    largest_sector_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    cash_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    position_count: int = Field(default=0, ge=0)
    beta: float | None = Field(default=None, description="Requires benchmark history; not yet ingested")
    warnings: tuple[str, ...] = ()
    breached_limits: tuple[str, ...] = ()


class PortfolioRecommendationContext(BaseModel):
    """The portfolio view handed to the AI for one candidate symbol.

    This is the model passed through ``AIAnalysisRequest.additional_inputs``
    under the ``"portfolio"`` key and rendered into a prompt section. It is a
    *projection* built by the service — never the raw portfolio — so the prompt
    receives decision-relevant facts and no incidental history.
    """

    model_config = ConfigDict(frozen=True)

    candidate_symbol: str
    statistics: PortfolioStatistics
    performance: PortfolioPerformance
    risk: PortfolioRisk
    allocations: tuple[Allocation, ...] = ()
    sector_exposure: tuple[SectorExposure, ...] = ()
    existing_holding: Holding | None = None
    candidate_sector: str | None = None
    candidate_sector_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    max_position_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    max_sector_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    min_cash_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    suggested_action: PortfolioAction = PortfolioAction.HOLD
    suggested_capital: float = Field(default=0.0, ge=0.0)
    constraint_notes: tuple[str, ...] = ()

    @field_validator("candidate_symbol")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return normalize_symbol(value)
