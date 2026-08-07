from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from portfolio.allocation_engine import AllocationEngine
from portfolio.portfolio_manager import PortfolioManager
from portfolio.portfolio_models import (
    AssetClass,
    Holding,
    Portfolio,
    PortfolioPerformance,
    PortfolioRisk,
    PortfolioStatistics,
    UNCLASSIFIED_SECTOR,
)
from portfolio.portfolio_service import PortfolioService
from portfolio.portfolio_validation import (
    HoldingNotFoundError,
    InsufficientFundsError,
    PortfolioError,
    PortfolioValidationError,
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# Composition root for the portfolio layer. The portfolio is held in memory:
# persistence is deliberately out of scope for V2, and this keeps the storage
# decision open rather than baking in a database. Everything below is stateless
# and injectable; only the portfolio value itself is retained here.
_portfolio: Portfolio | None = None
_manager = PortfolioManager()
_service = PortfolioService()


def get_portfolio() -> Portfolio:
    """Return the active portfolio, creating an empty one on first use."""
    global _portfolio
    if _portfolio is None:
        _portfolio = PortfolioManager.empty()
    return _portfolio


def set_portfolio(portfolio: Portfolio) -> Portfolio:
    """Replace the active portfolio and return it."""
    global _portfolio
    _portfolio = portfolio
    return _portfolio


class CreatePortfolioRequest(BaseModel):
    name: str = Field(default="default", min_length=1)
    cash: float = Field(default=0.0, ge=0.0)
    currency: str = Field(default="USD", min_length=3, max_length=3)


class AddHoldingRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    quantity: float = Field(..., gt=0.0)
    average_cost: float = Field(..., gt=0.0)
    current_price: float | None = Field(default=None, gt=0.0)
    sector: str = Field(default=UNCLASSIFIED_SECTOR, min_length=1)
    asset_class: AssetClass = AssetClass.EQUITY
    settle_cash: bool = Field(
        default=True,
        description="Deduct the cost from cash. False when importing an existing book.",
    )


class PortfolioSummaryResponse(BaseModel):
    name: str
    statistics: PortfolioStatistics
    performance: PortfolioPerformance
    risk: PortfolioRisk
    allocations: tuple
    sector_exposure: tuple


@router.get("", response_model=Portfolio)
async def read_portfolio() -> Portfolio:
    """Return the full portfolio: cash, holdings, closed positions, trades."""
    return get_portfolio()


@router.post("", response_model=Portfolio)
async def create_portfolio(request: CreatePortfolioRequest) -> Portfolio:
    """Create or replace the portfolio with an opening cash balance."""
    try:
        portfolio = PortfolioManager.empty(
            name=request.name, cash=request.cash, currency=request.currency
        )
    except PortfolioError as exc:
        raise _http_error(exc) from exc
    logger.info("portfolio created name=%s cash=%.2f", request.name, request.cash)
    return set_portfolio(portfolio)


@router.post("/holding", response_model=Portfolio)
async def add_holding(request: AddHoldingRequest) -> Portfolio:
    """Add a position, or average into an existing one."""
    try:
        holding = Holding(
            symbol=request.symbol,
            quantity=request.quantity,
            average_cost=request.average_cost,
            current_price=request.current_price or request.average_cost,
            sector=request.sector,
            asset_class=request.asset_class,
        )
        updated = _manager.add_holding(get_portfolio(), holding, settle_cash=request.settle_cash)
    except PortfolioError as exc:
        raise _http_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("holding added symbol=%s qty=%s", request.symbol, request.quantity)
    return set_portfolio(updated)


@router.delete("/holding/{symbol}", response_model=Portfolio)
async def remove_holding(
    symbol: str, quantity: float | None = None, exit_price: float | None = None
) -> Portfolio:
    """Sell all or part of a holding, recording the realized result."""
    try:
        updated = _manager.remove_holding(
            get_portfolio(), symbol, quantity=quantity, exit_price=exit_price
        )
    except PortfolioError as exc:
        raise _http_error(exc) from exc
    logger.info("holding removed symbol=%s", symbol)
    return set_portfolio(updated)


@router.get("/summary", response_model=PortfolioSummaryResponse)
async def portfolio_summary() -> PortfolioSummaryResponse:
    """Combined statistics, performance, risk, and allocation view."""
    portfolio = get_portfolio()
    return PortfolioSummaryResponse(
        name=portfolio.name,
        statistics=_service.statistics(portfolio),
        performance=_service.performance(portfolio),
        risk=_service.risk(portfolio),
        allocations=AllocationEngine(_service.limits).allocations(portfolio),
        sector_exposure=AllocationEngine(_service.limits).sector_exposure(portfolio),
    )


@router.get("/performance", response_model=PortfolioPerformance)
async def portfolio_performance() -> PortfolioPerformance:
    """Realized and unrealized results."""
    return _service.performance(get_portfolio())


@router.get("/risk", response_model=PortfolioRisk)
async def portfolio_risk() -> PortfolioRisk:
    """Portfolio-level risk score, warnings, and breached limits."""
    return _service.risk(get_portfolio())


def _http_error(exc: PortfolioError) -> HTTPException:
    """Map a domain error to its status. Ordered most-specific first."""
    if isinstance(exc, HoldingNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, InsufficientFundsError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, PortfolioValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))
