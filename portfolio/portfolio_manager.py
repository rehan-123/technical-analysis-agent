from __future__ import annotations

from datetime import datetime, timezone

from portfolio.cash_manager import CashManager
from portfolio.portfolio_models import (
    CashBalance,
    ClosedPosition,
    Holding,
    Portfolio,
    Trade,
    TradeSide,
)
from portfolio.portfolio_validation import (
    HoldingNotFoundError,
    PortfolioValidationError,
    normalize_symbol,
)


class PortfolioManager:
    """Applies state transitions to an immutable ``Portfolio``.

    Every operation is a pure function of (portfolio, arguments) returning a new
    portfolio — no hidden state, no in-place mutation, so callers control
    persistence and concurrency. Cash effects are delegated to ``CashManager``
    rather than reimplemented, keeping one definition of "can I afford this".
    """

    def __init__(self, cash_manager: CashManager | None = None) -> None:
        self._cash = cash_manager or CashManager()

    @staticmethod
    def empty(*, name: str = "default", cash: float = 0.0, currency: str = "USD") -> Portfolio:
        """A new portfolio holding only cash."""
        if cash < 0:
            raise PortfolioValidationError("initial cash must not be negative")
        return Portfolio(name=name, cash=CashBalance(amount=round(cash, 2), currency=currency))

    def add_holding(self, portfolio: Portfolio, holding: Holding, *, settle_cash: bool = True) -> Portfolio:
        """Add a new position, or average into an existing one.

        Averaging in is the correct behaviour for a repeat buy; rejecting it
        would force callers to reimplement cost-basis maths. ``settle_cash``
        exists for importing an existing book, where cash was spent elsewhere.
        """
        existing = portfolio.holding_for(holding.symbol)
        cash = self._cash.debit(portfolio.cash, holding.cost_basis) if settle_cash else portfolio.cash

        if existing is None:
            holdings = (*portfolio.holdings, holding)
        else:
            quantity = existing.quantity + holding.quantity
            merged_cost = (existing.cost_basis + holding.cost_basis) / quantity
            holdings = tuple(
                existing.model_copy(update={
                    "quantity": round(quantity, 6),
                    "average_cost": round(merged_cost, 6),
                    "current_price": holding.current_price,
                }) if h.symbol == existing.symbol else h
                for h in portfolio.holdings
            )

        trade = Trade(symbol=holding.symbol, side=TradeSide.BUY,
                      quantity=holding.quantity, price=holding.average_cost)
        return portfolio.model_copy(update={
            "holdings": holdings, "cash": cash, "trades": (*portfolio.trades, trade),
        })

    def remove_holding(
        self, portfolio: Portfolio, symbol: str, *,
        quantity: float | None = None, exit_price: float | None = None,
    ) -> Portfolio:
        """Sell all or part of a holding, recording the realized result.

        Raises:
            HoldingNotFoundError: if the symbol is not held.
            PortfolioValidationError: if the quantity is non-positive or exceeds
                the position.
        """
        target = normalize_symbol(symbol)
        existing = portfolio.holding_for(target)
        if existing is None:
            raise HoldingNotFoundError(f"portfolio does not hold {target}")

        sell_qty = existing.quantity if quantity is None else quantity
        if sell_qty <= 0:
            raise PortfolioValidationError("sell quantity must be positive")
        if sell_qty > existing.quantity:
            raise PortfolioValidationError(
                f"cannot sell {sell_qty} of {target}; only {existing.quantity} held"
            )

        price = exit_price if exit_price is not None else existing.current_price
        if price <= 0:
            raise PortfolioValidationError("exit price must be positive")

        proceeds = round(sell_qty * price, 2)
        remaining = round(existing.quantity - sell_qty, 6)
        holdings = tuple(h for h in portfolio.holdings if h.symbol != target)
        if remaining > 0:
            holdings = (*holdings, existing.model_copy(update={"quantity": remaining}))

        closed = ClosedPosition(
            symbol=target, quantity=sell_qty,
            entry_price=existing.average_cost, exit_price=price,
            closed_at=datetime.now(timezone.utc),
        )
        trade = Trade(symbol=target, side=TradeSide.SELL, quantity=sell_qty, price=price)
        return portfolio.model_copy(update={
            "holdings": holdings,
            "cash": self._cash.credit(portfolio.cash, proceeds),
            "closed_positions": (*portfolio.closed_positions, closed),
            "trades": (*portfolio.trades, trade),
        })

    @staticmethod
    def update_prices(portfolio: Portfolio, prices: dict[str, float]) -> Portfolio:
        """Mark holdings to market. Unknown symbols are ignored; non-positive
        prices are rejected rather than silently corrupting valuations."""
        normalized = {normalize_symbol(s): p for s, p in prices.items()}
        for symbol, price in normalized.items():
            if price <= 0:
                raise PortfolioValidationError(f"price for {symbol} must be positive")
        holdings = tuple(
            h.model_copy(update={"current_price": normalized[h.symbol]})
            if h.symbol in normalized else h
            for h in portfolio.holdings
        )
        return portfolio.model_copy(update={"holdings": holdings})

    def deposit(self, portfolio: Portfolio, amount: float) -> Portfolio:
        return portfolio.model_copy(update={"cash": self._cash.credit(portfolio.cash, amount)})

    def withdraw(self, portfolio: Portfolio, amount: float) -> Portfolio:
        return portfolio.model_copy(update={"cash": self._cash.debit(portfolio.cash, amount)})
