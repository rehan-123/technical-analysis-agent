from __future__ import annotations

import pytest
from pydantic import ValidationError

from portfolio.portfolio_models import (
    AssetClass,
    CashBalance,
    ClosedPosition,
    Holding,
    Portfolio,
    Trade,
    TradeSide,
)
from portfolio.portfolio_validation import (
    PortfolioValidationError,
    normalize_symbol,
    require_percentage,
    require_unique_symbols,
)


def _holding(symbol="AAPL", qty=10.0, cost=100.0, price=120.0, sector="Technology"):
    return Holding(symbol=symbol, quantity=qty, average_cost=cost,
                   current_price=price, sector=sector)


class TestSymbolValidation:
    @pytest.mark.parametrize("raw,expected", [("aapl", "AAPL"), (" msft ", "MSFT"),
                                              ("BRK.B", "BRK.B"), ("RDS-A", "RDS-A")])
    def test_normalizes(self, raw, expected):
        assert normalize_symbol(raw) == expected

    @pytest.mark.parametrize("bad", ["", "   ", "TOO$LONG", "A" * 16, "!!", ".LEAD"])
    def test_rejects_invalid(self, bad):
        with pytest.raises(PortfolioValidationError):
            normalize_symbol(bad)

    def test_rejects_duplicate_holdings(self):
        with pytest.raises(PortfolioValidationError, match="AAPL"):
            require_unique_symbols(["AAPL", "MSFT", "AAPL"])

    def test_accepts_unique(self):
        require_unique_symbols(["AAPL", "MSFT"])

    def test_percentage_bounds(self):
        assert require_percentage(50.0, label="x") == 50.0
        with pytest.raises(PortfolioValidationError):
            require_percentage(101.0, label="x")
        with pytest.raises(PortfolioValidationError):
            require_percentage(-1.0, label="x")


class TestCashBalance:
    def test_available_excludes_reserved(self):
        assert CashBalance(amount=1000.0, reserved=250.0).available == 750.0

    def test_rejects_negative_cash(self):
        with pytest.raises(ValidationError):
            CashBalance(amount=-1.0)

    def test_rejects_reserved_above_balance(self):
        with pytest.raises(ValidationError):
            CashBalance(amount=100.0, reserved=200.0)

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CashBalance(amount=10.0).amount = 20.0


class TestHolding:
    def test_computed_values(self):
        h = _holding()
        assert h.market_value == 1200.0
        assert h.cost_basis == 1000.0
        assert h.unrealized_pnl == 200.0
        assert h.unrealized_pnl_pct == 20.0

    def test_loss_is_negative(self):
        assert _holding(price=80.0).unrealized_pnl == -200.0

    def test_symbol_normalized(self):
        assert _holding(symbol="aapl").symbol == "AAPL"

    @pytest.mark.parametrize("field", ["quantity", "average_cost", "current_price"])
    def test_rejects_non_positive(self, field):
        with pytest.raises(ValidationError):
            _holding(**{{"quantity": "qty", "average_cost": "cost",
                         "current_price": "price"}[field]: 0.0})

    def test_defaults_to_equity_and_unclassified(self):
        h = Holding(symbol="X", quantity=1, average_cost=1, current_price=1)
        assert h.asset_class is AssetClass.EQUITY
        assert h.sector == "Unclassified"

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            _holding().quantity = 99


class TestTradeAndClosedPosition:
    def test_buy_costs_cash_including_fees(self):
        t = Trade(symbol="AAPL", side=TradeSide.BUY, quantity=10, price=100, fees=5)
        assert t.gross_value == 1000.0 and t.net_value == 1005.0

    def test_sell_returns_cash_net_of_fees(self):
        t = Trade(symbol="AAPL", side=TradeSide.SELL, quantity=10, price=100, fees=5)
        assert t.net_value == -995.0

    def test_rejects_negative_fees(self):
        with pytest.raises(ValidationError):
            Trade(symbol="AAPL", side=TradeSide.BUY, quantity=1, price=1, fees=-1)

    def test_closed_position_realized_pnl(self):
        c = ClosedPosition(symbol="AAPL", quantity=10, entry_price=100, exit_price=120, fees=10)
        assert c.realized_pnl == 190.0
        assert c.realized_pnl_pct == 19.0

    def test_closed_position_loss(self):
        c = ClosedPosition(symbol="AAPL", quantity=10, entry_price=100, exit_price=90)
        assert c.realized_pnl == -100.0


class TestPortfolio:
    def test_totals(self):
        p = Portfolio(cash=CashBalance(amount=1000.0), holdings=(_holding(),))
        assert p.holdings_value == 1200.0
        assert p.total_value == 2200.0
        assert p.invested_pct == pytest.approx(54.55, abs=0.01)
        assert p.cash_pct == pytest.approx(45.45, abs=0.01)

    def test_empty_portfolio_is_all_cash(self):
        p = Portfolio(cash=CashBalance(amount=500.0))
        assert p.total_value == 500.0 and p.cash_pct == 100.0 and p.invested_pct == 0.0

    def test_zero_value_portfolio_is_safe(self):
        p = Portfolio(cash=CashBalance(amount=0.0))
        assert p.total_value == 0.0 and p.cash_pct == 100.0

    def test_rejects_duplicate_symbols(self):
        with pytest.raises(ValidationError):
            Portfolio(cash=CashBalance(amount=0.0), holdings=(_holding(), _holding()))

    def test_holding_lookup_is_case_insensitive(self):
        p = Portfolio(cash=CashBalance(amount=0.0), holdings=(_holding(),))
        assert p.holding_for("aapl") is not None
        assert p.holding_for("MSFT") is None

    def test_is_frozen(self):
        p = Portfolio(cash=CashBalance(amount=0.0))
        with pytest.raises(ValidationError):
            p.name = "other"

    def test_serializes_round_trip(self):
        p = Portfolio(cash=CashBalance(amount=1000.0), holdings=(_holding(),))
        restored = Portfolio.model_validate_json(p.model_dump_json())
        assert restored.total_value == p.total_value
        assert restored.holdings[0].symbol == "AAPL"
