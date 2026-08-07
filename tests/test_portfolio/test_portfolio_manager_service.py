from __future__ import annotations

import pytest

from portfolio.portfolio_manager import PortfolioManager
from portfolio.portfolio_models import (
    CashBalance,
    ClosedPosition,
    Holding,
    Portfolio,
    PortfolioAction,
)
from portfolio.portfolio_service import PortfolioService
from portfolio.portfolio_validation import (
    HoldingNotFoundError,
    InsufficientFundsError,
    PortfolioValidationError,
)
from portfolio.risk_limits import RiskLimits


def _h(symbol="AAPL", qty=10.0, cost=100.0, price=100.0, sector="Technology"):
    return Holding(symbol=symbol, quantity=qty, average_cost=cost,
                   current_price=price, sector=sector)


class TestPortfolioManagerCreation:
    def test_empty_portfolio(self):
        pf = PortfolioManager.empty(name="growth", cash=5000.0)
        assert pf.name == "growth" and pf.cash.amount == 5000.0
        assert pf.holdings == () and pf.total_value == 5000.0

    def test_rejects_negative_opening_cash(self):
        with pytest.raises(PortfolioValidationError):
            PortfolioManager.empty(cash=-1.0)


class TestAddHolding:
    def test_settles_cash_and_records_trade(self):
        pf = PortfolioManager().add_holding(PortfolioManager.empty(cash=5000.0), _h())
        assert pf.cash.amount == 4000.0
        assert len(pf.holdings) == 1 and len(pf.trades) == 1
        assert pf.trades[0].side.value == "BUY"

    def test_rejects_purchase_beyond_cash(self):
        with pytest.raises(InsufficientFundsError):
            PortfolioManager().add_holding(PortfolioManager.empty(cash=100.0), _h())

    def test_import_mode_skips_cash_settlement(self):
        pf = PortfolioManager().add_holding(
            PortfolioManager.empty(cash=0.0), _h(), settle_cash=False)
        assert pf.cash.amount == 0.0 and len(pf.holdings) == 1

    def test_second_buy_averages_cost(self):
        m = PortfolioManager()
        pf = m.add_holding(PortfolioManager.empty(cash=10_000.0), _h(qty=10, cost=100))
        pf = m.add_holding(pf, _h(qty=10, cost=200, price=200))
        holding = pf.holding_for("AAPL")
        assert holding.quantity == 20.0
        assert holding.average_cost == pytest.approx(150.0)
        assert len(pf.holdings) == 1  # still one position, never a duplicate

    def test_does_not_mutate_the_input_portfolio(self):
        original = PortfolioManager.empty(cash=5000.0)
        PortfolioManager().add_holding(original, _h())
        assert original.holdings == () and original.cash.amount == 5000.0


class TestRemoveHolding:
    def _seeded(self):
        return PortfolioManager().add_holding(PortfolioManager.empty(cash=5000.0), _h())

    def test_full_exit_credits_cash_and_closes_position(self):
        pf = PortfolioManager().remove_holding(self._seeded(), "AAPL", exit_price=120.0)
        assert pf.holdings == ()
        assert pf.cash.amount == 5200.0
        assert len(pf.closed_positions) == 1
        assert pf.closed_positions[0].realized_pnl == 200.0

    def test_partial_exit_keeps_remainder(self):
        pf = PortfolioManager().remove_holding(self._seeded(), "AAPL", quantity=4.0, exit_price=110.0)
        assert pf.holding_for("AAPL").quantity == 6.0
        assert pf.closed_positions[0].quantity == 4.0

    def test_defaults_to_current_price(self):
        pf = PortfolioManager().remove_holding(self._seeded(), "AAPL")
        assert pf.closed_positions[0].exit_price == 100.0

    def test_unknown_symbol_raises(self):
        with pytest.raises(HoldingNotFoundError):
            PortfolioManager().remove_holding(self._seeded(), "MSFT")

    def test_oversell_is_rejected(self):
        with pytest.raises(PortfolioValidationError):
            PortfolioManager().remove_holding(self._seeded(), "AAPL", quantity=999.0)

    def test_non_positive_quantity_rejected(self):
        with pytest.raises(PortfolioValidationError):
            PortfolioManager().remove_holding(self._seeded(), "AAPL", quantity=0.0)

    def test_symbol_is_case_insensitive(self):
        pf = PortfolioManager().remove_holding(self._seeded(), "aapl")
        assert pf.holdings == ()


class TestPriceUpdatesAndCash:
    def test_update_prices_marks_to_market(self):
        pf = PortfolioManager().add_holding(PortfolioManager.empty(cash=5000.0), _h())
        pf = PortfolioManager.update_prices(pf, {"AAPL": 150.0})
        assert pf.holding_for("AAPL").market_value == 1500.0

    def test_unknown_symbols_ignored(self):
        pf = PortfolioManager().add_holding(PortfolioManager.empty(cash=5000.0), _h())
        assert PortfolioManager.update_prices(pf, {"MSFT": 50.0}).holding_for("AAPL").current_price == 100.0

    def test_non_positive_price_rejected(self):
        pf = PortfolioManager().add_holding(PortfolioManager.empty(cash=5000.0), _h())
        with pytest.raises(PortfolioValidationError):
            PortfolioManager.update_prices(pf, {"AAPL": 0.0})

    def test_deposit_and_withdraw(self):
        m = PortfolioManager()
        pf = m.deposit(PortfolioManager.empty(cash=100.0), 400.0)
        assert pf.cash.amount == 500.0
        assert m.withdraw(pf, 200.0).cash.amount == 300.0

    def test_withdraw_beyond_balance_rejected(self):
        with pytest.raises(InsufficientFundsError):
            PortfolioManager().withdraw(PortfolioManager.empty(cash=100.0), 500.0)


class TestPortfolioService:
    def _portfolio(self):
        return Portfolio(
            cash=CashBalance(amount=4000.0),
            holdings=(_h("AAPL", qty=10, cost=100, price=120),
                      _h("XOM", qty=10, cost=100, price=90, sector="Energy")),
            closed_positions=(ClosedPosition(symbol="OLD", quantity=5,
                                             entry_price=10, exit_price=20),),
        )

    def test_statistics(self):
        stats = PortfolioService().statistics(self._portfolio())
        assert stats.position_count == 2
        assert stats.total_value == 6100.0
        assert stats.largest_position.symbol == "AAPL"
        assert stats.sector_count == 2
        assert stats.closed_position_count == 1

    def test_performance_combines_realized_and_unrealized(self):
        perf = PortfolioService().performance(self._portfolio())
        assert perf.unrealized_pnl == 100.0
        assert perf.realized_pnl == 50.0
        assert perf.total_pnl == 150.0

    def test_risk_is_produced(self):
        risk = PortfolioService().risk(self._portfolio())
        assert 0 <= risk.risk_score <= 100
        assert risk.position_count == 2

    def test_summary_contains_every_view(self):
        summary = PortfolioService().summary(self._portfolio())
        assert set(summary) == {"name", "statistics", "performance", "risk",
                                "allocations", "sector_exposure"}

    def test_build_context_for_new_symbol(self):
        ctx = PortfolioService().build_context(self._portfolio(), symbol="nvda",
                                               sector="Technology")
        assert ctx.candidate_symbol == "NVDA"
        assert ctx.existing_holding is None
        assert ctx.suggested_action is PortfolioAction.ADD_NEW
        assert ctx.suggested_capital > 0

    def test_build_context_for_held_symbol(self):
        ctx = PortfolioService().build_context(self._portfolio(), symbol="AAPL")
        assert ctx.existing_holding is not None
        assert ctx.candidate_sector == "Technology"
        assert ctx.suggested_action in (PortfolioAction.INCREASE, PortfolioAction.HOLD,
                                        PortfolioAction.REDUCE)

    def test_no_headroom_suggests_staying_in_cash(self):
        pf = Portfolio(cash=CashBalance(amount=0.0), holdings=(_h("AAPL", qty=10),))
        ctx = PortfolioService().build_context(pf, symbol="NVDA", sector="Energy")
        assert ctx.suggested_capital == 0.0
        assert ctx.suggested_action is PortfolioAction.STAY_IN_CASH
        assert any("headroom" in n for n in ctx.constraint_notes)

    def test_context_carries_the_active_limits(self):
        limits = RiskLimits(max_position_pct=12.0, max_sector_pct=40.0, min_cash_pct=10.0)
        ctx = PortfolioService(limits=limits).build_context(self._portfolio(), symbol="NVDA")
        assert ctx.max_position_pct == 12.0
        assert ctx.max_sector_pct == 40.0
        assert ctx.min_cash_pct == 10.0

    def test_build_context_is_deterministic(self):
        service, pf = PortfolioService(), self._portfolio()
        assert service.build_context(pf, symbol="NVDA") == service.build_context(pf, symbol="NVDA")

    def test_invalid_symbol_rejected(self):
        with pytest.raises(PortfolioValidationError):
            PortfolioService().build_context(self._portfolio(), symbol="")

    def test_size_position_delegates_to_the_sizer(self):
        result = PortfolioService().size_position(
            self._portfolio(), symbol="NVDA", price=100.0, allocation_pct=5.0)
        assert result.symbol == "NVDA" and result.shares > 0
