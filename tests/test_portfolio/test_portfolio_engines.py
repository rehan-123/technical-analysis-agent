from __future__ import annotations

import pytest
from pydantic import ValidationError

from portfolio.allocation_engine import AllocationEngine
from portfolio.cash_manager import CashManager
from portfolio.performance_tracker import PerformanceTracker
from portfolio.portfolio_models import (
    CashBalance,
    ClosedPosition,
    Holding,
    Portfolio,
    RiskLevel,
)
from portfolio.portfolio_validation import InsufficientFundsError, PortfolioValidationError
from portfolio.position_sizer import PositionSizer, SizingMethod
from portfolio.risk_limits import RiskEngine, RiskLimits


def _h(symbol, value, sector="Technology"):
    """Holding worth exactly ``value`` at a price of 100."""
    return Holding(symbol=symbol, quantity=value / 100.0, average_cost=100.0,
                   current_price=100.0, sector=sector)


def _pf(cash=1000.0, holdings=(), closed=()):
    return Portfolio(cash=CashBalance(amount=cash), holdings=tuple(holdings),
                     closed_positions=tuple(closed))


class TestCashManager:
    def test_buying_power_is_available_cash(self):
        assert CashManager.buying_power(CashBalance(amount=500.0, reserved=100.0)) == 400.0

    def test_debit_reduces_balance(self):
        assert CashManager.debit(CashBalance(amount=500.0), 200.0).amount == 300.0

    def test_debit_beyond_available_is_rejected(self):
        with pytest.raises(InsufficientFundsError):
            CashManager.debit(CashBalance(amount=100.0), 200.0)

    def test_debit_respects_reservations(self):
        with pytest.raises(InsufficientFundsError):
            CashManager.debit(CashBalance(amount=500.0, reserved=450.0), 100.0)

    def test_credit_increases_balance(self):
        assert CashManager.credit(CashBalance(amount=100.0), 50.0).amount == 150.0

    def test_reserve_and_release(self):
        c = CashManager.reserve(CashBalance(amount=500.0), 200.0)
        assert c.available == 300.0
        assert CashManager.release(c, 200.0).available == 500.0

    def test_release_cannot_go_negative(self):
        assert CashManager.release(CashBalance(amount=100.0), 999.0).reserved == 0.0

    def test_negative_amounts_rejected(self):
        with pytest.raises(InsufficientFundsError):
            CashManager.credit(CashBalance(amount=10.0), -5.0)

    def test_operations_do_not_mutate_input(self):
        original = CashBalance(amount=500.0)
        CashManager.debit(original, 100.0)
        assert original.amount == 500.0


class TestAllocationEngine:
    def test_weights_sum_over_total_value(self):
        pf = _pf(cash=0.0, holdings=[_h("AAPL", 600.0), _h("MSFT", 400.0)])
        allocations = AllocationEngine().allocations(pf)
        assert [a.symbol for a in allocations] == ["AAPL", "MSFT"]
        assert allocations[0].weight_pct == 60.0 and allocations[1].weight_pct == 40.0

    def test_sorted_by_weight_descending(self):
        pf = _pf(cash=0.0, holdings=[_h("A", 100.0), _h("B", 900.0)])
        assert [a.symbol for a in AllocationEngine().allocations(pf)] == ["B", "A"]

    def test_sector_exposure_aggregates(self):
        pf = _pf(cash=0.0, holdings=[_h("AAPL", 300.0, "Technology"),
                                     _h("MSFT", 300.0, "Technology"),
                                     _h("XOM", 400.0, "Energy")])
        sectors = AllocationEngine().sector_exposure(pf)
        assert sectors[0].sector == "Technology" and sectors[0].weight_pct == 60.0
        assert sectors[0].symbols == ("AAPL", "MSFT")

    def test_sector_weight_lookup(self):
        pf = _pf(cash=0.0, holdings=[_h("XOM", 1000.0, "Energy")])
        engine = AllocationEngine()
        assert engine.sector_weight(pf, "Energy") == 100.0
        assert engine.sector_weight(pf, "Healthcare") == 0.0
        assert engine.sector_weight(pf, None) == 0.0

    def test_empty_portfolio_has_no_allocations(self):
        assert AllocationEngine().allocations(_pf()) == ()

    def test_diversification_notes_flag_breaches(self):
        pf = _pf(cash=0.0, holdings=[_h("AAPL", 1000.0)])
        notes = AllocationEngine().diversification_notes(pf)
        assert any("AAPL" in n for n in notes)
        assert any("Technology" in n for n in notes)

    def test_capacity_respects_tightest_constraint(self):
        pf = _pf(cash=10_000.0)
        capacity = AllocationEngine(RiskLimits(max_position_pct=10.0)).capacity_for(
            pf, symbol="NVDA", sector="Technology")
        assert capacity == 1000.0

    def test_capacity_zero_when_cash_exhausted(self):
        pf = _pf(cash=0.0, holdings=[_h("AAPL", 1000.0)])
        assert AllocationEngine().capacity_for(pf, symbol="NVDA", sector="Energy") == 0.0

    def test_capacity_accounts_for_existing_holding(self):
        pf = _pf(cash=10_000.0, holdings=[_h("NVDA", 1500.0)])
        engine = AllocationEngine(RiskLimits(max_position_pct=20.0, max_sector_pct=90.0))
        assert engine.capacity_for(pf, symbol="NVDA", sector="Technology") < 2300.0


class TestRiskEngine:
    def _assess(self, pf, limits=None):
        engine = RiskEngine(limits or RiskLimits())
        allocations = AllocationEngine(engine.limits).allocations(pf)
        sectors = AllocationEngine(engine.limits).sector_exposure(pf)
        return engine.assess(pf, allocations=allocations, sector_exposure=sectors)

    def test_diversified_portfolio_scores_low(self):
        holdings = [_h(f"S{i}", 100.0, f"Sector{i}") for i in range(6)]
        risk = self._assess(_pf(cash=1000.0, holdings=holdings))
        assert risk.risk_level is RiskLevel.LOW
        assert risk.breached_limits == ()

    def test_single_position_breaches_concentration(self):
        risk = self._assess(_pf(cash=0.0, holdings=[_h("AAPL", 1000.0)]))
        assert "max_position_pct" in risk.breached_limits
        assert risk.risk_score >= 35

    def test_low_cash_is_flagged(self):
        holdings = [_h(f"S{i}", 100.0, f"Sector{i}") for i in range(6)]
        risk = self._assess(_pf(cash=0.0, holdings=holdings))
        assert "min_cash_pct" in risk.breached_limits

    def test_warnings_are_human_readable(self):
        risk = self._assess(_pf(cash=0.0, holdings=[_h("AAPL", 1000.0)]))
        assert any("%" in w for w in risk.warnings)

    def test_score_is_capped_at_100(self):
        risk = self._assess(_pf(cash=0.0, holdings=[_h("AAPL", 1000.0)]))
        assert 0 <= risk.risk_score <= 100

    @pytest.mark.parametrize("score,level", [(0, RiskLevel.LOW), (25, RiskLevel.MODERATE),
                                             (50, RiskLevel.HIGH), (85, RiskLevel.CRITICAL)])
    def test_classification_thresholds(self, score, level):
        assert RiskEngine.classify(score) is level

    def test_beta_is_explicitly_unavailable(self):
        assert self._assess(_pf()).beta is None

    def test_is_deterministic(self):
        pf = _pf(cash=0.0, holdings=[_h("AAPL", 1000.0)])
        assert self._assess(pf) == self._assess(pf)


class TestRiskLimits:
    def test_rejects_position_limit_above_sector_limit(self):
        with pytest.raises(ValidationError):
            RiskLimits(max_position_pct=50.0, max_sector_pct=30.0)

    def test_rejects_impossible_cash_and_exposure_combination(self):
        with pytest.raises(ValidationError):
            RiskLimits(min_cash_pct=20.0, max_portfolio_exposure_pct=95.0)

    def test_defaults_are_coherent(self):
        limits = RiskLimits()
        assert limits.max_position_pct <= limits.max_sector_pct


class TestPositionSizer:
    def test_percent_allocation(self):
        result = PositionSizer().size(_pf(cash=10_000.0), symbol="NVDA", price=100.0,
                                      method=SizingMethod.PERCENT_ALLOCATION, allocation_pct=10.0)
        assert result.capital == 1000.0 and result.shares == 10.0
        assert result.is_actionable

    def test_fixed_capital(self):
        result = PositionSizer().size(_pf(cash=10_000.0), symbol="NVDA", price=50.0,
                                      method=SizingMethod.FIXED_CAPITAL, capital=500.0)
        assert result.capital == 500.0 and result.shares == 10.0

    def test_fixed_risk_scales_with_stop_distance(self):
        pf = _pf(cash=100_000.0)
        sizer = PositionSizer(RiskLimits(max_risk_per_trade_pct=1.0, max_position_pct=100.0,
                                         max_sector_pct=100.0, min_cash_pct=0.0,
                                         max_portfolio_exposure_pct=100.0))
        tight = sizer.size(pf, symbol="NVDA", price=100.0,
                           method=SizingMethod.FIXED_RISK, stop_price=95.0)
        wide = sizer.size(pf, symbol="NVDA", price=100.0,
                          method=SizingMethod.FIXED_RISK, stop_price=90.0)
        assert tight.shares > wide.shares
        assert tight.risk_amount == pytest.approx(1000.0, abs=1.0)

    def test_capped_by_max_position(self):
        result = PositionSizer(RiskLimits(max_position_pct=5.0)).size(
            _pf(cash=10_000.0), symbol="NVDA", price=100.0,
            method=SizingMethod.PERCENT_ALLOCATION, allocation_pct=50.0)
        assert result.capped_by == "max_position_pct"
        assert result.capital == 500.0

    def test_capped_by_cash_reserve(self):
        result = PositionSizer(RiskLimits(min_cash_pct=50.0, max_position_pct=90.0,
                                          max_sector_pct=90.0,
                                          max_portfolio_exposure_pct=50.0)).size(
            _pf(cash=1000.0), symbol="NVDA", price=10.0,
            method=SizingMethod.PERCENT_ALLOCATION, allocation_pct=90.0)
        assert result.capital <= 500.0

    def test_no_cash_yields_no_position(self):
        result = PositionSizer().size(_pf(cash=0.0), symbol="NVDA", price=100.0,
                                      method=SizingMethod.PERCENT_ALLOCATION, allocation_pct=10.0)
        assert result.shares == 0.0 and not result.is_actionable
        assert result.rejected_reason

    def test_rejects_non_positive_price(self):
        with pytest.raises(PortfolioValidationError):
            PositionSizer().size(_pf(), symbol="X", price=0.0)

    def test_fixed_risk_requires_stop(self):
        with pytest.raises(PortfolioValidationError):
            PositionSizer().size(_pf(), symbol="X", price=10.0, method=SizingMethod.FIXED_RISK)

    def test_stop_above_entry_rejected(self):
        with pytest.raises(PortfolioValidationError):
            PositionSizer().size(_pf(), symbol="X", price=10.0,
                                 method=SizingMethod.FIXED_RISK, stop_price=12.0)

    def test_percent_allocation_requires_pct(self):
        with pytest.raises(PortfolioValidationError):
            PositionSizer().size(_pf(), symbol="X", price=10.0,
                                 method=SizingMethod.PERCENT_ALLOCATION)

    @pytest.mark.parametrize("price,stop,target,expected",
                             [(100, 90, 130, 3.0), (100, 95, 110, 2.0), (100, 90, 95, None)])
    def test_risk_reward(self, price, stop, target, expected):
        assert PositionSizer.risk_reward(price, stop, target) == expected

    def test_risk_reward_requires_both_legs(self):
        assert PositionSizer.risk_reward(100, None, 120) is None

    def test_meets_risk_reward_threshold(self):
        sizer = PositionSizer()
        assert sizer.meets_risk_reward(3.0) is True
        assert sizer.meets_risk_reward(1.5) is False
        assert sizer.meets_risk_reward(None) is False


class TestPerformanceTracker:
    def test_unrealized_from_open_holdings(self):
        h = Holding(symbol="AAPL", quantity=10, average_cost=100, current_price=120)
        perf = PerformanceTracker().track(_pf(holdings=[h]))
        assert perf.unrealized_pnl == 200.0 and perf.realized_pnl == 0.0

    def test_realized_from_closed_positions(self):
        closed = [ClosedPosition(symbol="A", quantity=10, entry_price=100, exit_price=120),
                  ClosedPosition(symbol="B", quantity=10, entry_price=100, exit_price=90)]
        perf = PerformanceTracker().track(_pf(closed=closed))
        assert perf.realized_pnl == 100.0
        assert perf.win_count == 1 and perf.loss_count == 1
        assert perf.win_rate_pct == 50.0
        assert perf.best_trade_pnl == 200.0 and perf.worst_trade_pnl == -100.0

    def test_break_even_excluded_from_win_rate(self):
        closed = [ClosedPosition(symbol="A", quantity=10, entry_price=100, exit_price=120),
                  ClosedPosition(symbol="B", quantity=10, entry_price=100, exit_price=100)]
        perf = PerformanceTracker().track(_pf(closed=closed))
        assert perf.win_count == 1 and perf.loss_count == 0
        assert perf.win_rate_pct == 100.0

    def test_return_pct_uses_combined_cost_basis(self):
        h = Holding(symbol="AAPL", quantity=10, average_cost=100, current_price=110)
        perf = PerformanceTracker().track(_pf(holdings=[h]))
        assert perf.return_pct == 10.0

    def test_empty_portfolio_is_all_zero(self):
        perf = PerformanceTracker().track(_pf())
        assert perf.total_pnl == 0.0 and perf.return_pct == 0.0
        assert perf.best_trade_pnl is None
