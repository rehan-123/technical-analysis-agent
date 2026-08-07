from __future__ import annotations

from portfolio.portfolio_models import Portfolio, PortfolioPerformance


class PerformanceTracker:
    """Derives realized and unrealized results from portfolio state.

    Stateless and deterministic: it reads a ``Portfolio`` and returns a
    ``PortfolioPerformance``, holding no counters of its own. Unrealized P/L
    comes from open holdings, realized P/L from closed positions; the two are
    reported separately as well as combined, because conflating them hides
    whether a return has actually been banked.
    """

    def track(self, portfolio: Portfolio) -> PortfolioPerformance:
        """Compute performance for ``portfolio``."""
        unrealized = round(sum(h.unrealized_pnl for h in portfolio.holdings), 2)
        realized_values = [p.realized_pnl for p in portfolio.closed_positions]
        realized = round(sum(realized_values), 2)

        open_basis = sum(h.cost_basis for h in portfolio.holdings)
        closed_basis = sum(p.entry_price * p.quantity for p in portfolio.closed_positions)
        cost_basis = round(open_basis + closed_basis, 2)

        total = round(unrealized + realized, 2)
        wins = sum(1 for v in realized_values if v > 0)
        losses = sum(1 for v in realized_values if v < 0)
        decided = wins + losses

        return PortfolioPerformance(
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_pnl=total,
            cost_basis=cost_basis,
            return_pct=round((total / cost_basis) * 100.0, 2) if cost_basis else 0.0,
            win_count=wins,
            loss_count=losses,
            # Break-even closes are excluded from the denominator: they are
            # neither wins nor losses, and counting them would understate the rate.
            win_rate_pct=round((wins / decided) * 100.0, 2) if decided else 0.0,
            best_trade_pnl=max(realized_values) if realized_values else None,
            worst_trade_pnl=min(realized_values) if realized_values else None,
        )
