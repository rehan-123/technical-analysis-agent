from __future__ import annotations

from collections import defaultdict

from portfolio.portfolio_models import (
    Allocation,
    Portfolio,
    SectorExposure,
    UNCLASSIFIED_SECTOR,
)
from portfolio.risk_limits import RiskLimits


class AllocationEngine:
    """Computes how a portfolio is distributed, and whether that is allowed.

    All outputs are deterministically ordered — allocations and sectors descend
    by weight with the symbol as tie-breaker — so reports and prompts are stable
    across runs. Aggregation is single-pass (O(n)); no nested scans.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self._limits = limits or RiskLimits()

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def allocations(self, portfolio: Portfolio) -> tuple[Allocation, ...]:
        """Per-symbol weights of total portfolio value (cash included)."""
        total = portfolio.total_value
        rows = [
            Allocation(
                symbol=h.symbol,
                market_value=h.market_value,
                weight_pct=round((h.market_value / total) * 100.0, 2) if total else 0.0,
            )
            for h in portfolio.holdings
        ]
        return tuple(sorted(rows, key=lambda a: (-a.weight_pct, a.symbol)))

    def sector_exposure(self, portfolio: Portfolio) -> tuple[SectorExposure, ...]:
        """Per-sector weights, aggregated in one pass."""
        total = portfolio.total_value
        values: dict[str, float] = defaultdict(float)
        symbols: dict[str, list[str]] = defaultdict(list)
        for holding in portfolio.holdings:
            sector = holding.sector or UNCLASSIFIED_SECTOR
            values[sector] += holding.market_value
            symbols[sector].append(holding.symbol)

        rows = [
            SectorExposure(
                sector=sector,
                market_value=round(value, 2),
                weight_pct=round((value / total) * 100.0, 2) if total else 0.0,
                symbols=tuple(sorted(symbols[sector])),
            )
            for sector, value in values.items()
        ]
        return tuple(sorted(rows, key=lambda s: (-s.weight_pct, s.sector)))

    def sector_weight(self, portfolio: Portfolio, sector: str | None) -> float:
        """Current weight of one sector, 0.0 when absent or unknown."""
        if not sector:
            return 0.0
        return next(
            (s.weight_pct for s in self.sector_exposure(portfolio) if s.sector == sector), 0.0
        )

    def diversification_notes(self, portfolio: Portfolio) -> tuple[str, ...]:
        """Human-readable constraint observations, in a deterministic order.

        These are *observations*, not decisions: the AI weighs them, and the
        risk engine scores them. Kept as plain sentences because they are
        rendered directly into a prompt section.
        """
        notes: list[str] = []
        allocations = self.allocations(portfolio)
        sectors = self.sector_exposure(portfolio)

        for allocation in allocations:
            if allocation.weight_pct > self._limits.max_position_pct:
                notes.append(
                    f"{allocation.symbol} is {allocation.weight_pct:.1f}% of the portfolio, "
                    f"over the {self._limits.max_position_pct:.1f}% per-position limit."
                )
        for sector in sectors:
            if sector.weight_pct > self._limits.max_sector_pct:
                notes.append(
                    f"{sector.sector} exposure is {sector.weight_pct:.1f}%, over the "
                    f"{self._limits.max_sector_pct:.1f}% sector limit."
                )
        if portfolio.cash_pct < self._limits.min_cash_pct:
            notes.append(
                f"Cash reserve is {portfolio.cash_pct:.1f}%, under the "
                f"{self._limits.min_cash_pct:.1f}% minimum."
            )
        return tuple(notes)

    def capacity_for(self, portfolio: Portfolio, *, symbol: str, sector: str | None) -> float:
        """Remaining capital that could go into ``symbol`` without a breach.

        Takes the tightest of the per-position, per-sector, and cash-reserve
        headrooms — the same constraint set the sizer applies, expressed as
        available room rather than a share count.
        """
        total = portfolio.total_value
        if total <= 0:
            return 0.0

        existing = portfolio.holding_for(symbol)
        held_value = existing.market_value if existing else 0.0
        position_room = max(0.0, total * (self._limits.max_position_pct / 100.0) - held_value)

        sector_value = total * (self.sector_weight(portfolio, sector) / 100.0)
        sector_room = max(0.0, total * (self._limits.max_sector_pct / 100.0) - sector_value)

        reserve = total * (self._limits.min_cash_pct / 100.0)
        cash_room = max(0.0, portfolio.cash.available - reserve)

        return round(min(position_room, sector_room, cash_room), 2)
