from __future__ import annotations

from portfolio.portfolio_models import PortfolioRecommendationContext
from services.prompt_sections.base import RenderedSection, SectionRenderer


class PortfolioSectionRenderer(SectionRenderer):
    """Projects portfolio context into one prompt section.

    The bridge that lets the AI answer "should this be added to *my* portfolio?"
    instead of "is this a good stock?". It implements the existing
    ``SectionRenderer`` contract and is registered on the existing section
    registry, so ``PromptBuilder`` requires no change at all — this is the
    extension seam working as designed.

    A pure presentation layer, exactly like the technical and news renderers:
    it selects whitelisted fields and lays them out. It makes no allocation,
    sizing, or risk decision — those already happened in the portfolio engines.
    Deterministic: inputs arrive pre-sorted and no clock or randomness is read.
    """

    kind = "portfolio"
    version = "1.0"

    def render(self, model: PortfolioRecommendationContext, *, max_items: int) -> RenderedSection:
        """Render the portfolio section.

        ``max_items`` caps the holdings and sector lists so a large book cannot
        dominate the prompt; ``truncated`` records when it bit.
        """
        stats, risk = model.statistics, model.risk
        lines: list[str] = [
            f"Total value: {stats.total_value:.2f}",
            f"Cash available: {stats.cash_available:.2f} ({stats.cash_pct:.1f}% of portfolio)",
            f"Invested: {stats.invested_pct:.1f}% across {stats.position_count} position(s)",
            f"Unrealized P/L: {model.performance.unrealized_pnl:.2f} "
            f"({model.performance.return_pct:.2f}% on cost)",
            f"Risk: {risk.risk_level.value} (score {risk.risk_score}/100)",
        ]

        allocations = model.allocations[:max_items]
        truncated = len(model.allocations) > len(allocations)
        if allocations:
            lines.append("Current holdings:")
            lines.extend(
                f"  - {a.symbol}: {a.weight_pct:.1f}% ({a.market_value:.2f})" for a in allocations
            )

        sectors = model.sector_exposure[:max_items]
        truncated = truncated or len(model.sector_exposure) > len(sectors)
        if sectors:
            lines.append("Sector exposure:")
            lines.extend(f"  - {s.sector}: {s.weight_pct:.1f}%" for s in sectors)

        lines.append("")
        lines.append(f"Candidate under review: {model.candidate_symbol}")
        if model.existing_holding is not None:
            held = model.existing_holding
            lines.append(
                f"  Already held: {held.quantity:g} @ {held.average_cost:.2f} "
                f"(unrealized {held.unrealized_pnl:.2f}, {held.unrealized_pnl_pct:.2f}%)"
            )
        else:
            lines.append("  Not currently held.")
        if model.candidate_sector:
            lines.append(
                f"  Sector {model.candidate_sector} currently {model.candidate_sector_pct:.1f}% "
                f"of the portfolio."
            )
        lines.append(f"  Capital available for this position: {model.suggested_capital:.2f}")
        lines.append(f"  Portfolio-only starting view: {model.suggested_action.value}")

        lines.append("")
        lines.append(
            f"Limits: max {model.max_position_pct:.1f}% per position, "
            f"max {model.max_sector_pct:.1f}% per sector, "
            f"min {model.min_cash_pct:.1f}% cash."
        )
        if model.constraint_notes:
            lines.append("Constraints in effect:")
            notes = model.constraint_notes[:max_items]
            truncated = truncated or len(model.constraint_notes) > len(notes)
            lines.extend(f"  - {note}" for note in notes)
        if risk.warnings:
            lines.append("Risk warnings:")
            lines.extend(f"  - {w}" for w in risk.warnings[:max_items])

        return RenderedSection(
            kind=self.kind,
            title="Portfolio Context",
            body="\n".join(lines),
            item_count=len(allocations),
            truncated=truncated,
        )
