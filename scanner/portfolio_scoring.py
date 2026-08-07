from __future__ import annotations

from config.settings import Settings, get_settings
from portfolio.portfolio_models import PortfolioAction, PortfolioRecommendationContext

#: Base score per suggested action — the portfolio layer's own deterministic
#: read of what should happen to a candidate (see
#: ``PortfolioService._suggest_action``). Explicit and ordered so the score
#: is explainable, the same "every point traces to a named condition"
#: principle ``RiskLimits``' own weight constants already follow.
_ACTION_BASE_SCORE: dict[PortfolioAction, int] = {
    PortfolioAction.ADD_NEW: 80,
    PortfolioAction.INCREASE: 65,
    PortfolioAction.HOLD: 50,
    PortfolioAction.STAY_IN_CASH: 30,
    PortfolioAction.REDUCE: 15,
    PortfolioAction.EXIT: 5,
}


def score_portfolio_fit(
    context: PortfolioRecommendationContext | None,
    *,
    settings: Settings | None = None,
) -> tuple[int, list[str]]:
    """Deterministic ``[0, 100]`` read of how well a candidate fits the
    active portfolio's constraints — derived entirely from
    ``PortfolioService.build_context``'s output. The Scanner performs no
    allocation or risk-limit computation of its own; every input here was
    already produced by the (frozen, integration-only) Portfolio Engine.

    Neutral (50) when no portfolio is configured — the Market Scanner is
    useful even with no portfolio loaded, and a missing portfolio must not
    silently suppress every candidate the way a genuinely poor fit should.
    """
    settings = settings or get_settings()  # reserved for future weight tuning; unused today
    if context is None:
        return 50, ["No portfolio is configured for this scan; portfolio_score defaulted to neutral (50)."]

    base = _ACTION_BASE_SCORE.get(context.suggested_action, 50)
    notes = [f"Suggested portfolio action: {context.suggested_action.value}."]

    if context.suggested_capital <= 0 and context.suggested_action in (
        PortfolioAction.ADD_NEW,
        PortfolioAction.INCREASE,
    ):
        base = min(base, 25)
        notes.append("No capital headroom is available under the current risk limits.")

    if context.risk.breached_limits:
        base -= 15
        notes.append(f"Portfolio risk limits already breached: {', '.join(context.risk.breached_limits)}.")

    if context.max_sector_pct > 0 and context.candidate_sector_pct >= context.max_sector_pct:
        base -= 10
        notes.append("The candidate's sector is already at or above its configured exposure limit.")

    score = int(max(0, min(100, base)))
    return score, notes
