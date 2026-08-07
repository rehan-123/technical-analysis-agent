from __future__ import annotations

from config.settings import Settings, get_settings
from models.opportunity import Opportunity


class RankingEngine:
    """Combines an ``Opportunity``'s component scores into ``combined_score``
    and assigns final ``ranking`` order across a batch.

    Weights come from ``Settings.scanner_weight_*`` and are normalized here
    at combination time (they need not sum to 1 in configuration) — the same
    "declared, normalized internally" convention the confluence engine's
    ``weight_*`` block already uses.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def combine(
        self,
        *,
        technical_score: int,
        news_score: int,
        portfolio_score: int,
        opportunity_score: int,
    ) -> int:
        """Blend the four component scores into one ``[0, 100]`` ranking key."""
        s = self._settings
        weights = {
            "technical": max(0.0, s.scanner_weight_technical),
            "news": max(0.0, s.scanner_weight_news),
            "portfolio": max(0.0, s.scanner_weight_portfolio),
            "opportunity": max(0.0, s.scanner_weight_opportunity),
        }
        total_weight = sum(weights.values()) or 1.0
        blended = (
            weights["technical"] * technical_score
            + weights["news"] * news_score
            + weights["portfolio"] * portfolio_score
            + weights["opportunity"] * opportunity_score
        ) / total_weight
        return int(max(0, min(100, round(blended))))

    def rank(self, opportunities: list[Opportunity]) -> list[Opportunity]:
        """Sort descending by ``combined_score`` (ties broken by
        ``confidence``, then ``ticker`` for full determinism) and assign
        1-based ``ranking``.

        A single ``sorted()`` call — O(n log n), no nested scans — over
        already-computed scores. Returns a *new* list: ``Opportunity`` is
        frozen, so a re-ranked item is produced via
        ``model_copy(update=...)``, the same "immutable, copy to update"
        pattern ``PortfolioManager`` already uses for ``Portfolio``.
        """
        ordered = sorted(opportunities, key=lambda o: (-o.combined_score, -o.confidence, o.ticker))
        return [o.model_copy(update={"ranking": idx}) for idx, o in enumerate(ordered, start=1)]
