from __future__ import annotations

from models.analysis_result import TechnicalAnalysisResult
from models.strategy import StrategyDirection, StrategyName, StrategySignal
from strategy.base import Strategy, clamp_score, structure_of


class SwingStrategy(Strategy):
    """Buys pullbacks within an established trend.

    Applicable when the market structure engine still reads an established
    trend AND the pattern service has flagged a pullback (a short-term
    counter-move within that trend) — the classic swing-trade entry: trend
    intact, price temporarily discounted. Proximity to a mapped support
    (uptrend) or resistance (downtrend) level, and an RSI that is merely
    neutral rather than at a reversal extreme, both raise confidence without
    being required for applicability.
    """

    name = StrategyName.SWING
    description = "Buys pullbacks within an established trend, targeting a swing back toward the trend's prior extreme."

    def evaluate(self, technical: TechnicalAnalysisResult) -> StrategySignal:
        ind = technical.indicators
        structure = structure_of(technical)
        struct_label = structure.get("structure", "undetermined")
        pullback = bool(technical.patterns.pullback)
        levels = technical.levels
        s = self.settings

        near_support = self._near_any(ind.close, levels.support, s.strategy_level_proximity_pct)
        near_resistance = self._near_any(ind.close, levels.resistance, s.strategy_level_proximity_pct)
        rsi_neutral = s.pullback_rsi_low <= ind.rsi <= s.pullback_rsi_high + 15.0

        signals: list[str] = []
        reasoning: list[str] = []
        direction = StrategyDirection.NONE
        applicable = False

        if struct_label == "uptrend" and pullback:
            applicable = True
            direction = StrategyDirection.LONG
            signals.append("Pullback detected within an uptrend")
            reasoning.append("Market structure remains an uptrend while a pullback offers a lower-risk entry.")
            if near_support:
                signals.append("Price is near a mapped support level")
                reasoning.append("The pullback is holding near a mapped support level.")
        elif struct_label == "downtrend" and pullback:
            applicable = True
            direction = StrategyDirection.SHORT
            signals.append("Relief bounce detected within a downtrend")
            reasoning.append("Market structure remains a downtrend while a relief bounce offers a lower-risk short entry.")
            if near_resistance:
                signals.append("Price is near a mapped resistance level")
                reasoning.append("The bounce is stalling near a mapped resistance level.")
        else:
            reasoning.append("No qualifying pullback within an established trend was detected.")

        if applicable and rsi_neutral:
            signals.append(f"RSI {ind.rsi:.1f} is neutral, consistent with a pullback rather than a reversal")
        elif applicable:
            reasoning.append("RSI is outside the neutral zone — the pullback may be turning into a full reversal.")

        near_level = (direction is StrategyDirection.LONG and near_support) or (
            direction is StrategyDirection.SHORT and near_resistance
        )
        base_strength = technical.strength if direction is StrategyDirection.LONG else 100 - technical.strength
        bonus = (15.0 if near_level else 0.0) + (10.0 if (applicable and rsi_neutral) else 0.0)
        score = clamp_score(base_strength * 0.7 + bonus if applicable else base_strength * 0.25)

        if applicable:
            confidence = technical.confidence
            if near_level:
                confidence += 10
            if rsi_neutral:
                confidence += 5
            confidence = clamp_score(confidence)
        else:
            confidence = clamp_score(technical.confidence * 0.25)

        return StrategySignal(
            strategy=self.name,
            ticker=technical.ticker,
            applicable=applicable,
            direction=direction,
            confidence=confidence,
            score=score,
            signals=signals,
            reasoning=reasoning,
        )

    @staticmethod
    def _near_any(price: float, levels: list[float], tolerance_pct: float) -> bool:
        """Whether ``price`` sits within ``tolerance_pct`` of any level."""
        if not levels or price <= 0:
            return False
        return any(abs(price - level) / price <= tolerance_pct for level in levels)
