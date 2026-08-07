from __future__ import annotations

from models.analysis_result import TechnicalAnalysisResult
from models.strategy import StrategyDirection, StrategyName, StrategySignal
from strategy.base import Strategy, clamp_score, confluence_of, structure_of


class TrendFollowingStrategy(Strategy):
    """Rides an established primary trend.

    Applicable when EMA stacking (price/EMA20/EMA50/EMA200) and the market
    structure engine's read (higher-highs/higher-lows vs lower-highs/
    lower-lows) agree on a single direction, and confluence does not
    contradict it. This is the platform's slowest, highest-conviction
    strategy — it deliberately requires two independent signals (moving
    averages and swing structure) to agree before calling a trend.
    """

    name = StrategyName.TREND_FOLLOWING
    description = "Rides an established primary trend confirmed by EMA stacking and market structure."

    def evaluate(self, technical: TechnicalAnalysisResult) -> StrategySignal:
        ind = technical.indicators
        structure = structure_of(technical)
        confl = confluence_of(technical)

        struct_label = structure.get("structure", "undetermined")
        last_label = structure.get("last_label", "")
        break_of_structure = bool(structure.get("break_of_structure", False))
        change_of_character = bool(structure.get("change_of_character", False))
        net_bias = confl.get("net_bias", "neutral")
        bullish_score = float(confl.get("bullish_score", 0.0))
        bearish_score = float(confl.get("bearish_score", 0.0))

        bullish_stack = ind.close > ind.ema_20 > ind.ema_50 and (
            ind.ema_200 is None or ind.ema_50 >= ind.ema_200
        )
        bearish_stack = ind.close < ind.ema_20 < ind.ema_50 and (
            ind.ema_200 is None or ind.ema_50 <= ind.ema_200
        )

        signals: list[str] = []
        reasoning: list[str] = []
        direction = StrategyDirection.NONE
        applicable = False

        if struct_label == "uptrend" and bullish_stack and net_bias != "bearish":
            applicable = True
            direction = StrategyDirection.LONG
            signals.append("Price > EMA20 > EMA50" + (" > EMA200" if ind.ema_200 is not None else ""))
            signals.append(f"Market structure: uptrend ({last_label})" if last_label else "Market structure: uptrend")
            reasoning.append(
                "Price is stacked above EMA20 and EMA50 while market structure confirms "
                "higher highs and higher lows."
            )
        elif struct_label == "downtrend" and bearish_stack and net_bias != "bullish":
            applicable = True
            direction = StrategyDirection.SHORT
            signals.append("Price < EMA20 < EMA50" + (" < EMA200" if ind.ema_200 is not None else ""))
            signals.append(
                f"Market structure: downtrend ({last_label})" if last_label else "Market structure: downtrend"
            )
            reasoning.append(
                "Price is stacked below EMA20 and EMA50 while market structure confirms "
                "lower highs and lower lows."
            )
        else:
            reasoning.append(
                f"EMA stacking and market structure ('{struct_label}') do not agree on a "
                "single primary trend direction."
            )

        if applicable and break_of_structure:
            signals.append("Break of structure confirms continuation")
            reasoning.append("A recent break of structure supports the trend continuing.")
        if change_of_character:
            reasoning.append("A change of character was flagged on this ticker — treat continuation with caution.")

        base_strength = technical.strength if direction is StrategyDirection.LONG else 100 - technical.strength
        score = clamp_score(base_strength if applicable else base_strength * 0.35)

        agreement = bullish_score if direction is StrategyDirection.LONG else bearish_score
        if applicable:
            confidence = clamp_score(0.6 * technical.confidence + 0.4 * agreement)
            if change_of_character:
                confidence = clamp_score(confidence * 0.85)
        else:
            confidence = clamp_score(technical.confidence * 0.3)

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
