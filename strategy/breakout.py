from __future__ import annotations

from models.analysis_result import TechnicalAnalysisResult
from models.strategy import StrategyDirection, StrategyName, StrategySignal
from strategy.base import Strategy, clamp_score, volatility_of


class BreakoutStrategy(Strategy):
    """Confirms a price breakout beyond a mapped support/resistance level.

    The pattern service's ``breakout`` flag says *that* price has broken out;
    this strategy determines *which direction* (price beyond the highest
    mapped resistance vs. below the lowest mapped support, falling back to
    the headline trend when no level is available) and grades the breakout
    by volume expansion and a preceding volatility squeeze/ATR expansion —
    the two classic breakout confirmations.
    """

    name = StrategyName.BREAKOUT
    description = "Confirms a price breakout beyond a mapped support/resistance level with volume and volatility expansion."

    def evaluate(self, technical: TechnicalAnalysisResult) -> StrategySignal:
        ind = technical.indicators
        levels = technical.levels
        vol = volatility_of(technical)
        breakout_flag = bool(technical.patterns.breakout)

        signals: list[str] = []
        reasoning: list[str] = []
        direction = StrategyDirection.NONE

        if breakout_flag:
            broke_resistance = bool(levels.resistance) and ind.close > max(levels.resistance)
            broke_support = bool(levels.support) and ind.close < min(levels.support)
            if broke_resistance and not broke_support:
                direction = StrategyDirection.LONG
                signals.append(f"Price {ind.close:.2f} broke above resistance {max(levels.resistance):.2f}")
                reasoning.append("Price has broken above the highest mapped resistance level.")
            elif broke_support and not broke_resistance:
                direction = StrategyDirection.SHORT
                signals.append(f"Price {ind.close:.2f} broke below support {min(levels.support):.2f}")
                reasoning.append("Price has broken below the lowest mapped support level.")
            elif technical.trend in ("Bullish", "Strong Bullish"):
                direction = StrategyDirection.LONG
                reasoning.append("A breakout was flagged and the primary trend is bullish.")
            elif technical.trend in ("Bearish", "Strong Bearish"):
                direction = StrategyDirection.SHORT
                reasoning.append("A breakout was flagged and the primary trend is bearish.")
            else:
                reasoning.append(
                    "A breakout was flagged but its direction could not be determined from "
                    "mapped levels or the primary trend."
                )
        else:
            reasoning.append("No breakout pattern is currently flagged.")

        applicable = breakout_flag and direction is not StrategyDirection.NONE

        volume_spike = ind.relative_volume >= self.settings.volume_spike_multiplier
        squeeze_or_expansion = bool(vol.get("bollinger_squeeze", False)) or bool(vol.get("atr_expansion", False))
        breakout_probability = float(vol.get("breakout_probability", 0) or 0)

        if applicable and volume_spike:
            signals.append(f"Relative volume {ind.relative_volume:.2f}x confirms breakout")
            reasoning.append("Volume expanded well beyond its average, confirming the breakout is not a false move.")
        elif applicable:
            reasoning.append("Volume has not yet expanded to confirm the breakout.")

        if applicable and squeeze_or_expansion:
            signals.append("Volatility expansion / squeeze release supports the breakout")
            reasoning.append("The breakout follows a volatility squeeze or ATR expansion, a classic breakout precursor.")

        bonus = (15.0 if volume_spike else 0.0) + (10.0 if squeeze_or_expansion else 0.0) + breakout_probability * 0.2
        base_strength = technical.strength if direction is StrategyDirection.LONG else 100 - technical.strength
        score = clamp_score(base_strength * 0.6 + bonus if applicable else base_strength * 0.2)

        if applicable:
            confidence = technical.confidence
            if volume_spike:
                confidence += 10
            if squeeze_or_expansion:
                confidence += 5
            confidence = clamp_score(confidence)
        else:
            confidence = clamp_score(technical.confidence * 0.2)

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
