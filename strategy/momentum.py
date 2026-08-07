from __future__ import annotations

from models.analysis_result import TechnicalAnalysisResult
from models.strategy import StrategyDirection, StrategyName, StrategySignal
from strategy.base import Strategy, clamp_score


class MomentumStrategy(Strategy):
    """Confirms directional price momentum.

    Applicable when MACD (line vs. signal, histogram sign) and RSI agree on
    direction without RSI already being at a momentum-exhaustion extreme.
    Relative volume is used as a participation confirmation rather than a
    gate — a momentum move on light volume is weaker, not disqualified.
    """

    name = StrategyName.MOMENTUM
    description = "Confirms directional price momentum via RSI, MACD, and relative volume."

    def evaluate(self, technical: TechnicalAnalysisResult) -> StrategySignal:
        ind = technical.indicators
        s = self.settings

        macd_bullish = ind.macd_line > ind.macd_signal and ind.macd_histogram > 0
        macd_bearish = ind.macd_line < ind.macd_signal and ind.macd_histogram < 0
        # Momentum zone excludes the overbought/oversold extremes: those are
        # the Mean Reversion strategy's territory, not fresh momentum's.
        rsi_bullish_zone = 50.0 <= ind.rsi < s.rsi_overbought
        rsi_bearish_zone = s.rsi_oversold < ind.rsi <= 50.0
        volume_confirms = ind.relative_volume >= 1.0

        signals: list[str] = []
        reasoning: list[str] = []
        direction = StrategyDirection.NONE
        applicable = False

        if macd_bullish and rsi_bullish_zone:
            applicable = True
            direction = StrategyDirection.LONG
            signals.append(f"MACD bullish (line {ind.macd_line:.4f} > signal {ind.macd_signal:.4f})")
            signals.append(f"RSI {ind.rsi:.1f} in bullish momentum zone")
            reasoning.append(
                "MACD sits above its signal line with a positive histogram, and RSI confirms "
                "upside momentum without being overbought."
            )
        elif macd_bearish and rsi_bearish_zone:
            applicable = True
            direction = StrategyDirection.SHORT
            signals.append(f"MACD bearish (line {ind.macd_line:.4f} < signal {ind.macd_signal:.4f})")
            signals.append(f"RSI {ind.rsi:.1f} in bearish momentum zone")
            reasoning.append(
                "MACD sits below its signal line with a negative histogram, and RSI confirms "
                "downside momentum without being oversold."
            )
        else:
            reasoning.append("MACD and RSI do not currently agree on a momentum direction.")

        if applicable and volume_confirms:
            signals.append(f"Relative volume {ind.relative_volume:.2f}x confirms participation")
            reasoning.append("Relative volume is at or above its 20-period average, supporting the move.")
        elif applicable:
            reasoning.append("Relative volume is below average — this momentum move lacks strong participation.")

        # RSI distance from the neutral midpoint (50) as a smooth 0-100
        # directional-strength proxy; volume adds/removes a fixed bonus.
        rsi_strength = abs(ind.rsi - 50.0) * 2.0
        volume_adjustment = 10.0 if volume_confirms else -10.0
        base_score = rsi_strength + (volume_adjustment if applicable else 0.0)
        score = clamp_score(base_score if applicable else base_score * 0.3)

        if applicable:
            confidence = clamp_score(0.5 * technical.confidence + 0.5 * rsi_strength)
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
