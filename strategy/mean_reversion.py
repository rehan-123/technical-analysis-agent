from __future__ import annotations

from models.analysis_result import TechnicalAnalysisResult
from models.strategy import StrategyDirection, StrategyName, StrategySignal
from strategy.base import Strategy, clamp_score, structure_of


class MeanReversionStrategy(Strategy):
    """Fades RSI/Bollinger extremes, targeting reversion toward the mean.

    Applicable when RSI is at a configured overbought/oversold extreme *and*
    price is simultaneously pressed against the corresponding Bollinger Band
    — two independent extremes agreeing, not just one. Deliberately the
    platform's contrarian strategy: when it fires against the prevailing
    market-structure trend, that is flagged explicitly as a higher-risk,
    short-term, counter-trend call rather than silently scored the same as a
    reversion that agrees with the trend (e.g. a bounce inside an uptrend's
    own pullback).
    """

    name = StrategyName.MEAN_REVERSION
    description = "Fades RSI/Bollinger Band extremes, targeting reversion toward the mean."

    def evaluate(self, technical: TechnicalAnalysisResult) -> StrategySignal:
        ind = technical.indicators
        s = self.settings
        struct_label = structure_of(technical).get("structure", "undetermined")

        oversold = ind.rsi <= s.rsi_oversold
        overbought = ind.rsi >= s.rsi_overbought
        at_lower_band = ind.bb_percent_b <= 0.05
        at_upper_band = ind.bb_percent_b >= 0.95

        signals: list[str] = []
        reasoning: list[str] = []
        direction = StrategyDirection.NONE
        applicable = False

        if oversold and at_lower_band:
            applicable = True
            direction = StrategyDirection.LONG
            signals.append(f"RSI {ind.rsi:.1f} oversold")
            signals.append(f"Price at lower Bollinger Band (%B {ind.bb_percent_b:.2f})")
            reasoning.append(
                "RSI is oversold and price is pressed against the lower Bollinger Band, "
                "favoring a bounce toward the mean."
            )
        elif overbought and at_upper_band:
            applicable = True
            direction = StrategyDirection.SHORT
            signals.append(f"RSI {ind.rsi:.1f} overbought")
            signals.append(f"Price at upper Bollinger Band (%B {ind.bb_percent_b:.2f})")
            reasoning.append(
                "RSI is overbought and price is pressed against the upper Bollinger Band, "
                "favoring a pullback toward the mean."
            )
        else:
            reasoning.append("RSI and Bollinger Band position do not currently show an agreeing reversion extreme.")

        counter_trend = applicable and (
            (direction is StrategyDirection.LONG and struct_label == "downtrend")
            or (direction is StrategyDirection.SHORT and struct_label == "uptrend")
        )
        if counter_trend:
            reasoning.append(
                f"This call fades the primary trend ('{struct_label}') — treat as a "
                "short-term, higher-risk reversion trade rather than a new trend call."
            )

        extreme_distance = abs(ind.rsi - 50.0) * 2.0
        score = clamp_score(extreme_distance if applicable else extreme_distance * 0.3)
        if counter_trend:
            score = clamp_score(score * 0.85)

        if applicable:
            confidence = clamp_score(0.5 * technical.confidence + 0.5 * extreme_distance)
            if counter_trend:
                confidence = clamp_score(confidence * 0.75)
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
