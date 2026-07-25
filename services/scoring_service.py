from __future__ import annotations

import pandas as pd

from config.settings import Settings
from models.analysis_result import PatternFlags, Risk, SupportResistanceLevels, Trend


class ScoringService:
    """Translates trend/pattern/level analysis into actionable trade
    parameters: entry zone, stop loss, price targets, risk tier, and a
    confidence score. This is the only layer that produces "actionable"
    output — every upstream service produces descriptive analysis only.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _col(prefix: str, period: int) -> str:
        return f"{prefix}_{period}"

    def _risk_tier(self, atr_pct: float) -> Risk:
        s = self.settings
        if atr_pct <= s.risk_low_atr_pct:
            return "Low"
        if atr_pct <= s.risk_medium_atr_pct:
            return "Medium"
        return "High"

    def _entry_zone(self, close: float) -> tuple[float, float]:
        buf = self.settings.entry_zone_buffer_pct
        return (round(close * (1 - buf), 2), round(close * (1 + buf), 2))

    def _stop_loss(
        self,
        close: float,
        atr: float,
        bullish: bool,
        support: list[float],
        resistance: list[float],
    ) -> float:
        s = self.settings
        if bullish:
            atr_stop = close - s.atr_stop_multiplier * atr
            if support:
                structural_stop = support[0] * 0.997
                if structural_stop < close:
                    atr_stop = max(atr_stop, structural_stop)
        else:
            atr_stop = close + s.atr_stop_multiplier * atr
            if resistance:
                structural_stop = resistance[0] * 1.003
                if structural_stop > close:
                    atr_stop = min(atr_stop, structural_stop)
        return round(atr_stop, 2)

    def _targets(
        self, close: float, atr: float, bullish: bool, resistance: list[float], support: list[float]
    ) -> list[float]:
        s = self.settings
        atr_targets = [
            close + m * atr if bullish else close - m * atr for m in s.atr_target_multipliers
        ]
        structural = resistance if bullish else support

        candidates = sorted(t for t in [*structural, *atr_targets] if (t > close) == bullish)
        candidates = self._merge_close_values(candidates, tolerance_pct=0.003)
        candidates.sort(key=lambda t: abs(t - close))  # nearest target first

        if candidates:
            return [round(t, 2) for t in candidates[:3]]
        return sorted((round(t, 2) for t in atr_targets), key=lambda t: abs(t - close))

    @staticmethod
    def _merge_close_values(values: list[float], tolerance_pct: float) -> list[float]:
        """Collapses values that fall within `tolerance_pct` of each other,
        so a resistance level and an ATR-based target that land almost on
        top of one another don't show up as two separate targets."""
        if not values:
            return []
        merged: list[float] = [values[0]]
        for v in values[1:]:
            if abs(v - merged[-1]) / merged[-1] <= tolerance_pct:
                merged[-1] = (merged[-1] + v) / 2
            else:
                merged.append(v)
        return merged

    def _confidence(self, strength: int, patterns: PatternFlags, atr_pct: float) -> int:
        s = self.settings
        confidence = strength
        if patterns.breakout or patterns.trend_reversal:
            confidence += 5
        if patterns.consolidation:
            confidence -= 10  # low-conviction, range-bound structure
        if atr_pct > s.risk_medium_atr_pct:
            confidence -= 10  # high volatility reduces confidence in any single read
        return int(max(0, min(100, confidence)))

    def _summary(self, trend: Trend, patterns: PatternFlags, risk: Risk) -> str:
        descriptors = []
        if patterns.breakout:
            descriptors.append("a confirmed breakout")
        if patterns.pullback:
            descriptors.append("a pullback within the broader trend")
        if patterns.trend_reversal:
            descriptors.append("an emerging trend reversal")
        if patterns.consolidation:
            descriptors.append("range-bound consolidation")
        if patterns.high_volatility:
            descriptors.append("elevated volatility")

        detail = f" with {', '.join(descriptors)}" if descriptors else ""
        return f"Technical structure is {trend.lower()}{detail}. Overall risk assessed as {risk.lower()}."

    def build(
        self,
        df: pd.DataFrame,
        trend: Trend,
        strength: int,
        levels: SupportResistanceLevels,
        patterns: PatternFlags,
    ) -> dict:
        s = self.settings
        latest = df.iloc[-1]
        close = float(latest["close"])
        atr = float(latest[self._col("atr", s.atr_period)])
        atr_pct = float(latest["atr_pct"])
        bullish = strength >= s.bullish_strength_threshold

        risk = self._risk_tier(atr_pct)
        entry_zone = self._entry_zone(close)
        stop_loss = self._stop_loss(close, atr, bullish, levels.support, levels.resistance)
        targets = self._targets(close, atr, bullish, levels.resistance, levels.support)
        confidence = self._confidence(strength, patterns, atr_pct)
        summary = self._summary(trend, patterns, risk)

        return {
            "entry_zone": entry_zone,
            "stop_loss": stop_loss,
            "targets": targets,
            "risk": risk,
            "confidence": confidence,
            "summary": summary,
        }
