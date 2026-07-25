from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.settings import Settings


@dataclass
class VolatilityResult:
    atr_pct: float
    regime: str  # low / normal / high
    atr_expansion: bool = False
    atr_compression: bool = False
    bollinger_squeeze: bool = False
    breakout_probability: int = 0  # 0-100, heuristic
    trend_exhaustion: bool = False
    signal: str = "neutral"
    notes: list[str] | None = None

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


class VolatilityEngine:
    """Characterizes the current volatility regime.

    Combines ATR% (ATR relative to price) against its own recent history to
    detect expansion/compression, a Bollinger/Keltner-style squeeze to flag
    coiled ranges, and a simple breakout-probability heuristic: a squeeze
    with rising ATR raises the odds of an imminent expansion move.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(self, df: pd.DataFrame) -> VolatilityResult:
        s = self.settings
        latest = df.iloc[-1]
        atr_pct = float(latest["atr_pct"])

        hist = df["atr_pct"].tail(s.vol_regime_lookback).dropna()
        avg_atr_pct = float(hist.mean()) if len(hist) else atr_pct

        # Regime by percentile rank of current ATR% within its history.
        if len(hist) >= 20:
            rank = float((hist <= atr_pct).mean())
        else:
            rank = 0.5
        if rank >= 0.70:
            regime = "high"
        elif rank <= 0.30:
            regime = "low"
        else:
            regime = "normal"

        result = VolatilityResult(atr_pct=round(atr_pct, 4), regime=regime)

        if avg_atr_pct and atr_pct > avg_atr_pct * s.vol_expansion_multiplier:
            result.atr_expansion = True
            result.notes.append("ATR expanding above its recent average")
        if avg_atr_pct and atr_pct < avg_atr_pct / s.vol_expansion_multiplier:
            result.atr_compression = True
            result.notes.append("ATR compressing below its recent average")

        # Bollinger squeeze: bandwidth in the low tail of its recent range.
        bw = df["bb_bandwidth"].tail(s.vol_squeeze_lookback).dropna()
        if len(bw) >= 20:
            bw_rank = float((bw <= bw.iloc[-1]).mean())
            if bw_rank <= s.vol_squeeze_percentile:
                result.bollinger_squeeze = True
                result.notes.append("Bollinger squeeze: bands unusually tight")

        # Breakout probability heuristic: squeeze + early expansion.
        prob = 0
        if result.bollinger_squeeze:
            prob += 45
        if result.atr_compression:
            prob += 20
        if result.atr_expansion:
            prob += 25
        result.breakout_probability = min(100, prob)

        # Trend exhaustion: high regime + fading momentum histogram.
        if regime == "high" and len(df) >= 3:
            hist_series = df["macd_histogram"].tail(3)
            fading = abs(hist_series.iloc[-1]) < abs(hist_series.iloc[-2]) < abs(hist_series.iloc[-3])
            if fading:
                result.trend_exhaustion = True
                result.notes.append("Momentum fading at elevated volatility (possible exhaustion)")

        result.signal = "neutral"  # volatility is directionally agnostic
        return result
