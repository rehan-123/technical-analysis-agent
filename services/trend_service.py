from __future__ import annotations

import pandas as pd

from config.settings import Settings
from models.analysis_result import Trend


class TrendService:
    """Determines overall trend direction and a 0-100 strength score from
    EMA alignment, MACD momentum, RSI positioning, and volume behavior.

    The score starts at a neutral baseline of 50 and is nudged up or down
    by each independent signal, then clamped to [0, 100] and mapped onto
    a five-bucket trend label. All thresholds are configurable.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _col(prefix: str, period: int) -> str:
        return f"{prefix}_{period}"

    def evaluate(self, df: pd.DataFrame) -> tuple[Trend, int, list[str]]:
        s = self.settings
        latest = df.iloc[-1]
        signals: list[str] = []
        score = 50.0

        ema_fast_col = self._col("ema", s.ema_fast_period)
        ema_mid_col = self._col("ema", s.ema_medium_period)
        ema_long_col = self._col("ema", s.ema_long_period)

        close = latest["close"]
        ema_fast = latest[ema_fast_col]
        ema_mid = latest[ema_mid_col]
        ema_long = latest.get(ema_long_col)

        # --- EMA alignment ---
        if ema_fast > ema_mid:
            score += 15
            signals.append(f"EMA{s.ema_fast_period} above EMA{s.ema_medium_period}")
        else:
            score -= 15
            signals.append(f"EMA{s.ema_fast_period} below EMA{s.ema_medium_period}")

        if pd.notna(ema_long):
            if ema_mid > ema_long:
                score += 10
                signals.append(f"EMA{s.ema_medium_period} above EMA{s.ema_long_period}")
            else:
                score -= 10
                signals.append(f"EMA{s.ema_medium_period} below EMA{s.ema_long_period}")
            score += 10 if close > ema_long else -10

        # --- MACD momentum ---
        macd_hist = latest["macd_histogram"]
        if macd_hist > 0:
            score += 10
            signals.append("MACD histogram positive")
        else:
            score -= 10
            signals.append("MACD histogram negative")

        prev_hist = df["macd_histogram"].iloc[-2] if len(df) > 1 else macd_hist
        if prev_hist <= 0 < macd_hist:
            signals.append("MACD Bullish Cross")
            score += 5
        elif prev_hist >= 0 > macd_hist:
            signals.append("MACD Bearish Cross")
            score -= 5

        # --- RSI momentum ---
        rsi = latest[self._col("rsi", s.rsi_period)]
        signals.append(f"RSI {rsi:.0f}")
        if rsi >= 55:
            score += 5
        elif rsi <= 45:
            score -= 5

        # --- Volume behavior ---
        relative_volume = latest["relative_volume"]
        if pd.notna(relative_volume):
            if relative_volume >= s.volume_trend_high:
                signals.append("Volume Increasing")
                score += 3
            elif relative_volume <= s.volume_trend_low:
                signals.append("Volume Decreasing")
                score -= 3

        strength = int(max(0, min(100, round(score))))

        if strength >= 80:
            trend: Trend = "Strong Bullish"
        elif strength >= 60:
            trend = "Bullish"
        elif strength >= 40:
            trend = "Neutral"
        elif strength >= 20:
            trend = "Bearish"
        else:
            trend = "Strong Bearish"

        return trend, strength, signals
