from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.settings import Settings


@dataclass
class CandlestickResult:
    patterns: list[str] = field(default_factory=list)
    signal: str = "neutral"
    strength: int = 0


class CandlestickEngine:
    """Recognizes classic candlestick patterns on the most recent bars.

    Patterns are defined by geometric relationships between open/high/low/
    close of one to three consecutive candles, using configurable body and
    shadow ratios so nothing is hardcoded. Only patterns forming on the
    final (most recent) bar(s) are reported, since those are the actionable
    ones.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _parts(row: pd.Series) -> tuple[float, float, float, float]:
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        return o, h, l, c

    def evaluate(self, df: pd.DataFrame) -> CandlestickResult:
        s = self.settings
        result = CandlestickResult()
        if len(df) < 3:
            return result

        c1 = df.iloc[-3]  # oldest of the three
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]  # most recent

        o, h, l, c = self._parts(c3)
        rng = max(h - l, 1e-9)
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        bullish_candle = c > o
        bearish_candle = c < o

        bull_patterns: list[str] = []
        bear_patterns: list[str] = []

        # --- Single-candle ---
        if body <= s.candle_doji_body_pct * rng:
            result.patterns.append("Doji")
        if body <= 0.3 * rng and lower_shadow >= s.candle_shadow_ratio * body and upper_shadow <= body:
            bull_patterns.append("Hammer")
        if body <= 0.3 * rng and upper_shadow >= s.candle_shadow_ratio * body and lower_shadow <= body:
            bear_patterns.append("Shooting Star")
        if body >= s.candle_long_body_pct * rng and upper_shadow <= 0.1 * rng and lower_shadow <= 0.1 * rng:
            (bull_patterns if bullish_candle else bear_patterns).append("Marubozu")
        if body <= 0.3 * rng and upper_shadow > body and lower_shadow > body and "Doji" not in result.patterns:
            result.patterns.append("Spinning Top")

        # --- Two-candle ---
        o2, _, _, c2v = self._parts(c2)
        prev_bull = c2v > o2
        prev_bear = c2v < o2

        if bullish_candle and prev_bear and c >= o2 and o <= c2v:
            bull_patterns.append("Bullish Engulfing")
        if bearish_candle and prev_bull and o >= c2v and c <= o2:
            bear_patterns.append("Bearish Engulfing")
        if bullish_candle and prev_bear and o < c2v and c > (o2 + c2v) / 2 and c < o2:
            bull_patterns.append("Piercing Pattern")
        if bearish_candle and prev_bull and o > c2v and c < (o2 + c2v) / 2 and c > o2:
            bear_patterns.append("Dark Cloud Cover")
        if abs(c - o) < abs(c2v - o2) and max(o, c) < max(o2, c2v) and min(o, c) > min(o2, c2v):
            result.patterns.append("Harami")

        # --- Three-candle ---
        o1, _, _, c1v = self._parts(c1)
        if (
            c1v < o1  # bearish
            and abs(c2v - o2) <= 0.3 * max(c2["high"] - c2["low"], 1e-9)  # small middle
            and bullish_candle
            and c > (o1 + c1v) / 2
        ):
            bull_patterns.append("Morning Star")
        if (
            c1v > o1  # bullish
            and abs(c2v - o2) <= 0.3 * max(c2["high"] - c2["low"], 1e-9)
            and bearish_candle
            and c < (o1 + c1v) / 2
        ):
            bear_patterns.append("Evening Star")

        three_bull = all(df.iloc[i]["close"] > df.iloc[i]["open"] for i in (-3, -2, -1))
        rising = df["close"].iloc[-1] > df["close"].iloc[-2] > df["close"].iloc[-3]
        if three_bull and rising:
            bull_patterns.append("Three White Soldiers")

        three_bear = all(df.iloc[i]["close"] < df.iloc[i]["open"] for i in (-3, -2, -1))
        falling = df["close"].iloc[-1] < df["close"].iloc[-2] < df["close"].iloc[-3]
        if three_bear and falling:
            bear_patterns.append("Three Black Crows")

        result.patterns.extend(bull_patterns)
        result.patterns.extend(bear_patterns)

        net = len(bull_patterns) - len(bear_patterns)
        if net > 0:
            result.signal = "bullish"
            result.strength = min(100, 40 + 20 * net)
        elif net < 0:
            result.signal = "bearish"
            result.strength = min(100, 40 + 20 * abs(net))
        else:
            result.signal = "neutral"
            result.strength = 20 if result.patterns else 0

        return result
