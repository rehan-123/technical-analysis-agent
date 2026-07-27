from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.settings import Settings


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str  # "high" or "low"


@dataclass
class MarketStructureResult:
    """Structural read of the most recent price action."""

    swing_highs: list[float] = field(default_factory=list)
    swing_lows: list[float] = field(default_factory=list)
    structure: str = "undetermined"  # uptrend / downtrend / ranging
    last_label: str = ""  # HH / HL / LH / LL
    break_of_structure: bool = False
    change_of_character: bool = False
    signal: str = "neutral"
    notes: list[str] = field(default_factory=list)


class MarketStructureEngine:
    """Detects swing structure and higher-level structural events.

    A swing high/low is a local extreme over a symmetric window. The
    sequence of swings is then labeled HH/HL/LH/LL, which yields the trend
    structure. A **Break of Structure (BOS)** is a continuation event: in an
    uptrend, price takes out the prior swing high. A **Change of Character
    (CHoCH)** is the first counter-trend break: in an uptrend, price breaks
    below the most recent higher-low, the earliest structural hint of a
    reversal.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _find_swings(self, df: pd.DataFrame) -> list[SwingPoint]:
        order = self.settings.ms_swing_order
        highs, lows = df["high"].values, df["low"].values
        n = len(df)
        swings: list[SwingPoint] = []
        for i in range(order, n - order):
            hi_window = highs[i - order : i + order + 1]
            lo_window = lows[i - order : i + order + 1]
            if highs[i] == hi_window.max():
                swings.append(SwingPoint(i, float(highs[i]), "high"))
            if lows[i] == lo_window.min():
                swings.append(SwingPoint(i, float(lows[i]), "low"))
        swings.sort(key=lambda s: s.index)
        return swings

    def evaluate(self, df: pd.DataFrame) -> MarketStructureResult:
        window = df.tail(self.settings.ms_lookback_bars)
        window = window.reset_index(drop=True)
        swings = self._find_swings(window)

        result = MarketStructureResult()
        highs = [s for s in swings if s.kind == "high"]
        lows = [s for s in swings if s.kind == "low"]
        result.swing_highs = [round(s.price, 2) for s in highs[-3:]]
        result.swing_lows = [round(s.price, 2) for s in lows[-3:]]

        if len(highs) < 2 or len(lows) < 2:
            result.notes.append("Insufficient swing points to determine structure")
            return result

        last_high, prev_high = highs[-1], highs[-2]
        last_low, prev_low = lows[-1], lows[-2]

        higher_high = last_high.price > prev_high.price
        higher_low = last_low.price > prev_low.price
        lower_high = last_high.price < prev_high.price
        lower_low = last_low.price < prev_low.price

        if higher_high and higher_low:
            result.structure = "uptrend"
            result.last_label = "HH" if last_high.index > last_low.index else "HL"
            result.signal = "bullish"
        elif lower_high and lower_low:
            result.structure = "downtrend"
            result.last_label = "LL" if last_low.index > last_high.index else "LH"
            result.signal = "bearish"
        else:
            result.structure = "ranging"
            result.signal = "neutral"

        current_price = window["close"].iloc[-1]

        # BOS: continuation break of the prior swing in the trend direction.
        if result.structure == "uptrend" and current_price > last_high.price:
            result.break_of_structure = True
            result.notes.append("Bullish BOS: price cleared the last swing high")
        elif result.structure == "downtrend" and current_price < last_low.price:
            result.break_of_structure = True
            result.notes.append("Bearish BOS: price broke the last swing low")

        # CHoCH: first counter-trend break.
        if result.structure == "uptrend" and current_price < last_low.price:
            result.change_of_character = True
            result.signal = "bearish"
            result.notes.append("Bearish CHoCH: broke the most recent higher-low")
        elif result.structure == "downtrend" and current_price > last_high.price:
            result.change_of_character = True
            result.signal = "bullish"
            result.notes.append("Bullish CHoCH: reclaimed the most recent lower-high")

        return result
