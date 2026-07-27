from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.settings import Settings


@dataclass
class FairValueGap:
    top: float
    bottom: float
    direction: str  # bullish / bearish
    index: int
    filled: bool


@dataclass
class OrderBlock:
    top: float
    bottom: float
    direction: str
    index: int


@dataclass
class SMCResult:
    """Smart Money Concepts read.

    IMPORTANT: these are transparent v1 heuristics, not canonical
    algorithms — the SMC literature is not standardized. Each detection is
    mechanical and deterministic, and the whole result is flagged
    ``heuristic=True`` so downstream consumers weight it accordingly.
    """

    heuristic: bool = True
    fair_value_gaps: list[FairValueGap] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    equal_highs: list[float] = field(default_factory=list)
    equal_lows: list[float] = field(default_factory=list)
    market_zone: str = "equilibrium"  # premium / discount / equilibrium
    signal: str = "neutral"
    strength: int = 0
    notes: list[str] = field(default_factory=list)


class SMCEngine:
    """Detects a deterministic subset of Smart Money Concepts.

    - **Fair Value Gap (FVG):** a 3-bar imbalance where bar1.high < bar3.low
      (bullish) or bar1.low > bar3.high (bearish) — a price gap the middle
      bar left unfilled.
    - **Equal highs/lows:** clusters of swing highs/lows within a small
      tolerance, a proxy for resting liquidity.
    - **Premium/discount:** where current price sits within the recent
      dealing range (top half = premium, bottom half = discount).
    - **Order block (basic):** the last opposing candle before a strong
      displacement move.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _fair_value_gaps(self, df: pd.DataFrame) -> list[FairValueGap]:
        s = self.settings
        gaps: list[FairValueGap] = []
        window = df.tail(s.smc_lookback_bars).reset_index(drop=True)
        highs, lows, closes = window["high"].values, window["low"].values, window["close"].values
        for i in range(2, len(window)):
            b1_high, b1_low = highs[i - 2], lows[i - 2]
            b3_high, b3_low = highs[i], lows[i]
            if b1_high < b3_low and (b3_low - b1_high) / b1_high >= s.smc_fvg_min_gap_pct:
                filled = bool((closes[i:] <= b1_high).any())
                gaps.append(FairValueGap(round(b3_low, 2), round(b1_high, 2), "bullish", i, filled))
            elif b1_low > b3_high and (b1_low - b3_high) / b3_high >= s.smc_fvg_min_gap_pct:
                filled = bool((closes[i:] >= b1_low).any())
                gaps.append(FairValueGap(round(b1_low, 2), round(b3_high, 2), "bearish", i, filled))
        return gaps[-5:]

    def _equal_levels(self, series: pd.Series) -> list[float]:
        tol = self.settings.smc_equal_level_tolerance_pct
        values = sorted(series.tail(self.settings.smc_lookback_bars).values)
        clusters: list[list[float]] = []
        for v in values:
            if clusters and abs(v - clusters[-1][-1]) / clusters[-1][-1] <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [round(float(sum(c) / len(c)), 2) for c in clusters if len(c) >= 2]

    def evaluate(self, df: pd.DataFrame) -> SMCResult:
        result = SMCResult()
        window = df.tail(self.settings.smc_lookback_bars)

        result.fair_value_gaps = self._fair_value_gaps(df)
        result.equal_highs = self._equal_levels(window["high"])[:3]
        result.equal_lows = self._equal_levels(window["low"])[:3]

        # Premium / discount within the dealing range.
        hi, lo = float(window["high"].max()), float(window["low"].min())
        current = float(df["close"].iloc[-1])
        mid = (hi + lo) / 2
        if current > mid * 1.001:
            result.market_zone = "premium"
        elif current < mid * 0.999:
            result.market_zone = "discount"
        else:
            result.market_zone = "equilibrium"

        # Basic order block: last opposing candle before a >1.5% displacement.
        recent = df.tail(20).reset_index(drop=True)
        for i in range(1, len(recent)):
            move = (recent["close"].iloc[i] - recent["close"].iloc[i - 1]) / recent["close"].iloc[i - 1]
            if move > 0.015 and recent["close"].iloc[i - 1] < recent["open"].iloc[i - 1]:
                result.order_blocks.append(
                    OrderBlock(round(recent["open"].iloc[i - 1], 2), round(recent["low"].iloc[i - 1], 2), "bullish", i - 1)
                )
            elif move < -0.015 and recent["close"].iloc[i - 1] > recent["open"].iloc[i - 1]:
                result.order_blocks.append(
                    OrderBlock(round(recent["high"].iloc[i - 1], 2), round(recent["open"].iloc[i - 1], 2), "bearish", i - 1)
                )
        result.order_blocks = result.order_blocks[-3:]

        # Directional lean: discount + unfilled bullish FVG => bullish bias.
        unfilled_bull = [g for g in result.fair_value_gaps if g.direction == "bullish" and not g.filled]
        unfilled_bear = [g for g in result.fair_value_gaps if g.direction == "bearish" and not g.filled]
        score = 0
        if result.market_zone == "discount":
            score += 1
        if result.market_zone == "premium":
            score -= 1
        score += len(unfilled_bull) - len(unfilled_bear)
        if score > 0:
            result.signal = "bullish"
        elif score < 0:
            result.signal = "bearish"
        result.strength = min(100, 30 + 15 * abs(score))

        if result.equal_highs:
            result.notes.append("Equal highs detected (potential buy-side liquidity)")
        if result.equal_lows:
            result.notes.append("Equal lows detected (potential sell-side liquidity)")
        return result
