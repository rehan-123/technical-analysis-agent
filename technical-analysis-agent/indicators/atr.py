from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class ATRIndicator(Indicator):
    """Average True Range using Wilder's smoothing method."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.name = f"atr_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        return atr.rename(self.name)
