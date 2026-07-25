from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class SMAIndicator(Indicator):
    """Simple Moving Average."""

    def __init__(self, period: int) -> None:
        self.period = period
        self.name = f"sma_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return (
            df["close"]
            .rolling(window=self.period, min_periods=self.period)
            .mean()
            .rename(self.name)
        )


class EMAIndicator(Indicator):
    """Exponential Moving Average."""

    def __init__(self, period: int) -> None:
        self.period = period
        self.name = f"ema_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return (
            df["close"]
            .ewm(span=self.period, adjust=False, min_periods=self.period)
            .mean()
            .rename(self.name)
        )
