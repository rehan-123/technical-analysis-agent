from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class BollingerBandsIndicator(Indicator):
    """Bollinger Bands: upper/middle/lower, %B, and bandwidth."""

    def __init__(self, period: int = 20, std_dev: float = 2.0) -> None:
        self.period = period
        self.std_dev = std_dev
        self.name = "bollinger"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        middle = df["close"].rolling(window=self.period, min_periods=self.period).mean()
        std = df["close"].rolling(window=self.period, min_periods=self.period).std()

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        percent_b = (df["close"] - lower) / (upper - lower)
        bandwidth = (upper - lower) / middle

        return pd.DataFrame(
            {
                "bb_upper": upper,
                "bb_middle": middle,
                "bb_lower": lower,
                "bb_percent_b": percent_b,
                "bb_bandwidth": bandwidth,
            }
        )
