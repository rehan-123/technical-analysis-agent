from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.base import Indicator


class RSIIndicator(Indicator):
    """Relative Strength Index using Wilder's smoothing method."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.name = f"rsi_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(avg_loss != 0, 100.0)  # zero losses over the window -> RSI 100
        return rsi.rename(self.name)
