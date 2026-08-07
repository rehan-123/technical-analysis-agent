from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class MACDIndicator(Indicator):
    """Moving Average Convergence Divergence: line, signal, and histogram."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = "macd"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame(
            {
                "macd_line": macd_line,
                "macd_signal": signal_line,
                "macd_histogram": histogram,
            }
        )
