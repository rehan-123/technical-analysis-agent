from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class VolumeIndicator(Indicator):
    """Volume SMA and relative volume (current volume vs. its own average)."""

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self.name = "volume_analysis"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        volume_sma = df["volume"].rolling(window=self.period, min_periods=self.period).mean()
        relative_volume = df["volume"] / volume_sma

        return pd.DataFrame(
            {
                "volume_sma": volume_sma,
                "relative_volume": relative_volume,
            }
        )
