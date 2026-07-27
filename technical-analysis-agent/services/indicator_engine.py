from __future__ import annotations

import pandas as pd

from config.settings import Settings
from indicators.atr import ATRIndicator
from indicators.base import Indicator
from indicators.bollinger import BollingerBandsIndicator
from indicators.macd import MACDIndicator
from indicators.moving_averages import EMAIndicator, SMAIndicator
from indicators.rsi import RSIIndicator
from indicators.volume import VolumeIndicator
from utils.exceptions import IndicatorCalculationError
from utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorEngine:
    """Computes every configured indicator and merges the results into a
    single feature-enriched DataFrame, aligned to the original OHLCV index.

    Indicators are fully independent of one another (see
    ``indicators/base.py``) — adding a new one only requires adding an
    entry to ``_build_indicators``; nothing else in the codebase needs to
    change, and no indicator's logic is duplicated elsewhere.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._indicators = self._build_indicators()

    def _build_indicators(self) -> list[Indicator]:
        s = self.settings
        return [
            EMAIndicator(s.ema_fast_period),
            EMAIndicator(s.ema_medium_period),
            EMAIndicator(s.ema_long_period),
            SMAIndicator(s.sma_period),
            RSIIndicator(s.rsi_period),
            MACDIndicator(s.macd_fast, s.macd_slow, s.macd_signal),
            ATRIndicator(s.atr_period),
            BollingerBandsIndicator(s.bb_period, s.bb_std_dev),
            VolumeIndicator(s.volume_sma_period),
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        for indicator in self._indicators:
            try:
                result = indicator.calculate(df)
            except Exception as exc:  # noqa: BLE001
                raise IndicatorCalculationError(f"{indicator.name} failed: {exc}") from exc

            if isinstance(result, pd.Series):
                enriched[result.name] = result
            else:
                enriched = enriched.join(result)

        atr_col = f"atr_{self.settings.atr_period}"
        enriched["atr_pct"] = enriched[atr_col] / enriched["close"]
        return enriched
