from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import Settings
from services.indicator_engine import IndicatorEngine
from services.support_resistance_service import SupportResistanceService


def test_detects_levels_around_a_ranging_market(settings: Settings):
    n = 200
    x = np.linspace(0, 8 * np.pi, n)
    close = 100 + 5 * np.sin(x)  # oscillates between ~95 and ~105, ends near 100
    high, low = close + 0.5, close - 0.5
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.full(n, 2_000_000.0)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )

    enriched = IndicatorEngine(settings).compute(df)
    levels = SupportResistanceService(settings).evaluate(enriched)

    assert len(levels.support) > 0
    assert len(levels.resistance) > 0
    current_price = enriched["close"].iloc[-1]
    assert all(lvl < current_price for lvl in levels.support)
    assert all(lvl > current_price for lvl in levels.resistance)
