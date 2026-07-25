from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import Settings
from services.indicator_engine import IndicatorEngine
from services.pattern_service import PatternService
from services.support_resistance_service import SupportResistanceService


def test_breakout_detected_on_volume_confirmed_new_high(settings: Settings):
    rng = np.random.default_rng(2)
    base = 100.0 + rng.normal(0, 0.3, 145)  # flat consolidation base
    jump = np.full(5, 110.0) + rng.normal(0, 0.1, 5)  # sudden breakout leg
    close = np.concatenate([base, jump])
    n = len(close)
    high, low = close + 0.3, close - 0.3
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.full(n, 1_000_000.0)
    volume[-5:] = 3_000_000.0  # volume spike confirming the breakout
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )

    enriched = IndicatorEngine(settings).compute(df)
    levels = SupportResistanceService(settings).evaluate(enriched)
    flags, signals = PatternService(settings).evaluate(enriched, levels)

    assert flags.breakout is True
    assert any("Breakout" in s for s in signals)


def test_consolidation_detected_after_a_volatility_squeeze(settings: Settings):
    rng = np.random.default_rng(3)
    volatile = 100 * np.cumprod(1 + rng.normal(0, 0.02, 100))
    tight = np.full(50, volatile[-1]) + rng.normal(0, 0.05, 50)
    close = np.concatenate([volatile, tight])
    n = len(close)
    high, low = close + 0.1, close - 0.1
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.full(n, 1_000_000.0)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )

    enriched = IndicatorEngine(settings).compute(df)
    levels = SupportResistanceService(settings).evaluate(enriched)
    flags, _ = PatternService(settings).evaluate(enriched, levels)

    assert flags.consolidation is True
