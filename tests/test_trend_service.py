from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from services.indicator_engine import IndicatorEngine
from services.trend_service import TrendService


def _make_trending_df(direction: str, n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(0 if direction == "up" else 1)
    drift = 0.0015 if direction == "up" else -0.0015
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    high, low = close * 1.005, close * 0.995
    open_ = np.roll(close, 1)
    open_[0] = 100
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


def test_uptrend_is_classified_bullish(settings: Settings):
    df = _make_trending_df("up")
    enriched = IndicatorEngine(settings).compute(df)
    trend, strength, signals = TrendService(settings).evaluate(enriched)

    assert trend in ("Bullish", "Strong Bullish")
    assert strength >= 60
    assert any("EMA" in s for s in signals)


def test_downtrend_is_classified_bearish(settings: Settings):
    df = _make_trending_df("down")
    enriched = IndicatorEngine(settings).compute(df)
    trend, strength, _ = TrendService(settings).evaluate(enriched)

    assert trend in ("Bearish", "Strong Bearish")
    assert strength <= 40
