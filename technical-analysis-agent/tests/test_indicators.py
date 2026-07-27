from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.atr import ATRIndicator
from indicators.bollinger import BollingerBandsIndicator
from indicators.macd import MACDIndicator
from indicators.moving_averages import EMAIndicator, SMAIndicator
from indicators.rsi import RSIIndicator
from indicators.volume import VolumeIndicator


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 300
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    high = close * 1.01
    low = close * 0.99
    open_ = np.roll(close, 1)
    open_[0] = 100
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


def test_sma_matches_pandas_rolling_mean(sample_df):
    result = SMAIndicator(20).calculate(sample_df)
    expected = sample_df["close"].rolling(20).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_ema_is_bounded_by_recent_prices(sample_df):
    result = EMAIndicator(20).calculate(sample_df)
    tail = sample_df["close"].tail(60)
    assert result.dropna().iloc[-1] <= tail.max() * 1.001
    assert result.dropna().iloc[-1] >= tail.min() * 0.999


def test_rsi_is_within_0_100(sample_df):
    result = RSIIndicator(14).calculate(sample_df)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_hits_100_on_pure_uptrend():
    close = pd.Series(np.linspace(100, 200, 40))
    df = pd.DataFrame({"close": close})
    rsi = RSIIndicator(14).calculate(df)
    assert rsi.dropna().iloc[-1] == pytest.approx(100.0)


def test_macd_histogram_equals_line_minus_signal(sample_df):
    result = MACDIndicator(12, 26, 9).calculate(sample_df)
    diff = (result["macd_line"] - result["macd_signal"]) - result["macd_histogram"]
    assert np.allclose(diff.dropna(), 0.0, atol=1e-9)


def test_atr_is_non_negative(sample_df):
    result = ATRIndicator(14).calculate(sample_df)
    assert (result.dropna() >= 0).all()


def test_bollinger_upper_above_lower(sample_df):
    result = BollingerBandsIndicator(20, 2.0).calculate(sample_df)
    valid = result.dropna()
    assert (valid["bb_upper"] >= valid["bb_middle"]).all()
    assert (valid["bb_middle"] >= valid["bb_lower"]).all()


def test_volume_relative_volume_around_one_on_average(sample_df):
    result = VolumeIndicator(20).calculate(sample_df)
    assert result["relative_volume"].dropna().mean() == pytest.approx(1.0, abs=0.2)
