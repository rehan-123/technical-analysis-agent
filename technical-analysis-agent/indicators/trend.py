"""Trend and overlay indicators (pure, vectorized functions).

Each function takes a validated OHLCV DataFrame and returns a Series or
DataFrame aligned to the input index. Functions never mutate the input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wma(close: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average with linearly increasing weights."""
    weights = np.arange(1, period + 1)
    return close.rolling(period).apply(
        lambda w: np.dot(w, weights) / weights.sum(), raw=True
    )


def vwma(df: pd.DataFrame, period: int) -> pd.Series:
    """Volume-Weighted Moving Average."""
    pv = df["close"] * df["volume"]
    return pv.rolling(period).sum() / df["volume"].rolling(period).sum()


def vwap(df: pd.DataFrame, period: int) -> pd.Series:
    """Rolling Volume-Weighted Average Price over ``period`` bars.

    Note: canonical VWAP anchors at the session open on intraday data. On
    daily bars a rolling-window VWAP is the meaningful analogue and is what
    is provided here; the window length is configurable.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    return pv.rolling(period).sum() / df["volume"].rolling(period).sum()


def donchian_channels(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Donchian Channels: highest high / lowest low / midline over ``period``."""
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({"dc_upper": upper, "dc_middle": middle, "dc_lower": lower})


def keltner_channels(
    df: pd.DataFrame, ema_period: int, atr_period: int, multiplier: float, atr: pd.Series
) -> pd.DataFrame:
    """Keltner Channels: EMA midline +/- ``multiplier`` * ATR."""
    middle = df["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    return pd.DataFrame({"kc_upper": upper, "kc_middle": middle, "kc_lower": lower})


def supertrend(df: pd.DataFrame, period: int, multiplier: float, atr: pd.Series) -> pd.DataFrame:
    """SuperTrend indicator.

    Returns a DataFrame with the SuperTrend line and its direction
    (+1 = bullish/price above line, -1 = bearish/price below line).
    """
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    close = df["close"]

    for i in range(1, len(df)):
        final_upper.iloc[i] = (
            upper_band.iloc[i]
            if (upper_band.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1])
            else final_upper.iloc[i - 1]
        )
        final_lower.iloc[i] = (
            lower_band.iloc[i]
            if (lower_band.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1])
            else final_lower.iloc[i - 1]
        )

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    direction.iloc[0] = 1
    st.iloc[0] = final_lower.iloc[0]

    for i in range(1, len(df)):
        prev_dir = direction.iloc[i - 1]
        if prev_dir == 1:
            direction.iloc[i] = -1 if close.iloc[i] < final_lower.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1
        st.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return pd.DataFrame({"supertrend": st, "supertrend_direction": direction})


def parabolic_sar(
    df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2
) -> pd.Series:
    """Parabolic SAR (Stop and Reverse)."""
    high, low = df["high"].values, df["low"].values
    n = len(df)
    sar = np.zeros(n)
    if n == 0:
        return pd.Series(sar, index=df.index)

    bull = True
    af = step
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if bull:
            sar[i] = min(sar[i], low[i - 1], low[max(i - 2, 0)])
            if low[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = low[i]
                af = step
            elif high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
        else:
            sar[i] = max(sar[i], high[i - 1], high[max(i - 2, 0)])
            if high[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = high[i]
                af = step
            elif low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)

    return pd.Series(sar, index=df.index, name="parabolic_sar")


def ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku Cloud components.

    tenkan (conversion), kijun (base), senkou span A/B (cloud), chikou (lag).
    Senkou spans are shifted forward by ``kijun`` bars per the standard.
    """

    def _mid(period: int) -> pd.Series:
        return (df["high"].rolling(period).max() + df["low"].rolling(period).min()) / 2

    tenkan_sen = _mid(tenkan)
    kijun_sen = _mid(kijun)
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_line = _mid(senkou_b).shift(kijun)
    chikou = df["close"].shift(-kijun)

    return pd.DataFrame(
        {
            "ichimoku_tenkan": tenkan_sen,
            "ichimoku_kijun": kijun_sen,
            "ichimoku_senkou_a": senkou_a,
            "ichimoku_senkou_b": senkou_b_line,
            "ichimoku_chikou": chikou,
        }
    )
