"""Momentum indicators (pure, vectorized functions)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def roc(close: pd.Series, period: int) -> pd.Series:
    """Rate of Change as a percentage."""
    return (close / close.shift(period) - 1) * 100


def momentum(close: pd.Series, period: int) -> pd.Series:
    """Absolute price momentum (close minus close ``period`` bars ago)."""
    return close - close.shift(period)


def cci(df: pd.DataFrame, period: int) -> pd.Series:
    """Commodity Channel Index."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical.rolling(period).mean()
    mean_dev = typical.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (typical - sma) / (0.015 * mean_dev)


def stoch_rsi(
    rsi_series: pd.Series, stoch_period: int, smooth_k: int, smooth_d: int
) -> pd.DataFrame:
    """Stochastic RSI (%K and %D), derived from an existing RSI series."""
    lowest = rsi_series.rolling(stoch_period).min()
    highest = rsi_series.rolling(stoch_period).max()
    stoch = (rsi_series - lowest) / (highest - lowest)
    k = (stoch * 100).rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return pd.DataFrame({"stochrsi_k": k, "stochrsi_d": d})


def adx(df: pd.DataFrame, period: int, atr: pd.Series) -> pd.DataFrame:
    """Average Directional Index with +DI and -DI (Wilder)."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr_safe = atr.replace(0, np.nan)
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_safe
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_safe
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})
