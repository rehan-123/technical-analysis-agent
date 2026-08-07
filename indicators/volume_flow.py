"""Volume-flow indicators (pure, vectorized functions)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum().rename("obv")


def cmf(df: pd.DataFrame, period: int) -> pd.Series:
    """Chaikin Money Flow over ``period`` bars."""
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
    mf_volume = mf_multiplier * df["volume"]
    return (
        mf_volume.rolling(period).sum() / df["volume"].rolling(period).sum()
    ).rename("cmf")


def mfi(df: pd.DataFrame, period: int) -> pd.Series:
    """Money Flow Index (a volume-weighted RSI)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = typical * df["volume"]

    delta = typical.diff()
    positive_flow = raw_money_flow.where(delta > 0, 0.0)
    negative_flow = raw_money_flow.where(delta < 0, 0.0)

    pos = positive_flow.rolling(period).sum()
    neg = negative_flow.rolling(period).sum().replace(0, np.nan)

    money_ratio = pos / neg
    mfi_series = 100 - (100 / (1 + money_ratio))
    mfi_series = mfi_series.where(neg.notna(), 100.0)
    return mfi_series.rename("mfi")
