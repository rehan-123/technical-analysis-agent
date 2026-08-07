"""Pivot point levels (classic floor-trader method)."""
from __future__ import annotations

import pandas as pd


def classic_pivots(df: pd.DataFrame) -> dict[str, float]:
    """Classic pivot points computed from the most recent completed bar.

    Returns pivot (P), three resistance levels (R1–R3) and three support
    levels (S1–S3). Uses the last fully-formed bar as the reference period.
    """
    last = df.iloc[-1]
    high, low, close = float(last["high"]), float(last["low"]), float(last["close"])

    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    return {
        "pivot": round(pivot, 4),
        "r1": round(r1, 4),
        "r2": round(r2, 4),
        "r3": round(r3, 4),
        "s1": round(s1, 4),
        "s2": round(s2, 4),
        "s3": round(s3, 4),
    }
