from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate and normalize a raw OHLCV DataFrame from any data provider.

    Ensures required columns exist (case-insensitively), drops rows with
    missing close prices, and guarantees a sorted DatetimeIndex.

    Raises:
        ValueError: if the data is empty or missing required columns.
            Callers are expected to translate this into a domain-specific
            exception (e.g. ``DataFetchError``).
    """
    if df is None or df.empty:
        raise ValueError(f"No OHLCV data returned for '{ticker}'")

    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV data for '{ticker}' is missing columns: {missing}")

    df = df[list(REQUIRED_COLUMNS)]
    df = df.dropna(subset=["close"])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df
