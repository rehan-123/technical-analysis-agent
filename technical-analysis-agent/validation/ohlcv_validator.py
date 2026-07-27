from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class ValidationReport:
    """Outcome of validating a raw OHLCV frame.

    ``errors`` are fatal (analysis should not proceed); ``warnings`` are
    recoverable issues the frame was auto-repaired from, surfaced in the
    response metadata for transparency.
    """

    cleaned: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class OHLCVValidator:
    """Validates and repairs raw OHLCV data before indicators run.

    Checks, in order: presence/emptiness, required columns, negative or
    zero prices, high/low coherence, NaNs, duplicate timestamps, and a
    monotonic (sorted, de-duplicated) DatetimeIndex. Recoverable problems
    are repaired and recorded as warnings; unrecoverable ones become errors.
    """

    def validate(self, df: pd.DataFrame, ticker: str) -> ValidationReport:
        report = ValidationReport(cleaned=pd.DataFrame())

        if df is None or len(df) == 0:
            report.errors.append(f"No OHLCV data returned for '{ticker}'")
            return report

        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            report.errors.append(f"Missing required columns: {missing}")
            return report

        df = df[list(REQUIRED_COLUMNS)]

        # Timestamps
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:  # noqa: BLE001
            report.errors.append("Index could not be parsed as timestamps")
            return report

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            report.warnings.append("Timestamps were not sorted; auto-sorted ascending")

        dupes = int(df.index.duplicated().sum())
        if dupes:
            df = df[~df.index.duplicated(keep="last")]
            report.warnings.append(f"Removed {dupes} duplicate timestamp(s)")

        # NaNs in price columns
        price_cols = ["open", "high", "low", "close"]
        nan_rows = int(df[price_cols].isna().any(axis=1).sum())
        if nan_rows:
            df = df.dropna(subset=price_cols)
            report.warnings.append(f"Dropped {nan_rows} row(s) with NaN prices")

        # Negative / zero prices
        neg = int((df[price_cols] <= 0).any(axis=1).sum())
        if neg:
            df = df[~(df[price_cols] <= 0).any(axis=1)]
            report.warnings.append(f"Dropped {neg} row(s) with non-positive prices")

        # High/low coherence
        incoherent = int(
            ((df["high"] < df["low"]) | (df["high"] < df["close"]) | (df["low"] > df["close"])).sum()
        )
        if incoherent:
            mask = (df["high"] >= df["low"]) & (df["high"] >= df["close"]) & (df["low"] <= df["close"])
            df = df[mask]
            report.warnings.append(f"Dropped {incoherent} row(s) with incoherent OHLC bounds")

        # Fill NaN volume with 0 (non-fatal)
        if df["volume"].isna().any():
            df["volume"] = df["volume"].fillna(0.0)
            report.warnings.append("Filled NaN volume with 0")

        if len(df) == 0:
            report.errors.append("No valid rows remain after cleaning")
            return report

        report.cleaned = df
        return report
