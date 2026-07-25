from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import Settings
from indicators.volume_flow import cmf, mfi, obv


@dataclass
class VolumeProfile:
    point_of_control: float
    value_area_high: float
    value_area_low: float
    high_volume_nodes: list[float] = field(default_factory=list)
    low_volume_nodes: list[float] = field(default_factory=list)


@dataclass
class VolumeResult:
    relative_volume: float
    volume_trend: str  # increasing / decreasing / stable
    volume_spike: bool = False
    volume_climax: bool = False
    volume_dry_up: bool = False
    buying_pressure: bool = False
    selling_pressure: bool = False
    obv_trend: str = "flat"
    cmf: float = 0.0
    mfi: float = 0.0
    profile: VolumeProfile | None = None
    signal: str = "neutral"
    strength: int = 0
    notes: list[str] = field(default_factory=list)


class VolumeEngine:
    """Volume intelligence: participation, pressure, and a price-by-volume
    profile.

    Relative volume, spikes, climaxes, and dry-ups gauge participation.
    OBV slope, CMF, and MFI gauge whether that participation is net buying
    or selling. The volume profile bins traded volume by price to locate the
    Point of Control (most-traded price) and value area — approximated from
    OHLCV bars, since true profiles require intraday prints.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _volume_profile(self, df: pd.DataFrame) -> VolumeProfile:
        s = self.settings
        window = df.tail(s.sr_lookback_bars)
        typical = (window["high"] + window["low"] + window["close"]) / 3
        lo, hi = float(typical.min()), float(typical.max())
        if hi <= lo:
            poc = round(float(window["close"].iloc[-1]), 2)
            return VolumeProfile(poc, poc, poc)

        bins = np.linspace(lo, hi, s.volume_profile_bins + 1)
        idx = np.clip(np.digitize(typical.values, bins) - 1, 0, s.volume_profile_bins - 1)
        vol_by_bin = np.zeros(s.volume_profile_bins)
        for b, v in zip(idx, window["volume"].values):
            vol_by_bin[b] += v
        centers = (bins[:-1] + bins[1:]) / 2

        poc_bin = int(vol_by_bin.argmax())
        poc = float(centers[poc_bin])

        # Value area: expand out from POC until covering target % of volume.
        total = vol_by_bin.sum()
        target = total * s.volume_profile_value_area_pct
        included = {poc_bin}
        covered = vol_by_bin[poc_bin]
        lo_i = hi_i = poc_bin
        while covered < target and (lo_i > 0 or hi_i < s.volume_profile_bins - 1):
            left = vol_by_bin[lo_i - 1] if lo_i > 0 else -1
            right = vol_by_bin[hi_i + 1] if hi_i < s.volume_profile_bins - 1 else -1
            if right >= left:
                hi_i += 1
                included.add(hi_i)
                covered += vol_by_bin[hi_i]
            else:
                lo_i -= 1
                included.add(lo_i)
                covered += vol_by_bin[lo_i]

        va_high = float(centers[max(included)])
        va_low = float(centers[min(included)])

        mean_vol = vol_by_bin.mean()
        hvn = [round(float(centers[i]), 2) for i in range(len(centers)) if vol_by_bin[i] >= 1.5 * mean_vol]
        lvn = [round(float(centers[i]), 2) for i in range(len(centers)) if 0 < vol_by_bin[i] <= 0.5 * mean_vol]

        return VolumeProfile(
            point_of_control=round(poc, 2),
            value_area_high=round(va_high, 2),
            value_area_low=round(va_low, 2),
            high_volume_nodes=hvn[:5],
            low_volume_nodes=lvn[:5],
        )

    def evaluate(self, df: pd.DataFrame) -> VolumeResult:
        s = self.settings
        latest = df.iloc[-1]
        rvol = float(latest["relative_volume"]) if pd.notna(latest["relative_volume"]) else 1.0

        if rvol >= s.volume_trend_high:
            volume_trend = "increasing"
        elif rvol <= s.volume_trend_low:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"

        result = VolumeResult(relative_volume=round(rvol, 2), volume_trend=volume_trend)

        if rvol >= s.volume_spike_multiplier:
            result.volume_spike = True
            result.notes.append("Volume spike vs. average")

        price_change = abs(latest["close"] - df["close"].iloc[-2]) / df["close"].iloc[-2] if len(df) > 1 else 0
        if result.volume_spike and price_change < 0.005:
            result.volume_climax = True
            result.notes.append("Volume climax: heavy volume, little price progress")
        if rvol <= 0.5:
            result.volume_dry_up = True
            result.notes.append("Volume dry-up: participation unusually low")

        cmf_val = float(cmf(df, s.cmf_period).iloc[-1])
        mfi_val = float(mfi(df, s.mfi_period).iloc[-1])
        obv_series = obv(df)
        obv_slope = float(obv_series.iloc[-1] - obv_series.iloc[-min(len(obv_series), 10)])
        result.cmf = round(cmf_val, 4)
        result.mfi = round(mfi_val, 2)
        result.obv_trend = "rising" if obv_slope > 0 else "falling" if obv_slope < 0 else "flat"

        if cmf_val > 0.05 and obv_slope > 0:
            result.buying_pressure = True
            result.signal = "bullish"
        elif cmf_val < -0.05 and obv_slope < 0:
            result.selling_pressure = True
            result.signal = "bearish"

        result.profile = self._volume_profile(df)

        strength = 0
        if result.buying_pressure or result.selling_pressure:
            strength += 40
        if result.volume_spike:
            strength += 20
        if mfi_val >= s.mfi_overbought or mfi_val <= s.mfi_oversold:
            strength += 15
        result.strength = min(100, strength)
        return result
