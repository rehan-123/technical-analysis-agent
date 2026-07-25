from __future__ import annotations

import pandas as pd

from config.settings import Settings
from models.analysis_result import PatternFlags, SupportResistanceLevels


class PatternService:
    """Detects higher-level structural patterns on top of the raw
    indicators and levels: breakouts, pullbacks, trend reversals,
    consolidation, and high-volatility regimes.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _col(prefix: str, period: int) -> str:
        return f"{prefix}_{period}"

    def evaluate(
        self, df: pd.DataFrame, levels: SupportResistanceLevels
    ) -> tuple[PatternFlags, list[str]]:
        s = self.settings
        latest = df.iloc[-1]
        signals: list[str] = []
        flags = PatternFlags()

        close = latest["close"]
        volume = latest["volume"]
        volume_sma = latest["volume_sma"]
        rsi = latest[self._col("rsi", s.rsi_period)]
        ema_fast = latest[self._col("ema", s.ema_fast_period)]
        ema_mid = latest[self._col("ema", s.ema_medium_period)]

        # --- Breakout: close exceeds the highest high of the prior lookback
        # window (excluding a small recent buffer, so a breakout leg doesn't
        # inflate its own reference point), confirmed by above-average volume.
        volume_confirmed = pd.notna(volume_sma) and volume > volume_sma * s.volume_spike_multiplier
        exclude_recent = s.reversal_lookback_bars
        window = df["high"].iloc[-(s.sr_lookback_bars + exclude_recent + 1) : -(exclude_recent + 1)]
        prior_high = window.max() if len(window) > 0 else None

        if prior_high is not None and pd.notna(prior_high) and volume_confirmed:
            if close > prior_high * (1 + s.breakout_buffer_pct):
                flags.breakout = True
                signals.append(f"Breakout above {s.sr_lookback_bars}-bar high ({round(float(prior_high), 2)})")

        # --- Pullback: retracement toward the fast EMA within an uptrend ---
        if ema_fast > ema_mid and s.pullback_rsi_low <= rsi <= s.pullback_rsi_high:
            near_fast_ema = abs(close - ema_fast) / ema_fast <= s.entry_zone_buffer_pct * 2
            if near_fast_ema:
                flags.pullback = True
                signals.append(f"Pullback toward EMA{s.ema_fast_period}")

        # --- Trend reversal: fast/mid EMA cross within the recent lookback ---
        recent = df.tail(s.reversal_lookback_bars + 1)
        fast_col, mid_col = self._col("ema", s.ema_fast_period), self._col("ema", s.ema_medium_period)
        diff = recent[fast_col] - recent[mid_col]
        if len(diff) > 1 and (diff.iloc[:-1] * diff.iloc[-1] < 0).any():
            direction = "Bullish" if diff.iloc[-1] > 0 else "Bearish"
            flags.trend_reversal = True
            signals.append(f"{direction} EMA cross (trend reversal)")

        # --- Consolidation: Bollinger bandwidth near its own recent lows ---
        bandwidth_history = df["bb_bandwidth"].tail(s.sr_lookback_bars).dropna()
        if len(bandwidth_history) >= 20:
            percentile_rank = (bandwidth_history <= bandwidth_history.iloc[-1]).mean()
            if percentile_rank <= s.consolidation_bandwidth_percentile:
                flags.consolidation = True
                signals.append("Price consolidating (Bollinger squeeze)")

        # --- High volatility: ATR% expanded vs. its own recent average ---
        atr_pct_history = df["atr_pct"].tail(s.sr_lookback_bars).dropna()
        if len(atr_pct_history) >= 20:
            atr_pct_avg = atr_pct_history.mean()
            if atr_pct_avg and latest["atr_pct"] > atr_pct_avg * s.high_volatility_atr_multiplier:
                flags.high_volatility = True
                signals.append("High volatility (ATR expansion)")

        return flags, signals
