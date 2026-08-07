from __future__ import annotations

import pandas as pd

from config.settings import Settings
from indicators import momentum as mom
from indicators import trend as trd
from indicators import volume_flow as vf
from indicators.pivots import classic_pivots
from indicators.rsi import RSIIndicator
from models.indicator_result import IndicatorResult
from utils.compute_cache import ComputeCache


def _sig(bullish: bool, bearish: bool) -> str:
    return "bullish" if bullish else "bearish" if bearish else "neutral"


class IndicatorSuite:
    """Computes the full indicator set on an enriched frame and returns a
    dict of ``IndicatorResult`` — the uniform, self-describing contract the
    confluence and confidence engines consume.

    Uses a shared ``ComputeCache`` so base series (EMA, ATR, typical price)
    are computed once and reused across indicators.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def compute(self, df: pd.DataFrame) -> dict[str, IndicatorResult]:
        s = self.settings
        cache = ComputeCache(df)
        latest = df.iloc[-1]
        close = float(latest["close"])
        results: dict[str, IndicatorResult] = {}

        def add(r: IndicatorResult) -> None:
            results[r.name] = r

        # --- Moving-average stack (from enriched cols) ---
        ema20, ema50 = float(latest[f"ema_{s.ema_fast_period}"]), float(latest[f"ema_{s.ema_medium_period}"])
        ema200 = latest.get(f"ema_{s.ema_long_period}")
        add(IndicatorResult(
            name="ema_stack",
            value={"ema20": round(ema20, 2), "ema50": round(ema50, 2),
                   "ema200": round(float(ema200), 2) if pd.notna(ema200) else None},
            signal=_sig(ema20 > ema50, ema20 < ema50),
            strength=70 if abs(ema20 - ema50) / close > 0.01 else 45,
            interpretation=("Fast EMA above slow EMA (bullish stack)" if ema20 > ema50
                            else "Fast EMA below slow EMA (bearish stack)"),
        ))

        # --- WMA / VWMA / VWAP ---
        wma = float(trd.wma(df["close"], s.wma_period).iloc[-1])
        add(IndicatorResult(name="wma", value=round(wma, 2), signal=_sig(close > wma, close < wma),
                            strength=55, interpretation="Price above WMA" if close > wma else "Price below WMA"))
        vwma = float(trd.vwma(df, s.vwma_period).iloc[-1])
        add(IndicatorResult(name="vwma", value=round(vwma, 2), signal=_sig(close > vwma, close < vwma),
                            strength=55, interpretation="Price above VWMA" if close > vwma else "Price below VWMA"))
        vwap = float(trd.vwap(df, s.vwap_period).iloc[-1])
        add(IndicatorResult(name="vwap", value=round(vwap, 2), signal=_sig(close > vwap, close < vwap),
                            strength=60, interpretation="Trading above rolling VWAP" if close > vwap else "Trading below rolling VWAP"))

        # --- RSI ---
        rsi_series = RSIIndicator(s.rsi_period).calculate(df)
        rsi = float(rsi_series.iloc[-1])
        rsi_sig = "bullish" if rsi < s.rsi_oversold else "bearish" if rsi > s.rsi_overbought else ("bullish" if rsi >= 55 else "bearish" if rsi <= 45 else "neutral")
        add(IndicatorResult(name="rsi", value=round(rsi, 2), signal=rsi_sig,
                            strength=min(100, int(abs(rsi - 50) * 2)),
                            interpretation=f"RSI {rsi:.0f} ({'oversold' if rsi < s.rsi_oversold else 'overbought' if rsi > s.rsi_overbought else 'neutral zone'})"))

        # --- Stochastic RSI ---
        srsi = mom.stoch_rsi(rsi_series, s.stochrsi_period, s.stochrsi_smooth_k, s.stochrsi_smooth_d)
        k, d = srsi["stochrsi_k"].iloc[-1], srsi["stochrsi_d"].iloc[-1]
        if pd.notna(k) and pd.notna(d):
            add(IndicatorResult(name="stoch_rsi", value={"k": round(float(k), 2), "d": round(float(d), 2)},
                                signal=_sig(k > d and k < 80, k < d and k > 20), strength=50,
                                interpretation=f"StochRSI %K {k:.0f}/%D {d:.0f}"))

        # --- MACD (from enriched cols) ---
        macd_line, macd_sig, macd_hist = float(latest["macd_line"]), float(latest["macd_signal"]), float(latest["macd_histogram"])
        add(IndicatorResult(name="macd", value={"line": round(macd_line, 4), "signal": round(macd_sig, 4), "histogram": round(macd_hist, 4)},
                            signal=_sig(macd_hist > 0, macd_hist < 0), strength=min(100, int(abs(macd_hist) / close * 5000)),
                            interpretation="MACD histogram positive" if macd_hist > 0 else "MACD histogram negative"))

        # --- ADX / DI ---
        atr = cache.atr(s.adx_period)
        adx_df = mom.adx(df, s.adx_period, atr)
        adx_v, pdi, mdi = adx_df["adx"].iloc[-1], adx_df["plus_di"].iloc[-1], adx_df["minus_di"].iloc[-1]
        if pd.notna(adx_v):
            trending = adx_v >= s.adx_trend_threshold
            add(IndicatorResult(name="adx", value={"adx": round(float(adx_v), 2), "plus_di": round(float(pdi), 2), "minus_di": round(float(mdi), 2)},
                                signal=_sig(trending and pdi > mdi, trending and mdi > pdi),
                                strength=min(100, int(adx_v)),
                                interpretation=f"ADX {adx_v:.0f} ({'trending' if trending else 'weak/ranging'}), {'+DI leads' if pdi > mdi else '-DI leads'}"))

        # --- CCI / ROC / Momentum ---
        cci = float(mom.cci(df, s.cci_period).iloc[-1])
        add(IndicatorResult(name="cci", value=round(cci, 2),
                            signal=_sig(cci > 0, cci < 0), strength=min(100, int(abs(cci) / 2)),
                            interpretation=f"CCI {cci:.0f}"))
        roc = float(mom.roc(df["close"], s.roc_period).iloc[-1])
        add(IndicatorResult(name="roc", value=round(roc, 2), signal=_sig(roc > 0, roc < 0),
                            strength=min(100, int(abs(roc) * 5)), interpretation=f"ROC {roc:+.1f}%"))
        mmt = float(mom.momentum(df["close"], s.momentum_period).iloc[-1])
        add(IndicatorResult(name="momentum", value=round(mmt, 2), signal=_sig(mmt > 0, mmt < 0),
                            strength=50, interpretation=f"Momentum {mmt:+.2f}"))

        # --- Volume flow: MFI / OBV / CMF ---
        mfi = float(vf.mfi(df, s.mfi_period).iloc[-1])
        add(IndicatorResult(name="mfi", value=round(mfi, 2),
                            signal="bullish" if mfi <= s.mfi_oversold else "bearish" if mfi >= s.mfi_overbought else _sig(mfi >= 55, mfi <= 45),
                            strength=min(100, int(abs(mfi - 50) * 2)), interpretation=f"MFI {mfi:.0f}"))
        cmf = float(vf.cmf(df, s.cmf_period).iloc[-1])
        add(IndicatorResult(name="cmf", value=round(cmf, 4), signal=_sig(cmf > 0.05, cmf < -0.05),
                            strength=min(100, int(abs(cmf) * 200)), interpretation=f"CMF {cmf:+.2f} ({'accumulation' if cmf > 0 else 'distribution'})"))
        obv_series = vf.obv(df)
        obv_slope = float(obv_series.iloc[-1] - obv_series.iloc[-min(len(obv_series), 10)])
        add(IndicatorResult(name="obv", value=round(float(obv_series.iloc[-1]), 2), signal=_sig(obv_slope > 0, obv_slope < 0),
                            strength=55, interpretation="OBV rising" if obv_slope > 0 else "OBV falling"))

        # --- Bollinger (from enriched cols) ---
        pct_b = float(latest["bb_percent_b"])
        add(IndicatorResult(name="bollinger", value={"upper": round(float(latest["bb_upper"]), 2), "middle": round(float(latest["bb_middle"]), 2),
                                                     "lower": round(float(latest["bb_lower"]), 2), "percent_b": round(pct_b, 3)},
                            signal="bearish" if pct_b > 1 else "bullish" if pct_b < 0 else "neutral",
                            strength=60 if (pct_b > 1 or pct_b < 0) else 30,
                            interpretation=f"%B {pct_b:.2f} ({'above upper band' if pct_b > 1 else 'below lower band' if pct_b < 0 else 'within bands'})"))

        # --- Keltner / Donchian ---
        kc = trd.keltner_channels(df, s.keltner_ema_period, s.keltner_atr_period, s.keltner_multiplier, cache.atr(s.keltner_atr_period))
        kc_u, kc_l = float(kc["kc_upper"].iloc[-1]), float(kc["kc_lower"].iloc[-1])
        add(IndicatorResult(name="keltner", value={"upper": round(kc_u, 2), "lower": round(kc_l, 2)},
                            signal=_sig(close > kc_u, close < kc_l), strength=55,
                            interpretation="Above Keltner upper" if close > kc_u else "Below Keltner lower" if close < kc_l else "Inside Keltner channel"))
        dc = trd.donchian_channels(df, s.donchian_period)
        dc_u, dc_l = float(dc["dc_upper"].iloc[-1]), float(dc["dc_lower"].iloc[-1])
        add(IndicatorResult(name="donchian", value={"upper": round(dc_u, 2), "lower": round(dc_l, 2)},
                            signal=_sig(close >= dc_u, close <= dc_l), strength=60,
                            interpretation="At Donchian high (breakout)" if close >= dc_u else "At Donchian low (breakdown)" if close <= dc_l else "Mid Donchian range"))

        # --- SuperTrend ---
        st = trd.supertrend(df, s.supertrend_period, s.supertrend_multiplier, cache.atr(s.supertrend_period))
        st_dir = float(st["supertrend_direction"].iloc[-1])
        add(IndicatorResult(name="supertrend", value=round(float(st["supertrend"].iloc[-1]), 2),
                            signal=_sig(st_dir > 0, st_dir < 0), strength=70,
                            interpretation="SuperTrend bullish" if st_dir > 0 else "SuperTrend bearish"))

        # --- Parabolic SAR ---
        psar = float(trd.parabolic_sar(df, s.psar_step, s.psar_max_step).iloc[-1])
        add(IndicatorResult(name="parabolic_sar", value=round(psar, 2), signal=_sig(close > psar, close < psar),
                            strength=55, interpretation="Price above PSAR (uptrend)" if close > psar else "Price below PSAR (downtrend)"))

        # --- Ichimoku ---
        ich = trd.ichimoku(df, s.ichimoku_tenkan, s.ichimoku_kijun, s.ichimoku_senkou_b)
        span_a, span_b = ich["ichimoku_senkou_a"].iloc[-1], ich["ichimoku_senkou_b"].iloc[-1]
        if pd.notna(span_a) and pd.notna(span_b):
            cloud_top, cloud_bottom = max(span_a, span_b), min(span_a, span_b)
            ich_sig = _sig(close > cloud_top, close < cloud_bottom)
            add(IndicatorResult(name="ichimoku", value={"tenkan": round(float(ich["ichimoku_tenkan"].iloc[-1]), 2), "kijun": round(float(ich["ichimoku_kijun"].iloc[-1]), 2),
                                                        "senkou_a": round(float(span_a), 2), "senkou_b": round(float(span_b), 2)},
                                signal=ich_sig, strength=65,
                                interpretation="Price above the cloud (bullish)" if ich_sig == "bullish" else "Price below the cloud (bearish)" if ich_sig == "bearish" else "Price inside the cloud (indecision)"))

        # --- Pivot points (levels, directionally neutral) ---
        piv = classic_pivots(df)
        add(IndicatorResult(name="pivots", value=piv, signal="neutral", strength=0,
                            interpretation=f"Pivot {piv['pivot']}, R1 {piv['r1']}, S1 {piv['s1']}"))

        return results
