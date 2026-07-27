from __future__ import annotations

import time

import pandas as pd

from agent.base import BaseAgent
from config.settings import Settings, get_settings
from data.base import MarketDataProvider
from data.yfinance_provider import YFinanceProvider
from engines.candlestick import CandlestickEngine
from engines.confidence import ConfidenceEngine
from engines.confluence import ConfluenceEngine
from engines.explanation import ExplanationEngine
from engines.indicator_suite import IndicatorSuite
from engines.market_structure import MarketStructureEngine
from engines.risk import RiskEngine
from engines.smc import SMCEngine
from engines.volatility import VolatilityEngine
from engines.volume import VolumeEngine
from models.analysis_result import (
    AnalysisMetadata,
    IndicatorSnapshot,
    TechnicalAnalysisResult,
    Trend,
)
from services.indicator_engine import IndicatorEngine
from services.pattern_service import PatternService
from services.support_resistance_service import SupportResistanceService
from utils.exceptions import DataFetchError, InsufficientDataError
from utils.logger import get_logger
from validation.ohlcv_validator import OHLCVValidator

logger = get_logger(__name__)


def _dc(obj) -> dict:
    """Serialize a dataclass/pydantic/plain object to a JSON-safe dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj) if isinstance(obj, dict) else {"value": obj}


class TechnicalAnalysisAgent(BaseAgent):
    """Institutional-grade technical-analysis agent for a single ticker.

    Orchestrates the full engine stack — indicator suite, market structure,
    candlesticks, volume, volatility, SMC, confluence, confidence, risk, and
    explanation — behind one ``run()`` / ``analyze()`` call. The top-level
    result fields remain backward-compatible with the original contract; the
    richer engine outputs are additive so a Chief Decision Agent can consume
    exactly as much detail as it needs.
    """

    name = "technical_analysis_agent"

    def __init__(
        self,
        settings: Settings | None = None,
        data_provider: MarketDataProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.data_provider = data_provider or YFinanceProvider()
        self.validator = OHLCVValidator()

        # Descriptive base indicators (populate the snapshot + enriched frame).
        self.indicator_engine = IndicatorEngine(self.settings)
        self.support_resistance_service = SupportResistanceService(self.settings)
        self.pattern_service = PatternService(self.settings)

        # Institutional engine stack.
        self.suite = IndicatorSuite(self.settings)
        self.structure_engine = MarketStructureEngine(self.settings)
        self.candlestick_engine = CandlestickEngine(self.settings)
        self.volume_engine = VolumeEngine(self.settings)
        self.volatility_engine = VolatilityEngine(self.settings)
        self.smc_engine = SMCEngine(self.settings)
        self.confluence_engine = ConfluenceEngine(self.settings)
        self.confidence_engine = ConfidenceEngine(self.settings)
        self.risk_engine = RiskEngine(self.settings)
        self.explanation_engine = ExplanationEngine(self.settings)

    async def run(self, ticker: str, **kwargs) -> TechnicalAnalysisResult:
        return await self.analyze(
            ticker,
            period=kwargs.get("period", self.settings.default_period),
            interval=kwargs.get("interval", self.settings.default_interval),
        )

    async def analyze(
        self, ticker: str, period: str | None = None, interval: str | None = None
    ) -> TechnicalAnalysisResult:
        start = time.perf_counter()
        ticker = ticker.strip().upper()
        period = period or self.settings.default_period
        interval = interval or self.settings.default_interval
        logger.info("Analyzing %s (period=%s, interval=%s)", ticker, period, interval)

        raw_df = await self.data_provider.get_ohlcv(ticker, period, interval)

        report = self.validator.validate(raw_df, ticker)
        if not report.ok:
            raise DataFetchError(f"{ticker}: {'; '.join(report.errors)}")
        df = report.cleaned

        if len(df) < self.settings.min_bars_required:
            raise InsufficientDataError(
                f"Only {len(df)} bars available for '{ticker}', "
                f"need at least {self.settings.min_bars_required}"
            )

        enriched = self.indicator_engine.compute(df)
        warm_cols = [f"ema_{self.settings.ema_fast_period}", f"rsi_{self.settings.rsi_period}"]
        enriched = enriched.dropna(subset=warm_cols)
        if enriched.empty:
            raise InsufficientDataError(f"Indicators could not be computed for '{ticker}'")

        atr_pct = float(enriched.iloc[-1]["atr_pct"])

        # --- Run engines ---
        suite = self.suite.compute(enriched)
        structure = self.structure_engine.evaluate(enriched)
        candles = self.candlestick_engine.evaluate(enriched)
        volume = self.volume_engine.evaluate(enriched)
        volatility = self.volatility_engine.evaluate(enriched)
        smc = self.smc_engine.evaluate(enriched)
        levels = self.support_resistance_service.evaluate(enriched)
        patterns, pattern_signals = self.pattern_service.evaluate(enriched, levels)

        confluence = self.confluence_engine.evaluate(suite, structure, candles, volume, volatility, smc)
        confidence = self.confidence_engine.evaluate(confluence, volatility)

        # --- Derive ONE canonical bias, used consistently for the headline
        # trend label AND the risk plan direction. The headline trend is
        # anchored to the *primary* (slow) trend so a short-term
        # counter-trend bounce can't flip it on its own — it's blended
        # with, not replaced by, the richer confluence read. Deriving the
        # risk plan's direction from this same blended score (rather than
        # the unblended confluence.net_bias) matters: without it, a report
        # could say "trend: Bearish" while the risk plan quietly said "no
        # directional bias" because confluence alone hadn't cleared its
        # separation threshold — an internally inconsistent result.
        confluence_net = 50 + (confluence.bullish_score - confluence.bearish_score) / 2
        primary = self._primary_trend_score(enriched, structure)
        net_score = int(max(0, min(100, round(0.6 * primary + 0.4 * confluence_net))))
        trend = self._trend_label(net_score)
        bias = "bullish" if net_score >= 60 else "bearish" if net_score < 40 else "neutral"

        risk = self.risk_engine.evaluate(enriched, bias, levels, structure, atr_pct)
        summary, reasoning = self.explanation_engine.build(
            ticker, confluence, confidence, structure, candles, volume, volatility, smc, risk
        )

        signals = self._collect_signals(suite, structure, candles, volume, pattern_signals)
        snapshot = self._build_snapshot(enriched)

        execution_ms = round((time.perf_counter() - start) * 1000, 2)
        if self.settings.enable_timing_logs:
            logger.info("Analyzed %s in %.2f ms (%d bars)", ticker, execution_ms, len(enriched))

        metadata = AnalysisMetadata(
            execution_ms=execution_ms,
            bars_analyzed=len(enriched),
            period=period,
            interval=interval,
            warnings=report.warnings,
            validation_errors=report.errors,
            heuristic_components=["smc"],
        )

        return TechnicalAnalysisResult(
            ticker=ticker,
            trend=trend,
            strength=net_score,
            signals=signals,
            entry_zone=risk.entry_zone,
            stop_loss=risk.stop_loss,
            targets=risk.targets,
            risk=risk.risk_tier,
            confidence=confidence.confidence,
            summary=summary,
            indicators=snapshot,
            levels=levels,
            patterns=patterns,
            indicator_suite={k: v.model_dump() for k, v in suite.items()},
            market_structure=_dc(structure),
            volume_analysis=_dc(volume),
            volatility=_dc(volatility),
            smc=_dc(smc),
            confluence=_dc(confluence),
            confidence_breakdown=_dc(confidence),
            risk_plan=_dc(risk),
            reasoning=reasoning,
            metadata=metadata,
        )

    def _primary_trend_score(self, df: pd.DataFrame, structure) -> float:
        """A slow, hard-to-whipsaw directional score (0-100) from the
        primary trend: price vs. EMA200, EMA50 vs. EMA200, price vs. EMA20,
        and the market-structure read. This anchors the headline trend so a
        short-lived counter-trend bounce (which can flip several short-term
        indicators at once) can't on its own relabel a primary downtrend as
        bullish, or vice versa.
        """
        s = self.settings
        latest = df.iloc[-1]
        close = float(latest["close"])
        ema20 = float(latest[f"ema_{s.ema_fast_period}"])
        ema50 = float(latest[f"ema_{s.ema_medium_period}"])
        ema200 = latest.get(f"ema_{s.ema_long_period}")

        votes = 0
        total = 0
        votes += 1 if close > ema20 else -1
        total += 1
        votes += 1 if close > ema50 else -1
        total += 1
        if pd.notna(ema200):
            votes += 1 if close > ema200 else -1
            total += 1
            votes += 1 if ema50 > ema200 else -1
            total += 1
        if structure.structure == "uptrend":
            votes += 1
            total += 1
        elif structure.structure == "downtrend":
            votes -= 1
            total += 1

        lean = votes / total if total else 0.0  # [-1, 1]
        return 50 + lean * 50

    def _trend_label(self, score: int) -> Trend:
        if score >= 80:
            return "Strong Bullish"
        if score >= 60:
            return "Bullish"
        if score >= 40:
            return "Neutral"
        if score >= 20:
            return "Bearish"
        return "Strong Bearish"

    def _collect_signals(self, suite, structure, candles, volume, pattern_signals) -> list[str]:
        signals: list[str] = []
        for r in suite.values():
            if r.signal in ("bullish", "bearish"):
                signals.append(r.interpretation)
        if structure.structure != "undetermined":
            label = f"Structure {structure.structure}"
            if structure.break_of_structure:
                label += " (BOS)"
            if structure.change_of_character:
                label += " (CHoCH)"
            signals.append(label)
        signals.extend(candles.patterns)
        if volume.buying_pressure:
            signals.append("Buying pressure")
        elif volume.selling_pressure:
            signals.append("Selling pressure")
        signals.extend(pattern_signals)
        # De-dup preserving order.
        seen: set[str] = set()
        return [s for s in signals if not (s in seen or seen.add(s))]

    def _build_snapshot(self, df: pd.DataFrame) -> IndicatorSnapshot:
        s = self.settings
        latest = df.iloc[-1]
        ema_long_value = latest.get(f"ema_{s.ema_long_period}")
        return IndicatorSnapshot(
            close=round(float(latest["close"]), 2),
            ema_20=round(float(latest[f"ema_{s.ema_fast_period}"]), 2),
            ema_50=round(float(latest[f"ema_{s.ema_medium_period}"]), 2),
            ema_200=round(float(ema_long_value), 2) if pd.notna(ema_long_value) else None,
            sma=round(float(latest[f"sma_{s.sma_period}"]), 2),
            rsi=round(float(latest[f"rsi_{s.rsi_period}"]), 2),
            macd_line=round(float(latest["macd_line"]), 4),
            macd_signal=round(float(latest["macd_signal"]), 4),
            macd_histogram=round(float(latest["macd_histogram"]), 4),
            atr=round(float(latest[f"atr_{s.atr_period}"]), 2),
            atr_pct=round(float(latest["atr_pct"]), 4),
            bb_upper=round(float(latest["bb_upper"]), 2),
            bb_middle=round(float(latest["bb_middle"]), 2),
            bb_lower=round(float(latest["bb_lower"]), 2),
            bb_percent_b=round(float(latest["bb_percent_b"]), 4),
            bb_bandwidth=round(float(latest["bb_bandwidth"]), 4),
            volume=float(latest["volume"]),
            volume_sma=round(float(latest["volume_sma"]), 2),
            relative_volume=round(float(latest["relative_volume"]), 2),
        )
