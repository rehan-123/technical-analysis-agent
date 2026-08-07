from __future__ import annotations

from models.analysis_result import (
    IndicatorSnapshot,
    PatternFlags,
    SupportResistanceLevels,
    TechnicalAnalysisResult,
)


def make_indicators(**overrides: float | None) -> IndicatorSnapshot:
    defaults: dict = dict(
        close=100.0,
        ema_20=98.0,
        ema_50=95.0,
        ema_200=90.0,
        sma=96.0,
        rsi=55.0,
        macd_line=0.5,
        macd_signal=0.3,
        macd_histogram=0.2,
        atr=2.0,
        atr_pct=0.02,
        bb_upper=105.0,
        bb_middle=100.0,
        bb_lower=95.0,
        bb_percent_b=0.5,
        bb_bandwidth=0.1,
        volume=1_000_000.0,
        volume_sma=900_000.0,
        relative_volume=1.1,
    )
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


def make_technical(
    *,
    ticker: str = "TEST",
    trend: str = "Neutral",
    strength: int = 50,
    confidence: int = 50,
    risk: str = "Medium",
    indicators: IndicatorSnapshot | None = None,
    support: list[float] | None = None,
    resistance: list[float] | None = None,
    breakout: bool = False,
    pullback: bool = False,
    trend_reversal: bool = False,
    consolidation: bool = False,
    high_volatility: bool = False,
    market_structure: dict | None = None,
    confluence: dict | None = None,
    volatility: dict | None = None,
    confidence_breakdown: dict | None = None,
) -> TechnicalAnalysisResult:
    """Build a fully valid ``TechnicalAnalysisResult`` with sensible defaults,
    overridable per-field. Used to give strategy unit tests exact, precise
    control over the inputs that drive each branch of a strategy's decision
    logic — independent of whatever a stochastic synthetic price series
    happens to produce on a given seed.
    """
    return TechnicalAnalysisResult(
        ticker=ticker,
        trend=trend,  # type: ignore[arg-type]
        strength=strength,
        signals=["synthetic test signal"],
        entry_zone=(99.0, 100.0),
        stop_loss=95.0,
        targets=[105.0, 110.0],
        risk=risk,  # type: ignore[arg-type]
        confidence=confidence,
        summary="synthetic test summary",
        indicators=indicators or make_indicators(),
        levels=SupportResistanceLevels(support=support or [], resistance=resistance or []),
        patterns=PatternFlags(
            breakout=breakout,
            pullback=pullback,
            trend_reversal=trend_reversal,
            consolidation=consolidation,
            high_volatility=high_volatility,
        ),
        market_structure=market_structure or {},
        confluence=confluence or {},
        volatility=volatility or {},
        confidence_breakdown=confidence_breakdown or {},
    )
