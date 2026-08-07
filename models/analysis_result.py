from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Trend = Literal["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"]
Risk = Literal["Low", "Medium", "High"]


class IndicatorSnapshot(BaseModel):
    """Latest computed indicator values, exposed for downstream agents /
    the Chief Decision Agent to reason over directly without recomputation.
    """

    close: float
    ema_20: float
    ema_50: float
    ema_200: float | None = None
    sma: float
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    atr: float
    atr_pct: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_percent_b: float
    bb_bandwidth: float
    volume: float
    volume_sma: float
    relative_volume: float


class SupportResistanceLevels(BaseModel):
    support: list[float] = Field(default_factory=list)
    resistance: list[float] = Field(default_factory=list)


class PatternFlags(BaseModel):
    breakout: bool = False
    pullback: bool = False
    trend_reversal: bool = False
    consolidation: bool = False
    high_volatility: bool = False


class AnalysisMetadata(BaseModel):
    execution_ms: float
    bars_analyzed: int
    period: str
    interval: str
    warnings: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    heuristic_components: list[str] = Field(default_factory=list)


class TechnicalAnalysisResult(BaseModel):
    """Canonical output contract of the Technical Analysis Agent.

    Top-level fields (`trend`, `strength`, `signals`, `entry_zone`,
    `stop_loss`, `targets`, `risk`, `confidence`, `summary`) mirror the
    minimal contract used by the platform and are preserved for backward
    compatibility. The remaining fields expose the full institutional
    detail — structured indicators, market structure, volume, volatility,
    SMC, confluence scoring, confidence breakdown, a risk plan, and a
    human-readable reasoning chain — for a Chief Decision Agent to
    cross-reference against News, Risk, Macro, and Options agents.
    """

    agent: str = "technical_analysis_agent"
    ticker: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Backward-compatible top-level contract ---
    trend: Trend
    strength: int = Field(ge=0, le=100)
    signals: list[str]

    entry_zone: tuple[float, float]
    stop_loss: float
    targets: list[float]

    risk: Risk
    confidence: int = Field(ge=0, le=100)
    summary: str

    indicators: IndicatorSnapshot
    levels: SupportResistanceLevels
    patterns: PatternFlags

    # --- Institutional extensions (additive; free-form dicts to avoid
    #     tightly coupling the contract to every engine's internal shape) ---
    indicator_suite: dict = Field(default_factory=dict)
    market_structure: dict = Field(default_factory=dict)
    volume_analysis: dict = Field(default_factory=dict)
    volatility: dict = Field(default_factory=dict)
    smc: dict = Field(default_factory=dict)
    confluence: dict = Field(default_factory=dict)
    confidence_breakdown: dict = Field(default_factory=dict)
    risk_plan: dict = Field(default_factory=dict)
    reasoning: list[str] = Field(default_factory=list)
    metadata: AnalysisMetadata | None = None
