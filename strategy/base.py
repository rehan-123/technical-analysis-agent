from __future__ import annotations

from abc import ABC, abstractmethod

from config.settings import Settings, get_settings
from models.analysis_result import TechnicalAnalysisResult
from models.strategy import StrategyName, StrategySignal


def clamp_score(value: float) -> int:
    """Clamp and round a raw score into the [0, 100] integer range every
    strategy output is contracted to use."""
    return int(max(0, min(100, round(value))))


# ---------------------------------------------------------------------------
# Shared accessors into TechnicalAnalysisResult's dict-shaped engine outputs.
#
# The institutional engine fields (`market_structure`, `confluence`,
# `volatility`, `confidence_breakdown`, `risk_plan`) are deliberately typed as
# free-form dicts on the contract (see models/analysis_result.py) so the
# Technical Engine's internal dataclasses don't leak into the public schema.
# Centralizing the key names used to read them here — rather than repeating
# `.get(...)` calls with string literals in five separate strategy modules —
# means a single place tracks the sub-schema every strategy relies on.
# ---------------------------------------------------------------------------


def structure_of(technical: TechnicalAnalysisResult) -> dict:
    """``MarketStructureResult`` fields: structure, last_label,
    break_of_structure, change_of_character, signal, swing_highs, swing_lows."""
    return technical.market_structure or {}


def confluence_of(technical: TechnicalAnalysisResult) -> dict:
    """``ConfluenceResult`` fields: bullish_score, bearish_score, net_bias."""
    return technical.confluence or {}


def volatility_of(technical: TechnicalAnalysisResult) -> dict:
    """``VolatilityResult`` fields: regime, atr_expansion, atr_compression,
    bollinger_squeeze, breakout_probability, trend_exhaustion, signal."""
    return technical.volatility or {}


def confidence_of(technical: TechnicalAnalysisResult) -> dict:
    """``ConfidenceResult`` fields: confidence, directional_agreement,
    dominant_side, components, caveats."""
    return technical.confidence_breakdown or {}


class Strategy(ABC):
    """Extension seam for one reusable trading strategy.

    Mirrors ``services.prompt_sections.base.SectionRenderer``: a single
    abstract method, a class-level identity attribute, and a narrow
    responsibility boundary. Every strategy is a **pure function** of an
    already-computed ``TechnicalAnalysisResult`` —

      * it fetches no market data itself (the Market Scanner already called
        the Technical Agent for that),
      * it computes no indicators of its own (it reads the already-computed
        ``indicators`` / ``market_structure`` / ``confluence`` / ``volatility``
        / ``patterns`` / ``levels`` fields),
      * it performs no I/O, no randomness, and no wall-clock-dependent logic
        beyond stamping the result timestamp,

    so identical input always produces an identical ``StrategySignal`` —
    the deterministic-output contract every strategy must satisfy.

    Because every implementation returns the same ``StrategySignal`` shape,
    strategies are interchangeable: the Strategy Engine, the Ranking Engine,
    and the ``/strategy`` API can treat any of them uniformly without knowing
    which concrete strategy produced a given signal.
    """

    #: Registry key / wire identity. Concrete strategies override this.
    name: StrategyName
    #: One-line, human-readable description surfaced by ``GET /strategy``.
    description: str = ""

    def __init__(self, settings: Settings | None = None) -> None:
        """``settings`` is injected, exactly like every indicator/engine in
        this platform, so a strategy's thresholds (e.g. ``rsi_overbought``,
        ``pullback_rsi_low``) are reused from the single centralized
        ``Settings`` surface rather than duplicated as new magic numbers."""
        self.settings = settings or get_settings()

    @abstractmethod
    def evaluate(self, technical: TechnicalAnalysisResult) -> StrategySignal:
        """Evaluate this strategy's setup against ``technical`` and return a
        fully-populated ``StrategySignal`` for ``technical.ticker``.

        Must never raise for well-formed input: an unmet setup is expressed
        as ``applicable=False`` with a low score, not an exception, so the
        Strategy Engine can evaluate every strategy against every symbol
        uninterrupted.
        """
        raise NotImplementedError
