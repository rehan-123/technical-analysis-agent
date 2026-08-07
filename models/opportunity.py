from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from models.analysis_result import Risk, Trend
from models.strategy import StrategyDirection, StrategyName
from portfolio.position_sizer import PositionSizeResult

# ---------------------------------------------------------------------------
# Opportunity — the ONE canonical output of the Market Scanner / Strategy
# milestone. Every future milestone that needs to add scanner-derived fields
# extends this model (new optional fields with defaults); it is never
# replaced, per the architecture brief.
# ---------------------------------------------------------------------------


class Opportunity(BaseModel):
    """One ranked, actionable candidate produced by the Market Scanner.

    Assembled by ``MarketScannerService`` from already-computed specialist
    results — it performs no technical calculation, no news analysis, and no
    AI reasoning of its own; every score here is either copied straight from
    an upstream agent's result (``trend``, ``risk``, ``entry_zone``,
    ``stop_loss``, ``targets`` all come verbatim from the Technical Agent) or
    derived by a small, deterministic, documented heuristic that lives in the
    ``scanner`` package (``news_score``, ``portfolio_score``,
    ``combined_score``).

    Score semantics (0-100 throughout):
      * ``technical_score``   — the Technical Agent's own read (from
        ``TechnicalAnalysisResult.strength``/``confidence``).
      * ``news_score``        — a deterministic recency/coverage heuristic
        over the News Agent's articles. NOT sentiment: the News Agent
        performs zero sentiment analysis, and neither does the Scanner.
        Neutral (50) when news was not fetched or was unavailable.
      * ``portfolio_score``   — how well this candidate fits the active
        portfolio's constraints (headroom, concentration, cash), from
        ``PortfolioService.build_context``. Neutral (50) when no portfolio
        is configured.
      * ``opportunity_score`` — the *matched strategy's own* signal strength
        (``StrategySignal.score``) — how strong this specific strategy's
        setup is, independent of the other three domains.
      * ``combined_score``    — the weighted blend of all four
        (``config.settings.scanner_weight_*``); the Ranking Engine's primary
        sort key.
      * ``confidence``        — blended conviction (strategy + technical),
        distinct from ``combined_score`` (a ranking measure) — a signal can
        rank low but still be reported with a confident read, or vice versa.

    Frozen, like every other domain value object in the platform. The
    Ranking Engine assigns ``ranking`` via ``model_copy(update=...)`` after
    sorting a batch — the same "immutable, copy to update" pattern
    ``PortfolioManager`` already uses for ``Portfolio``.
    """

    model_config = ConfigDict(frozen=True)

    agent: str = "market_scanner"
    ticker: str
    exchange: str = "UNKNOWN"
    strategy: StrategyName
    direction: StrategyDirection = StrategyDirection.NONE

    opportunity_score: int = Field(..., ge=0, le=100)
    technical_score: int = Field(..., ge=0, le=100)
    news_score: int = Field(..., ge=0, le=100)
    portfolio_score: int = Field(..., ge=0, le=100)
    combined_score: int = Field(..., ge=0, le=100)

    confidence: int = Field(..., ge=0, le=100)
    risk: Risk
    trend: Trend

    entry_zone: tuple[float, float]
    stop_loss: float
    targets: list[float] = Field(default_factory=list)
    position_size: PositionSizeResult | None = None

    signals: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)

    #: 1-based position within its scan's ranked output; 0 until the Ranking
    #: Engine assigns it.
    ranking: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Watchlists
# ---------------------------------------------------------------------------


class Watchlist(BaseModel):
    """A named, ordered set of symbols a scan can target by name instead of
    an explicit symbol list on every call."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    symbols: tuple[str, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WatchlistRequest(BaseModel):
    """Input contract for ``POST /scanner/watchlist``. Replaces (creates or
    overwrites) the named watchlist's symbol set in full — the same
    create-or-replace semantics ``POST /portfolio`` already uses."""

    name: str = Field(..., min_length=1, max_length=100)
    symbols: list[str] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Scan results
# ---------------------------------------------------------------------------


class ScanFailure(BaseModel):
    """One symbol that could not be scanned, and why — surfaced instead of
    silently dropped, so a caller scanning thousands of symbols can see
    exactly which ones need attention (delisted tickers, thin data, etc.)."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    reason: str


class ScanSummary(BaseModel):
    """Provenance and outcome counts for one scan run — pure observability,
    mirroring ``PromptMetadata``'s role for the AI layer."""

    model_config = ConfigDict(frozen=True)

    requested: int = Field(..., ge=0)
    scanned: int = Field(..., ge=0)
    succeeded: int = Field(..., ge=0)
    failed: int = Field(..., ge=0)
    elapsed_ms: float = Field(..., ge=0.0)
    include_news: bool
    include_ai: bool
    ai_analyzed: int = Field(default=0, ge=0)
    strategy_filter: StrategyName | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScanResult(BaseModel):
    """Response envelope shared by ``/scanner/scan``, ``/scanner/top``, and
    ``/scanner/opportunities`` — one shape regardless of which of the three
    views produced it, so clients handle a single contract."""

    model_config = ConfigDict(frozen=True)

    summary: ScanSummary
    opportunities: list[Opportunity] = Field(default_factory=list)
    failures: list[ScanFailure] = Field(default_factory=list)
