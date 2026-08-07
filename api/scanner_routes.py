from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from agent.market_scanner_agent import MarketScannerAgent
from config.settings import get_settings
from llm.exceptions import LLMConfigurationError
from models.analysis_result import Risk, Trend
from models.opportunity import ScanResult, Watchlist, WatchlistRequest
from models.strategy import StrategyName
from portfolio.portfolio_validation import PortfolioValidationError
from scanner.exceptions import NoSymbolsProvidedError, ScanTooLargeError, WatchlistNotFoundError
from scanner.scanner_service import MarketScannerService
from scanner.watchlist_store import WatchlistStore
from utils.exceptions import DataFetchError, InsufficientDataError, TechnicalAgentError
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/scanner", tags=["scanner"])

# Composition root for the Market Scanner. Reuses the *existing* technical
# and news agent accessors from their own routers (get_agent / get_news_agent)
# rather than re-deriving their wiring — the same pattern api/ai_routes.py's
# get_ai_pipeline_agent() already uses. This module owns only what is new to
# this milestone: the watchlist store and the scanner service/agent built on
# top of the shared, already-composed agents.
_watchlist_store = WatchlistStore()
_scanner_service: MarketScannerService | None = None
_scanner_agent: MarketScannerAgent | None = None


def get_watchlist_store() -> WatchlistStore:
    return _watchlist_store


def get_scanner_service() -> MarketScannerService:
    """Build (once) and return the Market Scanner service.

    Construction is lazy and cached, mirroring the News/AI routes: the news
    agent accessor can raise if unconfigured, and building eagerly would
    abort application startup for any deployment running the scanner without
    a news backend, taking a working service down over an optional feature.
    """
    global _scanner_service
    if _scanner_service is None:
        from api.news_routes import get_news_agent
        from api.routes import get_agent as get_technical_agent
        from data.providers.news_exceptions import NewsAgentError

        settings = get_settings()
        try:
            news_agent = get_news_agent()
        except NewsAgentError as exc:
            logger.warning("scanner: news agent unavailable (%s); scans proceed without news", exc)
            news_agent = None

        _scanner_service = MarketScannerService(
            technical_agent=get_technical_agent(), news_agent=news_agent, settings=settings
        )
        logger.info("Market scanner service initialized")
    return _scanner_service


def get_scanner_agent() -> MarketScannerAgent:
    global _scanner_agent
    if _scanner_agent is None:
        _scanner_agent = MarketScannerAgent(get_scanner_service())
    return _scanner_agent


class ScanRequestBody(BaseModel):
    """Structured, bulk-friendly variant of the GET query parameters below —
    the same GET-convenience / POST-structured pairing every other router in
    this platform already offers (``/analyze``, ``/news``, ``/ai/analyze``)."""

    symbols: list[str] | None = Field(default=None, description="Explicit ticker list")
    watchlist: str | None = Field(default=None, description="Named watchlist to include, in addition to symbols")
    strategy: StrategyName | None = Field(
        default=None, description="Restrict to one strategy; omitted means auto-select the best applicable per symbol"
    )
    include_news: bool | None = Field(default=None, description="Defaults to TA_SCANNER_DEFAULT_INCLUDE_NEWS")
    include_ai: bool | None = Field(
        default=None, description="Optional AI enrichment of top-ranked opportunities; defaults to disabled"
    )
    period: str | None = None
    interval: str | None = None


@router.get("/health")
async def scanner_health() -> dict:
    return {"status": "ok", "agent": "market_scanner"}


@router.get("/scan", response_model=ScanResult)
async def scan_get(
    symbols: str | None = Query(default=None, description="Comma-separated ticker list"),
    watchlist: str | None = Query(default=None, description="Named watchlist to scan, in addition to symbols"),
    strategy: StrategyName | None = Query(default=None),
    include_news: bool | None = Query(default=None),
    include_ai: bool | None = Query(default=None),
    period: str | None = Query(default=None),
    interval: str | None = Query(default=None),
) -> ScanResult:
    """Run a scan over an explicit symbol list and/or a named watchlist.

    Orchestrates only — it calls the existing Technical/News/Portfolio/
    Strategy/Ranking layers and never computes an indicator, analyzes a
    headline, or reasons with an LLM itself.
    """
    symbol_list = _split_symbols(symbols)
    return await _run_scan(
        symbols=symbol_list,
        watchlist=watchlist,
        strategy=strategy,
        include_news=include_news,
        include_ai=include_ai,
        period=period,
        interval=interval,
    )


@router.post("/scan", response_model=ScanResult)
async def scan_post(request: ScanRequestBody) -> ScanResult:
    """Structured request body variant of ``GET /scanner/scan`` — useful for
    symbol lists too long to comfortably fit a query string."""
    return await _run_scan(
        symbols=request.symbols,
        watchlist=request.watchlist,
        strategy=request.strategy,
        include_news=request.include_news,
        include_ai=request.include_ai,
        period=request.period,
        interval=request.interval,
    )


@router.get("/top", response_model=ScanResult)
async def scan_top(
    n: int | None = Query(default=None, ge=1, le=500, description="Defaults to TA_SCANNER_DEFAULT_TOP_N"),
    symbols: str | None = Query(default=None, description="Comma-separated ticker list"),
    watchlist: str | None = Query(default=None),
    strategy: StrategyName | None = Query(default=None),
    include_news: bool | None = Query(default=None),
    include_ai: bool | None = Query(default=None),
    period: str | None = Query(default=None),
    interval: str | None = Query(default=None),
) -> ScanResult:
    """Convenience shortcut: run the same pipeline as ``/scanner/scan`` and
    return only the top ``n`` ranked opportunities."""
    settings = get_settings()
    top_n = n or settings.scanner_default_top_n
    result = await _run_scan(
        symbols=_split_symbols(symbols),
        watchlist=watchlist,
        strategy=strategy,
        include_news=include_news,
        include_ai=include_ai,
        period=period,
        interval=interval,
    )
    return ScanResult(summary=result.summary, opportunities=result.opportunities[:top_n], failures=result.failures)


@router.get("/opportunities", response_model=ScanResult)
async def scan_opportunities(
    symbols: str | None = Query(default=None, description="Comma-separated ticker list"),
    watchlist: str | None = Query(default=None),
    strategy: StrategyName | None = Query(default=None),
    include_news: bool | None = Query(default=None),
    include_ai: bool | None = Query(default=None),
    period: str | None = Query(default=None),
    interval: str | None = Query(default=None),
    min_score: int | None = Query(default=None, ge=0, le=100, description="Minimum combined_score"),
    risk: Risk | None = Query(default=None),
    trend: Trend | None = Query(default=None),
) -> ScanResult:
    """Filterable view over the same pipeline: run a scan, then narrow the
    result by minimum combined score, risk tier, and/or trend."""
    result = await _run_scan(
        symbols=_split_symbols(symbols),
        watchlist=watchlist,
        strategy=strategy,
        include_news=include_news,
        include_ai=include_ai,
        period=period,
        interval=interval,
    )
    filtered = result.opportunities
    if min_score is not None:
        filtered = [o for o in filtered if o.combined_score >= min_score]
    if risk is not None:
        filtered = [o for o in filtered if o.risk == risk]
    if trend is not None:
        filtered = [o for o in filtered if o.trend == trend]
    return ScanResult(summary=result.summary, opportunities=filtered, failures=result.failures)


@router.post("/watchlist", response_model=Watchlist)
async def upsert_watchlist(request: WatchlistRequest) -> Watchlist:
    """Create or fully replace a named watchlist's symbol set."""
    try:
        return _watchlist_store.upsert(request.name, request.symbols)
    except PortfolioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/watchlists", response_model=list[Watchlist])
async def list_watchlists() -> list[Watchlist]:
    return _watchlist_store.list_all()


def _split_symbols(symbols: str | None) -> list[str] | None:
    if not symbols:
        return None
    return [s.strip() for s in symbols.split(",") if s.strip()]


def _resolve_symbols(symbols: list[str] | None, watchlist: str | None) -> list[str]:
    """Union an explicit symbol list with a named watchlist's symbols.

    Falls back to a watchlist literally named ``"default"`` when neither is
    given, so a caller who has set one up can simply call ``/scanner/scan``
    with no parameters. Resolving to an empty list here is not itself an
    error — ``MarketScannerService.scan`` raises ``NoSymbolsProvidedError``
    for that, which :func:`_run_scan` maps to a 422 with a clear message.
    """
    resolved: list[str] = list(symbols or [])
    if watchlist:
        try:
            wl = _watchlist_store.get(watchlist)
        except WatchlistNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        resolved.extend(wl.symbols)
    if not resolved:
        default = _watchlist_store.get_or_none("default")
        if default is not None:
            resolved.extend(default.symbols)
    return resolved


async def _run_scan(
    *,
    symbols: list[str] | None,
    watchlist: str | None,
    strategy: StrategyName | None,
    include_news: bool | None,
    include_ai: bool | None,
    period: str | None,
    interval: str | None,
) -> ScanResult:
    """Shared implementation behind every scan-producing endpoint: resolve
    symbols, run the pipeline, optionally enrich with AI, and map domain
    errors to HTTP responses exactly like every other router in this
    platform."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    settings = get_settings()
    resolved_symbols = _resolve_symbols(symbols, watchlist)

    from api.portfolio_routes import get_portfolio

    portfolio = get_portfolio()
    agent = get_scanner_agent()

    logger.info(
        "scanner.scan start request_id=%s symbols=%d strategy=%s",
        request_id, len(resolved_symbols), strategy.value if strategy else None,
    )
    try:
        result = await agent.scan(
            resolved_symbols,
            portfolio=portfolio,
            strategy=strategy,
            include_news=include_news,
            period=period,
            interval=interval,
        )
    except NoSymbolsProvidedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ScanTooLargeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (InsufficientDataError, DataFetchError, TechnicalAgentError) as exc:
        # Only reachable if something outside the Scanner's per-symbol error
        # containment fails — ordinary per-symbol failures are already
        # captured in result.failures, not raised.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    include_ai_effective = settings.scanner_default_include_ai if include_ai is None else include_ai
    if include_ai_effective and result.opportunities:
        try:
            from api.ai_routes import get_ai_agent

            ai_agent = get_ai_agent()
            result = await get_scanner_service().enrich_with_ai(
                result, ai_agent=ai_agent, portfolio=portfolio, period=period, interval=interval
            )
        except LLMConfigurationError as exc:
            logger.warning("scanner.scan: AI enrichment skipped, LLM not configured: %s", exc)

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "scanner.scan ok request_id=%s opportunities=%d failures=%d elapsed_ms=%.1f",
        request_id, len(result.opportunities), len(result.failures), elapsed_ms,
    )
    return result
