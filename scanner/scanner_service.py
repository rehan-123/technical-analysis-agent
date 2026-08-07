from __future__ import annotations

import asyncio
import time
from typing import Iterable

from agent.ai_analysis_agent import AIAnalysisAgent
from agent.news_agent import NewsAgent
from agent.technical_analysis_agent import TechnicalAnalysisAgent
from config.settings import Settings, get_settings
from data.providers.news_exceptions import NewsAgentError
from models.analysis_result import TechnicalAnalysisResult
from models.news import NewsAnalysisResult
from models.opportunity import Opportunity, ScanFailure, ScanResult, ScanSummary
from models.strategy import StrategyDirection, StrategyName, StrategySignal
from portfolio.portfolio_models import Portfolio
from portfolio.portfolio_service import PortfolioService
from portfolio.position_sizer import PositionSizeResult, SizingMethod
from scanner.cache import TTLCache
from scanner.exceptions import NoSymbolsProvidedError, ScanTooLargeError
from scanner.exchange import infer_exchange
from scanner.news_scoring import score_news
from scanner.portfolio_scoring import score_portfolio_fit
from scanner.ranking_engine import RankingEngine
from strategy.base import clamp_score
from strategy.engine import StrategyEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class MarketScannerService:
    """Orchestrates the full Market Scanner pipeline for a batch of symbols.

    ``Market Scanner -> Technical Agent -> News Agent -> Portfolio Context ->
    Strategy Engine -> Ranking Engine -> Opportunity -> (Optional) AI
    Analysis`` — exactly the architecture diagram in the milestone brief.
    This class performs **no** technical calculation, **no** news analysis,
    and (by default, always opt-in) **no** AI reasoning of its own: every one
    of those is delegated to the existing agent/service it wraps. Its own
    logic is limited to orchestration (concurrency, caching, per-symbol error
    containment) and the two small, deterministic scoring heuristics that do
    not belong to any existing engine (``scanner.news_scoring``,
    ``scanner.portfolio_scoring``).

    Failure policy (deliberately different from ``AIPipelineAgent``'s
    single-ticker policy): a scan may cover thousands of symbols, so one
    symbol's failure — a delisted ticker, a thin data window, a transient
    provider error — must never abort the batch. Every per-symbol failure is
    caught, recorded as a ``ScanFailure``, and the scan continues.
    """

    def __init__(
        self,
        *,
        technical_agent: TechnicalAnalysisAgent,
        news_agent: NewsAgent | None = None,
        portfolio_service: PortfolioService | None = None,
        strategy_engine: StrategyEngine | None = None,
        ranking_engine: RankingEngine | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._technical_agent = technical_agent
        self._news_agent = news_agent
        self._portfolio_service = portfolio_service or PortfolioService()
        self._settings = settings or get_settings()
        self._strategy_engine = strategy_engine or StrategyEngine(settings=self._settings)
        self._ranking_engine = ranking_engine or RankingEngine(settings=self._settings)

        # Scanner-owned, process-local cache. Wraps calls this service makes
        # to the Technical Agent; does not touch the agent itself. See
        # scanner/cache.py.
        self._technical_cache: TTLCache[tuple[str, str, str], TechnicalAnalysisResult] = TTLCache(
            self._settings.scanner_technical_cache_ttl_seconds
        )

    # --- Public API ------------------------------------------------------

    async def scan(
        self,
        symbols: Iterable[str],
        *,
        portfolio: Portfolio | None = None,
        strategy: StrategyName | None = None,
        include_news: bool | None = None,
        period: str | None = None,
        interval: str | None = None,
    ) -> ScanResult:
        """Run the full pipeline over ``symbols`` and return every produced
        ``Opportunity``, ranked.

        Does not call the AI Analysis Service — that is the separate,
        explicit, opt-in :meth:`enrich_with_ai` step, consistent with
        "LLM analysis must be optional, disabled by default" and "never call
        the LLM for every stock".

        Raises:
            NoSymbolsProvidedError: ``symbols`` resolves to an empty set.
            ScanTooLargeError: more than ``scanner_max_symbols_per_scan``
                unique symbols were requested.
        """
        started = time.perf_counter()
        settings = self._settings
        period = period or settings.default_period
        interval = interval or settings.default_interval
        include_news = settings.scanner_default_include_news if include_news is None else include_news

        # De-dup while preserving first-seen order: a ticker present in two
        # watchlists (or passed twice) is scanned, and cached, exactly once.
        unique_symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
        if not unique_symbols:
            raise NoSymbolsProvidedError("No symbols were provided to scan.")
        if len(unique_symbols) > settings.scanner_max_symbols_per_scan:
            raise ScanTooLargeError(
                f"{len(unique_symbols)} symbols requested, exceeding the configured ceiling of "
                f"{settings.scanner_max_symbols_per_scan} (scanner_max_symbols_per_scan)."
            )

        technical_semaphore = asyncio.Semaphore(settings.scanner_max_concurrency)
        news_semaphore = asyncio.Semaphore(settings.scanner_news_concurrency)

        outcomes = await asyncio.gather(
            *(
                self._scan_one(
                    ticker,
                    portfolio=portfolio,
                    strategy=strategy,
                    include_news=include_news,
                    period=period,
                    interval=interval,
                    technical_semaphore=technical_semaphore,
                    news_semaphore=news_semaphore,
                )
                for ticker in unique_symbols
            )
        )

        opportunities: list[Opportunity] = []
        failures: list[ScanFailure] = []
        for outcome in outcomes:
            if isinstance(outcome, ScanFailure):
                failures.append(outcome)
            elif outcome is not None:
                opportunities.append(outcome)
            # None: scanned successfully but no strategy was applicable —
            # not a failure, simply not surfaced as an opportunity.

        ranked = self._ranking_engine.rank(opportunities)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        summary = ScanSummary(
            requested=len(unique_symbols),
            scanned=len(unique_symbols),
            succeeded=len(unique_symbols) - len(failures),
            failed=len(failures),
            elapsed_ms=elapsed_ms,
            include_news=include_news,
            include_ai=False,
            ai_analyzed=0,
            strategy_filter=strategy,
        )
        logger.info(
            "scan complete: requested=%d succeeded=%d failed=%d opportunities=%d elapsed_ms=%.1f",
            summary.requested, summary.succeeded, summary.failed, len(ranked), elapsed_ms,
        )
        return ScanResult(summary=summary, opportunities=ranked, failures=failures)

    async def enrich_with_ai(
        self,
        result: ScanResult,
        *,
        ai_agent: AIAnalysisAgent,
        portfolio: Portfolio | None = None,
        period: str | None = None,
        interval: str | None = None,
        max_opportunities: int | None = None,
    ) -> ScanResult:
        """Send only the top-ranked opportunities to the AI Analysis Service
        and fold each thesis's headline call into that opportunity's
        ``reasoning``.

        Bounded by ``scanner_ai_max_opportunities`` regardless of what a
        caller requests — "do NOT call the LLM for every stock" is enforced
        here, not left to caller discipline. Any individual AI failure
        (model unavailable, parse error, timeout) degrades that one
        opportunity — annotated with a warning — and never fails the batch,
        the same per-symbol containment policy :meth:`scan` uses.
        """
        settings = self._settings
        period = period or settings.default_period
        interval = interval or settings.default_interval
        cap = settings.scanner_ai_max_opportunities
        if max_opportunities is not None:
            cap = min(cap, max_opportunities)
        cap = max(0, cap)

        # `result.opportunities` is already ranking-sorted ascending by
        # RankingEngine.rank(); the first `cap` entries are the top `cap`.
        targets = result.opportunities[:cap]
        if not targets:
            return result

        technical_semaphore = asyncio.Semaphore(settings.scanner_max_concurrency)
        news_semaphore = asyncio.Semaphore(settings.scanner_news_concurrency)

        annotated: dict[str, Opportunity] = {}
        ai_success = 0
        for opp in targets:
            try:
                technical = await self._get_technical(
                    opp.ticker, period=period, interval=interval, semaphore=technical_semaphore
                )
                news = (
                    await self._get_news(opp.ticker, semaphore=news_semaphore)
                    if self._news_agent is not None
                    else None
                )
                portfolio_context = self._build_portfolio_context(portfolio, opp.ticker, technical.indicators.close)
                additional_inputs = {"portfolio": portfolio_context} if portfolio_context is not None else {}

                thesis = await ai_agent.analyze(
                    opp.ticker, technical=technical, news=news, additional_inputs=additional_inputs
                )
                annotated[opp.ticker] = opp.model_copy(
                    update={
                        "reasoning": [
                            *opp.reasoning,
                            f"AI thesis: {thesis.recommendation.value} "
                            f"({thesis.confidence}% confidence) — {thesis.investment_thesis}",
                        ]
                    }
                )
                ai_success += 1
            except Exception as exc:  # noqa: BLE001 — AI enrichment is best-effort per opportunity, see docstring
                logger.warning("scan: AI enrichment failed for %s: %s", opp.ticker, exc)
                annotated[opp.ticker] = opp.model_copy(
                    update={"warnings": [*opp.warnings, f"AI enrichment unavailable: {exc}"]}
                )

        new_opportunities = [annotated.get(o.ticker, o) for o in result.opportunities]
        new_summary = result.summary.model_copy(update={"include_ai": True, "ai_analyzed": ai_success})
        return ScanResult(summary=new_summary, opportunities=new_opportunities, failures=result.failures)

    # --- Per-symbol pipeline ----------------------------------------------

    async def _scan_one(
        self,
        ticker: str,
        *,
        portfolio: Portfolio | None,
        strategy: StrategyName | None,
        include_news: bool,
        period: str,
        interval: str,
        technical_semaphore: asyncio.Semaphore,
        news_semaphore: asyncio.Semaphore,
    ) -> Opportunity | ScanFailure | None:
        """Run one symbol through the pipeline. Never raises — any failure
        for this ticker is caught and returned as a ``ScanFailure`` so
        :meth:`scan`'s ``asyncio.gather`` never aborts over one bad symbol.
        """
        try:
            technical = await self._get_technical(
                ticker, period=period, interval=interval, semaphore=technical_semaphore
            )
        except Exception as exc:  # noqa: BLE001 — per-symbol containment boundary, see class docstring
            logger.warning("scan: %s failed technical analysis: %s", ticker, exc)
            return ScanFailure(ticker=ticker, reason=f"technical analysis failed: {exc}")

        news: NewsAnalysisResult | None = None
        if include_news and self._news_agent is not None:
            news = await self._get_news(ticker, semaphore=news_semaphore)

        try:
            return self._build_opportunity(technical, news, portfolio=portfolio, strategy=strategy)
        except Exception as exc:  # noqa: BLE001 — per-symbol containment boundary, see class docstring
            logger.warning("scan: %s failed opportunity assembly: %s", ticker, exc)
            return ScanFailure(ticker=ticker, reason=f"opportunity assembly failed: {exc}")

    async def _get_technical(
        self, ticker: str, *, period: str, interval: str, semaphore: asyncio.Semaphore
    ) -> TechnicalAnalysisResult:
        cache_key = (ticker, period, interval)
        cached = self._technical_cache.get(cache_key)
        if cached is not None:
            return cached
        async with semaphore:
            # Re-check after acquiring the slot: a concurrent scan for the
            # same ticker may have populated the cache while this task waited.
            cached = self._technical_cache.get(cache_key)
            if cached is not None:
                return cached
            result = await self._technical_agent.analyze(ticker, period=period, interval=interval)
        self._technical_cache.set(cache_key, result)
        return result

    async def _get_news(self, ticker: str, *, semaphore: asyncio.Semaphore) -> NewsAnalysisResult | None:
        """Fetch news for ``ticker``. Tolerates any news-domain failure —
        exactly the asymmetric policy ``AIPipelineAgent`` already applies:
        news is enrichment, never a hard dependency for a scan."""
        assert self._news_agent is not None
        try:
            async with semaphore:
                return await self._news_agent.run(ticker)
        except NewsAgentError as exc:
            logger.info("scan: news unavailable for %s (%s); continuing without it", ticker, type(exc).__name__)
            return None

    def _build_portfolio_context(self, portfolio: Portfolio | None, ticker: str, price: float):
        """Build the portfolio projection for ``ticker``, or ``None`` when no
        portfolio is meaningfully configured (no cash, no holdings) — an
        empty default portfolio must read as "not configured", not as a
        portfolio that rejects everything for lack of capital."""
        if portfolio is None or (not portfolio.holdings and portfolio.cash.amount <= 0):
            return None
        return self._portfolio_service.build_context(portfolio, symbol=ticker, price=price)

    def _select_strategy_signal(
        self, technical: TechnicalAnalysisResult, strategy: StrategyName | None
    ) -> StrategySignal | None:
        if strategy is not None:
            signal = self._strategy_engine.evaluate_one(technical, strategy)
            return signal if signal.is_actionable else None
        return self._strategy_engine.best(technical)

    def _size_position(
        self, portfolio: Portfolio | None, context, technical: TechnicalAnalysisResult
    ) -> PositionSizeResult | None:
        """Convert the portfolio context's already-computed capital headroom
        into a concrete share count via the existing ``PositionSizer`` —
        composes two existing methods rather than re-deriving capacity or
        sizing logic."""
        if portfolio is None or context is None or context.suggested_capital <= 0:
            return None
        return self._portfolio_service.size_position(
            portfolio,
            symbol=technical.ticker,
            price=technical.indicators.close,
            method=SizingMethod.FIXED_CAPITAL,
            capital=context.suggested_capital,
            stop_price=technical.stop_loss,
            target_price=technical.targets[0] if technical.targets else None,
        )

    def _build_opportunity(
        self,
        technical: TechnicalAnalysisResult,
        news: NewsAnalysisResult | None,
        *,
        portfolio: Portfolio | None,
        strategy: StrategyName | None,
    ) -> Opportunity | None:
        signal = self._select_strategy_signal(technical, strategy)
        if signal is None:
            return None

        # Normalize the headline technical read to the *signal's* direction
        # (100 - strength for a SHORT call) — the same convention every
        # strategy already applies to its own `strength`-derived score, so a
        # bearish opportunity is not penalized for a bullish-scaled strength.
        directional_strength = (
            technical.strength
            if signal.direction is StrategyDirection.LONG
            else (100 - technical.strength if signal.direction is StrategyDirection.SHORT else technical.strength)
        )
        technical_score = clamp_score(0.5 * directional_strength + 0.5 * technical.confidence)

        news_score, news_notes = score_news(news, settings=self._settings)

        portfolio_context = self._build_portfolio_context(portfolio, technical.ticker, technical.indicators.close)
        portfolio_score, portfolio_notes = score_portfolio_fit(portfolio_context, settings=self._settings)
        position_size = self._size_position(portfolio, portfolio_context, technical)

        combined_score = self._ranking_engine.combine(
            technical_score=technical_score,
            news_score=news_score,
            portfolio_score=portfolio_score,
            opportunity_score=signal.score,
        )
        confidence = clamp_score(0.5 * signal.confidence + 0.5 * technical.confidence)

        warnings: list[str] = []
        if technical.metadata is not None:
            warnings.extend(technical.metadata.warnings)
        if portfolio_context is not None:
            warnings.extend(portfolio_context.constraint_notes)

        reasoning: list[str] = [*signal.reasoning, *news_notes, *portfolio_notes]

        return Opportunity(
            ticker=technical.ticker,
            exchange=infer_exchange(technical.ticker),
            strategy=signal.strategy,
            direction=signal.direction,
            opportunity_score=signal.score,
            technical_score=technical_score,
            news_score=news_score,
            portfolio_score=portfolio_score,
            combined_score=combined_score,
            confidence=confidence,
            risk=technical.risk,
            trend=technical.trend,
            entry_zone=technical.entry_zone,
            stop_loss=technical.stop_loss,
            targets=list(technical.targets),
            position_size=position_size,
            signals=list(signal.signals),
            warnings=warnings,
            reasoning=reasoning,
        )
