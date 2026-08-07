from __future__ import annotations

from typing import Iterable

from agent.base import BaseAgent
from config.settings import Settings, get_settings
from models.opportunity import Opportunity, ScanResult
from models.strategy import StrategyName
from portfolio.portfolio_models import Portfolio
from scanner.scanner_service import MarketScannerService


class MarketScannerAgent(BaseAgent):
    """Thin ``BaseAgent`` adapter over ``MarketScannerService``.

    Mirrors ``NewsAgent`` / ``TechnicalAnalysisAgent``'s role exactly: all
    real orchestration lives in the injected service, and this class adds
    only the generic-interface adapter — so a future Chief Decision Agent
    can hold the Scanner in the same specialist roster as Technical, News,
    and AI, and invoke it polymorphically.

    The Scanner's real entry point is inherently plural (:meth:`scan`),
    since screening one symbol at a time defeats its purpose as a scanner.
    ``run(ticker)`` exists purely for roster uniformity with the other
    agents: it scans exactly that one ticker and returns its ``Opportunity``
    (or ``None`` if no strategy was applicable).
    """

    name = "market_scanner"

    def __init__(self, service: MarketScannerService, settings: Settings | None = None) -> None:
        self._service = service
        self._settings = settings or get_settings()

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
        """The Scanner's real entry point. Delegates entirely to
        ``MarketScannerService.scan``; see that method for the full contract."""
        return await self._service.scan(
            symbols,
            portfolio=portfolio,
            strategy=strategy,
            include_news=include_news,
            period=period,
            interval=interval,
        )

    async def run(self, ticker: str, **kwargs: object) -> Opportunity | None:
        """``BaseAgent`` entry point — scans exactly ``ticker`` and returns
        its ``Opportunity``, or ``None`` if no strategy was applicable."""
        result = await self.scan(
            [ticker],
            portfolio=kwargs.get("portfolio"),  # type: ignore[arg-type]
            strategy=kwargs.get("strategy"),  # type: ignore[arg-type]
            include_news=kwargs.get("include_news"),  # type: ignore[arg-type]
            period=kwargs.get("period"),  # type: ignore[arg-type]
            interval=kwargs.get("interval"),  # type: ignore[arg-type]
        )
        return result.opportunities[0] if result.opportunities else None
