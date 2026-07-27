from __future__ import annotations

from agent.base import BaseAgent
from config.settings import Settings, get_settings
from models.news import NewsAnalysisResult, NewsRequest
from services.news_service import NewsService
from utils.logger import get_logger

logger = get_logger(__name__)


class NewsAgent(BaseAgent):
    """Specialist agent that gathers financial news for a ticker.

    Deliberately thin. It owns exactly one responsibility: adapting the
    generic ``BaseAgent`` interface — ``run(ticker, **kwargs)`` — into the
    News domain's typed ``NewsRequest``, then delegating to ``NewsService``
    and returning its result unchanged.

    Everything else belongs elsewhere and is intentionally absent here:

    * retrieval, retries and transport      -> the ``NewsProvider`` implementation
    * filtering, deduplication, sorting     -> ``NewsService``
    * provider selection                    -> ``create_news_provider`` at the
                                               composition root
    * caching                               -> ``NewsCache`` (not implemented yet)

    It performs no AI/LLM work, no sentiment analysis, and produces no
    recommendations — it returns factual, deterministic article data only.

    Because it implements ``BaseAgent``, a future Chief Decision Agent can
    hold a roster of specialists (Technical, News, Risk, Macro, …) and invoke
    each one polymorphically without knowing anything about their internals.
    """

    name = "news_agent"

    def __init__(self, service: NewsService, settings: Settings | None = None) -> None:
        """Args:
        service: The injected news pipeline. Required — the agent never
            selects or constructs a provider itself, so wiring stays in the
            composition root and this class remains trivially testable with a
            service backed by a fake provider.
        settings: Injected configuration, used solely to supply request
            defaults. Falls back to the cached global settings when omitted.
        """
        self._service = service
        self._settings = settings or get_settings()

    def _build_request(self, ticker: str, **kwargs: object) -> NewsRequest:
        """Translate the generic agent call into a typed ``NewsRequest``.

        Values explicitly passed by the caller win; anything omitted falls
        back to the centralized configured default. This is request assembly,
        not business logic — no article is inspected, filtered, or reordered
        here. Validation and normalization (ticker upper-casing, range
        checks) are left entirely to ``NewsRequest`` so those rules live in
        exactly one place.
        """
        return NewsRequest(
            ticker=ticker,
            lookback_days=kwargs.get("lookback_days") or self._settings.news_default_lookback_days,
            limit=kwargs.get("limit") or self._settings.news_default_limit,
            language=kwargs.get("language") or self._settings.news_default_language,
        )

    async def get_news(self, ticker: str, **kwargs: object) -> NewsAnalysisResult:
        """Retrieve processed news for ``ticker``.

        The explicit, domain-named entry point. ``run`` delegates here,
        mirroring the ``run``/``analyze`` split used by the Technical Analysis
        Agent so every specialist in the platform reads the same way.

        Provider and service errors propagate untouched: this layer has no
        information with which to decide how a failure should be handled, and
        swallowing one would turn an outage into an indistinguishable "no news
        found".
        """
        request = self._build_request(ticker, **kwargs)
        logger.info(
            "NewsAgent retrieving news for %s (lookback_days=%d, limit=%d, language=%s)",
            request.ticker, request.lookback_days, request.limit, request.language or "any",
        )
        return await self._service.get_news(request)

    async def run(self, ticker: str, **kwargs: object) -> NewsAnalysisResult:
        """``BaseAgent`` entry point — the uniform interface an orchestrator
        calls. Delegates to :meth:`get_news`."""
        return await self.get_news(ticker, **kwargs)
