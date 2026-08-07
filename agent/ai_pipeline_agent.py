from __future__ import annotations

import asyncio

from agent.ai_analysis_agent import AIAnalysisAgent
from agent.base import BaseAgent
from agent.news_agent import NewsAgent
from agent.technical_analysis_agent import TechnicalAnalysisAgent
from config.settings import Settings, get_settings
from data.providers.news_exceptions import NewsAgentError
from models.ai_analysis import AIAnalysisResult
from utils.logger import get_logger

logger = get_logger(__name__)


class AIPipelineAgent(BaseAgent):
    """Composite agent: gathers a ticker's inputs, then produces an AI thesis.

    This is the one place in the platform that composes specialists. It runs
    the Technical and News agents, then hands their results to
    ``AIAnalysisAgent``, which remains a pure consumer that never fetches
    anything itself. That separation is deliberate: the AI layer stays
    decoupled and offline-testable, and all cross-agent coordination lives
    here — the seed of the future Chief Decision Agent.

    It contains no analysis logic of its own. It only sequences existing
    agents and forwards their output.

    Failure policy (asymmetric, by design):
      * **Technical failure propagates.** It is the primary input; a thesis
        built without it would be materially weaker, and the caller should see
        the real cause (``DataFetchError``, ``InsufficientDataError``, ...)
        mapped to the correct status.
      * **News failure is tolerated.** News is enrichment: a missing or
        misconfigured news backend degrades the thesis rather than failing the
        request, so ``/ai`` keeps working when only the news API key is absent.

    The two upstream agents are independent, so they run concurrently — the
    request costs roughly one network round trip rather than two.
    """

    name = "ai_pipeline_agent"

    def __init__(
        self,
        *,
        technical_agent: TechnicalAnalysisAgent,
        news_agent: NewsAgent | None = None,
        ai_agent: AIAnalysisAgent,
        settings: Settings | None = None,
    ) -> None:
        """Args:
        technical_agent: Required — supplies the primary input.
        news_agent: Optional. When omitted (or unavailable), the thesis is
            built from technical analysis alone.
        ai_agent: Required — performs the actual interpretation.
        settings: Injected configuration; falls back to cached globals.
        """
        self._technical_agent = technical_agent
        self._news_agent = news_agent
        self._ai_agent = ai_agent
        self._settings = settings or get_settings()

    async def analyze(
        self,
        ticker: str,
        *,
        period: str = "1y",
        interval: str = "1d",
        model: str | None = None,
    ) -> AIAnalysisResult:
        """Gather inputs for ``ticker`` and return the AI thesis."""
        technical, news = await self._gather_inputs(ticker, period=period, interval=interval)
        return await self._ai_agent.analyze(
            ticker, technical=technical, news=news, model=model
        )

    async def _gather_inputs(self, ticker: str, *, period: str, interval: str):
        """Run the Technical and News agents concurrently.

        ``return_exceptions=True`` lets each agent fail independently so the
        asymmetric policy above can be applied; without it, one failure would
        cancel the sibling task.
        """
        technical_result, news_result = await asyncio.gather(
            self._technical_agent.analyze(ticker, period=period, interval=interval),
            self._run_news(ticker),
            return_exceptions=True,
        )

        if isinstance(technical_result, BaseException):
            # Primary input — surface the original error unchanged so the API
            # layer can map it to its established status code.
            raise technical_result

        if isinstance(news_result, BaseException):
            # Tolerate only news-domain failures (missing key, provider down,
            # rate limit). Anything else — a programming error, a cancellation —
            # is a real defect and must not be silently swallowed.
            if not isinstance(news_result, NewsAgentError):
                raise news_result
            logger.warning(
                "AI pipeline for %s: news unavailable (%s); continuing without it",
                ticker, type(news_result).__name__,
            )
            news_result = None

        return technical_result, news_result

    async def _run_news(self, ticker: str):
        """Fetch news when a news agent is wired.

        Exceptions are intentionally not caught here — ``gather`` captures them
        so :meth:`_gather_inputs` can apply the tolerance policy in one place.
        """
        if self._news_agent is None:
            return None
        return await self._news_agent.run(ticker)

    async def run(self, ticker: str, **kwargs: object) -> AIAnalysisResult:
        """``BaseAgent`` entry point. Delegates to :meth:`analyze`."""
        return await self.analyze(
            ticker,
            period=kwargs.get("period", "1y"),  # type: ignore[arg-type]
            interval=kwargs.get("interval", "1d"),  # type: ignore[arg-type]
            model=kwargs.get("model"),  # type: ignore[arg-type]
        )
