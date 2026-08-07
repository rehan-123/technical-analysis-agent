from __future__ import annotations

from typing import Mapping

from pydantic import BaseModel

from agent.base import BaseAgent
from config.settings import Settings, get_settings
from models.ai_analysis import AIAnalysisRequest, AIAnalysisResult
from models.analysis_result import TechnicalAnalysisResult
from models.news import NewsAnalysisResult
from services.ai_analysis_service import AIAnalysisService
from utils.logger import get_logger

logger = get_logger(__name__)


class AIAnalysisAgent(BaseAgent):
    """Specialist agent that produces an AI investment thesis for a ticker.

    Deliberately thin, exactly like ``NewsAgent`` and ``TechnicalAnalysisAgent``.
    Its single responsibility is to adapt the generic ``BaseAgent`` interface —
    ``run(ticker, **kwargs)`` — into the AI domain's typed ``AIAnalysisRequest``,
    then delegate to ``AIAnalysisService`` and return its result unchanged.

    Everything else lives elsewhere and is intentionally absent here:

    * prompt construction        -> ``PromptBuilder``
    * LLM transport and retries  -> the ``LLMProvider`` implementation
    * response parsing/validation-> ``ResponseParser`` / ``AIAnalysisResult``
    * repair-retry orchestration -> ``AIAnalysisService``
    * provider selection         -> ``create_llm_provider`` at the composition root

    It is a pure *consumer*: it never fetches market data, computes indicators,
    or calls another agent. The technical and news results it reasons over are
    supplied to it (by the caller / a future Chief Decision Agent), consistent
    with the AI layer being an interpretation layer over already-built results.

    Because it implements ``BaseAgent``, a Chief Decision Agent can hold it in
    the same roster as the other specialists and invoke it polymorphically.
    """

    name = "ai_analysis_agent"

    def __init__(self, service: AIAnalysisService, settings: Settings | None = None) -> None:
        """Args:
        service: The injected AI analysis pipeline. Required — the agent never
            selects or constructs an LLM provider itself, so wiring stays in
            the composition root and this class stays trivially testable with a
            mocked service.
        settings: Injected configuration. Retained for symmetry with the other
            agents and any future request defaults; falls back to cached global
            settings when omitted.
        """
        self._service = service
        self._settings = settings or get_settings()

    def _build_request(
        self,
        ticker: str,
        *,
        technical: TechnicalAnalysisResult | None = None,
        news: NewsAnalysisResult | None = None,
        additional_inputs: Mapping[str, BaseModel] | None = None,
        model: str | None = None,
    ) -> AIAnalysisRequest:
        """Assemble the typed ``AIAnalysisRequest`` from the generic call.

        Pure request assembly — no interpretation of the inputs. Validation and
        normalization (ticker casing, the ``additional_inputs`` BaseModel bound)
        are enforced by ``AIAnalysisRequest`` so those rules live in one place.
        """
        return AIAnalysisRequest(
            ticker=ticker,
            technical=technical,
            news=news,
            additional_inputs=additional_inputs or {},
            model=model,
        )

    async def analyze(
        self,
        ticker: str,
        *,
        technical: TechnicalAnalysisResult | None = None,
        news: NewsAnalysisResult | None = None,
        additional_inputs: Mapping[str, BaseModel] | None = None,
        model: str | None = None,
    ) -> AIAnalysisResult:
        """Produce an AI thesis for ``ticker`` from the supplied inputs.

        The explicit, domain-named entry point. ``run`` delegates here,
        mirroring the ``run``/``analyze`` split used across the platform's
        agents. Errors from the service (LLM, prompt, parse/validation) are
        allowed to propagate; mapping them to responses is the API layer's job.
        """
        request = self._build_request(
            ticker,
            technical=technical,
            news=news,
            additional_inputs=additional_inputs,
            model=model,
        )
        logger.info("AIAnalysisAgent analyzing %s", request.ticker)
        return await self._service.analyze(request)

    async def run(self, ticker: str, **kwargs: object) -> AIAnalysisResult:
        """``BaseAgent`` entry point — the uniform interface an orchestrator
        calls. Delegates to :meth:`analyze`."""
        return await self.analyze(
            ticker,
            technical=kwargs.get("technical"),  # type: ignore[arg-type]
            news=kwargs.get("news"),  # type: ignore[arg-type]
            additional_inputs=kwargs.get("additional_inputs"),  # type: ignore[arg-type]
            model=kwargs.get("model"),  # type: ignore[arg-type]
        )
