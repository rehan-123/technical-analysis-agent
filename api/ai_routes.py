from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent.ai_analysis_agent import AIAnalysisAgent
from config.settings import get_settings
from llm.exceptions import (
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
)
from data.providers.news_exceptions import NewsAgentError
from llm.provider_factory import available_llm_providers, create_llm_provider
from models.ai_analysis import AIAnalysisRequest, AIAnalysisResult
from services.ai_analysis_service import AIAnalysisService
from services.ai_exceptions import InvalidAIResponse, ResponseParseError
from services.prompt_builder import PromptBuilder
from services.prompt_sections.exceptions import PromptBuildError
from services.response_parser import ResponseParser
from utils.exceptions import (
    DataFetchError,
    InsufficientDataError,
    TechnicalAgentError,
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ai", tags=["ai"])

# Composition root for the AI Agent. This module is the ONLY place that wires
# provider -> service -> agent together; every other AI module depends purely
# on abstractions. The agent is stateless, so a single shared instance is
# reused across requests.
_agent_singleton: AIAnalysisAgent | None = None
_pipeline_singleton: "AIPipelineAgent | None" = None


def get_ai_pipeline_agent() -> "AIPipelineAgent":
    """Build (once) and return the composite pipeline agent.

    Reuses the *existing* accessors from the technical and news routers rather
    than re-deriving their wiring, so provider/service composition stays
    defined in exactly one place per domain. This is a same-layer (api -> api)
    dependency and introduces no cycle: neither of those modules imports this
    one.

    The news agent is optional: if the news backend is unconfigured its
    accessor raises, and the pipeline proceeds with technical analysis alone
    rather than failing the request.
    """
    global _pipeline_singleton
    if _pipeline_singleton is None:
        from agent.ai_pipeline_agent import AIPipelineAgent
        from api.news_routes import get_news_agent
        from api.routes import get_agent as get_technical_agent

        settings = get_settings()
        try:
            news_agent = get_news_agent()
        except NewsAgentError as exc:
            logger.warning("AI pipeline: news agent unavailable (%s); proceeding without news", exc)
            news_agent = None

        _pipeline_singleton = AIPipelineAgent(
            technical_agent=get_technical_agent(),
            news_agent=news_agent,
            ai_agent=get_ai_agent(),
            settings=settings,
        )
        logger.info("AI pipeline agent initialized")
    return _pipeline_singleton


def get_ai_agent() -> AIAnalysisAgent:
    """Build (once) and return the AI Analysis Agent.

    Construction is lazy and cached, mirroring the News routes. The LLM
    provider validates configuration when it is created; building eagerly at
    import time would abort application startup for any deployment without a
    configured/available LLM backend, taking down unrelated endpoints. Deferring
    means the app always starts and a missing backend surfaces as a clean 503 on
    the AI endpoints only.
    """
    global _agent_singleton
    if _agent_singleton is None:
        settings = get_settings()
        # create_llm_provider reads settings.llm_provider — no hardcoded name.
        # Raises LLMConfigurationError if unconfigured/unknown; that propagates
        # to the caller's error boundary (health handles it; analyze maps 503).
        provider = create_llm_provider(settings)
        service = AIAnalysisService(
            prompt_builder=PromptBuilder(settings=settings),
            parser=ResponseParser(),
            provider=provider,
            settings=settings,
        )
        _agent_singleton = AIAnalysisAgent(service=service, settings=settings)
        logger.info("AI agent initialized")
    return _agent_singleton


class AIHealth(BaseModel):
    """Structured readiness report for the AI subsystem."""

    status: str
    agent: str = "ai_analysis_agent"
    configured: bool
    provider_available: bool
    selected_model: str | None = None
    parser_available: bool = True
    prompt_builder_available: bool = True
    ready: bool
    detail: str | None = None


@router.get("/health", response_model=AIHealth)
async def ai_health() -> AIHealth:
    """Report whether the AI subsystem is usable.

    Never raises and never leaks secrets: a misconfigured or unavailable
    backend is reported as ``configured=false`` / ``ready=false`` with a
    reason, so the endpoint stays usable as a diagnostic. API keys, secrets,
    and prompt contents are never included.
    """
    settings = get_settings()
    provider_name = (settings.llm_provider or "").strip().lower()

    if not provider_name:
        return AIHealth(
            status="ok", configured=False, provider_available=False,
            selected_model=None, ready=False,
            detail="No LLM provider configured (set TA_LLM_PROVIDER).",
        )

    provider_available = provider_name in available_llm_providers()
    if not provider_available:
        return AIHealth(
            status="ok", configured=False, provider_available=False,
            selected_model=settings.llm_model, ready=False,
            detail=f"Unknown LLM provider '{provider_name}'.",
        )

    # Confirm the agent can actually be composed (catches config errors like a
    # cloud backend selected without an API key) without performing any I/O.
    try:
        get_ai_agent()
    except LLMConfigurationError as exc:
        return AIHealth(
            status="ok", configured=False, provider_available=True,
            selected_model=settings.llm_model, ready=False, detail=str(exc),
        )

    return AIHealth(
        status="ok", configured=True, provider_available=True,
        selected_model=settings.llm_model, ready=True,
    )


@router.post("/analyze", response_model=AIAnalysisResult)
async def analyze(request: AIAnalysisRequest) -> AIAnalysisResult:
    """Produce an AI investment thesis from already-built inputs.

    The route is intentionally thin: it validates the request body (Pydantic),
    calls the agent, and returns the result. All orchestration, prompt handling,
    and business logic live below the agent.
    """
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    settings = get_settings()
    logger.info(
        "ai.analyze start request_id=%s ticker=%s provider=%s",
        request_id, request.ticker, settings.llm_provider,
    )
    try:
        agent = get_ai_agent()
        result = await agent.run(
            request.ticker,
            technical=request.technical,
            news=request.news,
            additional_inputs=request.additional_inputs,
            model=request.model,
        )
    except Exception as exc:  # noqa: BLE001 — mapped to structured HTTP below
        status_code, reason = _map_exception(exc)
        elapsed_ms = (time.perf_counter() - started) * 1000
        # Log the failure reason only — never prompts, responses, or secrets.
        logger.warning(
            "ai.analyze fail request_id=%s ticker=%s provider=%s status=%d after=%.1fms reason=%s",
            request_id, request.ticker, settings.llm_provider, status_code, elapsed_ms, reason,
        )
        raise HTTPException(status_code=status_code, detail=reason) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "ai.analyze ok request_id=%s ticker=%s provider=%s after=%.1fms",
        request_id, request.ticker, settings.llm_provider, elapsed_ms,
    )
    return result


@router.get("/analyze/{ticker}", response_model=AIAnalysisResult)
async def analyze_ticker(
    ticker: str,
    period: str = Query(default="1y", description="Look-back window for technical analysis"),
    interval: str = Query(default="1d", description="Candle interval for technical analysis"),
    model: str | None = Query(default=None, description="Optional per-request LLM model override"),
) -> AIAnalysisResult:
    """Fetch this ticker's inputs and return an AI investment thesis.

    The convenience entry point: it gathers technical analysis (and news, when
    a news backend is configured) itself, so callers supply only a ticker.
    Follows the same by-ticker convention as ``GET /analyze/{ticker}`` and
    ``GET /news/{ticker}`` — path parameter plus query parameters, no request
    body.

    ``POST /ai/analyze`` remains available and unchanged for callers that
    already hold computed technical/news results and want to skip the fetch.
    """
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    settings = get_settings()
    logger.info(
        "ai.analyze_ticker start request_id=%s ticker=%s provider=%s",
        request_id, ticker, settings.llm_provider,
    )
    try:
        pipeline = get_ai_pipeline_agent()
        result = await pipeline.analyze(ticker, period=period, interval=interval, model=model)
    except Exception as exc:  # noqa: BLE001 — mapped to structured HTTP below
        status_code, reason = _map_exception(exc)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning(
            "ai.analyze_ticker fail request_id=%s ticker=%s provider=%s status=%d after=%.1fms reason=%s",
            request_id, ticker, settings.llm_provider, status_code, elapsed_ms, reason,
        )
        raise HTTPException(status_code=status_code, detail=reason) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "ai.analyze_ticker ok request_id=%s ticker=%s provider=%s after=%.1fms",
        request_id, ticker, settings.llm_provider, elapsed_ms,
    )
    return result


def _map_exception(exc: Exception) -> tuple[int, str]:
    """Map a domain exception to an (HTTP status, safe message) pair.

    Reuses the existing exception hierarchy; order is load-bearing
    (``LLMRateLimitError`` and ``LLMConfigurationError`` subclass more general
    types, so they are checked first). Never returns a stack trace.
    """
    if isinstance(exc, LLMRateLimitError):
        return 429, str(exc)
    if isinstance(exc, LLMConfigurationError):
        return 503, str(exc)
    if isinstance(exc, (LLMProviderError, ResponseParseError, InvalidAIResponse)):
        return 502, str(exc)
    if isinstance(exc, LLMError):
        # Any other LLM-layer error that is not configuration/rate-limit.
        return 502, str(exc)
    if isinstance(exc, PromptBuildError):
        return 500, str(exc)
    # Upstream input-gathering failures, reachable only from the by-ticker
    # endpoint. Mapped to the same statuses the technical router already uses,
    # so one failure means one status code across the whole API.
    if isinstance(exc, InsufficientDataError):
        return 422, str(exc)
    if isinstance(exc, DataFetchError):
        return 502, str(exc)
    if isinstance(exc, NewsAgentError):
        return 502, str(exc)
    if isinstance(exc, TechnicalAgentError):
        return 500, str(exc)
    # Unexpected: return a generic message, not the exception text, to avoid
    # leaking internals.
    logger.exception("ai.analyze unexpected error")
    return 500, "Internal server error"
