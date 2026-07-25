from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from agent.news_agent import NewsAgent
from config.settings import get_settings
from data.providers.news_exceptions import (
    NewsAgentError,
    NewsConfigurationError,
    NewsProviderError,
    NewsRateLimitError,
    NewsValidationError,
)
from data.providers.provider_factory import create_news_provider
from models.news import NewsAnalysisResult, NewsRequest
from services.news_service import NewsService
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/news", tags=["news"])

# Composition root for the News Agent. This module is the ONLY place that
# knows how the provider, service, and agent fit together — every other News
# module depends purely on abstractions.
_agent_singleton: NewsAgent | None = None


def get_news_agent() -> NewsAgent:
    """Build (once) and return the News Agent.

    Construction is **lazy and cached**, deliberately unlike the Technical
    Analysis routes which build their agent eagerly at import time. The news
    provider validates its configuration in its constructor and raises
    ``NewsConfigurationError`` when no API key is present; building eagerly
    would therefore abort application startup for any deployment that runs
    technical analysis without a news API key, taking a working service down
    over an unrelated optional feature.

    Deferring means the app always starts, and a missing key surfaces as a
    clean 503 on the news endpoints only, while every other endpoint keeps
    working.

    The agent is stateless, so a single shared instance is safe to reuse
    across requests.
    """
    global _agent_singleton
    if _agent_singleton is None:
        settings = get_settings()
        provider = create_news_provider(settings)
        service = NewsService(provider=provider, settings=settings)
        _agent_singleton = NewsAgent(service=service, settings=settings)
        logger.info("News agent initialized")
    return _agent_singleton


@router.get("/health")
async def news_health() -> dict:
    """Report whether the News Agent is usable.

    Never raises: a misconfigured news subsystem is reported as
    ``configured: false`` with the reason, rather than as a failed health
    check, so this endpoint stays usable as a diagnostic.
    """
    try:
        agent = get_news_agent()
    except NewsConfigurationError as exc:
        return {"status": "ok", "agent": "news_agent", "configured": False, "detail": str(exc)}
    return {"status": "ok", "agent": agent.name, "configured": True}


@router.get("/{ticker}", response_model=NewsAnalysisResult)
async def get_news_for_ticker(
    ticker: str,
    lookback_days: int | None = Query(default=None, ge=1, le=365),
    limit: int | None = Query(default=None, ge=1, le=250),
    language: str | None = Query(default=None, min_length=2, max_length=5),
) -> NewsAnalysisResult:
    """Retrieve deterministic, deduplicated, newest-first news for a ticker.

    Omitted query parameters fall back to the configured defaults.
    """
    return await _run(ticker, lookback_days=lookback_days, limit=limit, language=language)


@router.post("", response_model=NewsAnalysisResult)
async def get_news(request: NewsRequest) -> NewsAnalysisResult:
    """Retrieve news using a fully specified, validated request body."""
    return await _run(
        request.ticker,
        lookback_days=request.lookback_days,
        limit=request.limit,
        language=request.language,
    )


async def _run(ticker: str, **kwargs: object) -> NewsAnalysisResult:
    """Invoke the agent and translate domain errors into HTTP responses.

    The agent is obtained *inside* this exception boundary rather than via a
    route-level ``Depends``. Agent construction can raise
    ``NewsConfigurationError`` (e.g. no API key), and a dependency raised
    during FastAPI's dependency-resolution phase would bypass this handler and
    surface as a bare 500. Building it here guarantees that configuration
    failure is mapped to a structured 503 exactly like any other domain error.

    Except-clause order is load-bearing: ``NewsRateLimitError`` subclasses
    ``NewsProviderError``, so it must be caught first or a 429 would be
    reported as a generic 502.
    """
    try:
        agent = get_news_agent()
        return await agent.run(ticker, **kwargs)
    except NewsRateLimitError as exc:
        # 429 mirrors the upstream's own signal, and Retry-After is forwarded
        # verbatim when the provider supplied one so clients can honour the
        # server's guidance instead of guessing.
        headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else None
        raise HTTPException(status_code=429, detail=str(exc), headers=headers) from exc
    except NewsConfigurationError as exc:
        # 503: the service itself is not set up (e.g. missing API key). Distinct
        # from 5xx-from-upstream because the fix is ours (configuration), not
        # the vendor's. This is reachable both from agent construction above and
        # from the agent call itself.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (NewsProviderError, NewsValidationError) as exc:
        # 502: we are a gateway and the upstream failed or returned something
        # unusable — consistent with how the technical-analysis routes map
        # DataFetchError.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NewsAgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
