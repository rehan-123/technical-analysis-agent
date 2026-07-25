from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from agent.technical_analysis_agent import TechnicalAnalysisAgent
from data.factory import create_data_provider
from models.analysis_result import TechnicalAnalysisResult
from models.requests import TechnicalAnalysisRequest
from utils.exceptions import DataFetchError, InsufficientDataError, TechnicalAgentError
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# A single agent instance is reused across requests. It is stateless aside
# from its (also stateless) service/indicator collaborators, so sharing it
# avoids rebuilding the indicator engine on every call.
#
# The agent is given the production data provider — a real-source fallback
# chain (yfinance -> stooq) — so a connection reset against one source
# transparently fails over to the next instead of surfacing a 502.
_agent_singleton = TechnicalAnalysisAgent(data_provider=create_data_provider())


def get_agent() -> TechnicalAnalysisAgent:
    return _agent_singleton


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "agent": _agent_singleton.name}


@router.get("/analyze/{ticker}", response_model=TechnicalAnalysisResult)
async def analyze_ticker(
    ticker: str,
    period: str = Query(default="1y"),
    interval: str = Query(default="1d"),
    agent: TechnicalAnalysisAgent = Depends(get_agent),
) -> TechnicalAnalysisResult:
    request = TechnicalAnalysisRequest(ticker=ticker, period=period, interval=interval)
    return await _run_analysis(agent, request)


@router.post("/analyze", response_model=TechnicalAnalysisResult)
async def analyze(
    request: TechnicalAnalysisRequest,
    agent: TechnicalAnalysisAgent = Depends(get_agent),
) -> TechnicalAnalysisResult:
    return await _run_analysis(agent, request)


async def _run_analysis(
    agent: TechnicalAnalysisAgent, request: TechnicalAnalysisRequest
) -> TechnicalAnalysisResult:
    try:
        return await agent.analyze(request.ticker, period=request.period, interval=request.interval)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except TechnicalAgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
