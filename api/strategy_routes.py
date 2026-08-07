from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from config.settings import get_settings
from models.strategy import StrategyName, StrategySignal
from strategy.engine import StrategyEngine, UnknownStrategyError
from strategy.registry import available_strategies
from utils.exceptions import DataFetchError, InsufficientDataError, TechnicalAgentError
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/strategy", tags=["strategy"])

# Composition root for the Strategy Engine. A single shared instance is
# reused across requests — strategies are stateless, deterministic
# transforms, exactly like the section renderers in
# services/prompt_sections/registry.py.
_strategy_engine: StrategyEngine | None = None


def get_strategy_engine() -> StrategyEngine:
    global _strategy_engine
    if _strategy_engine is None:
        _strategy_engine = StrategyEngine(settings=get_settings())
        logger.info("Strategy engine initialized")
    return _strategy_engine


class StrategyDescriptor(BaseModel):
    """One entry in the ``GET /strategy`` catalog."""

    name: StrategyName
    description: str


@router.get("", response_model=list[StrategyDescriptor])
async def list_strategies() -> list[StrategyDescriptor]:
    """List every strategy the Strategy Engine can evaluate."""
    engine = get_strategy_engine()
    return [
        StrategyDescriptor(name=name, description=engine.registry[name].description)
        for name in available_strategies()
    ]


@router.get("/{ticker}", response_model=list[StrategySignal])
async def evaluate_ticker(
    ticker: str,
    strategy: StrategyName | None = Query(
        default=None, description="Restrict to one strategy; omit to evaluate every registered strategy"
    ),
    period: str = Query(default="1y"),
    interval: str = Query(default="1d"),
) -> list[StrategySignal]:
    """Run the Technical Agent for ``ticker``, then evaluate one or every
    registered strategy against the result.

    Reuses the existing shared Technical Agent (``api.routes.get_agent``)
    rather than constructing a new one — the Strategy Engine never fetches
    market data or computes an indicator itself.
    """
    from api.routes import get_agent

    try:
        technical = await get_agent().analyze(ticker, period=period, interval=interval)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except TechnicalAgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    engine = get_strategy_engine()
    if strategy is not None:
        try:
            return [engine.evaluate_one(technical, strategy)]
        except UnknownStrategyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return engine.evaluate_all(technical)
