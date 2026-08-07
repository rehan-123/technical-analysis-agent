from __future__ import annotations

from fastapi import FastAPI

from api.ai_routes import router as ai_router
from api.news_routes import router as news_router
from api.portfolio_routes import router as portfolio_router
from api.routes import router
from api.scanner_routes import router as scanner_router
from api.strategy_routes import router as strategy_router
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Technical Analysis Agent",
    description=(
        "Standalone technical-analysis microservice, designed to be "
        "orchestrated by a Chief Decision Agent alongside News, Risk, "
        "Macro, and Options agents."
    ),
    version="1.0.0",
)

app.include_router(router)
# News Agent endpoints, mounted under /news. Registered here (the composition
# root) so the two agents stay independent modules; neither imports the other.
app.include_router(news_router)
# AI Analysis Agent endpoints, mounted under /ai. Registered here at the
# composition root, additively; existing routers are untouched.
app.include_router(ai_router)
# Portfolio Intelligence endpoints (V2), mounted under /portfolio.
app.include_router(portfolio_router)
# Market Scanner endpoints (Scanner & Strategy milestone), mounted under
# /scanner. Orchestrates the Technical/News/Portfolio/Strategy/AI layers
# above; introduces no new computation of its own.
app.include_router(scanner_router)
# Strategy Engine endpoints (Scanner & Strategy milestone), mounted under
# /strategy. Evaluates the reusable strategy roster against an already
# computed TechnicalAnalysisResult.
app.include_router(strategy_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
