from __future__ import annotations

from fastapi import FastAPI

from api.news_routes import router as news_router
from api.routes import router
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
