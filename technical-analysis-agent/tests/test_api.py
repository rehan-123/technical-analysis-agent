from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.routes as routes_module
from agent.technical_analysis_agent import TechnicalAnalysisAgent
from config.settings import Settings
from data.synthetic_provider import SyntheticDataProvider
from main import app


@pytest.fixture(autouse=True)
def use_synthetic_agent(monkeypatch):
    """Swap the module-level agent singleton for one backed by synthetic
    data, so API tests never attempt a real network call."""
    synthetic_agent = TechnicalAnalysisAgent(
        settings=Settings(),
        data_provider=SyntheticDataProvider(seed=7, start_price=150.0, drift=0.0009, volatility=0.014),
    )
    monkeypatch.setattr(routes_module, "_agent_singleton", synthetic_agent)
    yield


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_get_endpoint_returns_expected_shape():
    client = TestClient(app)
    response = client.get("/analyze/AAPL")
    assert response.status_code == 200

    body = response.json()
    for key in (
        "ticker", "trend", "strength", "signals", "entry_zone",
        "stop_loss", "targets", "risk", "confidence", "summary",
    ):
        assert key in body
    assert body["ticker"] == "AAPL"


def test_analyze_post_endpoint_normalizes_ticker():
    client = TestClient(app)
    response = client.post("/analyze", json={"ticker": "btc-usd", "period": "1y", "interval": "1d"})
    assert response.status_code == 200
    assert response.json()["ticker"] == "BTC-USD"


def test_analyze_endpoint_rejects_too_short_period():
    client = TestClient(app)
    response = client.get("/analyze/AAPL", params={"period": "1mo"})
    assert response.status_code == 422
