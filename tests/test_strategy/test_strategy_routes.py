from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.routes as routes_module
import api.strategy_routes as strategy_routes_module
from agent.technical_analysis_agent import TechnicalAnalysisAgent
from config.settings import Settings
from data.synthetic_provider import SyntheticDataProvider
from main import app
from models.strategy import StrategyName


@pytest.fixture(autouse=True)
def use_synthetic_agent(monkeypatch):
    synthetic_agent = TechnicalAnalysisAgent(
        settings=Settings(),
        data_provider=SyntheticDataProvider(seed=7, start_price=150.0, drift=0.0009, volatility=0.014),
    )
    monkeypatch.setattr(routes_module, "_agent_singleton", synthetic_agent)
    monkeypatch.setattr(strategy_routes_module, "_strategy_engine", None)
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_list_strategies_returns_all_five(client):
    response = client.get("/strategy")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    names = {entry["name"] for entry in body}
    assert names == {name.value for name in StrategyName}
    assert all(entry["description"] for entry in body)


def test_evaluate_ticker_returns_all_strategies_by_default(client):
    response = client.get("/strategy/AAPL")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    assert {s["strategy"] for s in body} == {name.value for name in StrategyName}
    for signal in body:
        assert signal["ticker"] == "AAPL"
        assert 0 <= signal["score"] <= 100
        assert 0 <= signal["confidence"] <= 100


def test_evaluate_ticker_single_strategy_filter(client):
    response = client.get("/strategy/AAPL", params={"strategy": "momentum"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["strategy"] == "momentum"


def test_evaluate_ticker_normalizes_lowercase_symbol(client):
    response = client.get("/strategy/aapl")
    assert response.status_code == 200
    assert response.json()[0]["ticker"] == "AAPL"


def test_evaluate_ticker_invalid_strategy_returns_422(client):
    response = client.get("/strategy/AAPL", params={"strategy": "not_a_strategy"})
    assert response.status_code == 422


def test_evaluate_ticker_rejects_too_short_period(client):
    response = client.get("/strategy/AAPL", params={"period": "1mo"})
    assert response.status_code == 422
