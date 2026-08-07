from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.routes as routes_module
import api.scanner_routes as scanner_routes_module
from agent.technical_analysis_agent import TechnicalAnalysisAgent
from config.settings import Settings
from data.synthetic_provider import SyntheticDataProvider
from main import app
from scanner.scanner_service import MarketScannerService
from scanner.watchlist_store import WatchlistStore


@pytest.fixture(autouse=True)
def use_synthetic_agent(monkeypatch):
    """Swap the shared technical agent for one backed by synthetic data, and
    reset the Scanner's lazily-cached service/agent/watchlist-store globals
    so every test rebuilds against it — mirrors tests/test_api.py's existing
    pattern for the same reason (never touch the network in this suite)."""
    synthetic_agent = TechnicalAnalysisAgent(
        settings=Settings(),
        data_provider=SyntheticDataProvider(seed=7, start_price=150.0, drift=0.0009, volatility=0.014),
    )
    monkeypatch.setattr(routes_module, "_agent_singleton", synthetic_agent)
    monkeypatch.setattr(scanner_routes_module, "_scanner_service", None)
    monkeypatch.setattr(scanner_routes_module, "_scanner_agent", None)
    monkeypatch.setattr(scanner_routes_module, "_watchlist_store", WatchlistStore())
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_scanner_health(client):
    response = client.get("/scanner/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_scan_get_returns_ranked_opportunities(client):
    response = client.get("/scanner/scan", params={"symbols": "AAA,BBB,CCC"})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["requested"] == 3
    scores = [o["combined_score"] for o in body["opportunities"]]
    assert scores == sorted(scores, reverse=True)
    assert [o["ranking"] for o in body["opportunities"]] == list(range(1, len(body["opportunities"]) + 1))


def test_scan_get_without_symbols_or_watchlist_returns_422(client):
    response = client.get("/scanner/scan")
    assert response.status_code == 422


def test_scan_post_structured_body(client):
    response = client.post("/scanner/scan", json={"symbols": ["AAA", "BBB"]})
    assert response.status_code == 200
    assert response.json()["summary"]["requested"] == 2


def test_scan_rejects_too_many_symbols(client, monkeypatch):
    tight_settings = Settings(scanner_max_symbols_per_scan=1)
    tight_service = MarketScannerService(technical_agent=routes_module._agent_singleton, settings=tight_settings)
    monkeypatch.setattr(scanner_routes_module, "_scanner_service", tight_service)
    monkeypatch.setattr(scanner_routes_module, "_scanner_agent", None)

    response = client.get("/scanner/scan", params={"symbols": "AAA,BBB"})
    assert response.status_code == 422


def test_watchlist_upsert_and_list(client):
    response = client.post("/scanner/watchlist", json={"name": "tech", "symbols": ["aapl", "msft"]})
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "tech"
    assert body["symbols"] == ["AAPL", "MSFT"]

    listing = client.get("/scanner/watchlists")
    assert listing.status_code == 200
    assert "tech" in [w["name"] for w in listing.json()]


def test_watchlist_upsert_rejects_invalid_symbol(client):
    response = client.post("/scanner/watchlist", json={"name": "bad", "symbols": ["not a symbol!"]})
    assert response.status_code == 422


def test_scan_by_watchlist_name(client):
    client.post("/scanner/watchlist", json={"name": "mylist", "symbols": ["AAA", "BBB"]})
    response = client.get("/scanner/scan", params={"watchlist": "mylist"})
    assert response.status_code == 200
    assert response.json()["summary"]["requested"] == 2


def test_scan_unknown_watchlist_returns_404(client):
    response = client.get("/scanner/scan", params={"watchlist": "does-not-exist"})
    assert response.status_code == 404


def test_scan_top_limits_results(client):
    response = client.get("/scanner/top", params={"symbols": "AAA,BBB,CCC", "n": 1})
    assert response.status_code == 200
    assert len(response.json()["opportunities"]) <= 1


def test_scan_opportunities_filters_by_min_score(client):
    baseline = client.get("/scanner/opportunities", params={"symbols": "AAA,BBB,CCC"})
    assert baseline.status_code == 200
    baseline_opportunities = baseline.json()["opportunities"]

    # A threshold strictly above every observed score, clamped to the
    # endpoint's legal 0-100 range, must filter out everything.
    max_score = max((o["combined_score"] for o in baseline_opportunities), default=0)
    threshold = min(100, max_score + 1)

    filtered = client.get("/scanner/opportunities", params={"symbols": "AAA,BBB,CCC", "min_score": threshold})
    assert filtered.status_code == 200
    if max_score < 100:
        assert filtered.json()["opportunities"] == []
    else:
        assert len(filtered.json()["opportunities"]) <= len(baseline_opportunities)


def test_scan_strategy_filter_only_returns_that_strategy(client):
    response = client.get("/scanner/scan", params={"symbols": "AAA,BBB,CCC", "strategy": "trend_following"})
    assert response.status_code == 200
    for o in response.json()["opportunities"]:
        assert o["strategy"] == "trend_following"


def test_scan_invalid_strategy_returns_422(client):
    response = client.get("/scanner/scan", params={"symbols": "AAA", "strategy": "not_a_strategy"})
    assert response.status_code == 422


def test_scan_ai_disabled_by_default(client):
    response = client.get("/scanner/scan", params={"symbols": "AAA"})
    assert response.status_code == 200
    assert response.json()["summary"]["include_ai"] is False


def test_scan_ai_requested_but_unavailable_degrades_gracefully(client):
    """No LLM backend is reachable in this test environment. Requesting AI
    enrichment must still return 200 — enrichment failure degrades the
    result, it never fails the scan."""
    response = client.get("/scanner/scan", params={"symbols": "AAA", "include_ai": "true"})
    assert response.status_code == 200
