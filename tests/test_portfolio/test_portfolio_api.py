from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.portfolio_routes as portfolio_routes
from main import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_portfolio():
    """Each test starts from a clean in-memory portfolio."""
    portfolio_routes._portfolio = None
    yield
    portfolio_routes._portfolio = None


def _create(cash=10_000.0, name="default"):
    return client.post("/portfolio", json={"name": name, "cash": cash})


def _add(symbol="AAPL", quantity=10.0, average_cost=100.0, **extra):
    body = {"symbol": symbol, "quantity": quantity, "average_cost": average_cost}
    body.update(extra)
    return client.post("/portfolio/holding", json=body)


class TestCreateAndRead:
    def test_read_returns_an_empty_portfolio_by_default(self):
        resp = client.get("/portfolio")
        assert resp.status_code == 200
        assert resp.json()["holdings"] == []

    def test_create_sets_opening_cash(self):
        resp = _create(cash=5000.0, name="growth")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "growth" and body["cash"]["amount"] == 5000.0

    def test_create_rejects_negative_cash(self):
        assert client.post("/portfolio", json={"cash": -5.0}).status_code == 422

    def test_created_portfolio_persists_across_requests(self):
        _create(cash=1234.0)
        assert client.get("/portfolio").json()["cash"]["amount"] == 1234.0


class TestHoldings:
    def test_add_holding_settles_cash(self):
        _create()
        resp = _add()
        assert resp.status_code == 200
        body = resp.json()
        assert body["cash"]["amount"] == 9000.0
        assert len(body["holdings"]) == 1
        assert body["holdings"][0]["symbol"] == "AAPL"

    def test_add_holding_beyond_cash_returns_409(self):
        _create(cash=100.0)
        resp = _add()
        assert resp.status_code == 409
        assert "insufficient" in resp.json()["detail"].lower()

    def test_import_mode_skips_cash_settlement(self):
        _create(cash=0.0)
        assert _add(settle_cash=False).status_code == 200

    def test_invalid_symbol_returns_422(self):
        _create()
        assert _add(symbol="!!!").status_code == 422

    def test_non_positive_quantity_returns_422(self):
        _create()
        assert _add(quantity=0.0).status_code == 422

    def test_second_buy_averages_rather_than_duplicating(self):
        _create()
        _add(quantity=10, average_cost=100)
        body = _add(quantity=10, average_cost=200, current_price=200).json()
        assert len(body["holdings"]) == 1
        assert body["holdings"][0]["quantity"] == 20.0

    def test_remove_holding_credits_cash(self):
        _create()
        _add()
        resp = client.delete("/portfolio/holding/AAPL?exit_price=120")
        assert resp.status_code == 200
        body = resp.json()
        assert body["holdings"] == []
        assert body["cash"]["amount"] == 10_200.0
        assert len(body["closed_positions"]) == 1

    def test_partial_sale_keeps_the_remainder(self):
        _create()
        _add()
        body = client.delete("/portfolio/holding/AAPL?quantity=4").json()
        assert body["holdings"][0]["quantity"] == 6.0

    def test_removing_an_unheld_symbol_returns_404(self):
        _create()
        assert client.delete("/portfolio/holding/MSFT").status_code == 404

    def test_overselling_returns_422(self):
        _create()
        _add()
        assert client.delete("/portfolio/holding/AAPL?quantity=999").status_code == 422


class TestAnalyticsEndpoints:
    def _seeded(self):
        _create()
        _add("AAPL", 10, 100, sector="Technology")
        _add("XOM", 10, 100, sector="Energy")

    def test_summary_returns_every_view(self):
        self._seeded()
        body = client.get("/portfolio/summary").json()
        assert set(body) >= {"name", "statistics", "performance", "risk",
                             "allocations", "sector_exposure"}
        assert body["statistics"]["position_count"] == 2

    def test_summary_allocations_are_sorted_by_weight(self):
        self._seeded()
        weights = [a["weight_pct"] for a in client.get("/portfolio/summary").json()["allocations"]]
        assert weights == sorted(weights, reverse=True)

    def test_performance_endpoint(self):
        self._seeded()
        body = client.get("/portfolio/performance").json()
        assert "unrealized_pnl" in body and "realized_pnl" in body and "return_pct" in body

    def test_risk_endpoint(self):
        self._seeded()
        body = client.get("/portfolio/risk").json()
        assert 0 <= body["risk_score"] <= 100
        assert body["risk_level"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
        assert body["beta"] is None  # honestly unavailable, not fabricated

    def test_analytics_work_on_an_empty_portfolio(self):
        assert client.get("/portfolio/summary").status_code == 200
        assert client.get("/portfolio/risk").status_code == 200


class TestBackwardCompatibility:
    def test_portfolio_routes_appear_in_openapi(self):
        paths = client.get("/openapi.json").json()["paths"]
        for path in ("/portfolio", "/portfolio/holding", "/portfolio/summary",
                     "/portfolio/performance", "/portfolio/risk"):
            assert path in paths

    def test_existing_routers_are_unaffected(self):
        paths = client.get("/openapi.json").json()["paths"]
        assert "/analyze/{ticker}" in paths
        assert "/ai/analyze" in paths
        assert any(p.startswith("/news") for p in paths)

    def test_existing_ai_contract_is_unchanged(self):
        spec = client.get("/openapi.json").json()
        ref = spec["paths"]["/ai/analyze"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        assert ref.endswith("AIAnalysisRequest")
