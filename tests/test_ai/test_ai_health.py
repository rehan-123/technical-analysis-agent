from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.ai_routes as ai_routes
from config.settings import Settings
from llm.exceptions import LLMConfigurationError
from main import app


@pytest.fixture(autouse=True)
def _reset_singleton():
    ai_routes._agent_singleton = None
    yield
    ai_routes._agent_singleton = None


client = TestClient(app, raise_server_exceptions=False)


def _set_settings(monkeypatch, **overrides):
    settings = Settings(**overrides)
    monkeypatch.setattr(ai_routes, "get_settings", lambda: settings)
    return settings


class TestHealthNeverCrashes:
    def test_unconfigured_provider_reports_not_configured(self, monkeypatch):
        _set_settings(monkeypatch, llm_provider="")
        resp = client.get("/ai/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is False
        assert data["ready"] is False
        assert data["detail"]

    def test_unknown_provider_reports_unavailable(self, monkeypatch):
        _set_settings(monkeypatch, llm_provider="does-not-exist")
        resp = client.get("/ai/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is False
        assert data["provider_available"] is False
        assert "does-not-exist" in data["detail"]

    def test_configuration_error_during_compose_is_caught(self, monkeypatch):
        _set_settings(monkeypatch, llm_provider="ollama")
        # Force agent composition to raise a config error.
        def _boom():
            raise LLMConfigurationError("missing api key")
        monkeypatch.setattr(ai_routes, "get_ai_agent", _boom)
        resp = client.get("/ai/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is False
        assert data["provider_available"] is True
        assert data["ready"] is False

    def test_healthy_when_provider_configured_and_composable(self, monkeypatch):
        _set_settings(monkeypatch, llm_provider="ollama", llm_model="qwen2.5:7b")
        monkeypatch.setattr(ai_routes, "get_ai_agent", lambda: object())
        resp = client.get("/ai/health")
        data = resp.json()
        assert data["configured"] is True
        assert data["provider_available"] is True
        assert data["ready"] is True
        assert data["selected_model"] == "qwen2.5:7b"


class TestHealthShape:
    def test_reports_component_availability(self, monkeypatch):
        _set_settings(monkeypatch, llm_provider="ollama")
        monkeypatch.setattr(ai_routes, "get_ai_agent", lambda: object())
        data = client.get("/ai/health").json()
        assert data["parser_available"] is True
        assert data["prompt_builder_available"] is True
        assert data["agent"] == "ai_analysis_agent"

    def test_never_leaks_secrets(self, monkeypatch):
        _set_settings(monkeypatch, llm_provider="ollama",
                      llm_api_key="SUPER_SECRET_KEY_123", llm_model="qwen2.5:7b")
        monkeypatch.setattr(ai_routes, "get_ai_agent", lambda: object())
        resp = client.get("/ai/health")
        assert "SUPER_SECRET_KEY_123" not in resp.text
