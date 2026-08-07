from __future__ import annotations

import json

import pytest

from config.settings import Settings
from llm.base import LLMProvider, LLMResponse
from llm.exceptions import LLMConfigurationError, LLMProviderError, LLMRateLimitError
from models.ai_analysis import AIAnalysisRequest, AIAnalysisResult, Recommendation
from models.analysis_result import (
    TechnicalAnalysisResult,
)
from services.ai_analysis_service import AIAnalysisService
from services.ai_exceptions import InvalidAIResponse, ResponseParseError
from services.prompt_sections.exceptions import PromptBuildError

# --- fakes --------------------------------------------------------------------

def _valid_json(**overrides) -> str:
    base = dict(
        recommendation="BUY", confidence=77, investment_thesis="Thesis.",
        news_sentiment="POSITIVE", technical_alignment="ALIGNED",
    )
    base.update(overrides)
    return json.dumps(base)

class FakeLLMProvider(LLMProvider):
    """Returns queued responses (or raises queued errors), recording calls.

    No network, no Ollama. Each ``generate`` pops the next scripted outcome.
    """

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def generate(self, prompt, *, system=None, schema=None, model=None) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(text=outcome, model=model or "fake-model", raw={}, latency_ms=1.0)

def _technical() -> TechnicalAnalysisResult:
    from tests.test_ai.test_technical_section import _result
    return _result()

def _request(**overrides) -> AIAnalysisRequest:
    base = dict(ticker="AAPL", technical=_technical())
    base.update(overrides)
    return AIAnalysisRequest(**base)

def _service(provider, **settings_overrides) -> AIAnalysisService:
    settings = Settings(**settings_overrides)
    return AIAnalysisService(provider=provider, settings=settings)

# --- happy path ---------------------------------------------------------------

class TestSuccessfulOrchestration:
    @pytest.mark.asyncio
    async def test_returns_validated_result(self):
        provider = FakeLLMProvider([_valid_json()])
        result = await _service(provider).analyze(_request())
        assert isinstance(result, AIAnalysisResult)
        assert result.recommendation is Recommendation.BUY
        assert result.confidence == 77
        assert result.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_provider_is_called_with_system_and_user_prompt(self):
        provider = FakeLLMProvider([_valid_json()])
        await _service(provider).analyze(_request())
        call = provider.calls[0]
        assert call["system"] and call["prompt"]
        assert "AAPL" in call["prompt"]

    @pytest.mark.asyncio
    async def test_model_override_from_request_is_forwarded(self):
        provider = FakeLLMProvider([_valid_json()])
        await _service(provider).analyze(_request(model="qwen2.5:14b"))
        assert provider.calls[0]["model"] == "qwen2.5:14b"

    @pytest.mark.asyncio
    async def test_model_used_recorded_on_result(self):
        provider = FakeLLMProvider([_valid_json()])
        result = await _service(provider).analyze(_request(model="custom-model"))
        assert result.model_used == "custom-model"

    @pytest.mark.asyncio
    async def test_markdown_wrapped_response_is_handled(self):
        provider = FakeLLMProvider([f"```json\n{_valid_json()}\n```"])
        result = await _service(provider).analyze(_request())
        assert result.confidence == 77

# --- repair retry (orchestration-level, NOT transport) ------------------------

class TestRepairRetry:
    @pytest.mark.asyncio
    async def test_invalid_then_valid_recovers_within_repair_budget(self):
        provider = FakeLLMProvider(["not json at all", _valid_json()])
        result = await _service(provider, llm_max_repair_attempts=1).analyze(_request())
        assert result.confidence == 77
        assert len(provider.calls) == 2  # initial + one repair

    @pytest.mark.asyncio
    async def test_repair_prompt_augments_the_user_prompt(self):
        provider = FakeLLMProvider(["garbage", _valid_json()])
        await _service(provider, llm_max_repair_attempts=1).analyze(_request())
        first, second = provider.calls[0]["prompt"], provider.calls[1]["prompt"]
        assert len(second) > len(first)
        assert "valid JSON" in second

    @pytest.mark.asyncio
    async def test_exhausted_repairs_raise_parse_error(self):
        provider = FakeLLMProvider(["garbage", "still garbage"])
        with pytest.raises(ResponseParseError):
            await _service(provider, llm_max_repair_attempts=1).analyze(_request())
        assert len(provider.calls) == 2

    @pytest.mark.asyncio
    async def test_zero_repair_budget_raises_on_first_invalid(self):
        provider = FakeLLMProvider(["garbage"])
        with pytest.raises(ResponseParseError):
            await _service(provider, llm_max_repair_attempts=0).analyze(_request())
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_semantic_invalid_then_valid_recovers(self):
        provider = FakeLLMProvider([_valid_json(confidence=999), _valid_json()])
        result = await _service(provider, llm_max_repair_attempts=1).analyze(_request())
        assert result.confidence == 77

    @pytest.mark.asyncio
    async def test_invalid_ai_response_raised_when_semantic_repairs_exhausted(self):
        provider = FakeLLMProvider([_valid_json(confidence=999), _valid_json(confidence=888)])
        with pytest.raises(InvalidAIResponse):
            await _service(provider, llm_max_repair_attempts=1).analyze(_request())

# --- transport failures: NOT retried by the service ---------------------------

class TestTransportFailuresNotRetriedHere:
    @pytest.mark.asyncio
    async def test_provider_error_propagates_without_service_retry(self):
        provider = FakeLLMProvider([LLMProviderError("upstream down", provider="ollama")])
        with pytest.raises(LLMProviderError):
            await _service(provider, llm_max_repair_attempts=2).analyze(_request())
        assert len(provider.calls) == 1  # NOT retried by the service

    @pytest.mark.asyncio
    async def test_rate_limit_propagates_without_service_retry(self):
        provider = FakeLLMProvider([LLMRateLimitError("limited", provider="ollama")])
        with pytest.raises(LLMRateLimitError):
            await _service(provider, llm_max_repair_attempts=2).analyze(_request())
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_transport_error_is_never_treated_as_a_repairable_parse_error(self):
        """A transport failure must not consume repair attempts."""
        provider = FakeLLMProvider([LLMProviderError("boom"), _valid_json()])
        with pytest.raises(LLMProviderError):
            await _service(provider, llm_max_repair_attempts=3).analyze(_request())
        assert len(provider.calls) == 1

# --- prompt build + provider selection ----------------------------------------

class TestPromptAndProvider:
    @pytest.mark.asyncio
    async def test_empty_request_raises_prompt_build_error(self):
        provider = FakeLLMProvider([_valid_json()])
        with pytest.raises(PromptBuildError):
            await _service(provider).analyze(AIAnalysisRequest(ticker="AAPL"))
        assert len(provider.calls) == 0  # never reached the provider

    @pytest.mark.asyncio
    async def test_lazy_provider_creation_uses_factory_and_can_fail_config(self):
        """With no injected provider and no configured backend, the factory
        raises LLMConfigurationError on first use (not at construction)."""
        service = AIAnalysisService(settings=Settings(llm_provider=""))
        with pytest.raises(LLMConfigurationError):
            await service.analyze(_request())

    def test_service_construction_performs_no_io(self):
        """Constructible without a backend or event loop."""
        assert AIAnalysisService(settings=Settings(llm_provider="")) is not None

# --- logging safety -----------------------------------------------------------

class TestLoggingSafety:
    @pytest.mark.asyncio
    async def test_prompt_contents_are_never_logged(self, caplog):
        import logging

        provider = FakeLLMProvider([_valid_json()])
        req = _request()
        with caplog.at_level(logging.DEBUG):
            await _service(provider).analyze(req)
        combined = " ".join(r.getMessage() for r in caplog.records)
        # the user prompt contains the section header text; it must not appear
        assert "Technical Analysis" not in combined
        assert "Security under analysis" not in combined

    @pytest.mark.asyncio
    async def test_ticker_and_provider_are_logged(self, caplog):
        import logging

        provider = FakeLLMProvider([_valid_json()])
        with caplog.at_level(logging.INFO):
            await _service(provider).analyze(_request())
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert "AAPL" in combined
        assert "FakeLLMProvider" in combined
