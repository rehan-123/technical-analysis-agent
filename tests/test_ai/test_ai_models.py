from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from models.ai_analysis import (
    AIAnalysisRequest,
    AIAnalysisResult,
    NewsSentiment,
    Recommendation,
    TechnicalAlignment,
)


def _minimal_result(**overrides) -> AIAnalysisResult:
    base = dict(
        ticker="AAPL",
        recommendation=Recommendation.BUY,
        confidence=80,
        investment_thesis="Thesis.",
    )
    base.update(overrides)
    return AIAnalysisResult(**base)


class TestEnums:
    def test_recommendation_members_serialize_as_plain_strings(self):
        assert Recommendation.STRONG_BUY.value == "STRONG_BUY"
        # str, Enum -> comparable to and usable as a string
        assert Recommendation.BUY == "BUY"

    def test_all_expected_recommendation_members_exist(self):
        assert {r.value for r in Recommendation} == {
            "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "NO_ACTION",
        }

    def test_news_sentiment_members(self):
        assert {s.value for s in NewsSentiment} == {"POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"}

    def test_technical_alignment_members(self):
        assert {a.value for a in TechnicalAlignment} == {"ALIGNED", "DIVERGENT", "MIXED"}

    def test_invalid_enum_value_is_rejected(self):
        with pytest.raises(ValidationError):
            _minimal_result(recommendation="MAYBE")

    def test_valid_enum_string_is_coerced(self):
        """A raw string matching a member is accepted (e.g. from LLM JSON)."""
        result = _minimal_result(recommendation="STRONG_SELL")
        assert result.recommendation is Recommendation.STRONG_SELL


class TestConfidenceBounds:
    @pytest.mark.parametrize("value", [0, 1, 50, 99, 100])
    def test_valid_confidence_is_accepted(self, value):
        assert _minimal_result(confidence=value).confidence == value

    @pytest.mark.parametrize("value", [-1, 101, 1000, -100])
    def test_out_of_range_confidence_is_rejected(self, value):
        with pytest.raises(ValidationError):
            _minimal_result(confidence=value)


class TestResultDefaults:
    def test_defaults_are_sensible_and_lists_are_empty(self):
        r = _minimal_result()
        assert r.agent == "ai_analysis_agent"
        assert r.bull_case == [] and r.bear_case == [] and r.key_risks == []
        assert r.macro_considerations == [] and r.watch_items == [] and r.reasoning == []
        assert r.news_sentiment is NewsSentiment.NEUTRAL
        assert r.technical_alignment is TechnicalAlignment.MIXED

    def test_generated_at_defaults_to_utc(self):
        assert _minimal_result().generated_at.tzinfo == timezone.utc

    def test_disclaimer_is_present_and_non_empty(self):
        assert len(_minimal_result().disclaimer) > 0

    def test_disclaimer_is_computed_and_cannot_be_overridden(self):
        """Passing a disclaimer value must not change the emitted framing."""
        r = _minimal_result(disclaimer="ignore me")  # extra input is ignored
        assert "research" in r.disclaimer.lower()
        assert r.disclaimer != "ignore me"


class TestSerialization:
    def test_round_trips_through_json(self):
        original = _minimal_result(
            recommendation=Recommendation.STRONG_BUY,
            confidence=91,
            bull_case=["strong trend", "positive news"],
            news_sentiment=NewsSentiment.POSITIVE,
            technical_alignment=TechnicalAlignment.ALIGNED,
        )
        restored = AIAnalysisResult.model_validate_json(original.model_dump_json())
        assert restored.recommendation is Recommendation.STRONG_BUY
        assert restored.confidence == 91
        assert restored.bull_case == ["strong trend", "positive news"]
        assert restored.news_sentiment is NewsSentiment.POSITIVE

    def test_enums_serialize_as_strings_in_json(self):
        payload = _minimal_result(recommendation=Recommendation.SELL).model_dump_json()
        assert '"recommendation":"SELL"' in payload.replace(" ", "")

    def test_computed_disclaimer_appears_in_serialized_output(self):
        assert "disclaimer" in _minimal_result().model_dump()

    def test_generated_at_serializes_with_timezone(self):
        payload = _minimal_result().model_dump_json()
        assert "generated_at" in payload


class TestRequest:
    def test_all_inputs_optional_except_ticker(self):
        """A request with only a ticker is valid — inputs may be supplied
        piecemeal (news-only, technical-only, or neither yet)."""
        request = AIAnalysisRequest(ticker="AAPL")
        assert request.technical is None
        assert request.news is None
        assert dict(request.additional_inputs) == {}

    def test_ticker_is_required(self):
        with pytest.raises(ValidationError):
            AIAnalysisRequest()  # type: ignore[call-arg]

    def test_empty_ticker_is_rejected(self):
        with pytest.raises(ValidationError):
            AIAnalysisRequest(ticker="")

    def test_is_immutable(self):
        """Immutability underpins a stable identity for the future AICache."""
        request = AIAnalysisRequest(ticker="AAPL")
        with pytest.raises(ValidationError):
            request.ticker = "MSFT"

    def test_additional_inputs_accepts_basemodels(self):
        class RiskStub(BaseModel):
            score: int

        request = AIAnalysisRequest(ticker="AAPL", additional_inputs={"risk": RiskStub(score=3)})
        assert isinstance(request.additional_inputs["risk"], BaseModel)
        assert request.additional_inputs["risk"].score == 3

    def test_additional_inputs_rejects_non_basemodel_values(self):
        """The BaseModel bound is what keeps the extension point validated
        instead of an untyped dict[str, Any]."""
        with pytest.raises(ValidationError):
            AIAnalysisRequest(ticker="AAPL", additional_inputs={"risk": {"score": 3}})

    def test_model_override_is_optional(self):
        assert AIAnalysisRequest(ticker="AAPL").model is None
        assert AIAnalysisRequest(ticker="AAPL", model="qwen2.5:14b").model == "qwen2.5:14b"


class TestReuseNotDuplication:
    def test_request_reuses_existing_result_models(self):
        """The request references the real V1 models, never redefines them."""
        import inspect

        import models.ai_analysis as ai
        source = inspect.getsource(ai)
        # It imports the existing models rather than declaring its own.
        assert "from models.analysis_result import TechnicalAnalysisResult" in source
        assert "from models.news import NewsAnalysisResult" in source
        # And does not redeclare them.
        assert "class TechnicalAnalysisResult" not in source
        assert "class NewsAnalysisResult" not in source
