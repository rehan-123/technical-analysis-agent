from __future__ import annotations

import json

import pytest

from models.ai_analysis import AIAnalysisResult, Recommendation
from services.ai_exceptions import InvalidAIResponse
from services.response_parser import ResponseParser

P = ResponseParser()


def _payload(**overrides) -> str:
    base = dict(
        recommendation="HOLD",
        confidence=50,
        investment_thesis="Neutral.",
        news_sentiment="NEUTRAL",
        technical_alignment="MIXED",
    )
    base.update(overrides)
    return json.dumps(base)


class TestConfidenceBounds:
    @pytest.mark.parametrize("value", [0, 1, 50, 99, 100])
    def test_valid_confidence_accepted(self, value):
        assert P.parse(_payload(confidence=value), ticker="AAPL").confidence == value

    @pytest.mark.parametrize("value", [-1, 101, 500, -50])
    def test_out_of_range_confidence_rejected(self, value):
        with pytest.raises(InvalidAIResponse):
            P.parse(_payload(confidence=value), ticker="AAPL")

    def test_non_integer_confidence_rejected(self):
        with pytest.raises(InvalidAIResponse):
            P.parse(_payload(confidence="high"), ticker="AAPL")


class TestRecommendationVocabulary:
    @pytest.mark.parametrize(
        "value", ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "NO_ACTION"],
    )
    def test_every_valid_recommendation_accepted(self, value):
        result = P.parse(_payload(recommendation=value), ticker="AAPL")
        assert result.recommendation == Recommendation(value)

    @pytest.mark.parametrize("value", ["buy", "Buy", "STRONGBUY", "MAYBE", "", "LONG"])
    def test_invalid_recommendation_rejected(self, value):
        with pytest.raises(InvalidAIResponse):
            P.parse(_payload(recommendation=value), ticker="AAPL")


class TestRequiredFields:
    def test_missing_recommendation_rejected(self):
        with pytest.raises(InvalidAIResponse):
            P.parse(json.dumps({"confidence": 50, "investment_thesis": "x"}), ticker="AAPL")

    def test_missing_thesis_rejected(self):
        with pytest.raises(InvalidAIResponse):
            P.parse(json.dumps({"recommendation": "BUY", "confidence": 50}), ticker="AAPL")

    def test_missing_confidence_rejected(self):
        with pytest.raises(InvalidAIResponse):
            P.parse(json.dumps({"recommendation": "BUY", "investment_thesis": "x"}), ticker="AAPL")


class TestValidationIsDelegatedNotDuplicated:
    def test_parser_does_not_reimplement_bounds_or_enum_logic(self):
        """The parser must route to the model, not re-implement its rules.

        Enforced structurally: the parser's source contains no confidence
        comparison and no hardcoded recommendation literals — those live only
        in AIAnalysisResult / the Recommendation enum.
        """
        import inspect

        import services.response_parser as mod
        source = inspect.getsource(mod)
        assert "confidence" not in source.replace("confidence bounds", "")  # only prose mention
        for literal in ("STRONG_BUY", "STRONG_SELL", "NO_ACTION"):
            assert literal not in source

    def test_optional_list_fields_default_when_absent(self):
        """Defaults come from the model, not the parser."""
        result = P.parse(_payload(), ticker="AAPL")
        assert result.bull_case == [] and result.reasoning == []

    def test_disclaimer_is_present_from_model(self):
        """The computed disclaimer is supplied by the model, unremovable."""
        result = P.parse(_payload(), ticker="AAPL")
        assert isinstance(result, AIAnalysisResult)
        assert len(result.disclaimer) > 0
