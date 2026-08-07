from __future__ import annotations

import json

import pytest

from models.ai_analysis import AIAnalysisResult, NewsSentiment, Recommendation, TechnicalAlignment
from services.ai_exceptions import AIAnalysisError, InvalidAIResponse, ResponseParseError
from services.response_parser import ResponseParser

P = ResponseParser()


def _valid_payload(**overrides) -> dict:
    base = dict(
        recommendation="BUY",
        confidence=80,
        investment_thesis="Solid setup.",
        bull_case=["trend up"],
        bear_case=["macro risk"],
        key_risks=["earnings"],
        macro_considerations=[],
        watch_items=[],
        reasoning=["a", "b"],
        news_sentiment="POSITIVE",
        technical_alignment="ALIGNED",
    )
    base.update(overrides)
    return base


def _json(**overrides) -> str:
    return json.dumps(_valid_payload(**overrides))


class TestValidJsonParsing:
    def test_plain_json_object(self):
        result = P.parse(_json(), ticker="AAPL")
        assert isinstance(result, AIAnalysisResult)
        assert result.recommendation is Recommendation.BUY
        assert result.confidence == 80

    def test_ticker_is_backfilled_when_missing(self):
        result = P.parse(_json(), ticker="MSFT")
        assert result.ticker == "MSFT"

    def test_model_returned_ticker_is_not_overridden(self):
        result = P.parse(_json(ticker="TSLA"), ticker="MSFT")
        assert result.ticker == "TSLA"

    def test_model_used_is_recorded(self):
        result = P.parse(_json(), ticker="AAPL", model_used="qwen2.5:7b")
        assert result.model_used == "qwen2.5:7b"

    def test_enums_are_coerced(self):
        result = P.parse(_json(recommendation="STRONG_SELL", news_sentiment="MIXED",
                               technical_alignment="DIVERGENT"), ticker="AAPL")
        assert result.recommendation is Recommendation.STRONG_SELL
        assert result.news_sentiment is NewsSentiment.MIXED
        assert result.technical_alignment is TechnicalAlignment.DIVERGENT


class TestTolerantExtraction:
    def test_markdown_fenced_json(self):
        text = f"```json\n{_json()}\n```"
        assert P.parse(text, ticker="AAPL").confidence == 80

    def test_bare_fenced_block(self):
        text = f"```\n{_json()}\n```"
        assert P.parse(text, ticker="AAPL").confidence == 80

    def test_leading_and_trailing_whitespace(self):
        text = f"\n\n   {_json()}   \n\n"
        assert P.parse(text, ticker="AAPL").confidence == 80

    def test_json_embedded_in_prose(self):
        text = f"Sure, here is the analysis you requested:\n{_json()}\nHope that helps!"
        assert P.parse(text, ticker="AAPL").confidence == 80

    def test_prose_with_fenced_json_prefers_fence(self):
        text = f"Here you go:\n```json\n{_json(confidence=42)}\n```\nDone."
        assert P.parse(text, ticker="AAPL").confidence == 42


class TestParseErrors:
    def test_empty_string_raises_parse_error(self):
        with pytest.raises(ResponseParseError):
            P.parse("", ticker="AAPL")

    def test_whitespace_only_raises_parse_error(self):
        with pytest.raises(ResponseParseError):
            P.parse("   \n  ", ticker="AAPL")

    def test_non_json_text_raises_parse_error(self):
        with pytest.raises(ResponseParseError):
            P.parse("I cannot help with that.", ticker="AAPL")

    def test_json_array_not_object_raises_parse_error(self):
        with pytest.raises(ResponseParseError):
            P.parse('[1, 2, 3]', ticker="AAPL")

    def test_json_scalar_raises_parse_error(self):
        with pytest.raises(ResponseParseError):
            P.parse('"just a string"', ticker="AAPL")

    def test_parse_error_is_an_ai_analysis_error(self):
        assert issubclass(ResponseParseError, AIAnalysisError)


class TestValidationErrors:
    def test_confidence_out_of_range_raises_invalid_ai_response(self):
        with pytest.raises(InvalidAIResponse):
            P.parse(_json(confidence=150), ticker="AAPL")

    def test_invalid_recommendation_enum_raises_invalid_ai_response(self):
        with pytest.raises(InvalidAIResponse):
            P.parse(_json(recommendation="MAYBE"), ticker="AAPL")

    def test_missing_required_field_raises_invalid_ai_response(self):
        payload = _valid_payload()
        del payload["investment_thesis"]
        with pytest.raises(InvalidAIResponse):
            P.parse(json.dumps(payload), ticker="AAPL")

    def test_empty_recommendation_raises_invalid_ai_response(self):
        with pytest.raises(InvalidAIResponse):
            P.parse(_json(recommendation=""), ticker="AAPL")

    def test_invalid_response_is_an_ai_analysis_error(self):
        assert issubclass(InvalidAIResponse, AIAnalysisError)

    def test_structural_vs_semantic_errors_are_distinct(self):
        """A parse failure and a validation failure are different types."""
        assert not issubclass(ResponseParseError, InvalidAIResponse)
        assert not issubclass(InvalidAIResponse, ResponseParseError)
