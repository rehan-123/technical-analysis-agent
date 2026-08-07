from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from config.settings import Settings
from models.ai_analysis import AIAnalysisRequest
from models.analysis_result import (
    TechnicalAnalysisResult,
)
from models.news import NewsAnalysisResult, NewsArticle
from models.prompt_package import PromptPackage
from services.prompt_builder import PromptBuilder
from services.prompt_sections.exceptions import PromptBuildError

# --- fixtures -----------------------------------------------------------------

def _technical(**overrides) -> TechnicalAnalysisResult:
    from tests.test_ai.test_technical_section import _result as tech_result  # reuse builder
    return tech_result(**overrides)

def _news(n: int = 2) -> NewsAnalysisResult:
    articles = [
        NewsArticle(
            title=f"Headline {i}", source="finnhub", publisher="Reuters",
            published_at=datetime(2026, 1, 15 - i, tzinfo=timezone.utc),
            url=f"https://e.com/{i}", summary="body", ticker="AAPL", language="en",
        )
        for i in range(n)
    ]
    return NewsAnalysisResult(ticker="AAPL", articles=articles)

def _builder() -> PromptBuilder:
    return PromptBuilder(settings=Settings())

def _request(technical=None, news=None, **overrides) -> AIAnalysisRequest:
    base = dict(ticker="AAPL", technical=technical, news=news)
    base.update(overrides)
    return AIAnalysisRequest(**base)

B = _builder()

# --- basic assembly -----------------------------------------------------------

class TestAssembly:
    def test_technical_only(self):
        pkg = B.build(_request(technical=_technical()))
        assert isinstance(pkg, PromptPackage)
        assert "Technical Analysis" in pkg.user_prompt
        assert "Recent News" not in pkg.user_prompt
        assert pkg.metadata.sections_included == ["technical"]
        assert "news" in pkg.metadata.sections_skipped or "news" not in pkg.metadata.sections_included

    def test_news_only(self):
        pkg = B.build(_request(news=_news()))
        assert "Recent News" in pkg.user_prompt
        assert "Technical Analysis" not in pkg.user_prompt
        assert pkg.metadata.sections_included == ["news"]

    def test_technical_and_news(self):
        pkg = B.build(_request(technical=_technical(), news=_news()))
        assert "Technical Analysis" in pkg.user_prompt
        assert "Recent News" in pkg.user_prompt
        assert pkg.metadata.sections_included == ["technical", "news"]

    def test_system_prompt_comes_from_template(self):
        pkg = B.build(_request(technical=_technical()))
        assert "JSON" in pkg.system_prompt
        assert "recommendation" in pkg.system_prompt      # schema key from template
        assert len(pkg.system_prompt) > 0

    def test_ticker_appears_in_user_prompt(self):
        assert "AAPL" in B.build(_request(technical=_technical())).user_prompt

# --- ordering & determinism ---------------------------------------------------

class TestOrderingAndDeterminism:
    def test_canonical_order_technical_before_news(self):
        pkg = B.build(_request(technical=_technical(), news=_news()))
        assert pkg.user_prompt.index("Technical Analysis") < pkg.user_prompt.index("Recent News")

    def test_identical_request_yields_identical_package(self):
        req1 = _request(technical=_technical(), news=_news())
        req2 = _request(technical=_technical(), news=_news())
        a = B.build(req1)
        b = B.build(req2)
        assert a.system_prompt == b.system_prompt
        assert a.user_prompt == b.user_prompt
        assert a.prompt_version == b.prompt_version
        assert a.metadata == b.metadata

    def test_additional_inputs_order_does_not_affect_output(self):
        """Canonical ordering makes assembly independent of input ordering."""
        a = B.build(_request(technical=_technical(), news=_news()))
        b = B.build(_request(news=_news(), technical=_technical()))
        assert a.user_prompt == b.user_prompt

# --- metadata -----------------------------------------------------------------

class TestMetadata:
    def test_prompt_version_propagates_to_package_and_metadata(self):
        pkg = B.build(_request(technical=_technical()))
        assert pkg.prompt_version == PromptBuilder.PROMPT_VERSION
        assert pkg.metadata.prompt_version == PromptBuilder.PROMPT_VERSION

    def test_renderer_versions_are_recorded(self):
        pkg = B.build(_request(technical=_technical(), news=_news()))
        assert pkg.metadata.renderer_versions["technical"] == "1.0"
        assert pkg.metadata.renderer_versions["news"] == "1.0"

    def test_news_counts_and_truncation_recorded(self):
        pkg = B.build(_request(news=_news(n=5)))
        assert pkg.metadata.news_articles_available == 5
        assert pkg.metadata.news_articles_included == 5
        assert pkg.metadata.news_truncated is False

    def test_news_truncation_reflected_in_metadata(self):
        builder = PromptBuilder(settings=Settings(llm_max_news_articles=2))
        pkg = builder.build(_request(news=_news(n=6)))
        assert pkg.metadata.news_articles_available == 6
        assert pkg.metadata.news_articles_included == 2
        assert pkg.metadata.news_truncated is True

    def test_char_counts_and_total(self):
        pkg = B.build(_request(technical=_technical()))
        md = pkg.metadata
        assert md.system_prompt_chars == len(pkg.system_prompt)
        assert md.user_prompt_chars == len(pkg.user_prompt)
        assert md.total_chars == md.system_prompt_chars + md.user_prompt_chars

    def test_caps_applied_recorded(self):
        pkg = B.build(_request(technical=_technical(), news=_news()))
        assert "technical" in pkg.metadata.list_caps_applied
        assert "news" in pkg.metadata.list_caps_applied

# --- model hints --------------------------------------------------------------

class TestModelHints:
    def test_no_hints_by_default(self):
        assert B.build(_request(technical=_technical())).model_hints == {}

    def test_model_override_becomes_a_hint(self):
        pkg = B.build(_request(technical=_technical(), model="qwen2.5:14b"))
        assert pkg.model_hints["model"] == "qwen2.5:14b"

# --- error paths --------------------------------------------------------------

class TestErrorPaths:
    def test_empty_request_raises_prompt_build_error(self):
        """No technical, no news, no additional inputs -> nothing to build."""
        with pytest.raises(PromptBuildError):
            B.build(_request())

    def test_unknown_additional_input_kind_raises(self):
        """A supplied input with no registered renderer is a real error."""
        class MysteryResult(BaseModel):
            value: int = 1

        req = _request(technical=_technical(), additional_inputs={"mystery": MysteryResult()})
        with pytest.raises(PromptBuildError):
            B.build(req)

    def test_missing_template_raises_prompt_build_error(self, monkeypatch):
        import services.prompt_builder as pb

        # Force the cached loader to look for a template that does not exist.
        monkeypatch.setattr(pb.PromptBuilder, "_SYSTEM_TEMPLATE", "does_not_exist.txt")
        pb._load_template.cache_clear()
        with pytest.raises(PromptBuildError):
            _builder().build(_request(technical=_technical()))
        pb._load_template.cache_clear()

# --- future extensibility -----------------------------------------------------

class TestExtensibility:
    def test_future_kind_slots_in_when_registered(self, monkeypatch):
        """A newly-registered renderer is picked up with no builder change."""
        import services.prompt_sections.registry as reg
        from services.prompt_sections.base import RenderedSection, SectionRenderer

        class RiskRenderer(SectionRenderer):
            kind = "risk"
            version = "0.1"

            def render(self, model, *, max_items):
                return RenderedSection(kind="risk", title="Risk", body="risk body")

        class RiskResult(BaseModel):
            score: int = 5

        extended = dict(reg._SECTION_REGISTRY)
        extended["risk"] = RiskRenderer()
        monkeypatch.setattr(reg, "_SECTION_REGISTRY", extended)

        pkg = B.build(_request(technical=_technical(), additional_inputs={"risk": RiskResult()}))
        assert "Risk" in pkg.user_prompt
        # canonical kinds still lead; the extra kind follows deterministically
        assert pkg.metadata.sections_included == ["technical", "risk"]
