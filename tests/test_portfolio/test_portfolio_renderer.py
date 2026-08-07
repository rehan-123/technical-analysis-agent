from __future__ import annotations

import pytest

from config.settings import Settings
from models.ai_analysis import AIAnalysisRequest
from portfolio.portfolio_models import CashBalance, Holding, Portfolio
from portfolio.portfolio_renderer import PortfolioSectionRenderer
from portfolio.portfolio_service import PortfolioService
from portfolio.risk_limits import RiskLimits
from services.prompt_builder import PromptBuilder
from services.prompt_sections.base import RenderedSection, SectionRenderer
from services.prompt_sections.registry import (
    available_section_kinds,
    get_section_renderer,
)

R = PortfolioSectionRenderer()


def _h(symbol="AAPL", qty=10.0, cost=100.0, price=120.0, sector="Technology"):
    return Holding(symbol=symbol, quantity=qty, average_cost=cost,
                   current_price=price, sector=sector)


def _portfolio(cash=4000.0, holdings=None):
    return Portfolio(cash=CashBalance(amount=cash),
                     holdings=tuple(holdings if holdings is not None else [_h()]))


def _context(symbol="NVDA", portfolio=None, **kwargs):
    return PortfolioService(**kwargs).build_context(portfolio or _portfolio(), symbol=symbol)


class TestRendererContract:
    def test_conforms_to_the_section_interface(self):
        assert isinstance(R, SectionRenderer)
        assert R.kind == "portfolio" and R.version == "1.0"

    def test_returns_a_rendered_section(self):
        section = R.render(_context(), max_items=8)
        assert isinstance(section, RenderedSection)
        assert section.kind == "portfolio"
        assert section.title == "Portfolio Context"


class TestRenderedContent:
    def test_includes_headline_portfolio_facts(self):
        body = R.render(_context(), max_items=8).body
        assert "Total value:" in body
        assert "Cash available:" in body
        assert "Invested:" in body
        assert "Risk:" in body

    def test_lists_current_holdings_and_sectors(self):
        body = R.render(_context(), max_items=8).body
        assert "Current holdings:" in body and "AAPL" in body
        assert "Sector exposure:" in body and "Technology" in body

    def test_states_the_candidate_and_whether_it_is_held(self):
        body = R.render(_context(symbol="NVDA"), max_items=8).body
        assert "Candidate under review: NVDA" in body
        assert "Not currently held." in body

    def test_reports_an_existing_position_with_its_pnl(self):
        body = R.render(_context(symbol="AAPL"), max_items=8).body
        assert "Already held:" in body
        assert "unrealized" in body

    def test_states_the_active_limits(self):
        body = R.render(_context(), max_items=8).body
        assert "max" in body and "per position" in body and "min" in body

    def test_surfaces_constraint_notes_when_breached(self):
        pf = _portfolio(cash=0.0, holdings=[_h(qty=100)])
        body = R.render(_context(symbol="NVDA", portfolio=pf), max_items=8).body
        assert "Constraints in effect:" in body or "Risk warnings:" in body

    def test_reports_capital_available_for_the_candidate(self):
        assert "Capital available for this position:" in R.render(_context(), max_items=8).body


class TestDeterminismAndBounds:
    def test_same_context_yields_identical_text(self):
        ctx = _context()
        assert R.render(ctx, max_items=8).body == R.render(ctx, max_items=8).body

    def test_holdings_are_capped_by_max_items(self):
        holdings = [_h(f"S{i}", sector=f"Sector{i}") for i in range(10)]
        section = R.render(_context(portfolio=_portfolio(holdings=holdings)), max_items=3)
        assert section.truncated is True
        assert section.item_count == 3

    def test_not_truncated_within_cap(self):
        assert R.render(_context(), max_items=20).truncated is False

    def test_body_is_curated_text_not_a_dump(self):
        body = R.render(_context(), max_items=8).body
        assert "{" not in body and "}" not in body
        assert "PortfolioRecommendationContext" not in body

    def test_no_trade_history_leaks_into_the_prompt(self):
        """The context is a projection: incidental history must not appear."""
        body = R.render(_context(), max_items=8).body
        assert "trades" not in body.lower()
        assert "closed_positions" not in body


class TestRegistryIntegration:
    def test_portfolio_kind_is_registered(self):
        assert "portfolio" in available_section_kinds()

    def test_registry_returns_the_portfolio_renderer(self):
        assert isinstance(get_section_renderer("portfolio"), PortfolioSectionRenderer)

    def test_existing_kinds_are_untouched(self):
        kinds = set(available_section_kinds())
        assert {"technical", "news"} <= kinds


class TestPromptBuilderIntegration:
    """Portfolio context must reach the prompt through additional_inputs alone —
    the whole point of the extension seam."""

    def _request(self, **kwargs):
        return AIAnalysisRequest(
            ticker="NVDA",
            additional_inputs={"portfolio": _context(symbol="NVDA")},
            **kwargs,
        )

    def test_portfolio_section_appears_in_the_prompt(self):
        package = PromptBuilder(settings=Settings()).build(self._request())
        assert "## Portfolio Context" in package.user_prompt
        assert "Candidate under review: NVDA" in package.user_prompt

    def test_metadata_records_the_portfolio_section(self):
        package = PromptBuilder(settings=Settings()).build(self._request())
        assert "portfolio" in package.metadata.sections_included
        assert package.metadata.renderer_versions["portfolio"] == "1.0"

    def test_portfolio_alone_is_enough_to_build_a_prompt(self):
        """No technical or news input required — the seam is independent."""
        package = PromptBuilder(settings=Settings()).build(self._request())
        assert package.metadata.sections_included == ["portfolio"]

    def test_renders_after_the_canonical_sections(self):
        from tests.test_ai.test_technical_section import _result as technical_result

        request = AIAnalysisRequest(
            ticker="NVDA",
            technical=technical_result(),
            additional_inputs={"portfolio": _context(symbol="NVDA")},
        )
        package = PromptBuilder(settings=Settings()).build(request)
        prompt = package.user_prompt
        assert prompt.index("Technical Analysis") < prompt.index("Portfolio Context")
        assert package.metadata.sections_included == ["technical", "portfolio"]

    def test_prompt_build_is_deterministic_with_portfolio(self):
        builder = PromptBuilder(settings=Settings())
        a = builder.build(self._request())
        b = builder.build(self._request())
        assert a.user_prompt == b.user_prompt

    def test_custom_limits_flow_through_to_the_prompt(self):
        limits = RiskLimits(max_position_pct=12.0, max_sector_pct=40.0)
        request = AIAnalysisRequest(
            ticker="NVDA",
            additional_inputs={"portfolio": _context(symbol="NVDA", limits=limits)},
        )
        package = PromptBuilder(settings=Settings()).build(request)
        assert "12.0% per position" in package.user_prompt

    def test_raw_portfolio_model_is_rejected_by_the_request(self):
        """additional_inputs is typed: only BaseModel instances are accepted."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AIAnalysisRequest(ticker="NVDA", additional_inputs={"portfolio": {"cash": 100}})
