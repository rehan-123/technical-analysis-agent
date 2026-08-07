from __future__ import annotations

from datetime import datetime, timedelta, timezone

from models.news import NewsAnalysisResult, NewsArticle
from services.prompt_sections.base import RenderedSection
from services.prompt_sections.news_section import NewsSectionRenderer

def _article(**overrides) -> NewsArticle:
    base = dict(
        title="Apple beats earnings estimates",
        source="finnhub",
        publisher="Reuters",
        published_at=datetime(2026, 1, 15, 9, 30, tzinfo=timezone.utc),
        url="https://example.com/apple-earnings",
        summary="THIS IS THE ARTICLE BODY AND MUST NEVER APPEAR IN THE PROMPT.",
        ticker="AAPL",
        language="en",
    )
    base.update(overrides)
    return NewsArticle(**base)

def _result(articles=None, **overrides) -> NewsAnalysisResult:
    if articles is None:
        articles = [
            _article(title="First headline", url="https://e.com/1",
                     published_at=datetime(2026, 1, 15, tzinfo=timezone.utc)),
            _article(title="Second headline", publisher="Bloomberg", url="https://e.com/2",
                     published_at=datetime(2026, 1, 14, tzinfo=timezone.utc)),
            _article(title="Third headline", publisher="CNBC", url="https://e.com/3",
                     published_at=datetime(2026, 1, 13, tzinfo=timezone.utc)),
        ]
    base = dict(ticker="AAPL", articles=articles)
    base.update(overrides)
    return NewsAnalysisResult(**base)

R = NewsSectionRenderer()

class TestBasicRendering:
    def test_returns_rendered_section_with_correct_kind_and_title(self):
        section = R.render(_result(), max_items=8)
        assert isinstance(section, RenderedSection)
        assert section.kind == "news"
        assert section.title == "Recent News"

    def test_titles_and_publishers_present(self):
        body = R.render(_result(), max_items=8).body
        assert "First headline" in body and "Second headline" in body
        assert "Reuters" in body and "Bloomberg" in body

    def test_date_is_rendered_as_iso_date_only(self):
        import re

        body = R.render(_result(), max_items=8).body
        assert "2026-01-15" in body
        # No time-of-day component should leak. Detect real ISO time patterns
        # (e.g. "T09:30", "09:30", "09:30:00") rather than the bare letter "T",
        # which legitimately appears inside article titles.
        assert "09:30" not in body
        assert not re.search(r"\d{2}:\d{2}", body)          # HH:MM
        assert not re.search(r"\d{4}-\d{2}-\d{2}T", body)    # ISO date+T timestamp

    def test_missing_publisher_falls_back(self):
        result = _result([_article(publisher="", url="https://e.com/x")])
        assert "Unknown source" in R.render(result, max_items=8).body

class TestDeterminism:
    def test_same_input_yields_identical_text(self):
        a = R.render(_result(), max_items=8)
        b = R.render(_result(), max_items=8)
        assert a.body == b.body and a == b

    def test_upstream_order_is_preserved(self):
        body = R.render(_result(), max_items=8).body
        assert body.index("First headline") < body.index("Second headline") < body.index("Third headline")

    def test_no_retrieved_at_timestamp_leaks(self):
        """Result-level retrieved_at is wall-clock and must not appear."""
        result = _result(retrieved_at=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
        body = R.render(result, max_items=8).body
        assert "2026-06-01" not in body

class TestWhitelistAndBodyLeakage:
    def test_article_body_never_leaks(self):
        body = R.render(_result(), max_items=8).body
        assert "ARTICLE BODY" not in body.upper()
        assert "must never appear".upper() not in body.upper()

    def test_url_is_not_included(self):
        body = R.render(_result(), max_items=8).body
        assert "https://" not in body and "example.com" not in body

    def test_source_and_language_not_included(self):
        body = R.render(_result(), max_items=8).body
        assert "finnhub" not in body
        assert "language" not in body and "\nen" not in body

    def test_article_count_not_dumped_as_text(self):
        body = R.render(_result(), max_items=8).body
        assert "article_count" not in body

    def test_no_raw_model_dump(self):
        body = R.render(_result(), max_items=8).body
        assert "{" not in body and "}" not in body
        assert "NewsArticle" not in body

    def test_new_field_content_does_not_leak(self):
        """Whitelist stability: excluded field contents never surface."""
        result = _result([_article(summary="SECRET_BODY_XYZ", url="https://e.com/z")])
        assert "SECRET_BODY_XYZ" not in R.render(result, max_items=8).body

class TestItemCountAndTruncation:
    def test_item_count_reflects_articles_shown(self):
        assert R.render(_result(), max_items=8).item_count == 3

    def test_not_truncated_within_cap(self):
        assert R.render(_result(), max_items=8).truncated is False

    def test_truncated_when_articles_exceed_cap(self):
        many = [_article(title=f"H{i}", url=f"https://e.com/{i}",
                         published_at=datetime(2026, 1, 15, tzinfo=timezone.utc) - timedelta(days=i))
                for i in range(10)]
        section = R.render(_result(many), max_items=3)
        assert section.truncated is True
        assert section.item_count == 3
        assert "H9" not in section.body      # clipped entries absent
        assert "H0" in section.body          # leading entries kept

    def test_truncation_keeps_leading_entries(self):
        arts = [_article(title=f"A{i}", url=f"https://e.com/{i}",
                         published_at=datetime(2026, 1, 15, tzinfo=timezone.utc) - timedelta(days=i))
                for i in range(4)]
        body = R.render(_result(arts), max_items=2).body
        assert "A0" in body and "A1" in body
        assert "A2" not in body and "A3" not in body

class TestEmpty:
    def test_empty_article_list_renders_placeholder(self):
        section = R.render(_result([]), max_items=8)
        assert "No recent news" in section.body
        assert section.item_count == 0
        assert section.truncated is False

    def test_empty_list_still_returns_valid_section(self):
        section = R.render(_result([]), max_items=8)
        assert section.kind == "news" and section.title == "Recent News"
