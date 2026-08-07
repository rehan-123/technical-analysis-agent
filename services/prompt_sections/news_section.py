from __future__ import annotations

from models.news import NewsAnalysisResult, NewsArticle
from services.prompt_sections.base import RenderedSection, SectionRenderer

# ---------------------------------------------------------------------------
# Field whitelist
#
# The renderer projects ONLY the fields below — an explicit allowlist, never a
# model dump. Fields added to NewsArticle / NewsAnalysisResult in the future
# will NOT appear in prompts unless this list is deliberately extended.
#
# INCLUDED (per article, and why):
#   title        -> the headline; the core signal a thesis reasons over.
#   publisher    -> source credibility/context (who reported it).
#   published_at -> recency context, rendered as a YYYY-MM-DD date only.
#
# EXCLUDED (and why):
#   summary   -> the article BODY/description. Explicitly excluded: it is long,
#       bloats the prompt, and is the raw content the AI layer must NOT be fed
#       to "summarize" (summarization/interpretation is not this pipeline's job;
#       the News Agent performs no sentiment analysis and neither do we).
#   url       -> not decision-relevant; adds noise/length and a small injection
#       surface. The headline carries the signal.
#   source    -> data-provider provenance ("finnhub"), an internal detail, not
#       something the LLM should reason about.
#   language  -> filtering concern handled upstream by the service.
#   ticker    -> already stated once at the section/result level; redundant per
#       article.
#   domain (computed)  -> derived from url, which is itself excluded.
#   retrieved_at (result-level)  -> a wall-clock fetch time. Non-deterministic;
#       including it would break the "identical input -> identical text"
#       guarantee. article_count is likewise omitted from the text (it is
#       surfaced via RenderedSection.item_count instead).
#
# published_at is rendered as a DATE (no time-of-day): recency is what matters,
# and a stable date keeps output deterministic and compact.
# ---------------------------------------------------------------------------


class NewsSectionRenderer(SectionRenderer):
    """Projects a ``NewsAnalysisResult`` into one prompt section.

    A pure presentation layer. The news has already been retrieved,
    normalized, deduplicated, and ordered upstream; this renderer only lays out
    a whitelisted view (headline, publisher, date) as concise text. It does not
    summarize, interpret sentiment, score, or decide anything, and it never
    includes article bodies.

    Deterministic by construction: it reads only stable fields and formats the
    publication date as ``YYYY-MM-DD`` (no time-of-day, no renderer-generated
    timestamp), so the same ``NewsAnalysisResult`` always yields identical text.
    Upstream article ordering is preserved.
    """

    kind = "news"
    version = "1.0"

    def render(self, model: NewsAnalysisResult, *, max_items: int) -> RenderedSection:
        """Render the news section.

        ``max_items`` caps how many articles are shown (the leading
        ``max_items``, preserving the upstream newest-first ordering). If more
        were available than shown, ``truncated`` is set. An empty article list
        renders a stable placeholder rather than an error.
        """
        available = len(model.articles)
        shown = model.articles[:max_items] if max_items >= 0 else list(model.articles)
        truncated = available > len(shown)

        lines: list[str] = []
        if not shown:
            lines.append("No recent news available.")
        else:
            for article in shown:
                lines.append(self._format_article(article))

        return RenderedSection(
            kind=self.kind,
            title="Recent News",
            body="\n".join(lines),
            item_count=len(shown),
            truncated=truncated,
        )

    @staticmethod
    def _format_article(article: NewsArticle) -> str:
        """Format one article as two lines: the headline, then publisher and
        date. Publisher falls back to a neutral label when absent so the layout
        stays stable."""
        publisher = article.publisher or "Unknown source"
        date = article.published_at.strftime("%Y-%m-%d")
        return f"- {article.title}\n  {publisher} \u2022 {date}"
