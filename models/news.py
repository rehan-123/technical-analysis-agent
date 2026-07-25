from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    computed_field,
    field_validator,
)

# Small tolerance for publisher/local clock skew when rejecting articles
# dated in the future. Kept as a named module constant rather than a magic
# literal so the intent is explicit and easy to adjust.
_MAX_FUTURE_SKEW = timedelta(minutes=5)


class NewsRequest(BaseModel):
    """Input contract for a news retrieval.

    Mirrors ``TechnicalAnalysisRequest`` in spirit: a small, validated,
    normalized request object the agent/service/provider all agree on. Field
    defaults are the API-level contract defaults; the agent may override them
    from centralized settings when constructing a request.

    Frozen (immutable) for two reasons:

      * **Deterministic cache-key derivation.** ``NewsCache`` derives its keys
        from a request's identity. If a request could be mutated after being
        used to look up or store an entry, the same object could yield two
        different keys and silently corrupt the cache. Immutability makes the
        identity stable for the object's whole lifetime.
      * **Hashability.** ``frozen=True`` makes Pydantic generate ``__hash__``,
        so a request can be used directly as a dict key by an in-memory cache
        implementation. All fields are hashable scalars (``str``/``int``/
        ``None``) — no lists or dicts — so that hash is well-defined.

    Because the field validators normalize (ticker upper-cased, language
    lower-cased), two semantically identical requests compare and hash equal.
    To vary a request, use ``model_copy(update=...)``, which returns a new
    instance rather than mutating the original.
    """

    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    ticker: str = Field(..., min_length=1, description="Company ticker symbol, e.g. AAPL")
    lookback_days: int = Field(default=7, ge=1, le=365, description="How many days back to retrieve")
    limit: int = Field(default=50, ge=1, le=250, description="Maximum number of articles to return")
    language: str | None = Field(default=None, description="Optional ISO-639-1 language filter, e.g. 'en'")

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str | None) -> str | None:
        return value.strip().lower() if value else None


class NewsArticle(BaseModel):
    """A single normalized news article.

    This model IS the validation gate: any raw provider item that cannot
    populate these fields (missing title, unparseable/absent URL, non
    timezone-aware timestamp) fails construction and is filtered out upstream
    by the provider. Instances are frozen — articles are immutable value
    objects, which keeps deduplication and sorting deterministic.

    ``source`` vs ``publisher`` are deliberately distinct:
      * ``source``   — the data provider we retrieved it through (e.g. "finnhub");
                       provenance of *how* we got the article.
      * ``publisher`` — the outlet that actually published it (e.g. "Reuters");
                        *who* wrote it.
    """

    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    title: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, description="Data provider the article was retrieved through")
    publisher: str = Field(default="", description="Publishing outlet as reported by the source")
    published_at: datetime = Field(..., description="Publication time, timezone-aware, normalized to UTC")
    url: HttpUrl = Field(..., description="Canonical article URL (validated http/https)")
    summary: str = Field(default="", description="Article summary/description as provided (never LLM-generated)")
    ticker: str = Field(..., min_length=1)
    language: str = Field(default="en")

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return (value or "en").strip().lower() or "en"

    @field_validator("published_at")
    @classmethod
    def _require_utc_aware(cls, value: datetime) -> datetime:
        # Reject naive datetimes (ambiguous instants) and normalize any
        # aware datetime to UTC so sorting/deduplication are well-defined.
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("published_at must be timezone-aware")
        value = value.astimezone(timezone.utc)
        # Reject implausibly future-dated articles (bad source data), allowing
        # a small tolerance for clock skew between us and the publisher.
        if value > datetime.now(timezone.utc) + _MAX_FUTURE_SKEW:
            raise ValueError("published_at is implausibly in the future")
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def domain(self) -> str:
        """Publisher domain extracted deterministically from the URL
        (lowercased host, no ``www.``)."""
        host = (self.url.host or "").lower()
        return host[4:] if host.startswith("www.") else host


class NewsAnalysisResult(BaseModel):
    """Structured output contract of the News Agent.

    Deliberately parallels ``TechnicalAnalysisResult``'s shape (``agent``,
    ``ticker``, a timestamp, and the payload) so a Chief Decision Agent can
    consume every specialist agent uniformly. Contains only deterministic,
    factual data — no sentiment, no scores, no LLM output.
    """

    agent: str = "news_agent"
    ticker: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    articles: list[NewsArticle] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def article_count(self) -> int:
        """Always consistent with ``articles`` — derived, never set directly."""
        return len(self.articles)
