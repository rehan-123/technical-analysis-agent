from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class PromptMetadata(BaseModel):
    """Structured provenance for a built prompt — pure observability.

    Records *how* a ``PromptPackage`` was assembled (which sections were
    included or skipped, what was truncated, how large the result is) so a
    prompt can be explained and reproduced without re-deriving it. Nothing
    here influences the prompt text; it is a description of the build, not an
    input to it.

    Deliberately free of any wall-clock or otherwise non-deterministic value:
    identical inputs must yield identical metadata, which is what keeps prompts
    snapshot-testable and makes a future prompt cache key meaningful.

    Frozen because it is a value object emitted alongside an (immutable)
    package; it should never be mutated after construction.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., min_length=1)

    # Versioning. ``prompt_version`` is duplicated onto the package (which
    # treats it as canonical); it is carried here too so a metadata record is
    # self-contained when logged on its own. The builder sets both from a
    # single source, and ``PromptPackage`` validates that they agree.
    prompt_version: str = Field(..., min_length=1)
    #: Per-section renderer versions, e.g. {"technical": "1.0", "news": "1.0"},
    #: so a single section can evolve without a global version bump.
    renderer_versions: dict[str, str] = Field(default_factory=dict)

    # Section selection — which input kinds contributed, and which were absent.
    sections_included: list[str] = Field(default_factory=list)
    sections_skipped: list[str] = Field(default_factory=list)

    # News bounding — available vs. actually rendered, and whether the cap bit.
    news_articles_available: int = Field(default=0, ge=0)
    news_articles_included: int = Field(default=0, ge=0)
    news_truncated: bool = False

    #: Which configured caps bound which sections, e.g. {"news": 15}.
    list_caps_applied: dict[str, int] = Field(default_factory=dict)

    # Size accounting — a cheap proxy for prompt cost / context-window risk.
    # Set by the builder from the actual assembled strings.
    system_prompt_chars: int = Field(default=0, ge=0)
    user_prompt_chars: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_chars(self) -> int:
        """Total assembled prompt size — derived, so it can never drift from
        its parts."""
        return self.system_prompt_chars + self.user_prompt_chars


class PromptPackage(BaseModel):
    """Immutable output of the Prompt Builder.

    Carries exactly what the Phase 4 AI service needs to call an
    ``LLMProvider`` and to record what it asked — and nothing more. It holds
    only flattened strings plus structured metadata: it intentionally does NOT
    hold the builder's internal ``RenderedSection`` objects, because those live
    in the services layer and referencing them here (a models-layer type)
    would invert the dependency graph. The section-level detail survives in
    ``metadata`` in string-safe form.

    Field mapping onto ``LLMProvider.generate`` (Phase 4):
        user_prompt   -> prompt
        system_prompt -> system
        model_hints   -> informs the ``model`` (and any provider knobs)

    Frozen: a package is a finished artifact; consumers read it, never mutate
    it. Immutability also gives it a stable identity for a future prompt cache.
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str
    user_prompt: str
    metadata: PromptMetadata
    prompt_version: str = Field(..., min_length=1)

    #: Optional, open-ended hints for the consuming service — e.g. a preferred
    #: model, temperature suggestion, or the machine-readable output schema.
    #: Typed ``dict[str, Any]`` on purpose: unlike ``AIAnalysisRequest``'s
    #: ``additional_inputs`` (a validated domain contract), these are
    #: heterogeneous, optional knobs the service consumes defensively, so an
    #: open bag is the right abstraction rather than a rigid model.
    model_hints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _versions_agree(self) -> "PromptPackage":
        """The package's canonical ``prompt_version`` must match the copy in
        ``metadata`` — guarding against a builder bug that sets them
        inconsistently."""
        if self.prompt_version != self.metadata.prompt_version:
            raise ValueError(
                "prompt_version mismatch: "
                f"package={self.prompt_version!r} metadata={self.metadata.prompt_version!r}"
            )
        return self
