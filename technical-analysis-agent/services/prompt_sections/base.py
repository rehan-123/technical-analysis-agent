from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class RenderedSection(BaseModel):
    """One completed prompt section produced by a single renderer.

    The intermediate, structured representation the ``PromptBuilder`` collects
    before flattening everything into the final prompt strings. Keeping this
    as a typed object (rather than a bare string) is what lets the builder
    order sections deterministically, record per-section statistics in prompt
    metadata, and reason about truncation — all without re-parsing text.

    Frozen: a rendered section is a finished value object. Once a renderer has
    produced it, it is read (assembled, measured), never mutated.
    """

    model_config = ConfigDict(frozen=True)

    kind: str = Field(..., min_length=1, description="Input kind this section was rendered from, e.g. 'technical'")
    title: str = Field(..., min_length=1, description="Human-readable section header")
    body: str = Field(..., description="The projected section text (may be empty if the source had no content)")
    item_count: int = Field(default=0, ge=0, description="How many source items this section represents (e.g. articles)")
    truncated: bool = Field(default=False, description="Whether a configured cap dropped some source items")


class SectionRenderer(ABC):
    """Abstract renderer: converts one domain model into one ``RenderedSection``.

    This is the extension seam of the Prompt Builder. Every input kind
    (technical analysis, news, and future risk/macro/options/portfolio
    results) has exactly one renderer implementing this interface, so adding a
    future agent means writing one new renderer — the builder never changes.

    Responsibility boundary (deliberately narrow):
      * A renderer performs *projection and formatting only* — it selects a
        whitelisted set of fields from its input model and lays them out as
        text. It does not interpret, score, or decide anything.
      * It knows nothing about the ``PromptBuilder`` that will call it, nothing
        about the LLM, HTTP transport, or AI reasoning, and nothing about other
        renderers or the registry. It is a pure, reusable transform.

    Implementations expose:
      * ``kind``    — the input kind they render (must match how the builder /
                      registry key this renderer), and the value stamped onto
                      the produced ``RenderedSection.kind``.
      * ``version`` — a per-renderer version, so one section's projection can
                      evolve and be recorded in metadata without a global
                      prompt-version bump.
      * ``render``  — the single transform method.
    """

    #: The input kind this renderer handles (e.g. "technical", "news").
    #: Concrete renderers override this with a class attribute.
    kind: str = ""

    #: Per-renderer projection version, surfaced in prompt metadata.
    version: str = "1.0"

    @abstractmethod
    def render(self, model: BaseModel, *, max_items: int) -> RenderedSection:
        """Project ``model`` into a single ``RenderedSection``.

        Args:
            model: The domain result model to project (e.g. a
                ``TechnicalAnalysisResult`` or ``NewsAnalysisResult``). Typed
                as the ``BaseModel`` base here so the interface stays neutral;
                each concrete renderer narrows to the specific model it
                understands.
            max_items: The configured cap on how many source items may appear
                in this section (e.g. the maximum number of news articles).
                Renderers that project a list honour it and set ``truncated``
                accordingly; renderers with no list nature may ignore it.

        Returns:
            A ``RenderedSection`` whose ``kind`` equals this renderer's
            ``kind``. Producing an empty ``body`` is a valid outcome when the
            source has no content; that decision belongs to the builder, not
            the renderer.

        This method is pure and deterministic: the same input and cap must
        always yield an identical section. It performs no I/O.
        """
        raise NotImplementedError
