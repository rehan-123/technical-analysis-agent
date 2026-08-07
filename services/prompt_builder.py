from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from config.settings import Settings, get_settings
from models.ai_analysis import AIAnalysisRequest
from models.prompt_package import PromptMetadata, PromptPackage
from services.prompt_sections.base import RenderedSection
from services.prompt_sections.exceptions import PromptBuildError
from services.prompt_sections.registry import get_section_renderer
from utils.logger import get_logger

logger = get_logger(__name__)

#: Directory holding the external system-prompt templates. Large prompt prose
#: lives in text files (editable, diffable, versioned by filename) rather than
#: being hardcoded in Python.
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "config" / "prompt_templates"

#: Canonical section ordering. Sections are always assembled in this order,
#: independent of how the request or its ``additional_inputs`` happen to be
#: ordered — the guarantee behind deterministic output. A future renderer joins
#: the sequence by being inserted here; nothing else in the builder changes.
_CANONICAL_ORDER: tuple[str, ...] = ("technical", "news")


@lru_cache(maxsize=None)
def _load_template(name: str) -> str:
    """Load and cache a system-prompt template by filename.

    Cached so the file is read once per process. Missing templates fail fast
    with ``PromptBuildError`` rather than surfacing a raw OSError later.
    """
    path = _TEMPLATE_DIR / name
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PromptBuildError(f"could not load prompt template {name!r}: {exc}") from exc


class PromptBuilder:
    """Assembles an ``AIAnalysisRequest`` into a ``PromptPackage``.

    The orchestration layer of prompt construction, and nothing more. It reads
    the request, asks the section registry for the renderer of each input that
    is present, collects the results into an internal ordered
    ``list[RenderedSection]``, then flattens that into the final system/user
    prompt strings and computes provenance metadata.

    It performs no I/O beyond reading a local template file, calls no LLM, does
    no AI reasoning, computes no indicators, and summarizes no news — every one
    of those belongs to another layer. It depends only on the
    ``SectionRenderer`` abstraction (via the registry), the request/package
    models, and settings.

    Deterministic by construction: sections are assembled in a fixed canonical
    order (never dict-iteration order), renderers are themselves deterministic,
    and no timestamp or random value enters the output. Identical requests
    therefore produce identical packages.
    """

    #: Bumped whenever produced prompt *text* changes (template wording, field
    #: whitelist, section order, cap semantics). Pure refactors do not bump it.
    PROMPT_VERSION = "1.0"

    #: System-prompt template filename (kept in step with PROMPT_VERSION).
    _SYSTEM_TEMPLATE = "system_v1.txt"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def build(self, request: AIAnalysisRequest) -> PromptPackage:
        """Build the prompt package for ``request``.

        Raises:
            PromptBuildError: if no renderable section is present (nothing to
                reason about), if a renderer lookup fails, or if the prompt
                otherwise cannot be constructed.
        """
        inputs = self._collect_inputs(request)
        if not inputs:
            raise PromptBuildError(
                f"cannot build a prompt for {request.ticker!r}: no technical, news, "
                "or additional inputs were supplied"
            )

        sections, skipped = self._render_sections(inputs)
        if not sections:
            # Every candidate input rendered nothing usable.
            raise PromptBuildError(
                f"cannot build a prompt for {request.ticker!r}: no section produced content"
            )

        system_prompt = _load_template(self._SYSTEM_TEMPLATE)
        user_prompt = self._assemble_user_prompt(request, sections)
        metadata = self._build_metadata(request, sections, skipped, system_prompt, user_prompt)

        logger.info(
            "Built prompt for %s: sections=%s version=%s chars=%d",
            request.ticker, [s.kind for s in sections], self.PROMPT_VERSION, metadata.total_chars,
        )

        return PromptPackage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=metadata,
            prompt_version=self.PROMPT_VERSION,
            model_hints=self._model_hints(request),
        )

    # --- Input collection -----------------------------------------------------

    def _collect_inputs(self, request: AIAnalysisRequest) -> dict[str, BaseModel]:
        """Map each present input to its section kind.

        Concrete fields map to their canonical kind; ``additional_inputs`` keys
        are taken as-is (a future 'risk' entry becomes the 'risk' section).
        Absent concrete inputs are simply not included here — they become
        ``sections_skipped`` later.
        """
        inputs: dict[str, BaseModel] = {}
        if request.technical is not None:
            inputs["technical"] = request.technical
        if request.news is not None:
            inputs["news"] = request.news
        for kind, model in request.additional_inputs.items():
            inputs[kind] = model
        return inputs

    def _ordered_kinds(self, present_kinds: set[str]) -> list[str]:
        """Return the present kinds in canonical order.

        Canonical kinds first (in their fixed sequence), then any extra kinds
        not yet in the canonical list, sorted for determinism — so an unknown
        future input still yields stable ordering without dict iteration.
        """
        ordered = [k for k in _CANONICAL_ORDER if k in present_kinds]
        extras = sorted(present_kinds - set(_CANONICAL_ORDER))
        return ordered + extras

    # --- Rendering ------------------------------------------------------------

    def _render_sections(
        self, inputs: dict[str, BaseModel]
    ) -> tuple[list[RenderedSection], list[str]]:
        """Render each input into a section, in canonical order.

        Returns the ordered sections plus the kinds that were skipped because a
        renderer produced an empty body. A missing renderer is fatal
        (``PromptBuildError`` from the registry) — an input the caller supplied
        but the system cannot render is a real error, not a silent drop.
        """
        sections: list[RenderedSection] = []
        skipped: list[str] = []

        for kind in self._ordered_kinds(set(inputs)):
            renderer = get_section_renderer(kind)  # raises PromptBuildError if unknown
            section = renderer.render(inputs[kind], max_items=self._cap_for(kind))
            if section.body:
                sections.append(section)
            else:
                skipped.append(kind)
        return sections, skipped

    def _cap_for(self, kind: str) -> int:
        """The item cap for a section kind, from settings."""
        if kind == "news":
            return self._settings.llm_max_news_articles
        return self._settings.llm_max_list_items

    # --- Assembly -------------------------------------------------------------

    def _assemble_user_prompt(
        self, request: AIAnalysisRequest, sections: list[RenderedSection]
    ) -> str:
        """Flatten the ordered sections into the user prompt string."""
        blocks: list[str] = [f"Security under analysis: {request.ticker}"]
        for section in sections:
            blocks.append(f"## {section.title}\n{section.body}")
        blocks.append(
            "Using only the information above, produce the JSON thesis exactly as "
            "specified in the system instructions."
        )
        return "\n\n".join(blocks)

    def _model_hints(self, request: AIAnalysisRequest) -> dict:
        """Optional hints for the consuming service. Only a caller-supplied
        model override is emitted; nothing is invented."""
        hints: dict = {}
        if request.model:
            hints["model"] = request.model
        return hints

    # --- Metadata -------------------------------------------------------------

    def _build_metadata(
        self,
        request: AIAnalysisRequest,
        sections: list[RenderedSection],
        skipped: list[str],
        system_prompt: str,
        user_prompt: str,
    ) -> PromptMetadata:
        """Compute deterministic provenance for the built prompt."""
        included = [s.kind for s in sections]
        renderer_versions = {kind: get_section_renderer(kind).version for kind in included}

        news_section = next((s for s in sections if s.kind == "news"), None)
        news_available = len(request.news.articles) if request.news is not None else 0
        news_included = news_section.item_count if news_section is not None else 0
        news_truncated = news_section.truncated if news_section is not None else False

        caps = {kind: self._cap_for(kind) for kind in included}

        return PromptMetadata(
            ticker=request.ticker,
            prompt_version=self.PROMPT_VERSION,
            renderer_versions=renderer_versions,
            sections_included=included,
            sections_skipped=skipped,
            news_articles_available=news_available,
            news_articles_included=news_included,
            news_truncated=news_truncated,
            list_caps_applied=caps,
            system_prompt_chars=len(system_prompt),
            user_prompt_chars=len(user_prompt),
        )
