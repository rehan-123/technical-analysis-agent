from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from services.prompt_sections.base import SectionRenderer
from services.prompt_sections.exceptions import PromptBuildError
from portfolio.portfolio_renderer import PortfolioSectionRenderer
from services.prompt_sections.news_section import NewsSectionRenderer
from services.prompt_sections.technical_section import TechnicalSectionRenderer

# ---------------------------------------------------------------------------
# Immutable section registry: maps a section ``kind`` to its renderer.
#
# Why singleton renderer instances (constructed once, here):
#   Renderers are stateless, deterministic transforms — they hold no per-call
#   state, so a single shared instance is safe to reuse across every prompt
#   build. Constructing them once avoids needless allocation and makes the
#   registry a plain value lookup rather than a factory. (This is the opposite
#   choice from the provider factories, which defer *construction* because
#   providers validate config and hold I/O resources; renderers have neither
#   concern, so eager instantiation is correct here.)
#
# Why immutable (MappingProxyType):
#   The set of known section kinds is fixed at import time and auditable by
#   reading this file. Immutability guarantees no code path can register,
#   replace, or remove a renderer at runtime, so a given deployment's section
#   vocabulary cannot drift and lookups are deterministic. No global mutable
#   state exists.
#
# Why the registry lives separately from PromptBuilder:
#   Separation of concerns. The registry answers "which renderer handles this
#   kind?"; the builder answers "how are sections ordered and assembled into a
#   prompt?". Keeping them apart means a future renderer is added here (one
#   entry) without touching the builder, and the builder can be tested against
#   a substitute mapping without this module. It is the open/closed seam.
# ---------------------------------------------------------------------------

_SECTION_REGISTRY: Final[Mapping[str, SectionRenderer]] = MappingProxyType(
    {
        TechnicalSectionRenderer.kind: TechnicalSectionRenderer(),
        NewsSectionRenderer.kind: NewsSectionRenderer(),
        PortfolioSectionRenderer.kind: PortfolioSectionRenderer(),
    }
)


def available_section_kinds() -> tuple[str, ...]:
    """Return every registered section kind, sorted.

    Useful for diagnostics and for error messages that tell a caller which
    kinds are actually renderable.
    """
    return tuple(sorted(_SECTION_REGISTRY))


def get_section_renderer(kind: str) -> SectionRenderer:
    """Return the renderer registered for ``kind``.

    Args:
        kind: The section kind to look up (e.g. ``"technical"``, ``"news"``).

    Returns:
        The shared ``SectionRenderer`` instance for that kind.

    Raises:
        PromptBuildError: if no renderer is registered for ``kind`` — a
            fail-fast signal that a caller asked for a section the system does
            not know how to render.
    """
    renderer = _SECTION_REGISTRY.get(kind)
    if renderer is None:
        raise PromptBuildError(
            f"No section renderer registered for kind {kind!r}. "
            f"Available kinds: {', '.join(available_section_kinds())}"
        )
    return renderer
