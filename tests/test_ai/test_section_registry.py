from __future__ import annotations

import pytest

from services.prompt_sections.base import SectionRenderer
from services.prompt_sections.exceptions import PromptBuildError, PromptError
from services.prompt_sections.news_section import NewsSectionRenderer
from services.prompt_sections.registry import (
    _SECTION_REGISTRY,
    available_section_kinds,
    get_section_renderer,
)
from services.prompt_sections.technical_section import TechnicalSectionRenderer


class TestAvailableKinds:
    def test_lists_the_registered_kinds_sorted(self):
        kinds = available_section_kinds()
        assert set(kinds) == {"technical", "news", "portfolio"}
        assert list(kinds) == sorted(kinds)

    def test_returns_a_tuple(self):
        assert isinstance(available_section_kinds(), tuple)


class TestLookup:
    def test_technical_lookup_returns_the_right_renderer(self):
        renderer = get_section_renderer("technical")
        assert isinstance(renderer, TechnicalSectionRenderer)
        assert isinstance(renderer, SectionRenderer)

    def test_news_lookup_returns_the_right_renderer(self):
        renderer = get_section_renderer("news")
        assert isinstance(renderer, NewsSectionRenderer)

    def test_lookup_is_deterministic_same_instance_each_time(self):
        """Renderers are stateless singletons; repeated lookups return the
        same shared instance."""
        assert get_section_renderer("technical") is get_section_renderer("technical")

    def test_registered_kind_matches_renderer_kind_attribute(self):
        """The registry key must equal the renderer's own declared kind."""
        for kind in available_section_kinds():
            assert get_section_renderer(kind).kind == kind


class TestUnknownKind:
    def test_unknown_kind_raises_prompt_build_error(self):
        with pytest.raises(PromptBuildError):
            get_section_renderer("options")

    def test_error_message_lists_available_kinds(self):
        with pytest.raises(PromptBuildError, match="technical"):
            get_section_renderer("nope")

    def test_prompt_build_error_is_a_prompt_error(self):
        assert issubclass(PromptBuildError, PromptError)

    def test_empty_kind_raises(self):
        with pytest.raises(PromptBuildError):
            get_section_renderer("")


class TestImmutabilityAndUniqueness:
    def test_registry_is_immutable_at_runtime(self):
        with pytest.raises(TypeError):
            _SECTION_REGISTRY["options"] = object()  # type: ignore[index]

    def test_registry_values_are_section_renderers(self):
        assert all(isinstance(v, SectionRenderer) for v in _SECTION_REGISTRY.values())

    def test_no_duplicate_registrations(self):
        """Each renderer's kind is unique — keys and their renderers' kinds
        line up one-to-one with no collisions."""
        keys = list(_SECTION_REGISTRY.keys())
        assert len(keys) == len(set(keys))
        renderer_kinds = [r.kind for r in _SECTION_REGISTRY.values()]
        assert len(renderer_kinds) == len(set(renderer_kinds))
