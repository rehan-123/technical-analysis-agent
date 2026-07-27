from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from services.prompt_sections.base import RenderedSection, SectionRenderer


def _section(**overrides) -> RenderedSection:
    base = dict(kind="technical", title="Technical Analysis", body="trend: up")
    base.update(overrides)
    return RenderedSection(**base)


class TestRenderedSection:
    def test_required_fields(self):
        s = _section()
        assert s.kind == "technical"
        assert s.title == "Technical Analysis"
        assert s.body == "trend: up"

    def test_optional_fields_default(self):
        s = _section()
        assert s.item_count == 0
        assert s.truncated is False

    def test_empty_body_is_allowed(self):
        """A source with no content legitimately yields an empty body."""
        assert _section(body="").body == ""

    def test_empty_kind_or_title_is_rejected(self):
        with pytest.raises(ValidationError):
            _section(kind="")
        with pytest.raises(ValidationError):
            _section(title="")

    def test_negative_item_count_is_rejected(self):
        with pytest.raises(ValidationError):
            _section(item_count=-1)

    def test_is_immutable(self):
        s = _section()
        with pytest.raises(ValidationError):
            s.body = "changed"
        with pytest.raises(ValidationError):
            s.truncated = True

    def test_round_trips_through_json(self):
        original = _section(item_count=5, truncated=True)
        restored = RenderedSection.model_validate_json(original.model_dump_json())
        assert restored == original


class TestSectionRendererContract:
    def test_cannot_instantiate_the_abstract_base(self):
        with pytest.raises(TypeError):
            SectionRenderer()  # type: ignore[abstract]

    def test_subclass_without_render_cannot_be_instantiated(self):
        class Incomplete(SectionRenderer):
            kind = "x"

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_conforms_and_renders(self):
        class DummyRenderer(SectionRenderer):
            kind = "dummy"
            version = "9.9"

            def render(self, model: BaseModel, *, max_items: int) -> RenderedSection:
                return RenderedSection(kind=self.kind, title="Dummy", body="ok")

        renderer = DummyRenderer()
        assert renderer.kind == "dummy"
        assert renderer.version == "9.9"

        class SomeModel(BaseModel):
            value: int = 1

        section = renderer.render(SomeModel(), max_items=5)
        assert isinstance(section, RenderedSection)
        assert section.kind == "dummy"

    def test_base_class_declares_expected_defaults(self):
        """The base provides neutral defaults concrete renderers override."""
        assert SectionRenderer.kind == ""
        assert SectionRenderer.version == "1.0"

    def test_render_signature_is_keyword_only_for_max_items(self):
        """max_items is keyword-only, so call sites read explicitly."""
        import inspect

        params = inspect.signature(SectionRenderer.render).parameters
        assert params["max_items"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_abstraction_has_no_internal_dependencies(self):
        """It must remain a reusable leaf: no builder, registry, llm, or
        transport imports."""
        import inspect

        import services.prompt_sections.base as mod
        source = inspect.getsource(mod)
        # executable import statements only
        import_lines = [ln for ln in source.splitlines() if ln.strip().startswith(("import ", "from "))]
        joined = "\n".join(import_lines)
        for forbidden in ("prompt_builder", "registry", "llm", "httpx", "fastapi", "api.", "ollama"):
            assert forbidden not in joined, f"unexpected dependency: {forbidden}"
