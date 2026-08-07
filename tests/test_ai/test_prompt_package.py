from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.prompt_package import PromptMetadata, PromptPackage


def _metadata(**overrides) -> PromptMetadata:
    base = dict(ticker="AAPL", prompt_version="1.0")
    base.update(overrides)
    return PromptMetadata(**base)


def _package(**overrides) -> PromptPackage:
    base = dict(
        system_prompt="SYS",
        user_prompt="USER",
        metadata=_metadata(),
        prompt_version="1.0",
    )
    base.update(overrides)
    return PromptPackage(**base)


class TestPromptMetadata:
    def test_required_fields(self):
        md = _metadata()
        assert md.ticker == "AAPL"
        assert md.prompt_version == "1.0"

    def test_ticker_and_version_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            PromptMetadata(ticker="", prompt_version="1.0")
        with pytest.raises(ValidationError):
            PromptMetadata(ticker="AAPL", prompt_version="")

    def test_collection_defaults_are_empty(self):
        md = _metadata()
        assert md.renderer_versions == {}
        assert md.sections_included == [] and md.sections_skipped == []
        assert md.list_caps_applied == {}

    def test_counts_default_to_zero_and_reject_negatives(self):
        md = _metadata()
        assert md.news_articles_available == 0 and md.news_articles_included == 0
        assert md.news_truncated is False
        with pytest.raises(ValidationError):
            _metadata(news_articles_available=-1)
        with pytest.raises(ValidationError):
            _metadata(system_prompt_chars=-5)

    def test_total_chars_is_computed_from_parts(self):
        md = _metadata(system_prompt_chars=30, user_prompt_chars=70)
        assert md.total_chars == 100

    def test_total_chars_cannot_be_set_directly(self):
        """It is derived, so it can never drift from its parts."""
        md = _metadata(system_prompt_chars=10, user_prompt_chars=5, total_chars=999)
        assert md.total_chars == 15

    def test_is_immutable(self):
        md = _metadata()
        with pytest.raises(ValidationError):
            md.ticker = "MSFT"

    def test_serializes_with_computed_total(self):
        payload = _metadata(system_prompt_chars=1, user_prompt_chars=2).model_dump()
        assert payload["total_chars"] == 3


class TestPromptPackage:
    def test_required_fields_present(self):
        pkg = _package()
        assert pkg.system_prompt == "SYS"
        assert pkg.user_prompt == "USER"
        assert pkg.prompt_version == "1.0"
        assert isinstance(pkg.metadata, PromptMetadata)

    def test_model_hints_default_empty(self):
        assert _package().model_hints == {}

    def test_model_hints_accepts_heterogeneous_values(self):
        pkg = _package(model_hints={"model": "qwen2.5:14b", "temperature": 0.1, "schema": {"type": "object"}})
        assert pkg.model_hints["model"] == "qwen2.5:14b"
        assert pkg.model_hints["temperature"] == 0.1

    def test_is_immutable(self):
        pkg = _package()
        with pytest.raises(ValidationError):
            pkg.user_prompt = "changed"

    def test_prompt_version_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            _package(prompt_version="")

    def test_version_mismatch_between_package_and_metadata_is_rejected(self):
        """The canonical version and the metadata copy must agree."""
        with pytest.raises(ValidationError):
            PromptPackage(
                system_prompt="s", user_prompt="u",
                metadata=_metadata(prompt_version="1.0"),
                prompt_version="2.0",
            )

    def test_matching_versions_are_accepted(self):
        pkg = PromptPackage(
            system_prompt="s", user_prompt="u",
            metadata=_metadata(prompt_version="3.1"),
            prompt_version="3.1",
        )
        assert pkg.prompt_version == pkg.metadata.prompt_version

    def test_round_trips_through_json(self):
        original = _package(
            model_hints={"model": "qwen2.5:7b"},
            metadata=_metadata(sections_included=["technical", "news"], news_articles_included=5),
        )
        restored = PromptPackage.model_validate_json(original.model_dump_json())
        assert restored.system_prompt == original.system_prompt
        assert restored.metadata.sections_included == ["technical", "news"]
        assert restored.metadata.news_articles_included == 5
        assert restored.model_hints == {"model": "qwen2.5:7b"}

    def test_package_holds_no_service_layer_types(self):
        """The package must stay a models-layer leaf: only strings, dicts, and
        its own metadata — never a services-layer RenderedSection — so the
        dependency graph stays downward.

        Inspected via AST (imports + field annotations), not a raw-source
        string match, so documentation prose mentioning ``RenderedSection`` or
        ``services`` does not produce a false positive.
        """
        import ast
        import inspect

        import models.prompt_package as mod

        tree = ast.parse(inspect.getsource(mod))

        # 1. No import pulls anything from the services layer.
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
        assert not any(m.split(".")[0] == "services" for m in imported_modules)

        # 2. No field on either model is annotated with a services-layer type.
        annotations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and node.annotation is not None:
                annotations.append(ast.unparse(node.annotation))
        assert not any("RenderedSection" in ann for ann in annotations)
