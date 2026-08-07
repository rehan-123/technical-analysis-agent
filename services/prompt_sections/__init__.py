"""Prompt section renderers for the AI Analysis Prompt Builder.

Each renderer is a small strategy that projects exactly one domain result
model (technical analysis, news, and future risk/macro/options/portfolio
results) into a single ``RenderedSection``. The ``PromptBuilder`` assembles
these sections into a final prompt; renderers themselves know nothing about
the builder, the LLM, HTTP, or AI reasoning.

This package depends only on ``models`` — never on the builder, a registry,
a concrete provider, or FastAPI — so the abstraction stays reusable and the
dependency graph stays downward.
"""
