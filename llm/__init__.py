"""LLM integration package for the AI Analysis Agent.

A distinct external-capability boundary — an LLM is neither a market-data nor
a news source — so it lives in its own top-level package rather than under
``data/``. Everything here depends only on ``config``, ``utils``, and
``models``; nothing in this package imports ``services``, ``agent``, or
``api``, which keeps the dependency graph acyclic and downward-only.

Phase 0 provides skeletons only. Concrete behaviour arrives in later phases:
  * Phase 2 — ``base`` (LLMProvider ABC + LLMResponse), ``exceptions``,
              ``ollama_provider``, ``provider_factory``.
"""
