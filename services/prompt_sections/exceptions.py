from __future__ import annotations


class PromptError(Exception):
    """Base class for every error raised within the prompt-building domain.

    Kept inside the prompt domain (not ``utils/``), mirroring how the News and
    LLM domains own their exceptions, so the boundaries stay structurally
    consistent without a shared catch-all module.
    """


class PromptBuildError(PromptError):
    """Raised when a prompt (or one of its sections) cannot be assembled.

    Covers, for example: a requested section ``kind`` with no registered
    renderer, or (later, in the builder) a request that carries no content to
    build a prompt from. Raised eagerly and with a clear message so a
    configuration/wiring mistake fails fast rather than producing a silently
    incomplete prompt.
    """
