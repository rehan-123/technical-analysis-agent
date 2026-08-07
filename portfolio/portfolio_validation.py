from __future__ import annotations

import re

#: Symbols are normalized to upper case and constrained to the characters used
#: by common venues (letters, digits, dot, hyphen). Deliberately venue-neutral:
#: no broker-specific formats are assumed.
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


class PortfolioError(Exception):
    """Base class for every error raised by the portfolio domain.

    Kept inside the domain, mirroring the News, LLM, and prompt packages, so
    boundaries stay consistent without a shared catch-all module.
    """


class PortfolioValidationError(PortfolioError, ValueError):
    """Raised when portfolio input violates a domain rule.

    Covers negative cash or quantities, duplicate holdings, impossible
    allocations, and invalid symbols. Model-level constraints (bounds, types)
    are enforced by Pydantic; this represents rules Pydantic cannot express.

    It subclasses ``ValueError`` as well as ``PortfolioError`` so that the two
    contexts it is raised from both behave correctly:

    * inside a Pydantic validator, Pydantic recognises ``ValueError`` and wraps
      it into a standard ``ValidationError`` — so constructing an invalid model
      fails the same way as any other model in the codebase;
    * called directly from the service or manager layer, it is still catchable
      as ``PortfolioValidationError`` (or ``PortfolioError``) with its full
      domain meaning intact.

    This keeps model semantics conventional without weakening the rule itself.
    """


class InsufficientFundsError(PortfolioError):
    """Raised when an operation would overdraw available cash."""


class HoldingNotFoundError(PortfolioError):
    """Raised when an operation targets a symbol the portfolio does not hold."""


def normalize_symbol(symbol: str) -> str:
    """Return the canonical form of ``symbol``.

    Raises:
        PortfolioValidationError: if the symbol is empty or malformed.
    """
    candidate = (symbol or "").strip().upper()
    if not candidate:
        raise PortfolioValidationError("symbol must not be empty")
    if not _SYMBOL_RE.match(candidate):
        raise PortfolioValidationError(f"invalid symbol: {symbol!r}")
    return candidate


def require_unique_symbols(symbols: list[str]) -> None:
    """Reject duplicate holdings — a portfolio holds one position per symbol.

    Raises:
        PortfolioValidationError: naming the duplicates, sorted for determinism.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for symbol in symbols:
        (duplicates if symbol in seen else seen).add(symbol)
    if duplicates:
        raise PortfolioValidationError(
            f"duplicate holdings: {', '.join(sorted(duplicates))}"
        )


def require_percentage(value: float, *, label: str) -> float:
    """Validate that ``value`` is a percentage in 0-100."""
    if not 0.0 <= value <= 100.0:
        raise PortfolioValidationError(f"{label} must be between 0 and 100, got {value}")
    return value
