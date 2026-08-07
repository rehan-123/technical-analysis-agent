from __future__ import annotations

from datetime import datetime, timezone

from models.opportunity import Watchlist
from portfolio.portfolio_validation import normalize_symbol
from scanner.exceptions import WatchlistNotFoundError


class WatchlistStore:
    """In-memory named-watchlist storage.

    Mirrors the existing Portfolio layer's storage decision (see
    ``api/portfolio_routes.py``'s module docstring): persistence is
    deliberately out of scope for this milestone. Unlike the Portfolio
    routes' module-level global, state here lives inside an injectable
    class, which keeps this store trivially unit-testable without
    monkeypatching module globals — a small, self-contained improvement on
    the existing pattern rather than a change to it.
    """

    def __init__(self) -> None:
        self._watchlists: dict[str, Watchlist] = {}

    def upsert(self, name: str, symbols: list[str]) -> Watchlist:
        """Create or fully replace the named watchlist's symbol set.

        Symbols are normalized (uppercased, validated) via the same
        ``normalize_symbol`` the Portfolio domain uses, and de-duplicated
        while preserving first-seen order.
        """
        normalized = tuple(dict.fromkeys(normalize_symbol(s) for s in symbols))
        existing = self._watchlists.get(name)
        now = datetime.now(timezone.utc)
        watchlist = Watchlist(
            name=name,
            symbols=normalized,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self._watchlists[name] = watchlist
        return watchlist

    def get(self, name: str) -> Watchlist:
        """Raises:
        WatchlistNotFoundError: if no watchlist named ``name`` exists.
        """
        try:
            return self._watchlists[name]
        except KeyError as exc:
            raise WatchlistNotFoundError(f"No watchlist named {name!r}.") from exc

    def get_or_none(self, name: str) -> Watchlist | None:
        return self._watchlists.get(name)

    def list_all(self) -> list[Watchlist]:
        return sorted(self._watchlists.values(), key=lambda w: w.name)

    def delete(self, name: str) -> None:
        self._watchlists.pop(name, None)
