from __future__ import annotations

import pytest

from portfolio.portfolio_validation import PortfolioValidationError
from scanner.exceptions import WatchlistNotFoundError
from scanner.watchlist_store import WatchlistStore


@pytest.fixture
def store() -> WatchlistStore:
    return WatchlistStore()


def test_upsert_creates_a_watchlist(store):
    watchlist = store.upsert("tech", ["aapl", "msft", "googl"])
    assert watchlist.name == "tech"
    assert watchlist.symbols == ("AAPL", "MSFT", "GOOGL")


def test_upsert_deduplicates_preserving_order(store):
    watchlist = store.upsert("dupes", ["AAPL", "msft", "aapl"])
    assert watchlist.symbols == ("AAPL", "MSFT")


def test_upsert_replaces_existing_symbols_entirely(store):
    store.upsert("tech", ["AAPL", "MSFT"])
    replaced = store.upsert("tech", ["GOOGL"])
    assert replaced.symbols == ("GOOGL",)


def test_upsert_preserves_created_at_across_replacement(store):
    first = store.upsert("tech", ["AAPL"])
    second = store.upsert("tech", ["MSFT"])
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


def test_get_returns_stored_watchlist(store):
    store.upsert("tech", ["AAPL"])
    assert store.get("tech").symbols == ("AAPL",)


def test_get_missing_watchlist_raises(store):
    with pytest.raises(WatchlistNotFoundError):
        store.get("does-not-exist")


def test_get_or_none_returns_none_for_missing(store):
    assert store.get_or_none("nope") is None


def test_list_all_sorted_by_name(store):
    store.upsert("zeta", ["AAPL"])
    store.upsert("alpha", ["MSFT"])
    names = [w.name for w in store.list_all()]
    assert names == ["alpha", "zeta"]


def test_delete_removes_watchlist(store):
    store.upsert("tech", ["AAPL"])
    store.delete("tech")
    assert store.get_or_none("tech") is None


def test_delete_missing_watchlist_is_a_noop(store):
    store.delete("never-existed")  # must not raise


def test_upsert_rejects_invalid_symbol(store):
    with pytest.raises(PortfolioValidationError):
        store.upsert("bad", ["not a valid symbol!"])
