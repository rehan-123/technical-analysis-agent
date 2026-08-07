from __future__ import annotations

import time

from scanner.cache import TTLCache


def test_set_and_get_within_ttl():
    cache: TTLCache[str, int] = TTLCache(ttl_seconds=10)
    cache.set("a", 1)
    assert cache.get("a") == 1


def test_get_missing_key_returns_none():
    cache: TTLCache[str, int] = TTLCache(ttl_seconds=10)
    assert cache.get("missing") is None


def test_zero_ttl_disables_caching():
    cache: TTLCache[str, int] = TTLCache(ttl_seconds=0)
    cache.set("a", 1)
    assert cache.get("a") is None
    assert len(cache) == 0


def test_expired_entry_returns_none_and_is_evicted():
    cache: TTLCache[str, int] = TTLCache(ttl_seconds=0.05)
    cache.set("a", 1)
    time.sleep(0.08)
    assert cache.get("a") is None
    assert len(cache) == 0


def test_clear_removes_all_entries():
    cache: TTLCache[str, int] = TTLCache(ttl_seconds=10)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert len(cache) == 0
