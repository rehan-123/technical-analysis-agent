from __future__ import annotations

import time
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """A minimal, process-local, time-based cache.

    Deliberately simple: entries expire lazily on read (no background sweep),
    and there is no LRU/size bound beyond natural key churn — the Scanner's
    keys are bounded by (ticker, period, interval) or by scan-parameter
    signatures, both small relative to typical process memory at the scale
    this milestone targets.

    Owned entirely by the ``scanner`` package. It wraps calls the Scanner
    itself makes to the Technical/News agents; it does not modify, subclass,
    or reach into those agents, consistent with "only integrate with them".
    A ``ttl_seconds`` of ``0`` disables caching outright (every ``get``
    misses), which is useful for tests that must never see stale data.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._store: dict[K, tuple[float, V]] = {}

    def get(self, key: K) -> V | None:
        if self._ttl <= 0:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: K, value: V) -> None:
        if self._ttl <= 0:
            return
        self._store[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
