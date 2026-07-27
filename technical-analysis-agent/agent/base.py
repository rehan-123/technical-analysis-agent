from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Common contract for every specialist agent in the platform
    (Technical, News, Risk, Macro, Options, ...).

    A future Chief Decision Agent orchestrates multiple ``BaseAgent``
    implementations and only ever talks to this interface — it never
    needs to know about an individual agent's internals, data sources,
    or scoring logic. Any agent that implements ``run()`` can be added
    to the orchestration roster without touching existing agents.
    """

    name: str

    @abstractmethod
    async def run(self, ticker: str, **kwargs: Any) -> Any:
        """Execute the agent's analysis for a given ticker and return its
        structured result."""
        raise NotImplementedError
