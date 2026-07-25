from __future__ import annotations

import logging
import sys
from functools import lru_cache

from config.settings import get_settings


def _configure_root_logger() -> None:
    settings = get_settings()
    root = logging.getLogger("ta_agent")
    if root.handlers:
        return  # already configured — avoid duplicate handlers on reimport

    root.setLevel(settings.log_level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.propagate = False


@lru_cache
def get_logger(name: str) -> logging.Logger:
    """Returns a namespaced logger under the ``ta_agent`` root logger."""
    _configure_root_logger()
    return logging.getLogger(f"ta_agent.{name}")
