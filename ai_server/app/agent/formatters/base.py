from __future__ import annotations

from typing import Any, Protocol


class Formatter(Protocol):
    def format(self, context: dict[str, Any]) -> str:
        ...
