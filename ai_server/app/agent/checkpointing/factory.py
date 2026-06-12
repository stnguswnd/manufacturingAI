from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver


class CheckpointerHandle:
    def __init__(self, *, context_manager: Any, checkpointer: Any):
        self.context_manager = context_manager
        self.checkpointer = checkpointer

    def close(self) -> None:
        self.context_manager.__exit__(None, None, None)


def create_sqlite_checkpointer(path: Path) -> CheckpointerHandle:
    path.parent.mkdir(parents=True, exist_ok=True)
    context_manager = SqliteSaver.from_conn_string(str(path))
    return CheckpointerHandle(context_manager=context_manager, checkpointer=context_manager.__enter__())
