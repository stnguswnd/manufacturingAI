from __future__ import annotations

import argparse
from pathlib import Path

from app.config import LANGGRAPH_CHECKPOINT_DB


def reset_sqlite_checkpoint(path: Path) -> bool:
    """Delete only the LangGraph SQLite checkpoint file.

    This is an explicit dev/test helper. It does not touch application tables,
    history stores, users, memories, or vector DB files.
    """

    if path.exists():
        path.unlink()
        return True
    return False


def main() -> None:
    default_path = LANGGRAPH_CHECKPOINT_DB.with_name(f'{LANGGRAPH_CHECKPOINT_DB.stem}_v2{LANGGRAPH_CHECKPOINT_DB.suffix}')
    parser = argparse.ArgumentParser(description='Reset only the v2 LangGraph SQLite checkpoint file.')
    parser.add_argument('--path', type=Path, default=default_path)
    args = parser.parse_args()
    deleted = reset_sqlite_checkpoint(args.path)
    status = 'deleted' if deleted else 'not_found'
    print(f'{status}: {args.path}')


if __name__ == '__main__':
    main()
