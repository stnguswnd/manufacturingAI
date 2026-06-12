from __future__ import annotations

from uuid import uuid4


def build_thread_id(*, user_id: str | None, session_id: str | None) -> str:
    safe_user = user_id or 'anonymous'
    safe_session = session_id or f'session_{uuid4().hex[:12]}'
    return f'{safe_user}:{safe_session}'
