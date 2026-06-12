from __future__ import annotations

from app.errors import AppError
from app.storage.sqlite_store import SQLiteStore


class UserNotFoundError(AppError):
    status_code = 404
    code = 'user_not_found'
    public_message = 'User not found'


class UserService:
    def __init__(self, store: SQLiteStore | None = None):
        self.store = store or SQLiteStore()

    def create(self, data: dict) -> dict:
        return self.store.create_user(data)

    def list(self, include_deleted: bool = False) -> list[dict]:
        return self.store.list_users(include_deleted=include_deleted)

    def get(self, user_id: str, include_deleted: bool = False) -> dict:
        user = self.store.get_user(user_id, include_deleted=include_deleted)
        if not user:
            raise UserNotFoundError()
        return user

    def update(self, user_id: str, data: dict) -> dict:
        user = self.store.update_user(user_id, data)
        if not user:
            raise UserNotFoundError()
        return user

    def delete(self, user_id: str, mode: str = 'hard') -> dict:
        self.get(user_id, include_deleted=True)
        return self.store.delete_user(user_id, mode=mode)

    def validate(self, user_id: str) -> dict:
        return self.get(user_id, include_deleted=False)

    def upsert_session(self, *, user_id: str, session_id: str, title: str | None = None) -> dict:
        self.validate(user_id)
        return self.store.upsert_session(user_id=user_id, session_id=session_id, title=title)


