from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import HISTORY_DB_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    """SQLite storage for users, context memories, sessions, and agent runs."""

    def __init__(self, path: Path | None = None):
        self.path = path or HISTORY_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    role TEXT,
                    department TEXT,
                    preferred_language TEXT DEFAULT 'ko',
                    report_style TEXT DEFAULT 'standard',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                )
                '''
            )
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
                '''
            )
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS user_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    memory_key TEXT,
                    content_json TEXT NOT NULL,
                    source_run_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    importance INTEGER DEFAULT 3,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
                '''
            )
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    risk_level TEXT,
                    route_json TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                '''
            )
            self._ensure_column(conn, 'agent_runs', 'user_id', 'TEXT')
            self._ensure_column(conn, 'user_memories', 'memory_key', 'TEXT')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_agent_runs_session_created ON agent_runs(session_id, created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_agent_runs_user_created ON agent_runs(user_id, created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_agent_runs_created ON agent_runs(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id, updated_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_memories_user ON user_memories(user_id, importance, updated_at)')
            conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_user_memory_unique ON user_memories(user_id, memory_type, memory_key)')

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row['name'] for row in conn.execute(f'PRAGMA table_info({table})').fetchall()}
        if column not in columns:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')

    def ready(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute('SELECT 1')
            return True
        except sqlite3.Error:
            return False

    def create_user(self, data: dict) -> dict:
        now = utc_now()
        user_id = data.get('user_id') or f'usr_{uuid4().hex[:12]}'
        record = {
            'user_id': user_id,
            'display_name': data['display_name'],
            'role': data.get('role'),
            'department': data.get('department'),
            'preferred_language': data.get('preferred_language') or 'ko',
            'report_style': data.get('report_style') or 'standard',
            'created_at': now,
            'updated_at': now,
            'deleted_at': None,
        }
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO users
                (user_id, display_name, role, department, preferred_language, report_style, created_at, updated_at, deleted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                tuple(record[k] for k in ['user_id', 'display_name', 'role', 'department', 'preferred_language', 'report_style', 'created_at', 'updated_at', 'deleted_at']),
            )
        return record

    def list_users(self, include_deleted: bool = False) -> list[dict]:
        where = '' if include_deleted else 'WHERE deleted_at IS NULL'
        with self._connect() as conn:
            rows = conn.execute(f'SELECT * FROM users {where} ORDER BY created_at DESC').fetchall()
        return [dict(row) for row in rows]

    def get_user(self, user_id: str, include_deleted: bool = False) -> dict | None:
        sql = 'SELECT * FROM users WHERE user_id = ?'
        params: tuple = (user_id,)
        if not include_deleted:
            sql += ' AND deleted_at IS NULL'
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def update_user(self, user_id: str, data: dict) -> dict | None:
        allowed = ['display_name', 'role', 'department', 'preferred_language', 'report_style']
        fields = [field for field in allowed if field in data and data[field] is not None]
        if not fields:
            return self.get_user(user_id)
        assignments = ', '.join([f'{field} = ?' for field in fields] + ['updated_at = ?'])
        values = [data[field] for field in fields] + [utc_now(), user_id]
        with self._connect() as conn:
            conn.execute(f'UPDATE users SET {assignments} WHERE user_id = ? AND deleted_at IS NULL', values)
        return self.get_user(user_id)

    def delete_user(self, user_id: str, mode: str = 'hard') -> dict:
        if mode not in {'hard', 'soft'}:
            mode = 'hard'
        with self._connect() as conn:
            if mode == 'soft':
                conn.execute('UPDATE users SET deleted_at = ?, updated_at = ? WHERE user_id = ?', (utc_now(), utc_now(), user_id))
                return {'deleted': True, 'mode': 'soft', 'deleted_counts': {'sessions': 0, 'memories': 0, 'runs': 0}}
            sessions = conn.execute('SELECT COUNT(*) AS count FROM user_sessions WHERE user_id = ?', (user_id,)).fetchone()['count']
            memories = conn.execute('SELECT COUNT(*) AS count FROM user_memories WHERE user_id = ?', (user_id,)).fetchone()['count']
            runs = conn.execute('SELECT COUNT(*) AS count FROM agent_runs WHERE user_id = ?', (user_id,)).fetchone()['count']
            conn.execute('DELETE FROM user_sessions WHERE user_id = ?', (user_id,))
            conn.execute('DELETE FROM user_memories WHERE user_id = ?', (user_id,))
            conn.execute('DELETE FROM agent_runs WHERE user_id = ?', (user_id,))
            conn.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
        return {'deleted': True, 'mode': 'hard', 'deleted_counts': {'sessions': sessions, 'memories': memories, 'runs': runs}}

    def upsert_session(self, *, user_id: str, session_id: str, title: str | None = None) -> dict:
        now = utc_now()
        with self._connect() as conn:
            existing = conn.execute('SELECT * FROM user_sessions WHERE session_id = ?', (session_id,)).fetchone()
            if existing:
                conn.execute('UPDATE user_sessions SET user_id = ?, title = COALESCE(?, title), updated_at = ? WHERE session_id = ?', (user_id, title, now, session_id))
            else:
                conn.execute(
                    'INSERT INTO user_sessions (session_id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)',
                    (session_id, user_id, title, now, now),
                )
            row = conn.execute('SELECT * FROM user_sessions WHERE session_id = ?', (session_id,)).fetchone()
        return dict(row)

    def upsert_memory(self, *, user_id: str, memory_type: str, memory_key: str, content: dict, source_run_id: str | None = None, confidence: float = 1.0, importance: int = 3) -> dict:
        now = utc_now()
        with self._connect() as conn:
            existing = conn.execute(
                'SELECT * FROM user_memories WHERE user_id = ? AND memory_type = ? AND memory_key = ?',
                (user_id, memory_type, memory_key),
            ).fetchone()
            if existing:
                old_content = json.loads(existing['content_json'])
                content['count'] = int(old_content.get('count') or 0) + int(content.get('count') or 1)
                conn.execute(
                    '''
                    UPDATE user_memories
                    SET content_json = ?, source_run_id = COALESCE(?, source_run_id), confidence = ?, importance = ?, updated_at = ?
                    WHERE memory_id = ?
                    ''',
                    (json.dumps(content, ensure_ascii=False, default=str), source_run_id, confidence, importance, now, existing['memory_id']),
                )
                memory_id = existing['memory_id']
            else:
                memory_id = f'mem_{uuid4().hex[:12]}'
                content.setdefault('count', 1)
                conn.execute(
                    '''
                    INSERT INTO user_memories
                    (memory_id, user_id, memory_type, memory_key, content_json, source_run_id, confidence, importance, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (memory_id, user_id, memory_type, memory_key, json.dumps(content, ensure_ascii=False, default=str), source_run_id, confidence, importance, now, now),
                )
            row = conn.execute('SELECT * FROM user_memories WHERE memory_id = ?', (memory_id,)).fetchone()
        return self._memory_row(row)

    def list_memories(self, user_id: str, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                '''
                SELECT * FROM user_memories
                WHERE user_id = ? AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                ''',
                (user_id, utc_now(), min(max(limit, 1), 200)),
            ).fetchall()
        return [self._memory_row(row) for row in rows]

    def delete_memories(self, user_id: str) -> int:
        with self._connect() as conn:
            count = conn.execute('SELECT COUNT(*) AS count FROM user_memories WHERE user_id = ?', (user_id,)).fetchone()['count']
            conn.execute('DELETE FROM user_memories WHERE user_id = ?', (user_id,))
        return count

    def append(self, record: dict) -> None:
        response = record.get('response') or {}
        request = record.get('request') or {}
        route = response.get('route') or []
        mfg = response.get('manufacturing_context') or {}
        risk_level = (mfg.get('risk_assessment') or {}).get('overall_priority') if isinstance(mfg, dict) else None
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT OR REPLACE INTO agent_runs
                (run_id, user_id, session_id, created_at, risk_level, route_json, request_json, response_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    record.get('run_id'),
                    record.get('user_id'),
                    record.get('session_id'),
                    utc_now(),
                    risk_level,
                    json.dumps(route, ensure_ascii=False, default=str),
                    json.dumps(request, ensure_ascii=False, default=str),
                    json.dumps(response, ensure_ascii=False, default=str),
                ),
            )

    def list(self, limit: int = 50, user_id: str | None = None) -> list[dict]:
        limit = min(max(int(limit or 50), 1), 500)
        where = 'WHERE user_id = ?' if user_id else ''
        params = (user_id, limit) if user_id else (limit,)
        with self._connect() as conn:
            rows = conn.execute(
                f'''
                SELECT run_id, user_id, session_id, created_at, risk_level, route_json, request_json, response_json
                FROM agent_runs
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                ''',
                params,
            ).fetchall()
        return [self._run_row(row) for row in rows]

    def get(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                '''
                SELECT run_id, user_id, session_id, created_at, risk_level, route_json, request_json, response_json
                FROM agent_runs
                WHERE run_id = ?
                ''',
                (run_id,),
            ).fetchone()
        return self._run_row(row) if row else None

    def recent_runs(self, user_id: str, limit: int = 3) -> list[dict]:
        return self.list(limit=limit, user_id=user_id)

    @staticmethod
    def _memory_row(row: sqlite3.Row) -> dict:
        record = dict(row)
        record['content'] = json.loads(record.pop('content_json'))
        return record

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict:
        return {
            'run_id': row['run_id'],
            'user_id': row['user_id'],
            'session_id': row['session_id'],
            'created_at': row['created_at'],
            'risk_level': row['risk_level'],
            'route': json.loads(row['route_json']),
            'request': json.loads(row['request_json']),
            'response': json.loads(row['response_json']),
        }


AgentRunStore = SQLiteStore
