from __future__ import annotations

from app.schemas import AgentRequest, AgentResponse
from app.services.user_service import UserService
from app.storage.sqlite_store import SQLiteStore


def test_agent_run_history_stores_user_id(tmp_path):
    store = SQLiteStore(tmp_path / 'test.sqlite3')
    user = UserService(store).create({'display_name': 'A'})
    request = AgentRequest(user_id=user['user_id'], question='q', session_id='s1')
    response = AgentResponse(run_id='r1', user_id=user['user_id'], session_id='s1', route=[], answer='ok', context_used={'user_id': user['user_id']})

    store.append({'run_id': response.run_id, 'user_id': user['user_id'], 'session_id': 's1', 'request': request.model_dump(), 'response': response.model_dump()})
    rows = store.list(user_id=user['user_id'])

    assert len(rows) == 1
    assert rows[0]['user_id'] == user['user_id']
    assert rows[0]['response']['context_used']['user_id'] == user['user_id']

