from __future__ import annotations

from app.agent.context import AnswerMemory, ContextResolver, RecommendedAction
from app.services.context_service import ContextService
from app.services.user_service import UserService
from app.storage.sqlite_store import SQLiteStore
from app.schemas import AgentRequest


def test_user_crud_and_hard_delete_cascade(tmp_path):
    store = SQLiteStore(tmp_path / 'test.sqlite3')
    users = UserService(store)
    user = users.create({'display_name': 'Engineer A', 'role': 'maintenance'})
    users.upsert_session(user_id=user['user_id'], session_id='s1', title='demo')
    store.upsert_memory(user_id=user['user_id'], memory_type='equipment_preference', memory_key='CNC', content={'summary': 'CNC'})
    store.append({'run_id': 'r1', 'user_id': user['user_id'], 'session_id': 's1', 'request': {'question': 'q'}, 'response': {'route': [], 'manufacturing_context': {}}})

    assert users.get(user['user_id'])['display_name'] == 'Engineer A'
    updated = users.update(user['user_id'], {'department': 'plant_1'})
    assert updated['department'] == 'plant_1'

    deleted = users.delete(user['user_id'], mode='hard')

    assert deleted['deleted_counts'] == {'sessions': 1, 'memories': 1, 'runs': 1}
    assert store.list(user_id=user['user_id']) == []


def test_user_isolation_context_does_not_mix_runs(tmp_path):
    store = SQLiteStore(tmp_path / 'test.sqlite3')
    users = UserService(store)
    a = users.create({'display_name': 'A'})
    b = users.create({'display_name': 'B'})
    store.append({
        'run_id': 'run_a',
        'user_id': a['user_id'],
        'session_id': 'sa',
        'request': {'question': 'CNC spindle 점검'},
        'response': {
            'route': [],
            'manufacturing_context': {
                'risk_assessment': {'overall_priority': 'high'},
                'failure_modes': [{'code': 'OSF'}],
                'safety_gates': [{'gate_id': 'loto_if_physical_maintenance'}],
            },
        },
    })

    context = ContextService(store).build(user_id=b['user_id'], session_id='sb', request=AgentRequest(user_id=b['user_id'], question='지난번처럼 해줘'))

    assert context['recent_runs'] == []
    assert context['similar_runs'] == []


def test_context_resolver_uses_answer_memory_for_pronoun_followup():
    memory = AnswerMemory(
        selected_path='fast_concept_answer',
        answer_type='definition',
        short_summary='토크는 회전시키는 힘의 효과입니다.',
        focus='토크',
        claims=['토크는 회전 부품의 부하를 이해할 때 중요하다'],
    )

    result = ContextResolver().resolve(
        current_user_message='이것의 단점이 뭐야?',
        last_answer_memory=memory,
        recent_turns=[],
        rolling_summary=None,
    )

    assert result.is_followup is True
    assert result.followup_type == 'previous_concept'
    assert result.followup_target == '토크'
    assert '토크' in result.standalone_query


def test_context_resolver_does_not_use_other_user_without_answer_memory():
    result = ContextResolver().resolve(
        current_user_message='이것의 단점은?',
        last_answer_memory=None,
        recent_turns=[],
        rolling_summary=None,
    )

    assert result.is_followup is True
    assert result.followup_type == 'ambiguous'
    assert result.confidence < 0.5


def test_context_resolver_action_recap_uses_memory_actions_only():
    memory = AnswerMemory(
        selected_path='heavy_analysis_answer',
        answer_type='diagnosis',
        short_summary='토크와 공구 마모가 큰 상황의 점검 권장조치',
        focus='점검 및 안전 절차',
        recommended_actions=[
            RecommendedAction(id='a1', title='LOTO 확인', priority=1),
            RecommendedAction(id='a2', title='공구 마모 확인', priority=2),
        ],
    )

    result = ContextResolver().resolve(
        current_user_message='방금 권장조치 중요한 순서대로 알려줘',
        last_answer_memory=memory,
        recent_turns=[],
        rolling_summary=None,
    )

    assert result.followup_type == 'previous_recommended_actions'
    assert result.context_needed == ['last_answer_memory.recommended_actions']


def test_context_resolver_new_concept_not_polluted_by_heavy_memory():
    memory = AnswerMemory(
        selected_path='heavy_analysis_answer',
        answer_type='diagnosis',
        short_summary='토크와 공구 마모가 큰 상황의 점검 권장조치',
        focus='점검 및 안전 절차',
        recommended_actions=[RecommendedAction(id='a1', title='LOTO 확인')],
    )

    result = ContextResolver().resolve(
        current_user_message='LOTO가 뭐야?',
        last_answer_memory=memory,
        recent_turns=[],
        rolling_summary=None,
    )

    assert result.is_followup is False
    assert result.followup_type == 'none'


def test_context_budget_limits_items(tmp_path, monkeypatch):
    store = SQLiteStore(tmp_path / 'test.sqlite3')
    user = UserService(store).create({'display_name': 'A'})
    for idx in range(10):
        store.upsert_memory(
            user_id=user['user_id'],
            memory_type='equipment_preference',
            memory_key=f'CNC-{idx}',
            content={'summary': 'CNC spindle coolant ' * 80},
            importance=idx % 5,
        )
    monkeypatch.setattr('app.services.context_service.MAX_CONTEXT_TOKENS', 120)

    context = ContextService(store).build(user_id=user['user_id'], session_id='s1', request=AgentRequest(user_id=user['user_id'], question='CNC'))

    assert context['estimated_context_tokens'] <= 120
