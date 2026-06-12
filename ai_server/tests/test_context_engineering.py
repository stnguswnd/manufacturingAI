from __future__ import annotations

import inspect
import importlib.util

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app.agent.checkpointing import build_thread_id, reset_sqlite_checkpoint
from app.agent.graph import ManufacturingAgentGraph
from app.agent.root_graph import RootManufacturingGraph
from app.agent.context import AnswerMemory, ContextCompressor, ContextPackBuilder, ContextResolver, ContextValidator, RecommendedAction
from app.agent.heavy import CitationBuilder, DiagnosticPlan, DiagnosticPlanToAgentPlanTranslator, DiagnosticPlanner, EvidenceGrader, PlanningResult, RagQueryPlanner
from app.agent.formatters import FormatterRegistry
from app.agent.safety import SafetyContext, SafetyFormatter
from app.schemas import AgentRequest, AgentSendRequest, RagChunk
from app.services.answer_formatter_service import AnswerFormatterService
from app.services.context_service import ContextService
from app.services.intent_gateway_service import IntentGatewayService
from app.services.memory_service import MemoryService
from app.services.prediction_service import PredictionService
from app.services.rag_service import RagService
from app.services.structured_output_schema import to_openai_strict_json_schema
from app.services.supervisor_service import SupervisorService
from app.services.user_service import UserService
from app.storage.sqlite_store import SQLiteStore


class NoopLLMService:
    model = 'fake-model'
    enabled = False
    last_error = None

    def generate_json(self, **kwargs):
        self.last_error = 'noop llm disabled'
        return None


def make_root(tmp_path):
    store = SQLiteStore(tmp_path / 'test.sqlite3')
    users = UserService(store)
    rag = RagService()
    llm = NoopLLMService()
    return RootManufacturingGraph(
        store=store,
        user_service=users,
        context_service=ContextService(store),
        memory_service=MemoryService(store),
        heavy_graph=ManufacturingAgentGraph(PredictionService(model_path=tmp_path / 'model.joblib'), rag, llm),
        llm_service=llm,
        rag_service=rag,
        checkpoint_path=tmp_path / 'checkpoints' / 'v2.sqlite3',
    ), users


def assert_json_like(value):
    if isinstance(value, BaseModel):
        raise AssertionError(f'Pydantic object leaked into checkpoint: {type(value).__name__}')
    if isinstance(value, dict):
        for item in value.values():
            assert_json_like(item)
        return
    if isinstance(value, list):
        for item in value:
            assert_json_like(item)
        return
    assert value is None or isinstance(value, (str, int, float, bool))


def memory_with_actions() -> AnswerMemory:
    return AnswerMemory(
        selected_path='heavy_analysis_answer',
        answer_type='diagnosis',
        short_summary='토크와 공구 마모가 큰 상황의 점검 권장조치',
        focus='점검 및 안전 절차',
        recommended_actions=[
            RecommendedAction(id='a1', title='LOTO 확인', rationale='물리 점검 전 예기치 않은 가동을 막기 위해 필요합니다.', safety_note='정비 전 에너지 차단 확인', priority=1),
            RecommendedAction(id='a2', title='공구 마모 확인', rationale='마모가 크면 절삭 저항과 품질 문제가 커질 수 있습니다.', priority=2),
            RecommendedAction(id='a3', title='스핀들 부하 확인', rationale='부하 상승 원인을 좁히기 위한 확인입니다.', priority=3),
        ],
    )


def test_context_resolver_classifies_reason_followup_with_answer_memory():
    memory = AnswerMemory(
        selected_path='fast_concept_answer',
        answer_type='watch_points',
        short_summary='공구 마모는 여러 지표와 함께 봐야 한다.',
        focus='공구 마모',
        claims=['공구 마모는 값 하나만 보지 말고 여러 지표와 함께 봐야 한다'],
    )

    resolved = ContextResolver().resolve(
        current_user_message='왜?',
        last_answer_memory=memory,
        recent_turns=[],
        rolling_summary=None,
    )

    assert resolved.is_followup is True
    assert resolved.followup_type == 'previous_answer_reason'
    assert resolved.followup_target == '공구 마모'
    assert '여러 지표' in resolved.standalone_query


def test_context_resolver_classifies_reason_without_memory_as_ambiguous():
    resolved = ContextResolver().resolve(
        current_user_message='왜?',
        last_answer_memory=None,
        recent_turns=[],
        rolling_summary=None,
    )

    assert resolved.is_followup is True
    assert resolved.followup_type == 'ambiguous'
    assert resolved.confidence < 0.5


def test_context_resolver_classifies_recommended_action_recap():
    resolved = ContextResolver().resolve(
        current_user_message='방금 권장조치 중요한 순서대로 알려줘',
        last_answer_memory=memory_with_actions(),
        recent_turns=[],
        rolling_summary=None,
    )

    assert resolved.followup_type == 'previous_recommended_actions'
    assert resolved.context_needed == ['last_answer_memory.recommended_actions']


def test_context_resolver_classifies_action_item_followup_with_index():
    resolved = ContextResolver().resolve(
        current_user_message='그중 2번은 왜 필요한데?',
        last_answer_memory=memory_with_actions(),
        recent_turns=[],
        rolling_summary=None,
    )

    assert resolved.followup_type == 'previous_recommended_action_item'
    assert resolved.followup_item_index == 2
    assert resolved.followup_target == '공구 마모 확인'


def test_context_resolver_treats_new_concept_as_standalone_even_with_heavy_memory():
    resolved = ContextResolver().resolve(
        current_user_message='LOTO가 뭐야?',
        last_answer_memory=memory_with_actions(),
        recent_turns=[],
        rolling_summary=None,
    )

    assert resolved.is_followup is False
    assert resolved.followup_type == 'none'


def test_context_pack_builder_excludes_raw_docs_and_messages_from_classifier_context():
    resolved = ContextResolver().resolve(
        current_user_message='방금 권장조치 순서대로 알려줘',
        last_answer_memory=memory_with_actions(),
        recent_turns=[],
        rolling_summary=None,
    )
    packs = ContextPackBuilder().build(
        current_user_message='방금 권장조치 순서대로 알려줘',
        context_resolution=resolved,
        compressed_context={'rolling_summary': '', 'recent_turns': [{'role': 'user', 'content': 'x'}]},
        last_answer_memory=memory_with_actions(),
        recent_turn_routes=[],
        process_data_policy={},
    )

    forbidden = {'retrieved_docs', 'retrieved_documents', 'rag_contexts', 'messages', 'full_conversation', 'internal_reason'}
    assert forbidden.isdisjoint(packs.classifier_context)
    assert packs.formatter_context['answer_type'] == 'recommended_action_recap'
    assert 'recommended_actions' in packs.formatter_context


def test_context_pack_builder_allows_action_item_only_for_item_explanation():
    resolved = ContextResolver().resolve(
        current_user_message='그중 2번은 왜 필요한데?',
        last_answer_memory=memory_with_actions(),
        recent_turns=[],
        rolling_summary=None,
    )
    packs = ContextPackBuilder().build(
        current_user_message='그중 2번은 왜 필요한데?',
        context_resolution=resolved,
        compressed_context={'rolling_summary': '', 'recent_turns': []},
        last_answer_memory=memory_with_actions(),
        recent_turn_routes=[],
        process_data_policy={},
    )

    assert packs.formatter_context['answer_type'] == 'recommended_action_item_explanation'
    assert packs.formatter_context['recommended_action_item']['title'] == '공구 마모 확인'
    assert 'recommended_actions' not in packs.formatter_context


def test_formatter_registry_renders_recommended_action_recap_only_when_requested():
    registry = FormatterRegistry()
    context = {
        'answer_type': 'recommended_action_recap',
        'recommended_actions': [action.model_dump() for action in memory_with_actions().recommended_actions],
    }

    answer = registry.format('recommended_action_recap', context)

    assert '1. LOTO 확인' in answer
    assert '2. 공구 마모 확인' in answer


def test_formatter_does_not_infer_recap_from_actions_on_general_path():
    registry = FormatterRegistry()
    answer = registry.format('general_lightweight_answer', {
        'answer_type': 'explanation',
        'recommended_actions': [action.model_dump() for action in memory_with_actions().recommended_actions],
    })

    assert '직전 답변의 권장조치' not in answer


def test_recommended_action_item_formatter_uses_item_rationale_and_safety_note():
    action = memory_with_actions().recommended_actions[0].model_dump()
    answer = FormatterRegistry().format('recommended_action_item_explanation', {
        'followup_item_index': 1,
        'recommended_action_item': action,
    })

    assert '예기치 않은 가동' in answer
    assert '정비 전 에너지 차단 확인' in answer


def test_fast_concept_formatter_does_not_leak_heavy_format():
    answer = FormatterRegistry().format('fast_concept_answer', {
        'answer_type': 'definition',
        'concept_payload': {
            'term': '토크',
            'definition': '토크는 물체를 회전시키는 힘의 효과입니다.',
            'manufacturing_context': '회전 부품 부하를 이해할 때 사용합니다.',
            'watch_points': ['회전수와 함께 볼 것'],
            'risks': ['발열'],
            'risk_boundary': '현재 설비 판단은 공정 데이터가 필요합니다.',
            'related_terms': ['회전수'],
        },
    })

    assert all(token not in answer for token in ['판정', '주요 근거', '위험도', '안전 확인', '권장 조치'])


def test_safety_formatter_includes_must_include_constraints():
    answer = SafetyFormatter().format(SafetyContext(
        must_include=['정비 전 LOTO 확인', '회전부 접근 전 방호장치 확인'],
        forbidden=['AI가 안전 상태를 보증한다고 말하지 않기'],
        disclaimer_level='strict',
    ))

    assert '정비 전 LOTO 확인' in answer
    assert 'AI가 안전 상태를 보증한다고 말하지 않기' in answer


def test_fallback_formatter_uses_public_reason_only():
    answer = AnswerFormatterService().unsupported_or_clarification(
        reason='BadRequestError invalid_json_schema additionalProperties missing'
    )

    assert 'BadRequestError' not in answer
    assert 'invalid_json_schema' not in answer
    assert '요청 의도나 참조 대상을 안정적으로 확정하지 못했습니다.' in answer


def test_answer_memory_stores_structured_actions_and_does_not_force_claims():
    memory = AnswerMemory(
        selected_path='heavy_analysis_answer',
        answer_type='diagnosis',
        short_summary='점검 권장',
        recommended_actions=[RecommendedAction(id='a1', title='LOTO 확인')],
    )

    assert memory.recommended_actions[0].title == 'LOTO 확인'
    assert memory.claims == []


def test_strict_schema_helper_handles_nested_objects_and_defaults():
    schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string', 'default': 'x'},
            'nested': {'type': 'object', 'properties': {'value': {'type': 'number', 'default': 1}}},
            'items': {'type': 'array', 'items': {'type': 'object', 'properties': {'label': {'type': 'string'}}}},
        },
    }

    strict = to_openai_strict_json_schema(schema)

    assert strict['additionalProperties'] is False
    assert set(strict['required']) == {'name', 'nested', 'items'}
    assert 'default' not in strict['properties']['name']
    assert strict['properties']['nested']['additionalProperties'] is False
    assert strict['properties']['items']['items']['additionalProperties'] is False


def test_context_compressor_rolls_up_older_turns():
    messages = []
    for idx in range(8):
        messages.append(HumanMessage(content=f'user turn {idx}'))
        messages.append(AIMessage(content=f'assistant turn {idx}'))

    compressed = ContextCompressor(max_recent_turns=3).compress(messages=messages)

    assert compressed.recent_turn_count == 6
    assert compressed.compressed_message_count == 10
    assert compressed.recent_turns[0]['content'] == 'user turn 5'


def test_context_validator_detects_fast_path_heavy_format_leak():
    warnings = ContextValidator().validate(
        context_resolution={'is_followup': False},
        context_packs={'classifier_context': {}},
        selected_path='fast_concept_answer',
        answer='판정\n위험도\n권장 조치',
    )

    assert 'heavy_answer_format_leaked_into_fast_concept' in warnings


def test_checkpoint_thread_id_policy_and_reset(tmp_path):
    path = tmp_path / 'checkpoints_v2.sqlite3'
    path.write_text('x')

    assert build_thread_id(user_id='u1', session_id='s1') == 'u1:s1'
    assert build_thread_id(user_id='u1', session_id='s2') != build_thread_id(user_id='u1', session_id='s1')
    assert reset_sqlite_checkpoint(path) is True
    assert not path.exists()


def test_clean_slate_has_no_v1_context_adapter_module():
    module_name = 'app.agent.context.' + 'legacy' + '_adapter'
    assert importlib.util.find_spec(module_name) is None


def test_integrated_recommended_action_recap_uses_dedicated_formatter():
    memory = memory_with_actions()
    resolution = ContextResolver().resolve(
        current_user_message='방금 권장조치 순서대로 알려줘',
        last_answer_memory=memory,
        recent_turns=[],
        rolling_summary=None,
    )
    packs = ContextPackBuilder().build(
        current_user_message='방금 권장조치 순서대로 알려줘',
        context_resolution=resolution,
        compressed_context={'rolling_summary': '', 'recent_turns': []},
        last_answer_memory=memory,
        recent_turn_routes=[],
        process_data_policy={},
    )
    gateway = IntentGatewayService().classify(
        request=AgentRequest(user_id='u1', session_id='s1', question=resolution.standalone_query),
        user_context={
            'context_resolution': resolution.model_dump(),
            'context_packs': packs.model_dump(),
            'last_answer_memory': memory.model_dump(),
        },
    )
    answer = FormatterRegistry().format(gateway['selected_path'], packs.formatter_context)

    assert gateway['selected_path'] == 'recommended_action_recap'
    assert '1. LOTO 확인' in answer
    assert '2. 공구 마모 확인' in answer


def test_integrated_recommended_action_item_uses_action_rationale():
    memory = memory_with_actions()
    resolution = ContextResolver().resolve(
        current_user_message='그중 2번은 왜 필요한데?',
        last_answer_memory=memory,
        recent_turns=[],
        rolling_summary=None,
    )
    packs = ContextPackBuilder().build(
        current_user_message='그중 2번은 왜 필요한데?',
        context_resolution=resolution,
        compressed_context={'rolling_summary': '', 'recent_turns': []},
        last_answer_memory=memory,
        recent_turn_routes=[],
        process_data_policy={},
    )
    answer = FormatterRegistry().format('recommended_action_item_explanation', packs.formatter_context)

    assert '공구 마모 확인' in answer
    assert '마모가 크면 절삭 저항과 품질 문제가 커질 수 있습니다.' in answer


def test_checkpoint_state_contains_only_json_like_values(tmp_path):
    root, users = make_root(tmp_path)
    user = users.create({'display_name': 'Checkpoint User'})

    root.run(AgentSendRequest(user_id=user['user_id'], session_id='s_json', message='토크란?'))
    values = root._checkpoint_values(user_id=user['user_id'], session_id='s_json')

    assert values['state_schema_version'] == 2
    assert_json_like(values)
    assert isinstance(values['request'], dict)
    assert isinstance(values['response'], dict)
    assert values['last_answer_memory']['focus'] == '토크'
    root.close()


def test_user_session_checkpoint_memory_isolation(tmp_path):
    root, users = make_root(tmp_path)
    user_a = users.create({'display_name': 'A'})
    user_b = users.create({'display_name': 'B'})

    root.run(AgentSendRequest(user_id=user_a['user_id'], session_id='same_session', message='토크란?'))
    preview = root.preview_route(AgentSendRequest(user_id=user_b['user_id'], session_id='same_session', message='이것의 단점은?'))
    values_b = root._checkpoint_values(user_id=user_b['user_id'], session_id='same_session')

    assert preview['selected_path'] == 'unsupported_or_clarification'
    assert values_b.get('last_answer_memory') in (None, {})
    root.close()


def test_supervisor_keyword_policy_is_hidden_behind_diagnostic_planner():
    planner = DiagnosticPlanner(SupervisorService(NoopLLMService()))
    request = AgentRequest(user_id='u1', session_id='s1', question='정비 전에 안전 절차와 LOTO 확인해야 해?')
    diagnostic = planner.structured_plan(request)
    root_source = inspect.getsource(RootManufacturingGraph._supervisor_planning_node)

    assert diagnostic.requires_safety is True
    assert diagnostic.requires_rag is True
    assert diagnostic.requires_prediction is False
    assert isinstance(diagnostic, DiagnosticPlan)
    assert 'SAFETY_KEYWORDS' not in root_source
    assert 'diagnostic_planner.plan' in root_source


def test_diagnostic_planner_returns_structured_plan():
    planner = DiagnosticPlanner(SupervisorService(NoopLLMService()))
    request = AgentRequest(user_id='u1', session_id='s1', question='이 조건 위험해?')

    result = planner.plan(request=request)
    diagnostic = result.diagnostic_plan

    assert isinstance(result, PlanningResult)
    assert diagnostic.requires_data is True
    assert diagnostic.requires_prediction is True
    assert diagnostic.missing_data_requirements == ['process_data']
    assert diagnostic.reason


def test_diagnostic_planner_has_no_request_instance_state():
    planner = DiagnosticPlanner(SupervisorService(NoopLLMService()))
    planner.plan(request=AgentRequest(user_id='u1', session_id='s1', question='이 조건 위험해?'))

    assert not hasattr(planner, 'last' + '_diagnostic' + '_plan')


def test_diagnostic_planner_does_not_call_supervisor_private_methods():
    source = inspect.getsource(DiagnosticPlanner)

    assert 'self.supervisor' + '._' not in source
    assert '_llm' + '_refine' not in source
    assert '._' + 'intent' not in source
    assert '._' + 'layers' not in source


def test_diagnostic_fallback_policy_uses_context_resolution_contract():
    import app.agent.heavy.diagnostic_planner as diagnostic_module

    source = inspect.getsource(diagnostic_module.DiagnosticFallbackPolicy)

    assert 'current' + '_turn' not in source
    assert 'question' + '_kind' not in source
    assert 'should' + '_use' + '_prediction' not in source


def test_auto_mode_process_data_prediction_does_not_force_rag():
    process_data = {
        'type': 'L',
        'air_temperature_k': 302.1,
        'process_temperature_k': 311.3,
        'rotational_speed_rpm': 1380,
        'torque_nm': 58.2,
        'tool_wear_min': 210,
    }
    request = AgentRequest(user_id='u1', session_id='s1', question='이 조건 위험해?', process_data=process_data)

    diagnostic = DiagnosticPlanner(SupervisorService(NoopLLMService())).structured_plan(request)

    assert diagnostic.requires_prediction is True
    assert diagnostic.requires_rag is False
    assert 'auto process-data prediction' in diagnostic.rag_reason


def test_diagnostic_plan_translator_builds_agent_plan_from_contract():
    diagnostic = DiagnosticPlan(
        requires_prediction=True,
        requires_rag=False,
        requires_safety=False,
        requires_report=False,
        requires_asset_context=True,
        requires_process_condition=True,
        requires_failure_mode=True,
        requires_safety_gate=True,
        requires_action_plan=True,
        rag_query='query',
        reason='contract',
    )

    plan = DiagnosticPlanToAgentPlanTranslator().translate(diagnostic)

    assert plan.prediction_required is True
    assert plan.rag_required is False
    assert 'Failure Mode Agent' in plan.required_nodes


def test_rag_query_planner_does_not_retrieve_directly(tmp_path):
    planner = RagQueryPlanner()
    request = AgentRequest(user_id='u1', session_id='s1', question='스핀들 점검 문서 찾아줘')
    mfg = ManufacturingAgentGraph(PredictionService(model_path=tmp_path / 'model.joblib'), RagService(), NoopLLMService()).domain_service.build_context(request, None, doc_count=0)

    retrieval_request = planner.plan(
        request=request,
        planned_query='spindle maintenance',
        prediction=None,
        manufacturing_context=mfg,
        top_k=3,
        filters={'source': 'Haas'},
    )

    assert retrieval_request.query
    assert retrieval_request.top_k == 3
    assert retrieval_request.filters == {'source': 'Haas'}
    assert not hasattr(planner, 'retrieve')


def test_evidence_grader_does_not_build_citations():
    grader = EvidenceGrader()
    chunks = [
        RagChunk(
            chunk_id='c1',
            source='Haas',
            document_title='Spindle troubleshooting',
            text='spindle load and tool wear maintenance guidance',
        )
    ]

    grade = grader.grade('spindle tool wear', chunks)

    assert grade.usable is True
    assert not hasattr(grader, 'build_citations')
    assert 'citation' not in inspect.getsource(EvidenceGrader.grade).lower()


def test_citation_builder_uses_graded_evidence():
    chunk = RagChunk(
        chunk_id='c1',
        source='OSHA',
        document_title='LOTO',
        text='Lockout tagout guidance',
        url='https://example.test/loto',
    )
    weak_grade = EvidenceGrader().grade('unrelated torque', [chunk])
    usable_grade = EvidenceGrader().grade('lockout tagout', [chunk])

    assert CitationBuilder().build([chunk], weak_grade) == []
    citations = CitationBuilder().build([chunk], usable_grade)
    assert citations[0]['chunk_id'] == 'c1'
    assert citations[0]['reason'] == 'graded_evidence'


def test_manufacturing_graph_legacy_wrappers_removed():
    removed = [
        '_' + 'make_rag_query',
        '_' + 'contexts_match_user_terms',
        '_' + 'collect_action_phrases',
        '_' + 'safety_guidance',
        '_' + 'warnings',
        '_' + 'llm_payload',
    ]

    for method_name in removed:
        assert not hasattr(ManufacturingAgentGraph, method_name)


def test_removed_rag_facade_module_is_absent():
    assert importlib.util.find_spec('app.agent.heavy.' + 'rag' + '_evidence' + '_planner') is None
