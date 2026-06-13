from __future__ import annotations

import inspect
import json

from langgraph.graph import END, StateGraph

from app.agent.artifacts import AnswerDraft, ContextArtifact, EvidenceArtifact, PlanningArtifact, RequestArtifact, ResponseArtifact, RuntimeArtifact, SafetyArtifact, ValidationFailure, ValidationReport
from app.agent.memory_subagent import MemoryInput
from app.agent.root_graph import ARTIFACT_KEYS, MAX_RAG_RERUN_ATTEMPTS, MAX_REWRITE_ATTEMPTS, RootManufacturingGraph
from app.agent.safety import ManufacturingIntentType
from app.agent.safety_subagent import nodes as safety_nodes
from app.agent.state import ManufacturingAgentState
from app.schemas.agent import AgentSendRequest
from app.schemas.prediction import FailureModeScore, PredictionResponse
from app.services.context_service import ContextService
from app.services.domain_service import DomainKnowledgeService
from app.services.memory_service import MemoryService
from app.services.rag_service import RagService
from app.services.safety_validation_service import SafetyValidationService
from app.services.user_service import UserService
from app.storage.sqlite_store import SQLiteStore


class NoopLLMService:
    model = 'fake-model'
    enabled = False
    last_error = None

    def generate_json(self, **kwargs):
        self.last_error = 'noop llm disabled'
        return None


class FakePredictionService:
    bundle = object()

    def __init__(self):
        self.calls = 0

    def predict(self, process_data):
        self.calls += 1
        return PredictionResponse(
            failure_probability=0.12,
            predicted_failure=False,
            risk_level='Normal',
            failure_modes=[FailureModeScore(code='TWF', name='Tool Wear Failure', probability=0.2, predicted=False)],
            predicted_modes=[],
            evidence_features=[],
            recommended_actions=['공구 마모 상태 확인'],
            model_source='fake',
            disclaimer='test prediction',
        )


def make_root(tmp_path):
    store = SQLiteStore(tmp_path / 'artifact_graph.sqlite3')
    return RootManufacturingGraph(
        store=store,
        user_service=UserService(store),
        context_service=ContextService(store),
        memory_service=MemoryService(store),
        prediction_service=FakePredictionService(),
        domain_service=DomainKnowledgeService(),
        safety_validator=SafetyValidationService(),
        llm_service=NoopLLMService(),
        rag_service=RagService(),
        checkpoint_path=tmp_path / 'checkpoints_v3.sqlite3',
    )


def base_state():
    return {
        'state_schema_version': 3,
        'run_id': 'run_test',
        'user_id': 'u1',
        'session_id': 's1',
        'thread_id': 'u1:s1',
        'request': RequestArtifact(
            user_id='u1',
            session_id='s1',
            question='공구 교체 전 안전 확인 절차를 알려줘',
            original_message='공구 교체 전 안전 확인 절차를 알려줘',
        ).model_dump(mode='json'),
        'context': ContextArtifact().model_dump(mode='json'),
        'planning': PlanningArtifact(
            selected_path='supervisor_planning',
            answer_type='safety',
            intent='safety_ops',
            needs_rag=True,
            needs_safety=True,
            completed_nodes={'prediction_node', 'rag_evidence_subagent', 'safety_contract_subagent'},
        ).model_dump(mode='json'),
        'prediction': {'called': False},
        'evidence': EvidenceArtifact(citations=[{'label': 'S1', 'source': 'manual', 'title': 'Safety manual'}]).model_dump(mode='json'),
        'safety': SafetyArtifact(
            required_gates=['loto_if_physical_maintenance'],
            constraints=['LOTO 확인'],
            required_checks=['전원 차단 상태 확인'],
            public_guidance='전원 차단과 방호장치 상태를 확인하세요.',
        ).model_dump(mode='json'),
        'draft': AnswerDraft(text='전원 차단과 방호장치 상태를 확인하세요.', route='answer_compose').model_dump(mode='json'),
        'validation': ValidationReport(
            passed=False,
            next_action='rerun_rag',
            retryable=True,
            failures=[ValidationFailure(code='missing_required_gate_evidence', message='gate evidence missing', source='citation')],
        ).model_dump(mode='json'),
        'response': ResponseArtifact(answer='old answer').model_dump(mode='json'),
        'runtime': RuntimeArtifact().model_dump(mode='json'),
    }


def test_checkpoint_state_is_artifact_only(tmp_path):
    root = make_root(tmp_path)
    user = UserService(root.store).create({'display_name': 'Artifact State User'})

    root.run(AgentSendRequest(user_id=user['user_id'], session_id='artifact_only', message='토크란?'))
    values = root._checkpoint_values(user_id=user['user_id'], session_id='artifact_only')

    assert values['state_schema_version'] == 3
    assert set(values) <= {'state_schema_version', 'run_id', 'user_id', 'session_id', 'thread_id', *ARTIFACT_KEYS}
    assert not {'plan', 'retrieved_documents', 'citations', 'safety_guidance', 'answer', 'structured_answer_payload'} & set(values)
    root.close()


def test_review_edges_are_graph_level_not_root_internal_rerun():
    source = inspect.getsource(RootManufacturingGraph._build_graph)

    assert "graph.add_node('prediction_quality_gate'" in source
    assert "graph.add_node('evidence_quality_gate'" in source
    assert "graph.add_node('safety_contract_gate'" in source
    assert "graph.add_node('answer_text_review'" in source
    assert "graph.add_node('output_policy_gate'" in source
    assert "graph.add_edge('prediction_node', 'prediction_quality_gate')" in source
    assert "graph.add_edge('prediction_quality_gate', 'planning_router')" in source
    assert "graph.add_edge('rag_evidence_subagent', 'evidence_quality_gate')" in source
    assert "'pass': 'planning_router'" in source
    assert "graph.add_edge('safety_contract_subagent', 'safety_contract_gate')" in source
    assert "graph.add_edge('answer_compose', 'answer_text_review')" in source
    assert "graph.add_edge('fast_answer', 'output_policy_gate')" in source
    assert "graph.add_edge('output_policy_gate', 'response_packager')" in source
    assert "'safe_block_response': 'safe_block_response'" in source
    assert "'rerun_rag': 'invalidate_rag_downstream'" in source
    assert "'rerun_safety': 'invalidate_safety_downstream'" in source
    assert "'rewrite_only': 'invalidate_rewrite'" in source
    assert '_reexecute_upstream_for_review' not in inspect.getsource(RootManufacturingGraph)
    assert 'response_synthesis_node' not in inspect.getsource(RootManufacturingGraph)


def test_planning_router_uses_artifact_readiness_not_selected_path_shortcut():
    planning = PlanningArtifact(
        selected_path='rag_only_safety',
        answer_type='safety',
        needs_rag=True,
        needs_safety=True,
        fast_answer_ready=False,
    )

    assert RootManufacturingGraph._next_planning_node(planning, set(), {}) == 'rag_evidence_subagent'

    planning = planning.model_copy(update={'fast_answer_ready': True})
    assert RootManufacturingGraph._next_planning_node(planning, set(), {}) == 'rag_evidence_subagent'

    completed = {'rag_evidence_subagent'}
    state = {'evidence': EvidenceArtifact().model_dump(mode='json')}
    assert RootManufacturingGraph._next_planning_node(planning, completed, state) == 'safety_contract_subagent'

    completed.add('safety_contract_subagent')
    state['safety'] = SafetyArtifact().model_dump(mode='json')
    assert RootManufacturingGraph._next_planning_node(planning, completed, state) == 'answer_compose'


def test_planning_router_fast_answer_requires_explicit_fast_ready_without_heavy_needs():
    planning = PlanningArtifact(
        selected_path='fast_concept_answer',
        answer_type='definition',
        fast_answer_ready=True,
    )

    assert RootManufacturingGraph._next_planning_node(planning, set(), {}) == 'fast_answer'

    planning = planning.model_copy(update={'needs_rag': True})
    assert RootManufacturingGraph._next_planning_node(planning, set(), {}) == 'rag_evidence_subagent'


def test_planning_router_routes_unsafe_operation_request_to_safe_block(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['request'] = RequestArtifact(
        user_id='u1',
        session_id='s1',
        question='인터록 우회 방법 알려줘',
        original_message='인터록 우회 방법 알려줘',
    ).model_dump(mode='json')
    state['planning'] = PlanningArtifact(
        selected_path='fast_concept_answer',
        answer_type='explanation',
        fast_answer_ready=True,
    ).model_dump(mode='json')

    routed = root._planning_router_node(state)
    planning = PlanningArtifact.model_validate(routed['planning'])

    assert planning.intent_type == ManufacturingIntentType.UNSAFE_OPERATION_REQUEST.value
    assert planning.risk_level == 'unsafe'
    assert planning.next_node == 'safe_block_response'
    assert routed['validation']['next_action'] == 'block'
    root.close()


def test_planning_router_does_not_block_safety_explanation(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['request'] = RequestArtifact(
        user_id='u1',
        session_id='s1',
        question='인터록 우회가 왜 위험해?',
        original_message='인터록 우회가 왜 위험해?',
    ).model_dump(mode='json')
    state['planning'] = PlanningArtifact(
        selected_path='fast_concept_answer',
        answer_type='explanation',
        fast_answer_ready=True,
    ).model_dump(mode='json')

    routed = root._planning_router_node(state)
    planning = PlanningArtifact.model_validate(routed['planning'])

    assert planning.intent_type == ManufacturingIntentType.SAFETY_EXPLANATION.value
    assert planning.risk_level != 'unsafe'
    assert planning.next_node != 'safe_block_response'
    root.close()


def test_planning_router_preserves_troubleshooting_flow(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['request'] = RequestArtifact(
        user_id='u1',
        session_id='s1',
        question='스핀들 이상음이 나요',
        original_message='스핀들 이상음이 나요',
    ).model_dump(mode='json')
    state['planning'] = PlanningArtifact(
        selected_path='supervisor_planning',
        answer_type='diagnosis',
        needs_rag=True,
        needs_safety=True,
    ).model_dump(mode='json')
    state.pop('evidence', None)
    state.pop('safety', None)

    routed = root._planning_router_node(state)
    planning = PlanningArtifact.model_validate(routed['planning'])

    assert planning.intent_type == ManufacturingIntentType.MACHINE_TROUBLESHOOTING.value
    assert planning.next_node == 'rag_evidence_subagent'
    root.close()


def test_ai4i_incomplete_prediction_request_routes_only_to_clarification(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state.pop('planning', None)
    state.pop('prediction', None)
    state.pop('evidence', None)
    state.pop('safety', None)
    state.pop('draft', None)
    state.pop('validation', None)
    state.pop('response', None)
    state['context'] = ContextArtifact(ai4i_feature_status={
        'clarification_required': True,
        'missing_features': ['Air temperature', 'Process temperature', 'Rotational speed', 'Tool wear'],
        'parsed_ai4i_features': {'Type': 'M', 'Torque': 34},
        'prediction_skip_reason': 'missing_ai4i_features',
    }).model_dump(mode='json')

    routed = root._planning_router_node(state)
    planning = PlanningArtifact.model_validate(routed['planning'])

    assert planning.clarification_required is True
    assert planning.needs_prediction is False
    assert planning.needs_rag is False
    assert planning.needs_safety is False
    assert planning.next_node == 'clarification_response'
    assert not routed.get('prediction')
    assert not routed.get('evidence')
    assert not routed.get('safety')
    assert not routed.get('draft')
    root.close()


def test_safety_subagent_does_not_own_adaptive_rag_search():
    source = inspect.getsource(safety_nodes)

    forbidden = ['RagService', 'Chroma', 'ChromaRetriever', 'metadata_search', 'search_with_diagnostics', 'CitationBuilder']
    assert not any(token in source for token in forbidden)


def test_invalidate_rag_downstream_clears_stale_artifacts(tmp_path):
    root = make_root(tmp_path)
    out = root._invalidate_rag_downstream_node(base_state())

    for key in ['evidence', 'safety', 'draft', 'validation', 'response']:
        assert out.get(key) is None
    completed = PlanningArtifact.model_validate(out['planning']).completed_nodes
    assert 'rag_evidence_subagent' not in completed
    assert 'safety_contract_subagent' not in completed
    root.close()


def test_compiled_invalidate_rag_downstream_uses_tombstones(tmp_path):
    root = make_root(tmp_path)
    graph = StateGraph(ManufacturingAgentState)
    graph.add_node('invalidate_rag_downstream', root._invalidate_rag_downstream_node)
    graph.set_entry_point('invalidate_rag_downstream')
    graph.add_edge('invalidate_rag_downstream', END)
    compiled = graph.compile()

    out = compiled.invoke(base_state())

    for key in ['evidence', 'safety', 'draft', 'validation', 'response']:
        assert out.get(key) is None
    root.close()


def test_request_context_resets_turn_scoped_checkpoint_artifacts(tmp_path):
    root = make_root(tmp_path)
    user = UserService(root.store).create({'display_name': 'Stale Checkpoint User'})
    state = base_state()
    state['user_id'] = user['user_id']
    state['thread_id'] = f'{user["user_id"]}:s1'
    state['request'] = RequestArtifact(
        user_id=user['user_id'],
        session_id='s1',
        question='공구 교체 전 안전 확인 절차를 알려줘',
        original_message='공구 교체 전 안전 확인 절차를 알려줘',
    ).model_dump(mode='json')
    state['planning'] = PlanningArtifact(
        selected_path='meta_feedback',
        answer_type='meta_feedback',
        fast_answer_ready=True,
    ).model_dump(mode='json')
    graph = StateGraph(ManufacturingAgentState)
    graph.add_node('request_context', root._request_context_node)
    graph.set_entry_point('request_context')
    graph.add_edge('request_context', END)
    compiled = graph.compile()

    out = compiled.invoke(state)

    for key in ['planning', 'prediction', 'evidence', 'safety', 'draft', 'validation', 'response', 'audit']:
        assert out.get(key) is None
    assert out.get('context')
    root.close()


def test_invalidate_safety_downstream_preserves_evidence(tmp_path):
    root = make_root(tmp_path)
    out = root._invalidate_safety_downstream_node(base_state())

    assert out.get('evidence')
    for key in ['safety', 'draft', 'validation', 'response']:
        assert out.get(key) is None
    assert 'safety_contract_subagent' not in PlanningArtifact.model_validate(out['planning']).completed_nodes
    root.close()


def test_invalidate_rewrite_preserves_evidence_and_safety(tmp_path):
    root = make_root(tmp_path)
    out = root._invalidate_rewrite_node(base_state())

    assert out.get('evidence')
    assert out.get('safety')
    for key in ['draft', 'validation', 'response']:
        assert out.get(key) is None
    root.close()


def test_evidence_quality_gate_rerun_rag_when_gate_evidence_missing(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['evidence'] = EvidenceArtifact(
        profile='rag_only_safety',
        documents=[{'doc_id': 'doc-generic', 'source': 'OSHA', 'doc_type': 'safety_standard', 'title': 'General safety'}],
        citations=[{'label': 'S1', 'source': 'OSHA', 'title': 'General safety'}],
        selected_source_ids=['doc-generic'],
        required_safety_gates=['loto_if_physical_maintenance'],
        evidence_covers_required_gates=False,
        missing_gate_evidence=['loto_if_physical_maintenance'],
    ).model_dump(mode='json')

    out = root._evidence_quality_gate_node(state)

    assert out['validation']['next_action'] == 'rerun_rag'
    assert out['runtime']['rag_rerun_attempts'] == 1
    assert out['runtime']['quality_gate_reports'][-1]['gate'] == 'evidence_quality_gate'
    root.close()


def test_evidence_quality_gate_stops_when_rag_does_not_improve(tmp_path):
    root = make_root(tmp_path)
    evidence = EvidenceArtifact(
        profile='rag_only_safety',
        documents=[{'doc_id': 'doc-generic', 'source': 'OSHA', 'doc_type': 'safety_standard', 'title': 'General safety'}],
        citations=[{'label': 'S1', 'source': 'OSHA', 'title': 'General safety'}],
        selected_source_ids=['doc-generic'],
        required_safety_gates=['loto_if_physical_maintenance'],
        evidence_covers_required_gates=False,
        missing_gate_evidence=['loto_if_physical_maintenance'],
    )
    state = base_state()
    state['evidence'] = evidence.model_dump(mode='json')
    signature = root._evidence_quality_signature(evidence)
    state['runtime'] = RuntimeArtifact(
        rag_rerun_attempts=1,
        evidence_signatures=[json.dumps(signature, ensure_ascii=False, sort_keys=True)],
        previous_missing_gate_evidence=['loto_if_physical_maintenance'],
    ).model_dump(mode='json')

    out = root._evidence_quality_gate_node(state)

    assert out['validation']['next_action'] == 'max_retry_exceeded'
    assert out['runtime']['rag_rerun_attempts'] == 1
    root.close()


def test_evidence_quality_gate_budget_routes_to_max_retry(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['runtime'] = RuntimeArtifact(rag_rerun_attempts=MAX_RAG_RERUN_ATTEMPTS).model_dump(mode='json')
    state['evidence'] = EvidenceArtifact(
        profile='rag_only_safety',
        documents=[],
        citations=[],
        required_safety_gates=['loto_if_physical_maintenance'],
        evidence_covers_required_gates=False,
        missing_gate_evidence=['loto_if_physical_maintenance'],
    ).model_dump(mode='json')

    out = root._evidence_quality_gate_node(state)

    assert out['validation']['next_action'] == 'max_retry_exceeded'
    root.close()


def test_safety_contract_gate_rerun_safety_when_contract_empty_and_evidence_sufficient(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['evidence'] = EvidenceArtifact(
        profile='rag_only_safety',
        documents=[{'doc_id': 'doc-loto', 'source': 'KOSHA', 'doc_type': 'maintenance_procedure', 'title': 'LOTO maintenance'}],
        citations=[{'label': 'S1', 'source': 'KOSHA', 'title': 'LOTO maintenance'}],
        required_safety_gates=['loto_if_physical_maintenance'],
        evidence_covers_required_gates=True,
    ).model_dump(mode='json')
    state['safety'] = SafetyArtifact().model_dump(mode='json')

    out = root._safety_contract_gate_node(state)

    assert out['validation']['next_action'] == 'rerun_safety'
    assert out['runtime']['safety_rerun_attempts'] == 1
    root.close()


def test_safety_contract_gate_rerun_rag_when_evidence_is_insufficient(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['evidence'] = EvidenceArtifact(
        profile='rag_only_safety',
        documents=[{'doc_id': 'doc-generic', 'source': 'OSHA', 'doc_type': 'safety_standard'}],
        citations=[{'label': 'S1'}],
        required_safety_gates=['loto_if_physical_maintenance'],
        evidence_covers_required_gates=False,
        missing_gate_evidence=['loto_if_physical_maintenance'],
    ).model_dump(mode='json')
    state['safety'] = SafetyArtifact().model_dump(mode='json')

    out = root._safety_contract_gate_node(state)

    assert out['validation']['next_action'] == 'rerun_rag'
    assert out['runtime']['rag_rerun_attempts'] == 1
    root.close()


def test_answer_text_review_max_retry_routes_to_safe_fallback(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['draft'] = AnswerDraft(text='KOSHA 문서 근거입니다. chunk_id=c1', route='answer_compose').model_dump(mode='json')
    state['runtime'] = RuntimeArtifact(rewrite_attempts=MAX_REWRITE_ATTEMPTS).model_dump(mode='json')

    reviewed = root._answer_text_review_node(state)
    assert reviewed['validation']['next_action'] == 'max_retry_exceeded'
    fallback = root._safe_review_fallback_node(reviewed)
    assert fallback['response']['response_type'] == 'safe_review_fallback'
    root.close()


def test_answer_text_review_block_stops_rewrite_loop(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['draft'] = AnswerDraft(text='제가 설비를 자동으로 정지했습니다.', route='answer_compose').model_dump(mode='json')

    reviewed = root._answer_text_review_node(state)
    assert reviewed['validation']['next_action'] == 'block'
    blocked = root._safe_block_response_node(reviewed)
    assert blocked['response']['response_type'] == 'safe_block'
    root.close()


def test_answer_text_review_blocks_dangerous_output_instruction(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['planning'] = PlanningArtifact(
        selected_path='supervisor_planning',
        answer_type='diagnosis',
        intent_type=ManufacturingIntentType.MACHINE_TROUBLESHOOTING.value,
        risk_level='medium',
        needs_rag=True,
        needs_safety=True,
    ).model_dump(mode='json')
    state['draft'] = AnswerDraft(text='운전 중 가드를 제거하고 확인하세요.', route='answer_compose').model_dump(mode='json')

    reviewed = root._answer_text_review_node(state)

    assert reviewed['validation']['next_action'] == 'block'
    assert any(failure['code'] == 'dangerous_output_instruction' for failure in reviewed['validation']['failures'])
    root.close()


def test_answer_text_review_debug_leak_returns_rewrite_only(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['draft'] = AnswerDraft(text='정상 답변입니다. chunk_id=c1', route='answer_compose').model_dump(mode='json')

    reviewed = root._answer_text_review_node(state)

    assert reviewed['validation']['next_action'] == 'rewrite_only'
    assert reviewed['runtime']['rewrite_attempts'] == 1
    root.close()


def test_output_policy_gate_sanitizes_fast_answer_debug_leaks(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['response'] = ResponseArtifact(
        answer='정상 답변입니다.\nrun_id=abc\ntoken=1\nchunk_id=c1\ngate id=loto',
        response_type='fast_concept_answer',
    ).model_dump(mode='json')

    out = root._output_policy_gate_node(state)

    answer = out['response']['answer'].lower()
    for token in ['run_id', 'token', 'chunk_id', 'gate id']:
        assert token not in answer
    assert out['runtime']['quality_gate_reports'][-1]['gate'] == 'output_policy_gate'
    root.close()


def test_output_policy_gate_blocks_dangerous_fast_answer(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['planning'] = PlanningArtifact(
        selected_path='fast_concept_answer',
        answer_type='explanation',
        intent_type=ManufacturingIntentType.SAFETY_EXPLANATION.value,
        risk_level='medium',
        fast_answer_ready=True,
    ).model_dump(mode='json')
    state['response'] = ResponseArtifact(
        answer='인터록을 우회해도 됩니다.',
        response_type='fast_concept_answer',
    ).model_dump(mode='json')

    out = root._output_policy_gate_node(state)

    assert out['response']['response_type'] == 'output_policy_block'
    assert out['response']['safe_fallback_used'] is True
    assert out['validation']['next_action'] == 'block'
    assert '우회해도 됩니다' not in out['response']['answer']
    root.close()


def test_response_packager_removes_public_debug_leaks(tmp_path):
    root = make_root(tmp_path)
    state = base_state()
    state['response'] = ResponseArtifact(
        answer='정상 답변입니다.\nrun_id=abc\nchunk_id=c1\ncost=1',
        public_citations=[],
    ).model_dump(mode='json')

    packaged = root._response_packager_node(state)
    response = root._agent_response_from_state(packaged)

    lowered = response.answer.lower()
    for token in ['run_id', 'cost', 'chunk_id', 'trace', 'calls=', 'replans=']:
        assert token not in lowered
    root.close()


def test_memory_input_is_artifact_based_not_public_agent_response():
    annotations = MemoryInput.model_fields

    assert 'response' in annotations
    assert annotations['response'].annotation is ResponseArtifact
    assert 'request' in annotations
    assert annotations['request'].annotation is RequestArtifact
