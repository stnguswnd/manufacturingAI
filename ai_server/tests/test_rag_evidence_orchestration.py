from __future__ import annotations

from pathlib import Path

from app.agent.artifacts import ContextArtifact, PlanningArtifact, RequestArtifact, RuntimeArtifact
from app.agent.rag_evidence import RagEvidenceDeps, RagEvidenceInput, RagEvidenceSubAgent
from app.agent.root_graph import RootManufacturingGraph
from app.agent.heavy import CitationBuilder, EvidenceFilter, EvidenceGrader, RagQueryPlanner
from app.agent.heavy.rag_query_planner import RagFanoutPolicy
from app.schemas.agent import AgentPlan, AgentRequest, AgentSendRequest
from app.schemas.prediction import FailureModeScore, PredictionResponse, ProcessData
from app.schemas.rag import RagChunk
from app.services.context_service import ContextService
from app.services.domain_service import DomainKnowledgeService
from app.services.memory_service import MemoryService
from app.services.prediction_service import PredictionService
from app.services.safety_validation_service import SafetyValidationService
from app.services.user_service import UserService
from app.storage.sqlite_store import SQLiteStore


class NoopLLMService:
    model = 'fake-model'
    enabled = False
    last_error = None

    def generate_json(self, **kwargs):
        return None


class FakeRagService:
    def __init__(self, responses, *, chroma_count=702):
        self.responses = list(responses)
        self.chroma_count = chroma_count
        self.queries = []
        self.metadata_terms = []

    def search_with_diagnostics(self, query, top_k=5, filters=None):
        self.queries.append({'query': query, 'top_k': top_k, 'filters': filters})
        chunks = self.responses.pop(0) if self.responses else []
        if isinstance(chunks, Exception):
            raise chunks
        return {'chunks': chunks, 'diagnostics': {'backend': 'chroma', 'error': None}}

    def metadata_search(self, terms, top_k=5, *, allow_restricted=False):
        self.metadata_terms.append({'terms': terms, 'top_k': top_k, 'allow_restricted': allow_restricted})
        return {'chunks': [], 'diagnostics': {'backend': 'chroma_metadata', 'error': None}}

    def corpus_health(self):
        return {'backend': 'chroma', 'chroma_count': self.chroma_count, 'error': None}


def chunk(chunk_id: str, *, text='토크 공구 마모 정비 점검 안전', source='KOSHA', safety_gate='maintenance_check') -> RagChunk:
    return RagChunk(
        chunk_id=chunk_id,
        source=source,
        document_title='정비 지침',
        title='정비 지침',
        doc_id='doc',
        text=text,
        safety_gate=safety_gate,
        failure_modes='OSF,TWF',
        related_signals='tool_wear_min,torque_nm',
        project_priority='high',
        retrieval_scope='default',
        score=0.9,
    )


def request():
    return AgentSendRequest(
        user_id='u1',
        session_id='s1',
        message='토크가 높고 공구 마모가 큰데 어떤 점검과 안전 절차를 확인해야 해?',
        process_data=ProcessData(
            type='L',
            air_temperature_k=302.1,
            process_temperature_k=311.3,
            rotational_speed_rpm=1380,
            torque_nm=58.2,
            tool_wear_min=210,
        ),
        top_k=4,
    )


def agent(fake_rag: FakeRagService) -> RagEvidenceSubAgent:
    return RagEvidenceSubAgent(RagEvidenceDeps(
        query_planner=RagQueryPlanner(),
        fanout_policy=RagFanoutPolicy(),
        rag_service=fake_rag,
        evidence_filter=EvidenceFilter(),
        evidence_grader=EvidenceGrader(),
        citation_builder=CitationBuilder(),
        domain_service=DomainKnowledgeService(),
    ))


def evidence_input():
    send = request()
    req = agent_request(send)
    plan = AgentPlan(
        intent='hybrid',
        rag_required=True,
        safety_required=True,
        safety_gate_required=True,
        prediction_required=True,
        rag_query='tool wear torque maintenance safety',
        required_nodes=['Procedure Retrieval Agent', 'Safety Gate Agent'],
    )
    context = DomainKnowledgeService().build_context(req, None, doc_count=0)
    return RagEvidenceInput(request=req, plan=plan, prediction=None, manufacturing_context=context, top_k=4)


def test_rag_evidence_subagent_runs_langgraph_flow():
    output = agent(FakeRagService([[chunk('c1'), chunk('c1'), chunk('c2', source='OSHA', safety_gate='loto')]] * 4)).invoke(evidence_input())

    assert output.retrieved_documents
    assert output.evidence_grade.usable is True
    assert output.evidence_artifact.profile in {'prediction_plus_rag', 'rag_only_safety', 'troubleshooting_rag', 'concept_explanation'}
    assert output.evidence_artifact.queries
    assert output.evidence_artifact.citations
    assert 'evidence_grade' in output.evidence_artifact.retrieval_diagnostics
    assert output.trace['query_spec_names'][0] == 'primary'
    assert output.trace['raw_count'] >= output.trace['filtered_count'] >= output.trace['selected_count']
    assert output.trace['citation_count'] == len(output.citations)


def test_rag_evidence_artifact_persists_final_health_diagnostics():
    output = agent(FakeRagService([[chunk('c1')]] * 4, chroma_count=701)).invoke(evidence_input())

    assert output.trace['corpus_count_mismatch'] is True
    assert output.evidence_artifact.generic_document_downgraded is False
    assert output.evidence_artifact.retrieval_diagnostics['corpus_count_mismatch'] is True
    assert any('Chroma collection count mismatch' in item for item in output.evidence_artifact.warnings)


def test_root_graph_passes_subagent_output_to_state(tmp_path: Path):
    root = make_root(tmp_path, FakeRagService([[chunk('c1')]] * 4))
    send = request()
    req = agent_request(send)
    plan = AgentPlan(intent='hybrid', rag_required=True, rag_query='maintenance', required_nodes=['Procedure Retrieval Agent'])
    state = {
        'state_schema_version': 3,
        'run_id': 'r1',
        'user_id': 'u1',
        'session_id': 's1',
        'thread_id': 'u1:s1',
        'request': RequestArtifact(
            user_id='u1',
            session_id='s1',
            question=send.message,
            original_message=send.message,
            top_k=4,
        ).model_dump(mode='json'),
        'context': ContextArtifact().model_dump(mode='json'),
        'planning': PlanningArtifact(
            selected_path='supervisor_planning',
            answer_type='diagnosis',
            intent='hybrid',
            needs_rag=True,
            agent_plan=plan.model_dump(mode='json'),
            route=plan.required_nodes,
        ).model_dump(mode='json'),
        'runtime': RuntimeArtifact().model_dump(mode='json'),
    }

    out = root._rag_evidence_node(state)

    assert out['evidence']['documents'][0]['chunk_id'] == 'c1'
    assert out['evidence']['citations']
    assert out['evidence']['profile'] is not None
    assert 'evidence_grade' in out['evidence']['retrieval_diagnostics']
    assert out['evidence']['retrieval_diagnostics']['selected_count'] == len(out['evidence']['documents'])


def test_empty_or_failed_retrieval_degrades_without_crash():
    output = agent(FakeRagService([RuntimeError('backend down'), [], [], []])).invoke(evidence_input())

    assert output.retrieved_documents == []
    assert output.citations == []
    assert output.evidence_grade.usable is False
    assert output.trace['selected_count'] == 0
    assert output.trace['retrieval_backend'] == 'error'


def test_trace_does_not_include_raw_chunk_text():
    raw_text = 'RAW_TEXT_SHOULD_NOT_BE_IN_TRACE'
    output = agent(FakeRagService([[chunk('c1', text=raw_text)]] * 4)).invoke(evidence_input())

    assert raw_text not in str(output.trace)


def test_fanout_is_bounded_to_four_specs():
    output = agent(FakeRagService([[chunk('c1')]] * 4)).invoke(evidence_input())

    assert len(output.trace['query_spec_names']) <= 4


def test_planned_query_internal_tokens_are_sanitized():
    req = AgentRequest(user_id='u1', session_id='s1', question='공구 교체 전 안전 절차 알려줘')
    plan_query = 'maintenance_manual troubleshooting_guide safety_standard failure_mode_catalog metadata planner route internal 공구 교체'
    context = DomainKnowledgeService().build_context(req, None, doc_count=0)

    retrieval = RagQueryPlanner().plan(
        request=req,
        planned_query=plan_query,
        prediction=None,
        manufacturing_context=context,
        top_k=5,
    )

    lowered = retrieval.query.lower()
    for token in ['maintenance_manual', 'troubleshooting_guide', 'safety_standard', 'failure_mode_catalog', 'metadata', 'planner', 'route', 'internal']:
        assert token not in lowered
    assert retrieval.query.startswith(req.question)


def test_prediction_plus_rag_profile_uses_prediction_hints():
    req = agent_request(request())
    prediction = PredictionResponse(
        failure_probability=0.12,
        predicted_failure=False,
        risk_level='Normal',
        failure_modes=[FailureModeScore(code='TWF', name='Tool Wear Failure', probability=0.2, predicted=False)],
        predicted_modes=['TWF'],
        evidence_features=[],
        recommended_actions=['공구 마모 상태 확인'],
        model_source='fake',
        disclaimer='test',
    )
    context = DomainKnowledgeService().build_context(req, prediction, doc_count=0)
    plan = AgentPlan(
        intent='hybrid',
        rag_required=True,
        prediction_required=True,
        safety_required=True,
        safety_gate_required=True,
        rag_query='failure_mode_catalog troubleshooting_guide tool wear',
        required_nodes=['RAG Evidence SubAgent'],
    )
    fake = FakeRagService([[chunk('twf', safety_gate='maintenance_check')]] * 4)

    output = agent(fake).invoke(RagEvidenceInput(request=req, plan=plan, prediction=prediction, manufacturing_context=context, top_k=4))

    assert output.trace['retrieval_profile'] == 'prediction_plus_rag'
    assert any('TWF' in (query['query'] or '') for query in fake.queries)
    assert all('failure_mode_catalog' not in (query['query'] or '') for query in fake.queries)


def test_safety_document_metadata_beats_specific_industry_body_match():
    req = AgentRequest(user_id='u1', session_id='s1', question='공구 교체 작업 전에 어떤 안전 절차와 확인 항목을 봐야 해?')
    plan = AgentPlan(
        intent='safety_ops',
        rag_required=True,
        safety_required=True,
        safety_gate_required=True,
        rag_query=req.question,
        required_nodes=['RAG Evidence SubAgent'],
    )
    context = DomainKnowledgeService().build_context(req, None, doc_count=0)
    broad_guard = chunk(
        'guard',
        text='공구 교체 안전 회전부 접근 전 방호장치 가드 인터록 비상정지 확인',
        safety_gate='rotating_parts_guard_check',
    ).model_copy(update={
        'doc_id': 'general_rotating_guard',
        'document_title': '회전기계 끼임 절단 방호장치 안전 지침',
        'title': '회전기계 끼임 절단 방호장치 안전 지침',
        'doc_type': 'korean_machine_safety',
        'score': 0.4,
    })
    industry = chunk(
        'industry',
        text='공구 교체 안전 점검 방호장치 확인',
        safety_gate='maintenance_check',
    ).model_copy(update={
        'doc_id': 'specific_industry_safety',
        'document_title': '조선업 안전점검 기술지침',
        'title': '조선업 안전점검 기술지침',
        'doc_type': 'korean_safety_reference',
        'score': 0.99,
    })
    fake = FakeRagService([[industry, broad_guard]] * 4)

    output = agent(fake).invoke(RagEvidenceInput(request=req, plan=plan, prediction=None, manufacturing_context=context, top_k=4))

    assert output.trace['retrieval_profile'] == 'rag_only_safety'
    assert output.retrieved_documents[0].doc_id == 'general_rotating_guard'
    assert len(output.retrieved_documents) <= 3


def test_equipment_title_match_prioritizes_machine_specific_safety_doc():
    req = AgentRequest(user_id='u1', session_id='s1', question='드릴기 작업 전에 공작물 고정 상태, 방호덮개, 비상정지장치를 어떻게 확인해야 하는지 알려줘.')
    plan = AgentPlan(
        intent='safety_ops',
        rag_required=True,
        safety_required=True,
        safety_gate_required=True,
        rag_query=req.question,
        required_nodes=['RAG Evidence SubAgent'],
    )
    context = DomainKnowledgeService().build_context(req, None, doc_count=0)
    drill_doc = chunk(
        'drill',
        text='공작물 고정 상태를 확인하고 드릴기 방호덮개와 비상정지장치를 점검한다.',
        safety_gate='rotating_parts_guard_check',
    ).model_copy(update={
        'doc_id': 'drill_machine_guarding',
        'document_title': '드릴기 방호조치 안전 지침',
        'title': '드릴기 방호조치 안전 지침',
        'doc_type': 'korean_machine_safety',
        'score': 0.35,
    })
    shipyard_doc = chunk(
        'shipyard',
        text='공작물 고정 상태 방호덮개 비상정지장치 안전점검',
        safety_gate='maintenance_check',
    ).model_copy(update={
        'doc_id': 'shipyard_general_safety',
        'document_title': '조선업 안전점검 기술지침',
        'title': '조선업 안전점검 기술지침',
        'doc_type': 'korean_safety_reference',
        'score': 0.99,
    })
    fake = FakeRagService([[shipyard_doc, drill_doc]] * 4)

    output = agent(fake).invoke(RagEvidenceInput(request=req, plan=plan, prediction=None, manufacturing_context=context, top_k=4))

    assert output.retrieved_documents[0].doc_id == 'drill_machine_guarding'
    assert output.citations[0]['doc_id'] == 'drill_machine_guarding'
    assert output.trace['retrieval_profile'] == 'rag_only_safety'


def test_evidence_filter_excludes_docs_without_required_context_overlap():
    good = chunk('toolwear', text='공구 마모 스핀들 점검', source='manual', safety_gate='qualified_maintenance')
    unrelated = chunk(
        'unrelated',
        text='배관 누설 압력 시험 밸브 점검',
        safety_gate='maintenance_check',
    ).model_copy(update={
        'document_title': '배관 압력 시험 정비 지침',
        'title': '배관 압력 시험 정비 지침',
        'doc_id': 'pipe_pressure_maintenance',
    })

    filtered = EvidenceFilter().filter(
        [unrelated, good],
        filters={
            'preferred_failure_modes': ['TWF'],
            'preferred_safety_gates': ['maintenance_check', 'qualified_maintenance'],
            'required_context_terms': ['공구', '마모', '스핀들', 'tool wear'],
        },
    )

    assert [item.chunk_id for item in filtered] == ['toolwear']


def make_root(tmp_path: Path, rag_service) -> RootManufacturingGraph:
    store = SQLiteStore(tmp_path / 'root.sqlite3')
    llm = NoopLLMService()
    return RootManufacturingGraph(
        store=store,
        user_service=UserService(store),
        context_service=ContextService(store),
        memory_service=MemoryService(store),
        prediction_service=PredictionService(model_path=tmp_path / 'model.joblib'),
        domain_service=DomainKnowledgeService(),
        safety_validator=SafetyValidationService(),
        llm_service=llm,
        rag_service=rag_service,
        checkpoint_path=tmp_path / 'checkpoints.sqlite3',
    )


def agent_request(send: AgentSendRequest) -> AgentRequest:
    return AgentRequest(
        user_id=send.user_id,
        session_id=send.session_id,
        question=send.message,
        process_data=send.process_data,
        top_k=send.top_k,
    )
