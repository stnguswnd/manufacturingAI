from __future__ import annotations

from app.agent.artifacts import AnswerDraft, EvidenceArtifact, PlanningArtifact, PredictionArtifact, SafetyArtifact, ValidationReport
from app.agent.validators.citation_verifier import CitationVerifier
from app.agent.validators.safety_critic import SafetyCritic
from app.schemas.agent import AgentRequest
from app.services.domain_service import DomainKnowledgeService
from app.services.safety_validation_service import SafetyValidationService


def safety_context():
    req = AgentRequest(question='공구 교체 전 LOTO와 방호장치 확인 절차를 알려줘')
    return DomainKnowledgeService().build_context(req, prediction=None)


def test_artifact_contracts_serialize_with_missing_optional_fields():
    artifacts = [
        PlanningArtifact(selected_path='supervisor_planning', answer_type='diagnosis'),
        PredictionArtifact(called=False, skip_reason='missing_ai4i_features', missing_features=['Tool wear']),
        EvidenceArtifact(profile='rag_only_safety'),
        SafetyArtifact(required_gates=['loto_if_physical_maintenance']),
        AnswerDraft(text='답변 초안', route='answer_compose'),
        ValidationReport.pass_report(),
    ]

    dumped = [artifact.model_dump() for artifact in artifacts]

    assert dumped[0]['needs_prediction'] is False
    assert dumped[2]['documents'] == []
    assert dumped[5]['next_action'] == 'pass'


def test_safety_critic_rewrite_only_when_required_safety_text_missing_but_evidence_exists():
    context = safety_context()
    report = SafetyCritic(SafetyValidationService()).review(
        AnswerDraft(text='일반 점검을 수행하세요.', route='answer_compose'),
        manufacturing_context=context,
        safety_artifact=SafetyArtifact(required_gates=[gate.gate_id for gate in context.safety_gates]),
        evidence_artifact=EvidenceArtifact(evidence_covers_required_gates=True),
    )

    assert report.passed is False
    assert report.next_action == 'rewrite_only'
    assert any(failure.code == 'required_safety_gate_missing' for failure in report.failures)


def test_safety_critic_rerun_rag_when_gate_evidence_is_missing():
    context = safety_context()
    report = SafetyCritic(SafetyValidationService()).review(
        AnswerDraft(text='일반 점검을 수행하세요.', route='answer_compose'),
        manufacturing_context=context,
        safety_artifact=SafetyArtifact(required_gates=[gate.gate_id for gate in context.safety_gates]),
        evidence_artifact=EvidenceArtifact(evidence_covers_required_gates=False, missing_gate_evidence=['loto_if_physical_maintenance']),
    )

    assert report.next_action == 'rerun_rag'
    assert report.required_reexecution == ['rag_evidence', 'safety_contract']


def test_safety_critic_rerun_safety_when_contract_is_missing():
    context = safety_context()
    report = SafetyCritic(SafetyValidationService()).review(
        AnswerDraft(text='일반 점검을 수행하세요.', route='answer_compose'),
        manufacturing_context=context,
        safety_artifact=SafetyArtifact(required_gates=[]),
        evidence_artifact=EvidenceArtifact(evidence_covers_required_gates=True),
    )

    assert report.next_action == 'rerun_safety'
    assert report.required_reexecution == ['safety_contract']


def test_safety_critic_blocks_forbidden_direct_control_phrase():
    context = safety_context()
    report = SafetyCritic(SafetyValidationService()).review(
        AnswerDraft(text='제가 설비를 자동으로 정지했습니다.', route='answer_compose'),
        manufacturing_context=context,
        safety_artifact=SafetyArtifact(required_gates=[gate.gate_id for gate in context.safety_gates]),
        evidence_artifact=EvidenceArtifact(evidence_covers_required_gates=True),
    )

    assert report.next_action == 'block'
    assert any(failure.code == 'forbidden_action' for failure in report.failures)


def test_citation_verifier_detects_missing_citation_and_public_debug_leak():
    report = CitationVerifier().verify(
        AnswerDraft(text='KOSHA 문서 근거로 판단했습니다. chunk_id=c1', route='answer_compose'),
        EvidenceArtifact(profile='rag_only_safety', citations=[]),
        needs_rag=True,
    )

    assert report.passed is False
    assert report.next_action in {'rerun_rag', 'rewrite_only'}
    assert {failure.code for failure in report.failures} >= {'citation_required_but_missing', 'public_debug_metadata_leak'}


def test_citation_verifier_routes_missing_citation_to_rerun_rag():
    report = CitationVerifier().verify(
        AnswerDraft(text='KOSHA 문서 근거로 판단했습니다.', route='answer_compose'),
        EvidenceArtifact(profile='rag_only_safety', citations=[]),
        needs_rag=True,
    )

    assert report.next_action == 'rerun_rag'
    assert report.required_reexecution == ['rag_evidence', 'safety_contract']


def test_citation_verifier_routes_debug_leak_to_rewrite_only():
    report = CitationVerifier().verify(
        AnswerDraft(text='답변입니다. chunk_id=c1', route='answer_compose'),
        EvidenceArtifact(profile='rag_only_safety', citations=[{'label': 'S1'}]),
        needs_rag=True,
    )

    assert report.next_action == 'rewrite_only'
    assert any(failure.source == 'debug_leak' for failure in report.failures)
