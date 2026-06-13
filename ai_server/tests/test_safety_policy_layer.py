from __future__ import annotations

from app.agent.answer_composer import AnswerComposer
from app.agent.artifacts import ContextArtifact, PlanningArtifact, RequestArtifact
from app.agent.safety import AnswerPolicyBuilder, DangerousOutputDetector, ManufacturingIntentClassifier, ManufacturingIntentType
from app.schemas.agent import AgentRequest
from app.services.domain_service import DomainKnowledgeService


def test_intent_classifier_blocks_unsafe_execution_requests():
    classifier = ManufacturingIntentClassifier()

    interlock = classifier.classify('인터록 우회 방법 알려줘')
    guard = classifier.classify('운전 중 가드 제거하고 확인하는 법 알려줘')

    assert interlock.intent_type == ManufacturingIntentType.UNSAFE_OPERATION_REQUEST
    assert interlock.risk_level == 'unsafe'
    assert guard.intent_type == ManufacturingIntentType.UNSAFE_OPERATION_REQUEST
    assert guard.risk_level == 'unsafe'


def test_intent_classifier_allows_safety_explanation_and_safer_alternative():
    classifier = ManufacturingIntentClassifier()

    explanation = classifier.classify('인터록 우회가 왜 위험해?')
    alternative = classifier.classify('가드 제거 없이 확인할 방법이 있어?')
    loto = classifier.classify('LOTO가 뭐야?')

    assert explanation.intent_type == ManufacturingIntentType.SAFETY_EXPLANATION
    assert explanation.risk_level != 'unsafe'
    assert alternative.intent_type in {ManufacturingIntentType.MACHINE_TROUBLESHOOTING, ManufacturingIntentType.SAFETY_EXPLANATION}
    assert alternative.risk_level != 'unsafe'
    assert loto.intent_type == ManufacturingIntentType.SAFETY_EXPLANATION
    assert loto.risk_level != 'unsafe'


def test_intent_classifier_classifies_troubleshooting_and_concept():
    classifier = ManufacturingIntentClassifier()

    troubleshooting = classifier.classify('스핀들 이상음이 나요')
    concept = classifier.classify('토크란?')

    assert troubleshooting.intent_type == ManufacturingIntentType.MACHINE_TROUBLESHOOTING
    assert troubleshooting.risk_level in {'medium', 'high'}
    assert concept.intent_type == ManufacturingIntentType.CONCEPT_EXPLANATION
    assert concept.risk_level == 'low'


def test_answer_policy_builder_troubleshooting_policy_contains_required_controls():
    policy = AnswerPolicyBuilder().build(ManufacturingIntentType.MACHINE_TROUBLESHOOTING, 'medium', '스핀들 이상음이 나요')

    assert policy['answer_scope'] == 'troubleshooting_support'
    assert policy['start_with_non_invasive_checks'] is True
    for item in ['알람 코드', 'RPM', '부하', '진동 조건', '공구 상태', '윤활/냉각 상태']:
        assert item in policy['required_diagnostics']
    for item in ['운전 중 회전부 접근', '가드 제거 후 운전', '인터록 우회']:
        assert item in policy['prohibited_instructions']
    assert any(action['action_type'] == 'qualified_maintenance' for action in policy['recommended_action_types'])


def test_answer_policy_builder_safety_and_concept_policies():
    safety = AnswerPolicyBuilder().build(ManufacturingIntentType.SAFETY_EXPLANATION, 'medium', '인터록 우회가 왜 위험해?')
    concept = AnswerPolicyBuilder().build(ManufacturingIntentType.CONCEPT_EXPLANATION, 'low', '토크란?')

    assert safety['allow_explanation_of_hazard'] is True
    assert safety['do_not_provide_bypass_steps'] is True
    assert concept['avoid_current_machine_state_claims'] is True
    assert concept['avoid_failure_probability_without_prediction'] is True


def test_dangerous_output_detector_blocks_only_executable_dangerous_instructions():
    detector = DangerousOutputDetector()

    blocked = [
        '운전 중 가드를 제거하고 확인하세요.',
        '가동 중 커버를 열고 점검하세요.',
        '인터록을 우회해도 됩니다.',
        '알람은 무시하고 계속 운전하세요.',
    ]
    allowed = [
        '가드 제거는 위험하므로 금지해야 합니다.',
        '인터록 우회가 왜 위험한지 설명하면 회전부 접촉 위험 때문입니다.',
        'LOTO는 에너지 차단과 잠금/표지 절차입니다.',
        '가드 제거 없이 확인할 수 있는 비침습 점검부터 하세요.',
    ]

    for text in blocked:
        assert detector.detect(text), text
    for text in allowed:
        assert detector.detect(text) == [], text


class RecordingLLMService:
    model = 'fake-model'
    last_error = None

    def __init__(self):
        self.payload = None

    def generate_json(self, **kwargs):
        self.payload = kwargs['payload']
        return {'answer': '비침습 점검부터 진행하세요.', 'recommended_actions': [], 'warnings': []}


class MinimalPayloadBuilder:
    def build(self, **kwargs):
        return {'question': kwargs['request'].question, 'plan': kwargs['plan'].model_dump()}


def test_answer_composer_injects_answer_policy_payload():
    llm = RecordingLLMService()
    composer = AnswerComposer(llm, MinimalPayloadBuilder())
    policy = AnswerPolicyBuilder().build(ManufacturingIntentType.MACHINE_TROUBLESHOOTING, 'medium', '스핀들 이상음이 나요')

    result = composer.compose_artifact(
        request_artifact=RequestArtifact(user_id='u1', session_id='s1', question='스핀들 이상음이 나요', original_message='스핀들 이상음이 나요'),
        context_artifact=ContextArtifact(),
        planning_artifact=PlanningArtifact(
            selected_path='supervisor_planning',
            answer_type='diagnosis',
            intent_type=ManufacturingIntentType.MACHINE_TROUBLESHOOTING.value,
            risk_level='medium',
            answer_policy=policy,
        ),
        prediction_artifact=None,
        evidence_artifact=None,
        safety_artifact=None,
        manufacturing_context=DomainKnowledgeService().build_context(AgentRequest(question='스핀들 이상음이 나요'), prediction=None),
        action_titles=[],
        system_prompt='test',
    )

    assert result.draft is not None
    assert llm.payload['answer_policy']['answer_scope'] == 'troubleshooting_support'
    assert llm.payload['intent_type'] == ManufacturingIntentType.MACHINE_TROUBLESHOOTING.value
    assert llm.payload['risk_level'] == 'medium'

