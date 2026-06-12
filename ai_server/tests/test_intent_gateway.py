from __future__ import annotations

from app.agent.context import AnswerMemory, ContextPackBuilder, ContextResolver, RecommendedAction
from app.agent.routing import GateContext, GateRegistry
from app.schemas import AgentRequest, ProcessData
from app.services.intent_classifier_service import IntentClassifierInput, IntentClassifierOutput, ResolvedReference
from app.services.intent_gateway_service import IntentGatewayService


class FakeClassifier:
    def __init__(self, output: IntentClassifierOutput | None = None):
        self.output = output
        self.calls = 0
        self.last_payload: IntentClassifierInput | None = None

    def classify(self, payload: IntentClassifierInput, **kwargs) -> IntentClassifierOutput:
        self.calls += 1
        self.last_payload = payload
        if self.output:
            return self.output
        return IntentClassifierOutput(
            selected_path='general_lightweight_answer',
            answer_type='explanation',
            resolved_reference=ResolvedReference(type='none', source='none', confidence=0.0),
            confidence=0.82,
            reason='fake lightweight explanation',
        )


def action_memory() -> AnswerMemory:
    return AnswerMemory(
        selected_path='heavy_analysis_answer',
        answer_type='diagnosis',
        short_summary='토크와 공구 마모가 큰 상황의 점검 권장조치',
        focus='점검 및 안전 절차',
        recommended_actions=[
            RecommendedAction(id='a1', title='LOTO 확인', rationale='물리 점검 전 예기치 않은 가동을 막기 위해 필요합니다.', priority=1),
            RecommendedAction(id='a2', title='공구 마모 확인', rationale='절삭 저항과 품질 문제를 확인하기 위해 필요합니다.', priority=2),
        ],
    )


def context_for(message: str, memory: AnswerMemory | None = None) -> dict:
    resolution = ContextResolver().resolve(
        current_user_message=message,
        last_answer_memory=memory,
        recent_turns=[],
        rolling_summary=None,
    )
    packs = ContextPackBuilder().build(
        current_user_message=message,
        context_resolution=resolution,
        compressed_context={'rolling_summary': '', 'recent_turns': []},
        last_answer_memory=memory,
        recent_turn_routes=[],
        process_data_policy={},
    )
    return {
        'context_resolution': resolution.model_dump(),
        'context_packs': packs.model_dump(),
        'last_answer_memory': memory.model_dump() if memory else {},
    }


def test_no_llm_glossary_concept_uses_fast_path():
    classifier = FakeClassifier()
    request = AgentRequest(user_id='u1', session_id='s1', question='LOTO가 뭐야?')

    result = IntentGatewayService(intent_classifier=classifier).classify(
        request=request,
        user_context=context_for('LOTO가 뭐야?', action_memory()),
    )

    assert result['selected_path'] == 'fast_concept_answer'
    assert result['answer_type'] == 'definition'
    assert result['requires_prediction'] is False
    assert classifier.calls == 0


def test_previous_heavy_memory_does_not_pollute_new_concept_question():
    request = AgentRequest(user_id='u1', session_id='s1', question='LOTO가 뭐야?')
    context = context_for('LOTO가 뭐야?', action_memory())

    result = IntentGatewayService(intent_classifier=FakeClassifier()).classify(
        request=request,
        user_context=context,
    )

    assert context['context_resolution']['is_followup'] is False
    assert result['selected_path'] == 'fast_concept_answer'
    assert result['turn_type'] == 'general_concept'


def test_recommended_action_followup_routes_to_recap_not_lightweight():
    request = AgentRequest(user_id='u1', session_id='s1', question='방금 권장조치 중요한 순서대로 알려줘')

    result = IntentGatewayService(intent_classifier=FakeClassifier()).classify(
        request=request,
        user_context=context_for('방금 권장조치 중요한 순서대로 알려줘', action_memory()),
    )

    assert result['selected_path'] == 'recommended_action_recap'
    assert result['answer_type'] == 'recommended_action_recap'
    assert result['selected_path'] != 'general_lightweight_answer'


def test_recommended_action_item_followup_routes_to_item_formatter():
    request = AgentRequest(user_id='u1', session_id='s1', question='그중 2번은 왜 필요한데?')

    result = IntentGatewayService(intent_classifier=FakeClassifier()).classify(
        request=request,
        user_context=context_for('그중 2번은 왜 필요한데?', action_memory()),
    )

    assert result['selected_path'] == 'recommended_action_item_explanation'
    assert result['answer_type'] == 'recommended_action_item_explanation'
    assert result['resolved_claim'] == '공구 마모 확인'


def test_classifier_receives_only_classifier_context_contract():
    classifier = FakeClassifier()
    request = AgentRequest(user_id='u1', session_id='s1', question='예시 보여줘')
    context = context_for('예시 보여줘', action_memory())
    context['context_packs']['classifier_context']['retrieved_docs'] = 'should not be copied by builder tests'

    IntentGatewayService(intent_classifier=classifier).classify(
        request=request,
        user_context=context_for('예시 보여줘', action_memory()),
    )

    payload = classifier.last_payload
    assert payload is not None
    dumped = payload.model_dump()
    forbidden = {'retrieved_docs', 'retrieved_documents', 'rag_contexts', 'messages', 'full_conversation'}
    assert forbidden.isdisjoint(dumped)
    assert payload.standalone_query == '예시 보여줘'


def test_process_data_risk_question_is_hard_gated_to_supervisor():
    process_data = ProcessData(
        type='L',
        air_temperature_k=302.1,
        process_temperature_k=311.3,
        rotational_speed_rpm=1380,
        torque_nm=58.2,
        tool_wear_min=210,
    )
    request = AgentRequest(user_id='u1', session_id='s1', question='이 조건 위험해?', process_data=process_data)

    result = IntentGatewayService(intent_classifier=FakeClassifier()).classify(
        request=request,
        user_context=context_for('이 조건 위험해?', action_memory()),
    )

    assert result['selected_path'] == 'supervisor_planning'
    assert result['requires_prediction'] is True


def test_safety_request_is_not_lightweight():
    request = AgentRequest(user_id='u1', session_id='s1', question='정비 전에 뭘 확인해야 해?')

    result = IntentGatewayService(intent_classifier=FakeClassifier()).classify(
        request=request,
        user_context=context_for('정비 전에 뭘 확인해야 해?', action_memory()),
    )

    assert result['selected_path'] == 'supervisor_planning'
    assert result['requires_safety'] is True


def test_meta_feedback_preserves_focus_policy():
    request = AgentRequest(user_id='u1', session_id='s1', question='내가 이걸이라고 한 거는 전 대화 맥락 보고 판단할 수 있잖아.')

    result = IntentGatewayService(intent_classifier=FakeClassifier()).classify(
        request=request,
        user_context=context_for('내가 이걸이라고 한 거는 전 대화 맥락 보고 판단할 수 있잖아.', action_memory()),
    )

    assert result['selected_path'] == 'meta_feedback'
    assert result['focus_update_policy'] == 'preserve'


def test_classifier_fallback_does_not_expose_raw_exception():
    class BrokenClassifier:
        def classify(self, payload: IntentClassifierInput, **kwargs):
            raise ValueError('BadRequestError invalid_json_schema additionalProperties')

    request = AgentRequest(user_id='u1', session_id='s1', question='예시 보여줘')

    result = IntentGatewayService(intent_classifier=BrokenClassifier()).classify(
        request=request,
        user_context=context_for('예시 보여줘', None),
    )

    assert result['selected_path'] == 'unsupported_or_clarification'
    assert 'BadRequestError' not in result['reason']
    assert 'invalid_json_schema' not in result['reason']


def test_followup_candidate_gate_does_not_make_final_route_decision():
    context = GateContext(
        question='왜?',
        original_question='왜?',
        compact_question='왜?',
        has_process_data=False,
    )

    results = GateRegistry().evaluate_all(context)
    final_result = GateRegistry().evaluate(context)

    followup = [item for item in results if item.gate_name == 'followup_candidate_signal'][0]
    assert followup.matched is True
    assert followup.is_final is False
    assert followup.selected_path is None
    assert final_result.matched is False
