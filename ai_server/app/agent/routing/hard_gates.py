from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from app.agent.routing.gate_schemas import GateContext, GateResult


def contains_any(compact_question: str, terms: Iterable[str]) -> bool:
    return any(str(term).lower().replace(' ', '') in compact_question for term in terms)


def concept_answer_type(compact_question: str) -> str:
    if any(term in compact_question for term in ['주의', '조심', '볼때', '봐야', '언제확인', '중요', '판단할때', '어떤값', '값을봐']):
        return 'watch_points'
    if any(term in compact_question for term in ['단점', '문제점', '한계', '안좋']):
        return 'disadvantages'
    return 'definition'


class HardGate(ABC):
    name: str

    @abstractmethod
    def evaluate(self, context: GateContext) -> GateResult:
        raise NotImplementedError


class EmptyQuestionGate(HardGate):
    name = 'empty_question'

    def evaluate(self, context: GateContext) -> GateResult:
        if context.original_question.strip():
            return GateResult()
        return GateResult(
            matched=True,
            gate_name=self.name,
            selected_path='unsupported_or_clarification',
            answer_type='clarification',
            reason='empty question',
            confidence=1.0,
            is_final=True,
            category='empty',
            focus_update_policy='skip',
        )


class ControlScopeGate(HardGate):
    name = 'control_scope_guard'
    terms = ['멈춰', '정지시켜', '꺼줘', '가동해', '제어해', 'reset', 'start machine', 'stop machine']

    def evaluate(self, context: GateContext) -> GateResult:
        if not contains_any(context.compact_question, self.terms):
            return GateResult()
        return GateResult(
            matched=True,
            gate_name=self.name,
            selected_path='unsupported_or_clarification',
            answer_type='clarification',
            reason='설비 제어 요청은 AI Agent가 수행할 수 없습니다.',
            confidence=1.0,
            is_final=True,
            category='scope_guard',
            turn_type='unsupported_control',
            focus_update_policy='skip',
        )


class MetaFeedbackGate(HardGate):
    name = 'meta_feedback'
    terms = ['맥락', '전 대화', '직전 대화', '판단할 수 있잖아', '잘못', '수정', '버그', 'resolved_target', 'resolved=false']

    def evaluate(self, context: GateContext) -> GateResult:
        if not contains_any(context.compact_question, self.terms):
            return GateResult()
        return GateResult(
            matched=True,
            gate_name=self.name,
            selected_path='meta_feedback',
            answer_type='meta_feedback',
            reason='사용자가 Agent의 대화 맥락/지시어 해석 동작에 대해 피드백했습니다.',
            confidence=0.95,
            is_final=True,
            category='meta',
            turn_type='meta_feedback',
            focus_update_policy='preserve',
        )


class RecommendedActionFollowupGate(HardGate):
    name = 'recommended_action_followup'

    def evaluate(self, context: GateContext) -> GateResult:
        followup_type = context.context_resolution.get('followup_type')
        if followup_type == 'previous_recommended_actions':
            return GateResult(
                matched=True,
                gate_name=self.name,
                selected_path='recommended_action_recap',
                answer_type='recommended_action_recap',
                reason='ContextResolver가 직전 답변의 권장조치 정렬 요청으로 해석했습니다.',
                confidence=float(context.context_resolution.get('confidence') or 0.85),
                is_final=True,
                category='followup',
                turn_type='action_order_followup',
                resolved_reference={'type': 'previous_answer_claim', 'text': '권장조치', 'normalized': 'recommended_actions', 'source': 'context_resolution', 'confidence': float(context.context_resolution.get('confidence') or 0.85)},
                resolved_claim=context.last_answer_memory.get('short_summary'),
                focus_update_policy='preserve',
            )
        if followup_type == 'previous_recommended_action_item':
            return GateResult(
                matched=True,
                gate_name=self.name,
                selected_path='recommended_action_item_explanation',
                answer_type='recommended_action_item_explanation',
                reason='ContextResolver가 직전 권장조치 특정 항목 설명 요청으로 해석했습니다.',
                confidence=float(context.context_resolution.get('confidence') or 0.85),
                is_final=True,
                category='followup',
                turn_type='action_item_followup',
                resolved_reference={'type': 'previous_recommended_action', 'text': context.context_resolution.get('followup_target'), 'normalized': 'recommended_action_item', 'source': 'context_resolution', 'confidence': float(context.context_resolution.get('confidence') or 0.85)},
                resolved_claim=context.context_resolution.get('followup_target'),
                focus_update_policy='preserve',
            )
        return GateResult()


class ProcessDataDiagnosisGate(HardGate):
    name = 'process_data_diagnosis'
    terms = ['위험', '위험해', '고장', '이상', '판단', '분석', '점검', '조치', '현재 값', '이 조건', '이 데이터', '공정 데이터']

    def evaluate(self, context: GateContext) -> GateResult:
        if not (context.has_process_data and contains_any(context.compact_question, self.terms)):
            return GateResult()
        return GateResult(
            matched=True,
            gate_name=self.name,
            selected_path='supervisor_planning',
            answer_type='diagnosis',
            reason='현재 공정 데이터에 대한 위험/이상 판단 요청입니다.',
            confidence=0.95,
            is_final=True,
            category='diagnosis',
            turn_type='prediction_request',
            requires_prediction=True,
            requires_rag=True,
            requires_safety=True,
            resolved_reference={'type': 'process_data', 'text': '현재 공정 조건', 'normalized': 'current_process_data', 'source': 'current_question', 'confidence': 0.95},
            focus_update_policy='update',
        )


class ReportGate(HardGate):
    name = 'report_request'
    terms = ['보고서', '리포트', '초안', '기록', '문서화']

    def evaluate(self, context: GateContext) -> GateResult:
        if not (context.generate_report or contains_any(context.compact_question, self.terms)):
            return GateResult()
        return GateResult(
            matched=True,
            gate_name=self.name,
            selected_path='supervisor_planning',
            answer_type='report',
            reason='보고서/기록 생성 요청입니다.',
            confidence=0.9,
            is_final=True,
            category='report',
            turn_type='report_request',
            requires_rag=True,
            requires_report=True,
            resolved_reference={'type': 'report', 'text': '보고서', 'normalized': 'report', 'source': 'current_question', 'confidence': 0.9},
            focus_update_policy='update',
        )


class GlossaryConceptGate(HardGate):
    name = 'glossary_concept'
    simple_terms = ['뭐야', '무엇', '정의', '설명', '란', '이란', '단점', '장점', '한계', '주의', '주의점', '볼 때', '봐야', '언제 확인']

    def evaluate(self, context: GateContext) -> GateResult:
        if not context.glossary_hit:
            return GateResult()
        if not contains_any(context.compact_question, self.simple_terms):
            return GateResult()
        is_followup = bool(context.context_resolution.get('is_followup'))
        return GateResult(
            matched=True,
            gate_name=self.name,
            selected_path='fast_concept_answer',
            answer_type=concept_answer_type(context.compact_question),
            reason='명확한 glossary concept 질문입니다.',
            confidence=0.95,
            is_final=True,
            category='concept',
            turn_type='concept_followup' if is_followup else 'general_concept',
            resolved_reference={'type': 'concept', 'text': context.glossary_hit.get('matched_text'), 'normalized': context.glossary_hit.get('canonical'), 'source': 'current_question', 'confidence': 0.95},
            focus_update_policy='preserve' if is_followup else 'update',
        )


class SafetyRequestGate(HardGate):
    name = 'safety_request'
    terms = ['안전', '정비 전', '정비전', 'loto', 'lockout', 'tagout', '방호', '가드', '보호구']

    def evaluate(self, context: GateContext) -> GateResult:
        if not contains_any(context.compact_question, self.terms):
            return GateResult()
        return GateResult(
            matched=True,
            gate_name=self.name,
            selected_path='supervisor_planning',
            answer_type='diagnosis',
            reason='안전/정비 hard gate 요청입니다.',
            confidence=0.85,
            is_final=True,
            category='safety_guard',
            turn_type='safety_request',
            requires_rag=True,
            requires_safety=True,
            resolved_reference={'type': 'concept', 'text': '안전/정비 절차', 'normalized': 'safety_maintenance', 'source': 'current_question', 'confidence': 0.85},
            focus_update_policy='preserve',
        )


class DocumentRequestGate(HardGate):
    name = 'document_request'
    terms = ['문서', '매뉴얼', '기준', '규정', '근거', '출처']

    def evaluate(self, context: GateContext) -> GateResult:
        if not contains_any(context.compact_question, self.terms):
            return GateResult()
        return GateResult(
            matched=True,
            gate_name=self.name,
            selected_path='lightweight_rag_answer',
            answer_type='explanation',
            reason='문서 근거가 필요한 가벼운 지식 질문입니다.',
            confidence=0.8,
            is_final=True,
            category='document',
            turn_type='knowledge_qa',
            requires_rag=True,
            resolved_reference={'type': 'document', 'text': '문서 근거', 'normalized': 'document_context', 'source': 'current_question', 'confidence': 0.8},
            focus_update_policy='preserve',
        )


class FollowupCandidateGate(HardGate):
    name = 'followup_candidate_signal'
    terms = ['이것', '이걸', '이건', '그것', '그걸', '그건', '방금', '앞에서', '직전', '위에서', '그중', '그 중', '왜', '이유']

    def evaluate(self, context: GateContext) -> GateResult:
        if not contains_any(context.compact_question, self.terms):
            return GateResult()
        return GateResult(
            matched=True,
            gate_name=self.name,
            selected_path=None,
            answer_type=None,
            reason='후속 질문 가능성이 있는 후보 신호입니다. 최종 라우팅은 ContextResolver/IntentClassifier가 결정합니다.',
            confidence=0.55,
            is_final=False,
            category='followup',
        )
