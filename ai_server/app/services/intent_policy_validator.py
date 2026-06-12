from __future__ import annotations

from app.services.intent_classifier_service import IntentClassifierInput, IntentClassifierOutput, ResolvedReference


DIAGNOSIS_GUARD_TERMS = [
    '위험',
    '고장',
    '이상',
    '판단',
    '분석',
    '점검',
    '정비',
    '조치',
    '알람',
    '현재값',
    '현재 값',
    '이조건',
    '이 조건',
    '공정데이터',
    '공정 데이터',
]
SAFETY_GUARD_TERMS = ['안전', '정비전', '정비 전', 'loto', 'lockout', 'tagout', '방호', '가드', '보호구']
REPORT_GUARD_TERMS = ['보고서', '리포트', '초안', '기록', '문서화']


class IntentPolicyValidator:
    def validate(self, output: IntentClassifierOutput, payload: IntentClassifierInput) -> IntentClassifierOutput:
        question = payload.current_question or ''
        compact = question.lower().replace(' ', '')

        if output.confidence < 0.6:
            return IntentClassifierOutput(
                selected_path='unsupported_or_clarification',
                answer_type='clarification',
                resolved_reference=ResolvedReference(type='none', source='none', confidence=0.0),
                focus_update_policy='skip',
                confidence=output.confidence,
                reason='Classifier confidence below threshold; clarification required.',
            )

        if payload.has_current_process_data and self._contains_any(compact, DIAGNOSIS_GUARD_TERMS):
            return output.model_copy(update={
                'selected_path': 'supervisor_planning',
                'answer_type': 'diagnosis',
                'requires_prediction': True,
                'requires_rag': True,
                'requires_safety': True,
                'focus_update_policy': 'update',
                'resolved_reference': ResolvedReference(type='process_data', text='현재 공정 조건', normalized='current_process_data', source='current_question', confidence=max(output.confidence, 0.85)),
                'reason': f'Hard gate override: current process data diagnosis request. {self._safe_reason(output.reason)}',
            })

        if self._contains_any(compact, REPORT_GUARD_TERMS):
            return output.model_copy(update={
                'selected_path': 'supervisor_planning',
                'answer_type': 'report',
                'requires_report': True,
                'requires_rag': True,
                'focus_update_policy': 'update',
                'reason': f'Hard gate override: report request. {self._safe_reason(output.reason)}',
            })

        if self._contains_any(compact, SAFETY_GUARD_TERMS):
            if output.selected_path == 'general_lightweight_answer':
                return output.model_copy(update={
                    'selected_path': 'supervisor_planning',
                    'answer_type': 'diagnosis',
                    'requires_rag': True,
                    'requires_safety': True,
                    'reason': f'Hard gate override: safety/maintenance request. {self._safe_reason(output.reason)}',
                })
            return output.model_copy(update={'requires_safety': True})

        if output.selected_path == 'general_lightweight_answer' and (output.requires_prediction or output.requires_safety or output.requires_report):
            return output.model_copy(update={
                'selected_path': 'supervisor_planning',
                'answer_type': 'diagnosis' if output.requires_prediction else output.answer_type,
                'requires_rag': output.requires_rag or output.requires_safety or output.requires_report,
                'reason': f'Hard gate override: lightweight path had heavy requirements. {self._safe_reason(output.reason)}',
            })

        if output.selected_path == 'meta_feedback':
            return output.model_copy(update={'focus_update_policy': 'preserve'})
        if output.answer_type == 'rationale':
            return output.model_copy(update={'focus_update_policy': 'preserve'})
        return output

    @staticmethod
    def _contains_any(compact_question: str, terms: list[str]) -> bool:
        compact_terms = [term.lower().replace(' ', '') for term in terms]
        return any(term in compact_question for term in compact_terms)

    @staticmethod
    def _safe_reason(reason: str) -> str:
        text = str(reason or '').strip()
        blocked = ['badrequesterror', 'invalid_json_schema', 'traceback', 'stack trace', 'valueerror', 'additionalproperties']
        if not text or any(token in text.lower() for token in blocked):
            return 'policy validation selected a safer route'
        return text
