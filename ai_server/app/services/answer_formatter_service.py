from __future__ import annotations

from typing import Any

from app.agent.formatters import FormatterRegistry
from app.agent.safety import SafetyContext, SafetyContextBuilder


class AnswerFormatterService:
    """Compatibility facade over FormatterRegistry.

    New formatting behavior should live in app.agent.formatters or
    app.agent.safety. This service remains to keep existing call sites stable
    while root_graph is migrated toward direct registry use.
    """

    def __init__(self, registry: FormatterRegistry | None = None):
        self.registry = registry or FormatterRegistry()
        self.safety_context_builder = SafetyContextBuilder()

    def fast_concept_answer(self, public_context: dict[str, Any]) -> str:
        return self.registry.format('fast_concept_answer', public_context)

    def general_lightweight_answer(self, public_context: dict[str, Any]) -> str:
        return self.registry.format('general_lightweight_answer', public_context)

    def recommended_action_recap(self, formatter_context: dict[str, Any]) -> str:
        return self.registry.format('recommended_action_recap', formatter_context)

    def recommended_action_item_explanation(self, formatter_context: dict[str, Any]) -> str:
        return self.registry.format('recommended_action_item_explanation', formatter_context)

    def unsupported_or_clarification(self, *, reason: str, missing_info: str | None = None) -> str:
        return self.registry.format('clarification', {
            'public_reason': self._safe_public_reason(reason),
            'missing_info': missing_info,
        })

    def safety_answer(self, public_context: dict[str, Any]) -> str:
        safety_context = public_context.get('safety_context')
        if isinstance(safety_context, SafetyContext):
            context = safety_context
        elif isinstance(safety_context, dict):
            context = SafetyContext.model_validate(safety_context)
        else:
            constraints = public_context.get('safety_constraints') or {}
            context = self.safety_context_builder.policy.build_context(
                must_include=list(constraints.get('must_include') or []),
                forbidden=list(constraints.get('forbidden') or []),
                strict=True,
            )
        return self.registry.format('safety_answer', {'safety_context': context})

    @staticmethod
    def meta_feedback(public_context: dict[str, Any]) -> str:
        focus_text = public_context.get('answer_memory_focus')
        if focus_text:
            first = f'맞습니다. 이 경우 "이걸"은 직전 대화의 "{focus_text}"로 해석하는 것이 자연스럽습니다.'
        else:
            first = '맞습니다. 이런 경우에는 직전 대화의 핵심 주제를 먼저 참조해서 지시어를 해석해야 합니다.'
        return (
            f'{first}\n\n'
            '수정 방향은 단순히 이전 실행 이력을 다시 검색하는 것이 아니라, 직전 답변의 핵심 기억을 `AnswerMemory`로 저장하고 다음 턴에서 "이것/이걸/그거" 같은 지시어가 나오면 그 값을 먼저 참조하도록 만드는 것입니다.\n\n'
            '또한 이런 시스템 동작 피드백에는 제조 분석 보고서 형식을 붙이지 않고, 짧게 문제를 인정하고 수정 방향만 설명해야 합니다.'
        )

    @staticmethod
    def sanitize_public_answer(answer: str) -> str:
        blocked = [
            'resolved=false',
            'resolved_target',
            'question_kind',
            'context_policy',
            'rag_contexts',
            'safety_gates',
            'recent_runs',
            'similar_runs',
            'audit_notes',
            'action_plan',
            'current turn information',
            'current_turn',
            'internal_reason',
            'badrequesterror',
            'invalid_json_schema',
        ]
        lines = []
        for line in (answer or '').splitlines():
            lowered = line.lower()
            if any(token.lower() in lowered for token in blocked):
                continue
            lines.append(line)
        return '\n'.join(lines).strip()

    @staticmethod
    def _safe_public_reason(reason: str) -> str:
        text = str(reason or '').strip()
        internal_tokens = [
            'badrequesterror',
            'invalid_json_schema',
            'stack trace',
            'traceback',
            'valueerror',
            'schema for response_format',
            'additionalproperties',
            'raw exception',
        ]
        if not text or any(token in text.lower() for token in internal_tokens):
            return '요청 의도나 참조 대상을 안정적으로 확정하지 못했습니다.'
        return text
