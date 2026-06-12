from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agent.context.followup_signals import FollowupSignalDetector
from app.agent.context.schemas import AnswerMemory, ContextResolution


class ContextResolver:
    """Resolves follow-up turns into standalone queries.

    The resolver consumes structured AnswerMemory instead of legacy focus or
    claim fields. Surface-keyword detection is isolated in
    FollowupSignalDetector and treated only as candidate evidence.
    """

    def resolve(
        self,
        *,
        current_user_message: str,
        last_answer_memory: Optional[AnswerMemory] = None,
        recent_turns: Optional[List[Dict[str, Any]]] = None,
        rolling_summary: Optional[str] = None,
    ) -> ContextResolution:
        message = (current_user_message or '').strip()
        signal = FollowupSignalDetector.detect(message)
        memory = last_answer_memory
        recent_turns = recent_turns or []
        rolling_summary = rolling_summary or ''

        if signal.is_clear_new_concept and not signal.has_followup_marker:
            return ContextResolution(
                is_followup=False,
                followup_type='none',
                standalone_query=message,
                context_needed=[],
                confidence=0.9,
                reason='clear_new_concept_question',
            )

        if not memory:
            if signal.has_followup_marker or signal.asks_reason or signal.is_short_ambiguous:
                return ContextResolution(
                    is_followup=True,
                    followup_type='ambiguous',
                    followup_target=None,
                    standalone_query=message,
                    context_needed=[],
                    confidence=0.35,
                    reason='followup_signal_without_answer_memory',
                )
            return self._standalone(message)

        if signal.asks_recommended_action_item and signal.item_index is not None:
            action = self._action_at(memory, signal.item_index)
            if action:
                return ContextResolution(
                    is_followup=True,
                    followup_type='previous_recommended_action_item',
                    followup_target=action,
                    followup_item_index=signal.item_index,
                    standalone_query=f'직전 답변의 권장조치 {signal.item_index}번 "{action}"이 필요한 이유를 설명해줘',
                    context_needed=['last_answer_memory.recommended_actions'],
                    confidence=0.88,
                    reason='recommended_action_item_followup_from_answer_memory',
                )
            return ContextResolution(
                is_followup=True,
                followup_type='ambiguous',
                followup_target='권장조치',
                followup_item_index=signal.item_index,
                standalone_query=message,
                context_needed=['last_answer_memory.recommended_actions'],
                confidence=0.42,
                reason='recommended_action_item_index_missing',
            )

        if signal.asks_recommended_actions and memory.recommended_actions:
            return ContextResolution(
                is_followup=True,
                followup_type='previous_recommended_actions',
                followup_target=memory.short_summary or memory.focus or '직전 답변',
                standalone_query='직전 답변의 권장조치를 중요한 순서대로 정리해줘',
                context_needed=['last_answer_memory.recommended_actions'],
                confidence=0.86,
                reason='recommended_actions_followup_from_answer_memory',
            )

        if signal.asks_reason and (memory.claims or memory.short_summary):
            claim = memory.claims[0] if memory.claims else memory.short_summary
            return ContextResolution(
                is_followup=True,
                followup_type='previous_answer_reason',
                followup_target=memory.focus or memory.short_summary,
                standalone_query=f'{claim}에 대한 이유를 설명해줘',
                context_needed=['last_answer_memory.claims', 'last_answer_memory.short_summary'],
                confidence=0.82,
                reason='reason_followup_from_answer_memory',
            )

        if signal.has_followup_marker and memory.focus:
            return ContextResolution(
                is_followup=True,
                followup_type='previous_concept',
                followup_target=memory.focus,
                standalone_query=f'{memory.focus}: {message}',
                context_needed=['last_answer_memory.focus'],
                confidence=0.76,
                reason='pronoun_followup_from_answer_memory_focus',
            )

        if signal.has_followup_marker and (recent_turns or rolling_summary):
            return ContextResolution(
                is_followup=True,
                followup_type='ambiguous',
                followup_target=None,
                standalone_query=message,
                context_needed=['recent_turns', 'rolling_summary'],
                confidence=0.45,
                reason='followup_signal_without_specific_memory_target',
            )

        return self._standalone(message)

    @staticmethod
    def _standalone(message: str) -> ContextResolution:
        return ContextResolution(
            is_followup=False,
            followup_type='none',
            followup_target=None,
            standalone_query=message,
            context_needed=[],
            confidence=0.9,
            reason='standalone_question',
        )

    @staticmethod
    def _action_at(memory: AnswerMemory, index: int) -> Optional[str]:
        if index <= 0:
            return None
        try:
            return memory.recommended_actions[index - 1].title
        except IndexError:
            return None
