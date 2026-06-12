from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agent.context.schemas import AnswerMemory, ContextPacks, ContextResolution


class ContextPackBuilder:
    """Builds node-specific context contracts.

    Each downstream node sees only the context it is allowed to use. This is the
    boundary that prevents retrieved documents, safety manuals, or raw state
    from leaking into the intent classifier and lightweight formatters.
    """

    def build(
        self,
        *,
        current_user_message: str,
        context_resolution: ContextResolution,
        compressed_context: Dict[str, Any],
        last_answer_memory: Optional[AnswerMemory],
        recent_turn_routes: List[Dict[str, Any]],
        process_data_policy: Dict[str, Any],
        glossary_summary: Optional[Dict[str, Any]] = None,
    ) -> ContextPacks:
        memory = last_answer_memory
        classifier_context = {
            'current_user_message': current_user_message,
            'standalone_query': context_resolution.standalone_query,
            'is_followup': context_resolution.is_followup,
            'followup_type': context_resolution.followup_type,
            'followup_target': context_resolution.followup_target,
            'last_answer_summary': memory.short_summary if memory else None,
            'last_answer_focus': memory.focus if memory else None,
            'recent_turn_intents': recent_turn_routes[-5:],
            'glossary_summary': glossary_summary or {},
        }
        answer_context = {
            'standalone_query': context_resolution.standalone_query,
            'context_resolution': context_resolution.model_dump(),
            'relevant_answer_memory': self._public_memory(memory),
            'rolling_summary': compressed_context.get('rolling_summary') or '',
            'recent_turns': compressed_context.get('recent_turns') or [],
            'process_data_reference_policy': process_data_policy,
        }
        rag_context = {
            'query': context_resolution.standalone_query,
            'followup_type': context_resolution.followup_type,
            'followup_target': context_resolution.followup_target,
        }
        safety_context = {
            'standalone_query': context_resolution.standalone_query,
            'followup_type': context_resolution.followup_type,
            'process_data_reference_policy': process_data_policy,
        }
        formatter_context = self._formatter_context(context_resolution, memory)
        memory_writer_context = {
            'current_user_message': current_user_message,
            'standalone_query': context_resolution.standalone_query,
            'context_resolution': context_resolution.model_dump(),
        }
        return ContextPacks(
            classifier_context=classifier_context,
            answer_context=answer_context,
            rag_context=rag_context,
            safety_context=safety_context,
            formatter_context=formatter_context,
            memory_writer_context=memory_writer_context,
        )

    @staticmethod
    def _public_memory(memory: Optional[AnswerMemory]) -> Optional[Dict[str, Any]]:
        if not memory:
            return None
        return {
            'selected_path': memory.selected_path,
            'answer_type': memory.answer_type,
            'short_summary': memory.short_summary,
            'focus': memory.focus,
            'key_points': memory.key_points[:8],
            'claims': memory.claims[:5],
            'recommended_actions': [item.model_dump() for item in memory.recommended_actions[:10]],
            'decisions': memory.decisions[:8],
            'source_refs': memory.source_refs[:6],
            'safety_level': memory.safety_level,
        }

    @staticmethod
    def _formatter_context(resolution: ContextResolution, memory: Optional[AnswerMemory]) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            'standalone_query': resolution.standalone_query,
            'followup_type': resolution.followup_type,
            'followup_target': resolution.followup_target,
            'public_reason': None,
        }
        if resolution.followup_type == 'previous_recommended_actions' and memory:
            base.update({
                'selected_path': 'recommended_action_recap',
                'answer_type': 'recommended_action_recap',
            'recommended_actions': [item.model_dump() for item in memory.recommended_actions[:10]],
            })
        elif resolution.followup_type == 'previous_recommended_action_item' and memory:
            action = None
            if resolution.followup_item_index and 0 < resolution.followup_item_index <= len(memory.recommended_actions):
                action = memory.recommended_actions[resolution.followup_item_index - 1].model_dump()
            base.update({
                'selected_path': 'recommended_action_item_explanation',
                'answer_type': 'recommended_action_item_explanation',
                'recommended_action_item': action,
                'followup_item_index': resolution.followup_item_index,
            })
        return base
