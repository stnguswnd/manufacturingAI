from __future__ import annotations

from typing import Any


class ContextValidator:
    HEAVY_FORMAT_TOKENS = ['판정', '주요 근거', '위험도', '안전 확인', '권장 조치']

    def validate(
        self,
        *,
        context_resolution: dict[str, Any],
        context_packs: dict[str, Any],
        selected_path: str | None = None,
        answer: str | None = None,
    ) -> list[str]:
        warnings: list[str] = []
        classifier_context = (context_packs or {}).get('classifier_context') or {}

        if not context_resolution.get('is_followup') and classifier_context.get('last_answer_summary') and selected_path == 'supervisor_planning':
            warnings.append('new_question_may_be_overusing_previous_context')

        if context_resolution.get('is_followup'):
            answer_context = (context_packs or {}).get('answer_context') or {}
            if not answer_context.get('last_answer_memory') and 'last_answer_memory' in (context_resolution.get('context_needed') or []):
                warnings.append('followup_missing_last_answer_memory')

        if 'retrieved_docs' in classifier_context or 'retrieved_documents' in classifier_context or 'rag_contexts' in classifier_context:
            warnings.append('classifier_context_contains_rag_documents')

        if len(str(classifier_context)) > 4000:
            warnings.append('classifier_context_too_long')

        if selected_path == 'fast_concept_answer' and answer:
            if any(token in answer for token in self.HEAVY_FORMAT_TOKENS):
                warnings.append('heavy_answer_format_leaked_into_fast_concept')

        return warnings
