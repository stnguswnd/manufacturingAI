from __future__ import annotations

from typing import Any, Callable

from app.agent.routing import GateContext, GateRegistry, GateResult
from app.schemas import LLMUsageRecord
from app.services.glossary_answer_service import GlossaryAnswerService
from app.services.intent_classifier_service import (
    IntentClassifierInput,
    IntentClassifierOutput,
    IntentClassifierService,
    ResolvedReference,
)
from app.services.intent_policy_validator import IntentPolicyValidator


class IntentGatewayService:
    def __init__(
        self,
        *,
        intent_classifier: IntentClassifierService | None = None,
        policy_validator: IntentPolicyValidator | None = None,
        glossary: GlossaryAnswerService | None = None,
        gate_registry: GateRegistry | None = None,
    ):
        self.intent_classifier = intent_classifier
        self.policy_validator = policy_validator or IntentPolicyValidator()
        self.glossary = glossary or GlossaryAnswerService()
        self.gate_registry = gate_registry or GateRegistry()

    def classify(self, *, request, user_context: dict[str, Any], usage_callback: Callable[[LLMUsageRecord], None] | None = None) -> dict[str, Any]:
        context_packs = user_context.get('context_packs') or {}
        classifier_context = context_packs.get('classifier_context') or user_context.get('classifier_context') or {}
        context_resolution = user_context.get('context_resolution') or classifier_context.get('context_resolution') or {}
        question = classifier_context.get('standalone_query') or request.question or ''
        original = request.question or ''
        compact = self._compact(question)
        glossary_hit = self.glossary.resolve_term(question)
        gate_result = self.gate_registry.evaluate(GateContext(
            question=question,
            original_question=original,
            compact_question=compact,
            has_process_data=request.process_data is not None,
            generate_report=bool(request.generate_report),
            context_resolution=context_resolution,
            last_answer_memory=user_context.get('last_answer_memory') or {},
            glossary_hit=glossary_hit,
        ))
        if gate_result.matched and gate_result.is_final:
            return self._dict_from_gate(gate_result)

        classifier_input = self._classifier_input(request=request, user_context=user_context, question=question)
        if self.intent_classifier is None:
            classifier_output = IntentClassifierService.safe_fallback(classifier_input, reason='intent classifier not configured')
        else:
            try:
                classifier_output = self.intent_classifier.classify(classifier_input, usage_callback=usage_callback)
            except TypeError:
                try:
                    classifier_output = self.intent_classifier.classify(classifier_input)
                except Exception:
                    classifier_output = IntentClassifierService.safe_fallback(
                        classifier_input,
                        reason='intent classifier unavailable; safe fallback selected',
                    )
            except Exception:
                classifier_output = IntentClassifierService.safe_fallback(
                    classifier_input,
                    reason='intent classifier unavailable; safe fallback selected',
                )
        final_output = self.policy_validator.validate(classifier_output, classifier_input)
        return self._dict_from_output(final_output, current_turn={}, turn_type=self._turn_type(final_output))

    def _classifier_input(self, *, request, user_context: dict[str, Any], question: str) -> IntentClassifierInput:
        context_packs = user_context.get('context_packs') or {}
        classifier_context = context_packs.get('classifier_context') or user_context.get('classifier_context') or {}
        classifier_question = classifier_context.get('standalone_query') or question
        return IntentClassifierInput(
            current_question=question,
            standalone_query=classifier_question,
            last_answer_summary=classifier_context.get('last_answer_summary'),
            last_answer_focus=classifier_context.get('last_answer_focus'),
            is_followup=bool(classifier_context.get('is_followup')),
            followup_type=classifier_context.get('followup_type') or 'none',
            followup_target=classifier_context.get('followup_target'),
            recent_turn_intents=list(classifier_context.get('recent_turn_intents') or []),
            has_current_process_data=request.process_data is not None,
            has_previous_process_data=bool(user_context.get('process_data_reference_policy', {}).get('previous_turn_process_data_used')),
            has_session_last_process_data=bool(user_context.get('process_data_reference_policy', {}).get('session_last_process_data_available')),
            current_process_data_summary=request.process_data.model_dump() if request.process_data else None,
            previous_process_data_summary=user_context.get('previous_turn_process_data_summary'),
            glossary_candidates=self._glossary_candidates(question),
            current_thread_id=f'{request.user_id}:{request.session_id}',
        )

    def _glossary_candidates(self, question: str) -> list[str]:
        text = (question or '').lower()
        candidates: list[str] = []
        for term, data in self.glossary.canonical_terms().items():
            if term.lower() in text or any(str(alias).lower() in text for alias in data.get('aliases') or []):
                candidates.append(term)
        return candidates

    @staticmethod
    def _dict_from_output(output: IntentClassifierOutput, *, current_turn: dict[str, Any], turn_type: str | None = None) -> dict[str, Any]:
        data = {
            'turn_type': turn_type or output.answer_type,
            'selected_path': output.selected_path,
            'answer_type': output.answer_type,
            'resolved': False,
            'resolved_question': current_turn.get('original_question'),
            'resolved_target': None,
            'resolved_reference': output.resolved_reference.model_dump(),
            'requires_prediction': output.requires_prediction,
            'requires_rag': output.requires_rag,
            'requires_safety': output.requires_safety,
            'requires_report': output.requires_report,
            'resolved_claim': output.resolved_claim,
            'phrase_repair': output.phrase_repair.model_dump() if output.phrase_repair else None,
            'focus_update_policy': output.focus_update_policy,
            'confidence': output.confidence,
            'reason': output.reason,
        }
        if not data['resolved_target'] and output.resolved_reference.type == 'concept':
            data['resolved_target'] = {
                'label': output.resolved_reference.normalized or output.resolved_reference.text,
                'type': 'concept',
                'source': output.resolved_reference.source,
                'confidence': output.resolved_reference.confidence,
            }
        return data

    @staticmethod
    def _turn_type(output: IntentClassifierOutput) -> str:
        if output.selected_path == 'general_lightweight_answer':
            return 'rationale_followup' if output.answer_type == 'rationale' else output.answer_type
        if output.selected_path == 'fast_concept_answer':
            return 'concept_followup' if output.focus_update_policy == 'preserve' else 'general_concept'
        if output.selected_path == 'supervisor_planning':
            return 'prediction_request' if output.requires_prediction else 'maintenance_request'
        if output.selected_path in {'recommended_action_recap', 'recommended_action_item_explanation'}:
            return output.selected_path
        return output.selected_path

    @staticmethod
    def _compact(question: str) -> str:
        return ''.join((question or '').split()).lower()

    @staticmethod
    def _dict_from_gate(result: GateResult) -> dict[str, Any]:
        reference = result.resolved_reference or {'type': 'none', 'source': 'none', 'confidence': 0.0}
        output = IntentClassifierOutput(
            selected_path=result.selected_path or 'unsupported_or_clarification',
            answer_type=result.answer_type or 'clarification',
            resolved_reference=ResolvedReference.model_validate(reference),
            resolved_claim=result.resolved_claim,
            requires_prediction=result.requires_prediction,
            requires_rag=result.requires_rag,
            requires_safety=result.requires_safety,
            requires_report=result.requires_report,
            focus_update_policy=result.focus_update_policy,
            confidence=result.confidence,
            reason=result.reason,
        )
        return IntentGatewayService._dict_from_output(output, current_turn={}, turn_type=result.turn_type)
