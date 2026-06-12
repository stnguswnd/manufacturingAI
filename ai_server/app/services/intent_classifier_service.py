from __future__ import annotations

from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from app.prompts.intent_classifier_prompt import INTENT_CLASSIFIER_SYSTEM_PROMPT
from app.schemas import LLMUsageRecord
from app.services.llm_service import LLMService
from app.services.structured_output_schema import to_openai_strict_json_schema


SelectedPath = Literal[
    'fast_concept_answer',
    'general_lightweight_answer',
    'lightweight_rag_answer',
    'supervisor_planning',
    'report_answer',
    'meta_feedback',
    'unsupported_or_clarification',
    'recommended_action_recap',
    'recommended_action_item_explanation',
    'safety_answer',
    'rag_answer',
    'heavy_analysis_answer',
]

AnswerType = Literal[
    'definition',
    'watch_points',
    'disadvantages',
    'rationale',
    'chart_guidance',
    'explanation',
    'diagnosis',
    'report',
    'clarification',
    'meta_feedback',
    'recommended_action_recap',
    'recommended_action_item_explanation',
]

ReferenceType = Literal[
    'concept',
    'process_data',
    'previous_answer_claim',
    'previous_recommended_action',
    'report',
    'document',
    'none',
]

ReferenceSource = Literal[
    'current_question',
    'answer_memory',
    'context_resolution',
    'previous_process_data',
    'session_last_process_data',
    'none',
]

FocusUpdatePolicy = Literal['update', 'preserve', 'skip']


class IntentClassifierInput(BaseModel):
    current_question: str
    standalone_query: str | None = None
    last_answer_summary: str | None = None
    last_answer_focus: str | None = None
    is_followup: bool = False
    followup_type: str = 'none'
    followup_target: str | None = None
    recent_turn_intents: list[dict[str, Any]] = Field(default_factory=list)
    has_current_process_data: bool = False
    has_previous_process_data: bool = False
    has_session_last_process_data: bool = False
    current_process_data_summary: dict[str, Any] | None = None
    previous_process_data_summary: dict[str, Any] | None = None
    glossary_candidates: list[str] = Field(default_factory=list)
    current_thread_id: str | None = None


class ResolvedReference(BaseModel):
    type: ReferenceType
    text: str | None = None
    normalized: str | None = None
    domain_focus: str | None = None
    source: ReferenceSource
    confidence: float = Field(ge=0.0, le=1.0)


class PhraseRepair(BaseModel):
    surface_text: str | None = None
    resolved_phrase: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class IntentClassifierOutput(BaseModel):
    selected_path: SelectedPath
    answer_type: AnswerType
    resolved_reference: ResolvedReference
    resolved_claim: str | None = None
    phrase_repair: PhraseRepair | None = None
    requires_prediction: bool = False
    requires_rag: bool = False
    requires_safety: bool = False
    requires_report: bool = False
    focus_update_policy: FocusUpdatePolicy = 'preserve'
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class IntentClassifierService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.last_error: str | None = None

    def classify(self, payload: IntentClassifierInput, usage_callback: Callable[[LLMUsageRecord], None] | None = None) -> IntentClassifierOutput:
        self.last_error = None
        try:
            raw = self.llm_service.generate_json(
                schema_name='intent_classifier_output',
                schema=self.output_schema(),
                system_prompt=INTENT_CLASSIFIER_SYSTEM_PROMPT,
                payload=payload.model_dump(),
                operation='intent_classifier',
                usage_callback=usage_callback,
            )
            if raw is None:
                raise ValueError(self.llm_service.last_error or 'intent classifier returned empty output')
            return IntentClassifierOutput.model_validate(raw)
        except (ValidationError, ValueError, Exception) as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
            return self.safe_fallback(payload, reason='intent classifier unavailable; safe fallback selected')

    @classmethod
    def output_schema(cls) -> dict[str, Any]:
        return to_openai_strict_json_schema(IntentClassifierOutput.model_json_schema())

    @staticmethod
    def safe_fallback(payload: IntentClassifierInput, *, reason: str = 'classifier_fallback') -> IntentClassifierOutput:
        if payload.has_current_process_data:
            return IntentClassifierOutput(
                selected_path='supervisor_planning',
                answer_type='diagnosis',
                resolved_reference=ResolvedReference(type='process_data', text='현재 공정 조건', normalized='current_process_data', source='current_question', confidence=0.7),
                requires_prediction=True,
                requires_rag=True,
                requires_safety=True,
                focus_update_policy='update',
                confidence=0.7,
                reason=f'safe fallback: {reason}',
            )
        if payload.last_answer_focus:
            return IntentClassifierOutput(
                selected_path='fast_concept_answer',
                answer_type='watch_points',
                resolved_reference=ResolvedReference(
                    type='concept',
                    text=payload.last_answer_focus,
                    normalized=payload.last_answer_focus,
                    source='answer_memory',
                    confidence=0.65,
                ),
                focus_update_policy='preserve',
                confidence=0.65,
                reason=f'safe fallback using answer memory focus: {reason}',
            )
        return IntentClassifierOutput(
            selected_path='unsupported_or_clarification',
            answer_type='clarification',
            resolved_reference=ResolvedReference(type='none', source='none', confidence=0.0),
            focus_update_policy='skip',
            confidence=0.5,
            reason=f'safe fallback clarification: {reason}',
        )
