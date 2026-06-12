from __future__ import annotations

from app.schemas import AgentRequest, ManufacturingContext, PredictionResponse, RagChunk


class StructuredAnswerPayloadBuilder:
    """Builds the LLM answer payload from already-selected heavy-path facts."""

    def build(
        self,
        *,
        request: AgentRequest,
        plan,
        prediction: PredictionResponse | None,
        manufacturing_context: ManufacturingContext,
        contexts: list[RagChunk],
        action_titles: list[str],
        safety_guidance: str | None,
        audit_feedback: list[str] | None = None,
    ) -> dict:
        return {
            'question': request.question,
            'context_resolution': (request.user_context or {}).get('context_resolution') or {},
            'inspection_notes': request.inspection_notes,
            'process_data': request.process_data.model_dump() if request.process_data else None,
            'plan': plan.model_dump() if plan else None,
            'prediction': prediction.model_dump() if prediction else None,
            'manufacturing_context': manufacturing_context.model_dump(),
            'rag_contexts': [chunk.model_dump() for chunk in contexts],
            'recommended_actions': action_titles,
            'safety_guidance': safety_guidance,
            'context_packs': (request.user_context or {}).get('context_packs') or {},
            'audit_feedback': audit_feedback or [],
            'output_policy': {
                'language': 'ko',
                'sections': ['판정', '주요 근거', '위험도', '안전 확인', '권장 조치', '주의 사항'],
                'must_include_citations': True,
                'no_equipment_control': True,
                'must_respect_safety_gates': True,
            },
        }
