from __future__ import annotations

from app.schemas import AgentRequest, ManufacturingContext, PredictionResponse

from app.agent.heavy.rag_schemas import RetrievalRequest


class RagQueryPlanner:
    """Plans retrieval queries only; it never executes retrieval."""

    def plan(
        self,
        *,
        request: AgentRequest,
        planned_query: str,
        prediction: PredictionResponse | None,
        manufacturing_context: ManufacturingContext,
        top_k: int,
        filters: dict | None = None,
    ) -> RetrievalRequest:
        parts = [planned_query or request.question or '']
        if prediction:
            parts.extend(prediction.predicted_modes)
            parts.extend([feature.feature for feature in prediction.evidence_features])
            parts.extend(prediction.recommended_actions[:4])
        parts.extend(manufacturing_context.document_search_terms)
        if request.inspection_notes:
            parts.append(request.inspection_notes)
        query = ' '.join(part for part in parts if part).strip() or 'manufacturing safety maintenance troubleshooting'
        return RetrievalRequest(
            query=query,
            top_k=top_k,
            filters=filters,
            reason='plan.rag_query, prediction evidence, domain search terms, and inspection notes were combined.',
        )

