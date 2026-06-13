from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.agent.artifacts import EvidenceArtifact
from app.agent.heavy.rag_schemas import EvidenceGrade
from app.schemas.agent import AgentPlan, AgentRequest
from app.schemas.domain import ManufacturingContext
from app.schemas.prediction import PredictionResponse
from app.schemas.rag import RagChunk


class RagEvidenceState(TypedDict, total=False):
    request: dict[str, Any]
    plan: dict[str, Any]
    prediction: dict[str, Any] | None
    manufacturing_context: dict[str, Any]
    top_k: int
    query_specs: list[dict[str, Any]]
    raw_chunks: list[dict[str, Any]]
    filtered_chunks: list[dict[str, Any]]
    selected_chunks: list[dict[str, Any]]
    evidence_grade: dict[str, Any]
    citations: list[dict[str, Any]]
    evidence_artifact: dict[str, Any]
    warnings: list[str]
    trace: dict[str, Any]
    output: dict[str, Any]
    replan_count_delta: int


class RagEvidenceInput(BaseModel):
    request: AgentRequest
    plan: AgentPlan
    prediction: PredictionResponse | None = None
    manufacturing_context: ManufacturingContext
    top_k: int = Field(default=5, ge=1, le=20)


class RagEvidenceOutput(BaseModel):
    plan: AgentPlan
    route: list[str] = Field(default_factory=list)
    retrieved_documents: list[RagChunk] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    evidence_grade: EvidenceGrade
    evidence_artifact: EvidenceArtifact = Field(default_factory=EvidenceArtifact)
    manufacturing_context: ManufacturingContext
    warnings: list[str] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    replan_count_delta: int = 0


def to_state(input_data: RagEvidenceInput) -> RagEvidenceState:
    return {
        'request': input_data.request.model_dump(),
        'plan': input_data.plan.model_dump(),
        'prediction': input_data.prediction.model_dump() if input_data.prediction else None,
        'manufacturing_context': input_data.manufacturing_context.model_dump(),
        'top_k': input_data.top_k,
        'warnings': [],
        'trace': {},
        'replan_count_delta': 0,
    }


def to_output(state: RagEvidenceState) -> RagEvidenceOutput:
    output = state.get('output') or {}
    return RagEvidenceOutput(
        plan=AgentPlan.model_validate(output.get('plan') or state['plan']),
        route=list(output.get('route') or []),
        retrieved_documents=[RagChunk.model_validate(item) for item in output.get('retrieved_documents') or []],
        citations=list(output.get('citations') or []),
        evidence_grade=EvidenceGrade.model_validate(output.get('evidence_grade') or state.get('evidence_grade') or {'usable': False, 'weak_reason': 'no_grade'}),
        evidence_artifact=EvidenceArtifact.model_validate(output.get('evidence_artifact') or state.get('evidence_artifact') or {}),
        manufacturing_context=ManufacturingContext.model_validate(output.get('manufacturing_context') or state['manufacturing_context']),
        warnings=list(output.get('warnings') or state.get('warnings') or []),
        trace=dict(output.get('trace') or state.get('trace') or {}),
        replan_count_delta=int(output.get('replan_count_delta') or state.get('replan_count_delta') or 0),
    )
