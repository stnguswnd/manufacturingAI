from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.agent.artifacts import SafetyArtifact
from app.schemas.agent import AgentRequest
from app.schemas.domain import ManufacturingContext
from app.schemas.prediction import PredictionResponse
from app.schemas.rag import RagChunk


class SafetyState(TypedDict, total=False):
    request: dict[str, Any]
    prediction: dict[str, Any] | None
    manufacturing_context: dict[str, Any] | None
    retrieved_documents: list[dict[str, Any]]
    structured_answer_payload: dict[str, Any]
    safety_artifact: dict[str, Any]
    safety_guidance: str | None
    safety_warnings: list[str]
    trace: dict[str, Any]
    output: dict[str, Any]


class SafetyInput(BaseModel):
    request: AgentRequest
    prediction: PredictionResponse | None = None
    manufacturing_context: ManufacturingContext | None = None
    retrieved_documents: list[RagChunk] = Field(default_factory=list)
    structured_answer_payload: dict[str, Any] = Field(default_factory=dict)


class SafetyOutput(BaseModel):
    manufacturing_context: ManufacturingContext
    structured_answer_payload: dict[str, Any] = Field(default_factory=dict)
    safety_artifact: SafetyArtifact = Field(default_factory=SafetyArtifact)
    safety_guidance: str | None = None
    safety_warnings: list[str] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)


def to_state(input_data: SafetyInput) -> SafetyState:
    return {
        'request': input_data.request.model_dump(),
        'prediction': input_data.prediction.model_dump() if input_data.prediction else None,
        'manufacturing_context': input_data.manufacturing_context.model_dump() if input_data.manufacturing_context else None,
        'retrieved_documents': [chunk.model_dump() for chunk in input_data.retrieved_documents],
        'structured_answer_payload': dict(input_data.structured_answer_payload),
        'safety_artifact': {},
        'safety_warnings': [],
        'trace': {},
    }


def to_output(state: SafetyState) -> SafetyOutput:
    output = state.get('output') or {}
    return SafetyOutput(
        manufacturing_context=ManufacturingContext.model_validate(output.get('manufacturing_context') or state['manufacturing_context']),
        structured_answer_payload=dict(output.get('structured_answer_payload') or state.get('structured_answer_payload') or {}),
        safety_artifact=SafetyArtifact.model_validate(output.get('safety_artifact') or state.get('safety_artifact') or {}),
        safety_guidance=output.get('safety_guidance', state.get('safety_guidance')),
        safety_warnings=list(output.get('safety_warnings') or state.get('safety_warnings') or []),
        trace=dict(output.get('trace') or state.get('trace') or {}),
    )
