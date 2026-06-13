from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.agent.artifacts import AnswerDraft, ContextArtifact, EvidenceArtifact, PlanningArtifact, PredictionArtifact, RequestArtifact, ResponseArtifact, RuntimeArtifact, SafetyArtifact


class MemoryState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]
    context: dict[str, Any]
    planning: dict[str, Any]
    prediction: dict[str, Any] | None
    evidence: dict[str, Any] | None
    safety: dict[str, Any] | None
    draft: dict[str, Any] | None
    response: dict[str, Any]
    runtime: dict[str, Any]
    user_id: str
    answer_memory: dict[str, Any]
    session_last_process_data: dict[str, Any] | None
    warnings: list[str]
    diagnostics: dict[str, Any]
    trace: dict[str, Any]
    output: dict[str, Any]


class MemoryInput(BaseModel):
    run_id: str
    request: RequestArtifact
    context: ContextArtifact = Field(default_factory=ContextArtifact)
    planning: PlanningArtifact | None = None
    prediction: PredictionArtifact | None = None
    evidence: EvidenceArtifact | None = None
    safety: SafetyArtifact | None = None
    draft: AnswerDraft | None = None
    response: ResponseArtifact
    runtime: RuntimeArtifact = Field(default_factory=RuntimeArtifact)
    user_id: str


class MemoryOutput(BaseModel):
    last_answer_memory: dict[str, Any] = Field(default_factory=dict)
    recent_turn_routes: list[dict[str, Any]] = Field(default_factory=list)
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    session_last_process_data: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)


def to_state(input_data: MemoryInput) -> MemoryState:
    return {
        'run_id': input_data.run_id,
        'request': input_data.request.model_dump(mode='json'),
        'context': input_data.context.model_dump(mode='json'),
        'planning': input_data.planning.model_dump(mode='json') if input_data.planning else None,
        'prediction': input_data.prediction.model_dump(mode='json') if input_data.prediction else None,
        'evidence': input_data.evidence.model_dump(mode='json') if input_data.evidence else None,
        'safety': input_data.safety.model_dump(mode='json') if input_data.safety else None,
        'draft': input_data.draft.model_dump(mode='json') if input_data.draft else None,
        'response': input_data.response.model_dump(mode='json'),
        'runtime': input_data.runtime.model_dump(mode='json'),
        'user_id': input_data.user_id,
        'warnings': [],
        'diagnostics': {},
        'trace': {},
    }


def to_output(state: MemoryState) -> MemoryOutput:
    output = state.get('output') or {}
    return MemoryOutput(
        last_answer_memory=dict(output.get('last_answer_memory') or state.get('answer_memory') or {}),
        recent_turn_routes=list(output.get('recent_turn_routes') or state.get('recent_turn_routes') or []),
        recent_turns=list(output.get('recent_turns') or state.get('recent_turns') or []),
        session_last_process_data=output.get('session_last_process_data', state.get('session_last_process_data')),
        warnings=list(output.get('warnings') or state.get('warnings') or []),
        diagnostics=dict(output.get('diagnostics') or state.get('diagnostics') or {}),
        trace=dict(output.get('trace') or state.get('trace') or {}),
    )
