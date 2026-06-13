from __future__ import annotations

from typing_extensions import NotRequired, TypedDict

from app.agent.artifacts import (
    AnswerDraft,
    AuditArtifact,
    ContextArtifact,
    EvidenceArtifact,
    MemoryArtifact,
    PlanningArtifact,
    PredictionArtifact,
    RequestArtifact,
    ResponseArtifact,
    RuntimeArtifact,
    SafetyArtifact,
    ValidationReport,
)


class ManufacturingAgentState(TypedDict, total=False):
    state_schema_version: int
    run_id: str
    user_id: str
    session_id: str
    thread_id: str

    request: RequestArtifact | None
    context: ContextArtifact | None
    planning: PlanningArtifact | None
    prediction: PredictionArtifact | None
    evidence: EvidenceArtifact | None
    safety: SafetyArtifact | None
    draft: AnswerDraft | None
    validation: ValidationReport | None
    response: ResponseArtifact | None
    memory: MemoryArtifact | None
    audit: AuditArtifact | None
    runtime: RuntimeArtifact | None
