from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ReviewAction = Literal[
    'pass',
    'rewrite_only',
    'rerun_rag',
    'rerun_safety',
    'clarification_required',
    'block',
    'max_retry_exceeded',
]


class RequestArtifact(BaseModel):
    user_id: str
    session_id: str
    question: str
    original_message: str
    normalized_message: str | None = None
    process_data: dict[str, Any] | None = None
    inspection_notes: str | None = None
    top_k: int | None = None
    mode: str = 'auto'
    llm_model: str | None = None


class ContextArtifact(BaseModel):
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    recent_turn_routes: list[dict[str, Any]] = Field(default_factory=list)
    rolling_summary: str = ''
    last_answer_memory: dict[str, Any] = Field(default_factory=dict)
    context_resolution: dict[str, Any] = Field(default_factory=dict)
    context_packs: dict[str, Any] = Field(default_factory=dict)
    compressed_context: dict[str, Any] = Field(default_factory=dict)
    user_context: dict[str, Any] = Field(default_factory=dict)
    turn_context: dict[str, Any] = Field(default_factory=dict)
    context_validation_warnings: list[str] = Field(default_factory=list)
    ai4i_feature_status: dict[str, Any] = Field(default_factory=dict)
    turn_process_data: dict[str, Any] | None = None
    previous_turn_process_data: dict[str, Any] | None = None
    session_last_process_data: dict[str, Any] | None = None
    process_data_reference_policy: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PlanningArtifact(BaseModel):
    selected_path: str
    answer_type: str
    intent: str | None = None
    needs_prediction: bool = False
    needs_rag: bool = False
    needs_safety: bool = False
    draft_ready: bool = False
    clarification_required: bool = False
    fast_answer_ready: bool = False
    missing_features: list[str] = Field(default_factory=list)
    completed_nodes: set[str] = Field(default_factory=set)
    next_node: str | None = None
    agent_plan: dict[str, Any] = Field(default_factory=dict)
    diagnostic_plan: dict[str, Any] = Field(default_factory=dict)
    route: list[str] = Field(default_factory=list)
    reasoning_summary: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PredictionArtifact(BaseModel):
    called: bool = False
    result: dict[str, Any] | None = None
    skip_reason: str | None = None
    parsed_features: dict[str, Any] = Field(default_factory=dict)
    missing_features: list[str] = Field(default_factory=list)
    ambiguous_features: list[str] = Field(default_factory=list)
    invalid_features: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvidenceArtifact(BaseModel):
    profile: str | None = None
    queries: list[str] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    selected_source_ids: list[str] = Field(default_factory=list)
    required_safety_gates: list[str] = Field(default_factory=list)
    evidence_covers_required_gates: bool = True
    missing_gate_evidence: list[str] = Field(default_factory=list)
    generic_document_downgraded: bool = False
    retrieval_diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SafetyArtifact(BaseModel):
    required_gates: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)
    public_guidance: str | None = None
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AnswerDraft(BaseModel):
    text: str
    route: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_used: bool = False
    llm_error: str | None = None
    recommended_actions: list[str] = Field(default_factory=list)
    safety_guidance: str | None = None


class ValidationFailure(BaseModel):
    code: str
    message: str
    severity: Literal['info', 'warning', 'error', 'critical'] = 'warning'
    source: Literal['citation', 'safety', 'format', 'prediction', 'debug_leak', 'unknown'] = 'unknown'


class ValidationReport(BaseModel):
    passed: bool
    failures: list[ValidationFailure] = Field(default_factory=list)
    retryable: bool = False
    next_action: ReviewAction = 'pass'
    required_reexecution: list[str] = Field(default_factory=list)

    @classmethod
    def pass_report(cls) -> 'ValidationReport':
        return cls(passed=True, retryable=False, next_action='pass')


class ResponseArtifact(BaseModel):
    answer: str
    route: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    public_citations: list[dict[str, Any]] = Field(default_factory=list)
    saved: bool = True
    response_type: str = 'answer'
    llm_used: bool = False
    llm_error: str | None = None
    report: str | None = None
    safe_fallback_used: bool = False


class MemoryArtifact(BaseModel):
    last_answer_memory: dict[str, Any] = Field(default_factory=dict)
    recent_turn_routes: list[dict[str, Any]] = Field(default_factory=list)
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    session_last_process_data: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AuditArtifact(BaseModel):
    persisted: bool = False
    warnings: list[str] = Field(default_factory=list)


class RuntimeArtifact(BaseModel):
    review_iteration: int = 0
    replan_count: int = 0
    rewrite_attempts: int = 0
    rag_rerun_attempts: int = 0
    safety_rerun_attempts: int = 0
    evidence_signatures: list[str] = Field(default_factory=list)
    previous_missing_gate_evidence: list[str] = Field(default_factory=list)
    quality_gate_reports: list[dict[str, Any]] = Field(default_factory=list)
    usage_records: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
