from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class ProcessData(BaseModel):
    type: Literal['L', 'M', 'H'] = Field(default='L', description='AI4I Type: L/M/H')
    air_temperature_k: float = Field(ge=250, le=350)
    process_temperature_k: float = Field(ge=250, le=400)
    rotational_speed_rpm: int = Field(ge=1, le=10000)
    torque_nm: float = Field(ge=0, le=1000)
    tool_wear_min: int = Field(ge=0, le=100000)

class PredictionRequest(BaseModel):
    process_data: ProcessData

class EvidenceFeature(BaseModel):
    feature: str
    direction: Literal['high','low','normal']
    reason: str
    value: float | int | str | None = None
    tag: str | None = None

class FailureModeScore(BaseModel):
    code: str
    name: str
    probability: float
    predicted: bool

class PredictionResponse(BaseModel):
    failure_probability: float
    predicted_failure: bool
    risk_level: str
    failure_modes: list[FailureModeScore]
    predicted_modes: list[str]
    evidence_features: list[EvidenceFeature]
    recommended_actions: list[str]
    input_warnings: list[str] = []
    model_source: str
    disclaimer: str

class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] | None = None

class RagChunk(BaseModel):
    chunk_id: str
    source: str
    document_title: str
    text: str
    doc_type: str | None = None
    equipment_type: str | None = None
    section: str | None = None
    language: str | None = None
    url: str | None = None
    score: float | None = None

AgentMode = Literal['auto','prediction','knowledge_qa','safety_ops','documentation','hybrid']
AgentIntent = Literal['prediction','knowledge_qa','safety_ops','documentation','hybrid','general']
RiskLevel = Literal['low','medium','high','critical','unknown']
ResolvedTargetType = Literal['concept','equipment','component','failure_mode','process_condition','document','report','previous_answer','unknown']
QuestionKind = Literal['general_concept_followup','current_state_followup','document_followup','safety_followup','report_followup','comparison_followup','unknown_followup']

class ResolvedTarget(BaseModel):
    label: str
    type: ResolvedTargetType = 'unknown'
    source: str
    reference_run_id: str | None = None
    reference_session_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class ReferenceResolutionResult(BaseModel):
    original_question: str
    resolved_question: str
    resolved: bool = False
    resolved_target: ResolvedTarget | None = None
    question_kind: QuestionKind = 'unknown_followup'
    should_use_prediction: bool = False
    should_use_rag: bool = False
    should_use_safety: bool = False
    should_generate_report: bool = False
    clarification_question: str | None = None
    reason: str = ''
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class AgentRequest(BaseModel):
    user_id: str | None = Field(default=None, max_length=128)
    question: str = Field(default='', max_length=4000)
    process_data: ProcessData | None = None
    inspection_notes: str | None = Field(default=None, max_length=10000)
    generate_report: bool = False
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = Field(default=None, max_length=128)
    mode: AgentMode = 'auto'
    llm_model: str | None = Field(default=None, max_length=80)
    user_context: dict[str, Any] | None = None

class AgentSendRequest(BaseModel):
    """User-facing message API request.

    /agent/send is the recommended endpoint for frontend or external clients.
    It keeps the interface message-centric while still allowing process data,
    inspection notes, and report generation flags.
    """
    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(default='', max_length=4000, description='사용자 메시지 또는 질문')
    session_id: str | None = Field(default=None, max_length=128)
    process_data: ProcessData | None = None
    inspection_notes: str | None = Field(default=None, max_length=10000)
    generate_report: bool = False
    top_k: int = Field(default=5, ge=1, le=20)
    mode: AgentMode = 'auto'
    llm_model: str | None = Field(default=None, max_length=80)

class UserCreateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    role: str | None = Field(default=None, max_length=120)
    department: str | None = Field(default=None, max_length=120)
    preferred_language: str = Field(default='ko', max_length=20)
    report_style: str = Field(default='standard', max_length=80)

class UserUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    role: str | None = Field(default=None, max_length=120)
    department: str | None = Field(default=None, max_length=120)
    preferred_language: str | None = Field(default=None, max_length=20)
    report_style: str | None = Field(default=None, max_length=80)

class UserResponse(BaseModel):
    user_id: str
    display_name: str
    role: str | None = None
    department: str | None = None
    preferred_language: str = 'ko'
    report_style: str = 'standard'
    created_at: str
    updated_at: str
    deleted_at: str | None = None

class AgentTraceStep(BaseModel):
    step: str
    detail: str

class AgentLayer(BaseModel):
    name: str
    nodes: list[str]
    purpose: str

class AssetContext(BaseModel):
    equipment_type: str = 'GeneralMachine'
    equipment_label_ko: str = '일반 설비'
    inferred_subsystems: list[str] = []
    inferred_components: list[str] = []
    hazards: list[str] = []
    confidence: float = 0.0
    rationale: str = ''

class ProcessCondition(BaseModel):
    tag: str
    label_ko: str
    severity: RiskLevel = 'unknown'
    source_feature: str | None = None
    value: float | int | str | None = None
    explanation: str = ''

class FailureModeDetail(BaseModel):
    code: str
    name_ko: str
    description_ko: str
    confidence: float = 0.0
    related_features: list[str] = []
    related_subsystems: list[str] = []
    recommended_checks: list[str] = []
    safety_gates: list[str] = []
    source: str = 'catalog'

class RiskAxis(BaseModel):
    axis: str
    level: RiskLevel
    rationale: str

class RiskAssessment(BaseModel):
    quality: RiskAxis
    equipment: RiskAxis
    safety: RiskAxis
    production: RiskAxis
    document_confidence: RiskAxis
    overall_priority: RiskLevel
    escalation_required: bool = False
    rationale: str = ''

class SafetyGateResult(BaseModel):
    gate_id: str
    name_ko: str
    severity: RiskLevel
    description_ko: str
    required_checks: list[str]
    forbidden_agent_actions: list[str]
    escalation: str | None = None
    triggered_by: list[str] = []

class ActionStep(BaseModel):
    action_id: str
    label_ko: str
    description_ko: str
    output_phrase: str
    priority: RiskLevel = 'medium'
    related_failure_modes: list[str] = []
    related_subsystems: list[str] = []
    requires_machine_stop: bool = False
    requires_loto: bool | str = False
    requires_authorized_person: bool = True
    safety_gate_ids: list[str] = []

class ManufacturingContext(BaseModel):
    asset_context: AssetContext
    process_conditions: list[ProcessCondition] = []
    failure_modes: list[FailureModeDetail] = []
    risk_assessment: RiskAssessment
    safety_gates: list[SafetyGateResult] = []
    action_plan: list[ActionStep] = []
    document_search_terms: list[str] = []
    audit_notes: list[str] = []

class AgentPlan(BaseModel):
    intent: AgentIntent
    confidence: float = 0.0
    prediction_required: bool = False
    rag_required: bool = False
    safety_required: bool = False
    report_required: bool = False
    domain_context_required: bool = True
    asset_context_required: bool = True
    process_condition_required: bool = False
    failure_mode_required: bool = False
    risk_priority_required: bool = True
    safety_gate_required: bool = False
    action_plan_required: bool = True
    required_nodes: list[str] = []
    layers: list[AgentLayer] = []
    rag_query: str = ''
    rag_filters: dict[str, Any] | None = None
    document_scope: list[str] = []
    rationale: str = ''
    supervisor_source: Literal['deterministic','llm_refined'] = 'deterministic'

class LLMUsageRecord(BaseModel):
    provider: str = 'openai'
    model: str
    operation: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    estimated_cost_krw: float = 0.0
    usd_krw_exchange_rate: float = 0.0
    latency_ms: float = 0.0

class LLMUsageSummary(BaseModel):
    calls: int = 0
    replan_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    estimated_cost_krw: float = 0.0
    usd_krw_exchange_rate: float = 0.0
    records: list[LLMUsageRecord] = []

class AgentResponse(BaseModel):
    run_id: str
    user_id: str | None = None
    route: list[str]
    answer: str
    prediction: PredictionResponse | None = None
    manufacturing_context: ManufacturingContext | None = None
    retrieved_documents: list[RagChunk] = []
    safety_guidance: str | None = None
    report: str | None = None
    citations: list[dict[str, Any]] = []
    warnings: list[str] = []
    trace: list[AgentTraceStep] = []
    saved: bool = False
    session_id: str | None = None
    plan: AgentPlan | None = None
    llm_used: bool = False
    llm_provider: str = 'openai'
    llm_model: str | None = None
    llm_usage: LLMUsageSummary | None = None
    llm_error: str | None = None
    context_used: dict[str, Any] | None = None

class EvaluationRequest(BaseModel):
    agent_answer: str
    expected_contract: dict[str, Any]
    route: list[str] | None = None
    manufacturing_context: ManufacturingContext | None = None

class EvaluationResponse(BaseModel):
    scores: dict[str, float]
    weighted_total: float
    passed: bool
    comments: list[str]
