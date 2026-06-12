from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import NotRequired, TypedDict

from app.schemas import AgentPlan, AgentRequest, AgentResponse, AgentSendRequest, LLMUsageRecord, ManufacturingContext, PredictionResponse, RagChunk


SelectedPath = Literal[
    'fast_concept_answer',
    'general_lightweight_answer',
    'lightweight_rag_answer',
    'unsupported_or_clarification',
    'meta_feedback',
    'report_answer',
    'supervisor_planning',
    'recommended_action_recap',
    'recommended_action_item_explanation',
    'safety_answer',
    'rag_answer',
    'heavy_analysis_answer',
]


class AgentState(TypedDict):
    state_schema_version: int
    run_id: str
    user_id: str
    session_id: str
    thread_id: str
    current_user_message: str
    turn_context: NotRequired[Dict[str, Any]]
    recent_turns: NotRequired[List[Dict[str, Any]]]
    rolling_summary: NotRequired[str]
    send_request: AgentSendRequest
    request: NotRequired[AgentRequest]
    user_context: NotRequired[Dict[str, Any]]
    context_resolution: NotRequired[Dict[str, Any]]
    context_packs: NotRequired[Dict[str, Any]]
    compressed_context: NotRequired[Dict[str, Any]]
    recent_turn_routes: NotRequired[List[Dict[str, Any]]]
    last_answer_memory: NotRequired[Dict[str, Any]]
    context_validation_warnings: NotRequired[List[str]]
    turn_process_data: NotRequired[Optional[Dict[str, Any]]]
    previous_turn_process_data: NotRequired[Optional[Dict[str, Any]]]
    session_last_process_data: NotRequired[Optional[Dict[str, Any]]]
    process_data_reference_policy: NotRequired[Dict[str, Any]]
    intent_gateway: NotRequired[Dict[str, Any]]
    selected_path: NotRequired[SelectedPath]
    answer_type: NotRequired[str]
    structured_answer_payload: NotRequired[Dict[str, Any]]
    formatter_context: NotRequired[Dict[str, Any]]
    safety_context: NotRequired[Dict[str, Any]]
    plan: NotRequired[AgentPlan]
    route: NotRequired[List[str]]
    prediction: NotRequired[Optional[PredictionResponse]]
    manufacturing_context: NotRequired[ManufacturingContext]
    retrieved_documents: NotRequired[List[RagChunk]]
    safety_guidance: NotRequired[Optional[str]]
    answer: NotRequired[str]
    report: NotRequired[Optional[str]]
    citations: NotRequired[List[Dict[str, Any]]]
    llm_used: NotRequired[bool]
    llm_error: NotRequired[Optional[str]]
    response: NotRequired[AgentResponse]
    warnings: List[str]
    errors: List[Dict[str, Any]]
    usage_records: List[LLMUsageRecord]
    trace: List[Dict[str, Any]]
    replan_count: int
