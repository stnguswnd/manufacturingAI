from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


FollowupType = Literal[
    'none',
    'previous_answer_reason',
    'previous_recommended_actions',
    'previous_recommended_action_item',
    'previous_code',
    'previous_table',
    'previous_decision',
    'previous_claim',
    'previous_concept',
    'previous_source',
    'ambiguous',
]


class ContextResolution(BaseModel):
    is_followup: bool
    followup_type: FollowupType = 'none'
    followup_target: Optional[str] = None
    followup_item_index: Optional[int] = None
    standalone_query: str
    context_needed: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str


class RecommendedAction(BaseModel):
    id: str
    title: str
    rationale: Optional[str] = None
    safety_note: Optional[str] = None
    priority: Optional[int] = None


class AnswerMemory(BaseModel):
    selected_path: str
    answer_type: Optional[str] = None
    user_intent: Optional[str] = None
    short_summary: str
    focus: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)
    recommended_actions: List[RecommendedAction] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    code_changes: List[str] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    mentioned_entities: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)
    safety_level: Optional[str] = None
    created_at: Optional[str] = None
    expires_after_turns: int = 5


class ContextPacks(BaseModel):
    classifier_context: Dict[str, Any] = Field(default_factory=dict)
    answer_context: Dict[str, Any] = Field(default_factory=dict)
    rag_context: Dict[str, Any] = Field(default_factory=dict)
    safety_context: Dict[str, Any] = Field(default_factory=dict)
    formatter_context: Dict[str, Any] = Field(default_factory=dict)
    memory_writer_context: Dict[str, Any] = Field(default_factory=dict)


class CompressedContext(BaseModel):
    rolling_summary: str = ''
    recent_turns: List[Dict[str, str]] = Field(default_factory=list)
    recent_turn_count: int = 0
    compressed_message_count: int = 0
    max_recent_turns: int = 5


class FallbackReason(BaseModel):
    category: Literal[
        'intent_uncertain',
        'schema_unavailable',
        'missing_context',
        'retrieval_failed',
        'safety_limited',
        'tool_unavailable',
        'unknown',
    ] = 'unknown'
    public_reason: str
    internal_reason: Optional[str] = None
