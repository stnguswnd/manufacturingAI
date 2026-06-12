from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


GateCategory = Literal['scope_guard', 'safety_guard', 'report', 'diagnosis', 'document', 'concept', 'followup', 'meta', 'empty']


class GateContext(BaseModel):
    question: str
    original_question: str
    compact_question: str
    has_process_data: bool = False
    generate_report: bool = False
    context_resolution: dict[str, Any] = Field(default_factory=dict)
    last_answer_memory: dict[str, Any] = Field(default_factory=dict)
    glossary_hit: dict[str, Any] | None = None


class GateResult(BaseModel):
    matched: bool = False
    gate_name: str = ''
    selected_path: str | None = None
    answer_type: str | None = None
    reason: str = ''
    confidence: float = 0.0
    is_final: bool = False
    category: GateCategory = 'empty'
    turn_type: str | None = None
    requires_prediction: bool = False
    requires_rag: bool = False
    requires_safety: bool = False
    requires_report: bool = False
    resolved_reference: dict[str, Any] = Field(default_factory=dict)
    resolved_claim: str | None = None
    focus_update_policy: str = 'preserve'
