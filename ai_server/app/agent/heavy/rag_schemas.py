from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] | None = None
    reason: str = ''


class EvidenceGrade(BaseModel):
    usable: bool
    usable_chunks: int = 0
    weak_reason: str | None = None
    relevance_label: str = 'unknown'

