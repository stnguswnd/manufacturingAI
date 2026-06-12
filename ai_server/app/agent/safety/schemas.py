from __future__ import annotations

from typing import Literal, List

from pydantic import BaseModel, Field


class SafetyContext(BaseModel):
    must_include: List[str] = Field(default_factory=list)
    forbidden: List[str] = Field(default_factory=list)
    disclaimer_level: Literal['none', 'light', 'strict'] = 'light'
    requires_professional_review: bool = True
    allowed_scope: List[str] = Field(default_factory=list)
