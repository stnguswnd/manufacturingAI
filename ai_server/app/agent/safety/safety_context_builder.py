from __future__ import annotations

from app.agent.safety.safety_policy import SafetyPolicy
from app.agent.safety.schemas import SafetyContext


class SafetyContextBuilder:
    def __init__(self, policy: SafetyPolicy | None = None):
        self.policy = policy or SafetyPolicy()

    def build(self, *, safety_guidance: str | None = None, strict: bool = False) -> SafetyContext:
        must_include = None
        if safety_guidance:
            must_include = [line.strip() for line in safety_guidance.splitlines() if line.strip()]
        return self.policy.build_context(must_include=must_include, strict=strict)
