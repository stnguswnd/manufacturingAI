from __future__ import annotations

from typing import Any

from app.agent.formatters.fallback_formatter import FallbackFormatter
from app.agent.formatters.fast_concept_formatter import FastConceptFormatter
from app.agent.formatters.general_lightweight_formatter import GeneralLightweightFormatter
from app.agent.formatters.recommended_action_formatter import RecommendedActionFormatter, RecommendedActionItemFormatter
from app.agent.safety import SafetyContext, SafetyFormatter


class FormatterRegistry:
    def __init__(self):
        self._formatters = {
            'fast_concept_answer': FastConceptFormatter(),
            'general_lightweight_answer': GeneralLightweightFormatter(),
            'recommended_action_recap': RecommendedActionFormatter(),
            'recommended_action_item_explanation': RecommendedActionItemFormatter(),
            'clarification': FallbackFormatter(),
            'fallback': FallbackFormatter(),
        }
        self._safety_formatter = SafetyFormatter()

    def format(self, key: str, context: dict[str, Any]) -> str:
        if key == 'safety_answer':
            safety_context = context.get('safety_context')
            if isinstance(safety_context, SafetyContext):
                return self._safety_formatter.format(safety_context)
            return self._safety_formatter.format(SafetyContext.model_validate(safety_context or {}))
        formatter = self._formatters.get(key) or self._formatters['fallback']
        return formatter.format(context)
