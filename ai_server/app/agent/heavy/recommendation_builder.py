from __future__ import annotations

from typing import Any

from app.schemas import ManufacturingContext, PredictionResponse


class RecommendationBuilder:
    """Creates action phrases and structured action payloads."""

    def collect_action_phrases(self, prediction: PredictionResponse | None, manufacturing_context: ManufacturingContext) -> list[str]:
        actions = [action.output_phrase for action in manufacturing_context.action_plan if action.output_phrase]
        if prediction:
            actions.extend(prediction.recommended_actions)
        for gate in manufacturing_context.safety_gates:
            actions.append(f'{gate.name_ko}: ' + '; '.join(gate.required_checks[:3]))
        return list(dict.fromkeys(actions)) or ['추가 데이터와 관련 문서를 확인한 뒤 담당자가 점검 여부를 판단하세요.']

    def to_action_dicts(self, actions: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for idx, action in enumerate(actions or [], start=1):
            if isinstance(action, dict):
                title = str(action.get('title') or action.get('action') or action.get('text') or '').strip()
                if not title:
                    continue
                normalized.append({
                    'id': str(action.get('id') or f'action_{idx}'),
                    'title': title,
                    'rationale': action.get('rationale'),
                    'safety_note': action.get('safety_note'),
                    'priority': action.get('priority') or idx,
                })
            else:
                title = str(action or '').strip()
                if title:
                    normalized.append({
                        'id': f'action_{idx}',
                        'title': title,
                        'rationale': None,
                        'safety_note': None,
                        'priority': idx,
                    })
        return normalized
