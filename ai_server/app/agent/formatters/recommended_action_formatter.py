from __future__ import annotations

from typing import Any


class RecommendedActionFormatter:
    def format(self, context: dict[str, Any]) -> str:
        actions = [self._title(item) for item in (context.get('recommended_actions') or []) if self._title(item)]
        if not actions:
            return '직전 답변에서 정리할 권장조치를 찾지 못했습니다. 어떤 조치를 기준으로 정렬할지 다시 지정해 주세요.'
        lines = '\n'.join(f'{idx}. {item}' for idx, item in enumerate(actions[:10], start=1))
        return (
            '직전 답변의 권장조치를 중요한 순서대로 정리하면 다음과 같습니다.\n\n'
            f'{lines}\n\n'
            '이 순서는 일반적인 우선순위 정리이며, 실제 작업 순서는 현장 절차와 자격 있는 담당자의 판단을 따라야 합니다.'
        )

    @staticmethod
    def _title(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get('title') or '').strip()
        return str(item or '').strip()


class RecommendedActionItemFormatter:
    def format(self, context: dict[str, Any]) -> str:
        index = context.get('followup_item_index')
        action = context.get('recommended_action_item')
        title = action.get('title') if isinstance(action, dict) else str(action or '')
        if not action:
            return '직전 권장조치에서 해당 번호의 항목을 찾지 못했습니다. 다시 항목 번호를 지정해 주세요.'
        rationale = action.get('rationale') if isinstance(action, dict) else None
        safety_note = action.get('safety_note') if isinstance(action, dict) else None
        reason = rationale or '이 항목은 원인을 확정하기 전에 작업자 안전과 설비 손상 가능성을 먼저 낮추기 위해 필요합니다.'
        safety = f'\n\n안전 주의\n{safety_note}' if safety_note else ''
        return (
            f'직전 권장조치 {index}번은 "{title}"입니다.\n\n'
            f'{reason}'
            f'{safety}'
        )
