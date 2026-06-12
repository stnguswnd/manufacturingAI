from __future__ import annotations

from typing import Any


class FallbackFormatter:
    def format(self, context: dict[str, Any]) -> str:
        reason = context.get('public_reason') or '요청 의도나 참조 대상을 안정적으로 확정하지 못했습니다.'
        info = context.get('missing_info') or '대상 설비, 개념, 문서, 또는 현재 공정 데이터를 더 구체적으로 지정해 주세요.'
        return (
            '바로 판단하기 어려운 이유\n'
            f'{reason}\n\n'
            '필요한 추가 정보\n'
            f'{info}\n\n'
            '안전/범위 제한\n'
            '이 Agent는 설비 제어, 안전 보증, 법적 최종 판단을 수행하지 않습니다.'
        )
