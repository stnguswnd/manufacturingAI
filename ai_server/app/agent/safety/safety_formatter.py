from __future__ import annotations

from app.agent.safety.schemas import SafetyContext


class SafetyFormatter:
    def format(self, context: SafetyContext) -> str:
        checks = '\n'.join(f'- {item}' for item in context.must_include) or '- 현장 안전 절차와 담당자 확인'
        forbidden = '\n'.join(f'- {item}' for item in context.forbidden) or '- AI가 안전 상태를 보증했다고 표현하지 않기'
        review = ''
        if context.requires_professional_review:
            review = '\n\n전문가 확인\n정비, 분해, 설비 접근, 법적 안전 판단은 자격 있는 담당자와 현장 절차를 따라야 합니다.'
        return (
            '안전 확인\n'
            f'{checks}\n\n'
            '금지/제한 사항\n'
            f'{forbidden}'
            f'{review}'
        )
