from __future__ import annotations

from app.agent.safety.schemas import SafetyContext


class SafetyPolicy:
    """Builds safety constraints after a route has already required safety."""

    def build_context(self, *, must_include: list[str] | None = None, forbidden: list[str] | None = None, strict: bool = False) -> SafetyContext:
        return SafetyContext(
            must_include=must_include or [
                '정비 전 에너지 차단 및 LOTO 확인',
                '회전부 접근 전 방호장치와 무에너지 상태 확인',
                '자격 있는 담당자 또는 안전관리자 확인',
            ],
            forbidden=forbidden or [
                'AI가 설비를 정지했다고 표현하기',
                'AI가 안전 상태를 보증한다고 표현하기',
                '잠금/표지 절차를 생략하라고 안내하기',
            ],
            disclaimer_level='strict' if strict else 'light',
            requires_professional_review=True,
            allowed_scope=['안전 확인 항목 안내', '현장 담당자 검토 필요 사항 정리'],
        )
