from __future__ import annotations

from typing import Any


class GeneralLightweightFormatter:
    def format(self, context: dict[str, Any]) -> str:
        answer_type = context.get('answer_type') or 'explanation'
        target = context.get('target') or context.get('followup_target') or '그 대상'
        raw_claim = context.get('resolved_claim') or {}
        claim_text = raw_claim.get('claim') if isinstance(raw_claim, dict) else str(raw_claim or '')
        phrase_repair = context.get('phrase_repair') or {}

        if answer_type == 'chart_guidance':
            return (
                '도표나 그래프를 고를 때는 먼저 보고 싶은 목적을 정해야 합니다.\n\n'
                '시간에 따른 변화를 보려면 선 그래프가 적합하고, 구간별 크기를 비교하려면 막대 그래프가 낫습니다. '
                '여러 지표가 같이 움직이는지 보려면 같은 시간축에 토크, 회전수, 온도, 마모 같은 값을 함께 놓고 추세를 비교하는 방식이 좋습니다.\n\n'
                '다만 그래프는 판단을 돕는 도구일 뿐이고, 실제 설비 이상 여부는 공정 데이터와 현장 기준값을 함께 확인해야 합니다.'
            )

        if answer_type == 'rationale':
            intro = ''
            if phrase_repair:
                surface = phrase_repair.get('surface_text') or '현재 표현'
                resolved = phrase_repair.get('resolved_phrase') or '직전 답변의 표현'
                intro = f'여기서 "{surface}"은 직전 답변의 "{resolved}"을 말한 것으로 보고 답변하겠습니다.\n\n'
            elif claim_text:
                intro = f'직전 답변의 "{claim_text}"라는 설명에 대한 이유로 답변하겠습니다.\n\n'
            return (
                f'{intro}'
                f'{target}를 볼 때 여러 지표를 함께 보는 이유는, 한 가지 값만으로는 원인을 확정하기 어렵기 때문입니다.\n\n'
                f'예를 들어 {target}가 커지거나 나빠져도 그 원인이 항상 하나로 정해지는 것은 아닙니다. '
                '토크 상승, 회전수 조건 변화, 공구 고정 상태, 냉각 상태, 소재 변화, 진동이나 소음 변화가 비슷한 현상을 만들 수 있습니다.\n\n'
                '그래서 토크, 회전수, 진동, 온도, 표면 품질, 공구 사용 시간 같은 지표를 함께 봅니다. '
                '여러 지표가 같은 방향으로 변하면 단일 값의 착각을 줄이고 원인을 더 신뢰 있게 좁힐 수 있습니다.\n\n'
                '즉, 여러 지표를 같이 보는 이유는 변화의 원인을 더 정확히 구분하기 위해서입니다.'
            )

        return (
            '이 질문은 현재 설비 상태 판단이라기보다 일반 설명 질문으로 보입니다.\n\n'
            '간단히 말하면, 변화가 있는 데이터를 함께 보는 이유는 흐름, 원인, 예외를 더 빨리 파악하기 위해서입니다. '
            '한 시점의 숫자만 보면 우연한 튐인지 실제 변화인지 구분하기 어렵지만, 관련 값들을 함께 보면 패턴과 원인을 좁힐 수 있습니다.\n\n'
            '현재 설비의 위험 여부는 실제 공정 데이터와 현장 기준값이 있어야 판단할 수 있습니다.'
        )
