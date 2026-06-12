from __future__ import annotations

from typing import Any


class FastConceptFormatter:
    def format(self, context: dict[str, Any]) -> str:
        payload = context['concept_payload']
        term = payload['term']
        risks = [str(item) for item in payload.get('risks') or []]
        watch_points = [str(item) for item in payload.get('watch_points') or []]
        related = [str(item) for item in payload.get('related_terms') or []]
        answer_type = context.get('answer_type') or 'definition'
        intro = ''
        if context.get('reference_note'):
            intro = f'{context["reference_note"]}\n\n'
        risk_title = '주의점 또는 단점' if answer_type == 'disadvantages' else '주의점'
        risk_body = ', '.join(risks) if risks else '현재 공정 조건에 따라 영향이 달라질 수 있습니다.'
        watch_body = '\n'.join(f'- {item}' for item in watch_points)
        if not watch_body:
            watch_body = f'- {term} 하나만 보지 말고 관련 지표와 추세를 함께 보세요.'
        related_body = ', '.join(related) if related else '관련 용어 없음'
        if answer_type == 'watch_points':
            middle = (
                f'{risk_title}\n'
                f'{term}를 볼 때는 값 하나만 보지 말고 함께 움직이는 지표를 같이 봐야 합니다.\n'
                f'{watch_body}\n\n'
                f'과도하거나 제어되지 않은 {term}는 {risk_body} 같은 문제와 연결될 수 있습니다.\n\n'
            )
        elif answer_type == 'related_terms':
            middle = (
                f'관련해서 같이 볼 용어\n'
                f'{related_body}\n\n'
                f'주의점\n'
                f'{term}를 해석할 때는 관련 용어와 현장 기준값을 함께 봐야 합니다.\n\n'
            )
        else:
            middle = (
                f'{risk_title}\n'
                f'{term} 자체가 항상 문제라는 뜻은 아닙니다. 다만 과도하거나 제어되지 않으면 {risk_body} 같은 문제가 생길 수 있습니다.\n\n'
            )
        return (
            f'{intro}'
            f'정의\n'
            f'{payload["definition"]}\n\n'
            f'제조 문맥에서의 의미\n'
            f'{payload["manufacturing_context"]}\n\n'
            f'{middle}'
            f'관련 용어\n'
            f'{related_body}\n\n'
            f'현재 설비 판단과의 경계\n'
            f'{payload["risk_boundary"]}'
        )
